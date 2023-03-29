
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/limits.h>
#include <jem/base/Class.h>
#include <jem/util/WeakHashMap.h>
#include <jem/util/MapEnumerator.h>
#include <jive/util/import.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include "private/TempData.h"


JIVE_BEGIN_PACKAGE( util )


using jem::dynamicCast;
using jem::checkedCast;
using jem::newInstance;
using jem::Class;
using jem::util::ObjectMap;


//=======================================================================
//   class Globdat
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Globdat::CONSTRAINTS        = "constraints";
const char*  Globdat::CONSTRAINTS_PARSER = "temp.conParser";
const char*  Globdat::DATABASES          = "databases";
const char*  Globdat::DOF_SPACE          = "dofSpace";
const char*  Globdat::FUNCTIONS          = "functions";
const char*  Globdat::ITEM_GROUPS        = "itemGroups";
const char*  Globdat::ITEM_SETS          = "itemSets";
const char*  Globdat::RANDOM             = "var.rand";
const char*  Globdat::RELATED_DATA       = "relatedData";
const char*  Globdat::RUNTIME            = "var.runtime";
const char*  Globdat::TABLES             = "tables";
const char*  Globdat::TEMP               = "temp";
const char*  Globdat::TIME               = "var.time";
const char*  Globdat::TIME_STEP          = "var.timeStep";
const char*  Globdat::OLD_TIME           = "var.oldTime";
const char*  Globdat::OLD_TIME_STEP      = "var.oldTimeStep";
const char*  Globdat::VARIABLES          = "var";
const char*  Globdat::VECTORS            = "vectors";


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void Globdat::init ( const Properties& globdat )
{
  globdat.set ( TEMP, jem::newInstance<TempData>() );
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Properties Globdat::dup ( const Properties& globdat )
{
  using jem::util::mergeNoReplace;
  using jem::util::MapEnum;

  Ref<Dict>       globdict2 = globdat.getContents ();
  Ref<ObjectMap>  relmap;


  globdict2 = checkedCast<Dict>

    ( globdict2->getClass()->newInstance() );

  globdict2->insert ( TEMP, jem::newInstance<TempData>() );

  if ( globdat.find( relmap, RELATED_DATA ) )
  {
    Ref<MapEnum>    e       = relmap->enumerate ();
    Ref<ObjectMap>  relmap2 = checkedCast<ObjectMap>

      ( relmap->getClass()->newInstance() );

    relmap2->reserve ( relmap->size() );

    for ( ; ! e->atEnd(); e->toNext() )
    {
      Ref<Object>  key = e->getKey ();

      if ( ! key )
      {
        continue;
      }

      Ref<Dict>  reldict = dynamicCast<Dict> ( e->getValue() );

      if ( reldict == nullptr || reldict->size() == 0 )
      {
        continue;
      }

      Properties  reldat2 =

        dup ( Properties( reldict, RELATED_DATA ) );

      relmap2->insert ( key, reldat2.getContents() );
    }

    globdict2->insert ( RELATED_DATA, relmap2 );
  }

  Properties  globdat2 ( globdict2, globdat.getName() );

  mergeNoReplace ( globdat2, globdat );

  return globdat2;
}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


Properties Globdat::newInstance ( const String& name )
{
  Properties  globdat ( name );

  init ( globdat );

  return globdat;
}


//-----------------------------------------------------------------------
//   newTempData
//-----------------------------------------------------------------------


Properties Globdat::newTempData ()
{
  return Properties ( jem::newInstance<TempData>(), TEMP );
}


//-----------------------------------------------------------------------
//   getVariables
//-----------------------------------------------------------------------


Properties Globdat::getVariables ( const Properties& globdat )
{
  return globdat.makeProps ( VARIABLES );
}


Properties Globdat::getVariables

  ( const String&      name,
    const Properties&  globdat )

{
  Properties  vars = globdat.makeProps ( VARIABLES );

  return vars.makeProps ( name );
}


//-----------------------------------------------------------------------
//   findVariables
//-----------------------------------------------------------------------


Properties Globdat::findVariables

  ( const String&      name,
    const Properties&  globdat )

{
  Properties  vars = globdat.findProps ( VARIABLES );

  return vars.findProps ( name );
}


//-----------------------------------------------------------------------
//   hasVariable
//-----------------------------------------------------------------------


bool Globdat::hasVariable

  ( const String&      name,
    const Properties&  globdat )

{
  Properties  vars = globdat.findProps ( VARIABLES );

  return vars.contains ( name );
}


//-----------------------------------------------------------------------
//   findDataFor
//-----------------------------------------------------------------------


Properties Globdat::findDataFor

  ( const Ref<Object>&  context,
    const Properties&   globdat )

{
  JEM_PRECHECK2 ( context, "NULL context object" );

  Ref<ObjectMap>  relmap;

  if ( globdat.find( relmap, RELATED_DATA ) )
  {
    Ref<Dict>  reldict =

      dynamicCast<Dict> ( relmap->get( context ) );

    if ( reldict )
    {
      String  name = joinNames ( globdat.getName(), RELATED_DATA );

      return Properties ( reldict, name );
    }
  }

  return Properties ();
}


//-----------------------------------------------------------------------
//   getDataFor
//-----------------------------------------------------------------------


Properties Globdat::getDataFor

  ( const Ref<Object>&  context,
    const Properties&   globdat )

{
  using jem::util::WeakHashMap;

  JEM_PRECHECK2 ( context, "NULL context object" );

  Ref<ObjectMap>  relmap;

  if ( ! globdat.find( relmap, RELATED_DATA ) )
  {
    relmap = jem::newInstance<WeakHashMap> ();

    globdat.set ( RELATED_DATA, relmap );
  }

  Ref<Dict>  reldict =

    dynamicCast<Dict> ( relmap->get( context ) );

  if ( ! reldict )
  {
    reldict = globdat.getContents ();
    reldict = checkedCast<Dict>

      ( reldict->getClass()->newInstance() );

    reldict->insert ( TEMP, jem::newInstance<TempData>() );
    relmap ->insert ( context, reldict );
  }

  String  name = joinNames ( globdat.getName(), RELATED_DATA );

  return Properties ( reldict, name );
}


//-----------------------------------------------------------------------
//   eraseDataFor
//-----------------------------------------------------------------------


void Globdat::eraseDataFor

  ( const Ref<Object>&  context,
    const Properties&   globdat )

{
  JEM_PRECHECK2 ( context, "NULL context object" );

  Ref<ObjectMap>  relmap;

  if ( globdat.find( relmap, RELATED_DATA ) )
  {
    relmap->erase ( context );
  }
}


//-----------------------------------------------------------------------
//   initTime
//-----------------------------------------------------------------------


void Globdat::initTime ( const Properties& globdat )
{
  double  t = 0.0;

  if ( ! globdat.find( t, Globdat::TIME ) )
  {
    globdat.set ( Globdat::TIME, t );
  }

  globdat.set ( Globdat::OLD_TIME, t );
}


//-----------------------------------------------------------------------
//   advanceTime
//-----------------------------------------------------------------------


void Globdat::advanceTime

  ( double             delta,
    const Properties&  globdat )

{
  double  t, t0;

  globdat.get ( t0, Globdat::OLD_TIME );

  t = t0 + delta;

  globdat.set ( Globdat::TIME, t );
}


//-----------------------------------------------------------------------
//   restoreTime
//-----------------------------------------------------------------------


void Globdat::restoreTime ( const Properties& globdat )
{
  double  t0;

  globdat.get ( t0, Globdat::OLD_TIME );
  globdat.set ( Globdat::TIME, t0 );
}


//-----------------------------------------------------------------------
//   commitTime
//-----------------------------------------------------------------------


void Globdat::commitTime ( const Properties& globdat )
{
  double  t;

  globdat.get ( t, Globdat::TIME );
  globdat.set ( Globdat::OLD_TIME, t );
}


//-----------------------------------------------------------------------
//   initStep
//-----------------------------------------------------------------------


void Globdat::initStep ( const Properties& globdat )
{
  idx_t  i = 0;

  if ( ! globdat.find( i, Globdat::TIME_STEP ) )
  {
    globdat.set ( Globdat::TIME_STEP, i );
  }

  globdat.set ( Globdat::OLD_TIME_STEP, i );
}


//-----------------------------------------------------------------------
//   advanceStep
//-----------------------------------------------------------------------


void Globdat::advanceStep ( const Properties&  globdat )
{
  idx_t  i, i0;

  globdat.get ( i0, Globdat::OLD_TIME_STEP );

  i = i0;

  if ( i < jem::maxOf( i ) )
  {
    i++;
  }

  globdat.set ( Globdat::TIME_STEP, i );
}


//-----------------------------------------------------------------------
//   restoreStep
//-----------------------------------------------------------------------


void Globdat::restoreStep ( const Properties& globdat )
{
  idx_t  i0;

  globdat.get ( i0, Globdat::OLD_TIME_STEP );
  globdat.set ( Globdat::TIME_STEP, i0 );
}


//-----------------------------------------------------------------------
//   commitStep
//-----------------------------------------------------------------------


void Globdat::commitStep ( const Properties& globdat )
{
  idx_t  i;

  globdat.get ( i, Globdat::TIME_STEP );
  globdat.set ( Globdat::OLD_TIME_STEP, i );
}


JIVE_END_PACKAGE( util )
