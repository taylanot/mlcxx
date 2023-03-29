
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


#include <cctype>
#include <jem/base/assert.h>
#include <jem/base/StringBuffer.h>
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jive/util/CtorMap.h>
#include <jive/geom/Names.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/StdShapeFactory.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::util::Pattern;
using jive::util::CtorMap;


//=======================================================================
//   class StdShapeFactory::CtorMap_
//=======================================================================


class StdShapeFactory::CtorMap_ : public CtorMap<Constructor>
{};


//=======================================================================
//   class StdShapeFactory
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char  StdShapeFactory::GEOM_SEPARATOR_ = '#';


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdShapeFactory::declare

  ( const String&  type,
    const String&  geom,
    Constructor    ctor )

{
  JEM_PRECHECK2 ( ctor, "NULL construction function pointer" );

  CtorMap_::insert ( makeKey_( type, geom ), ctor );
}


//-----------------------------------------------------------------------
//   exists
//-----------------------------------------------------------------------


bool StdShapeFactory::exists

  ( const String&  type,
    const String&  geom )

{
  if ( CtorMap_::contains( makeKey_( type, geom ) ) )
  {
    return true;
  }
  else
  {
    return CtorMap_::contains ( makeKey_( type, String( '*' ) ) );
  }
}


//-----------------------------------------------------------------------
//   listKnownTypes
//-----------------------------------------------------------------------


StringVector StdShapeFactory::listKnownTypes

  ( const String& geom )

{
  StringVector  mangledTypes = CtorMap_::listKnownTypes ();

  const idx_t   n = mangledTypes.size ();

  StringVector  typeList ( n );
  String        suffix;
  String        type;
  idx_t         i, j, k;


  for ( i = j = 0; i < n; i++ )
  {
    k = mangledTypes[i].rfind ( GEOM_SEPARATOR_ );

    if ( k >= 0 )
    {
      type   = mangledTypes[i][slice(BEGIN,k)];
      suffix = mangledTypes[i][slice(k + 1,END)];
    }
    else
    {
      type   = mangledTypes[i];
      suffix = "";
    }

    if ( geom.equalsIgnoreCase( suffix ) ||
         suffix == "*" || geom == "*" )
    {
      typeList[j++] = type;
    }
  }

  if ( j < n )
  {
    return typeList[slice(BEGIN,j)].clone ();
  }
  else
  {
    return typeList;
  }
}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


Ref<StdShape> StdShapeFactory::newInstance

  ( const String&      type,
    const String&      geom,
    const Properties&  conf,
    const Properties&  props )
{
  Constructor ctor = CtorMap_::find ( makeKey_( type, geom ) );

  if ( ! ctor )
  {
    ctor = CtorMap_::find ( makeKey_( type, String( '*' ) ) );
  }

  if ( ctor )
  {
    return ctor ( geom, conf, props );
  }
  else
  {
    return nullptr;
  }
}


Ref<StdShape> StdShapeFactory::newInstance

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  Ref<StdShape>  shape;
  String         type;


  props.get ( type, TYPE_PROP );
  conf .set ( TYPE_PROP, type );

  shape = newInstance ( type, geom, conf, props );

  if ( ! shape )
  {
    noSuchTypeError (
      props.getContext ( TYPE_PROP ),
      type,
      String::format ( "shape funcs (%s)", geom ),
      listKnownTypes ( geom )
    );
  }

  return shape;
}


//-----------------------------------------------------------------------
//   findShapeFuncs
//-----------------------------------------------------------------------


Ref<StdShape> StdShapeFactory::findShapeFuncs

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props,
    const String&      defaultType,
    idx_t              rank )

{
  Properties     sfConf = conf.makeProps ( PropNames::SHAPE_FUNCS );
  Properties     sfProps;

  Ref<StdShape>  sfuncs;


  if ( props.find( sfProps, PropNames::SHAPE_FUNCS ) )
  {
    sfuncs = newInstance ( geom, sfConf, sfProps );

    if ( rank >= 0 && sfuncs->rank() != rank )
    {
      props.propertyError (
        PropNames::SHAPE_FUNCS,
        String::format (
          "shape functions have rank %d but should have rank %d",
          sfuncs->rank (),
          rank
        )
      );
    }
  }
  else
  {
    sfConf.set ( TYPE_PROP, defaultType );
  }

  return sfuncs;
}


//-----------------------------------------------------------------------
//   makeKey_
//-----------------------------------------------------------------------


String StdShapeFactory::makeKey_

  ( const String&  type,
    const String&  geom )

{
  using jem::StringBuffer;

  StringBuffer  buf;

  buf.reserve ( type.size() + 1 + geom.size() );

  buf += type;
  buf += GEOM_SEPARATOR_;
  buf += geom.toUpper ();

  return buf.toString ();
}


JIVE_END_PACKAGE( geom )
