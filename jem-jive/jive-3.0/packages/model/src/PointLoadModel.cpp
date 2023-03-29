
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


#include <jem/base/assert.h>
#include <jem/base/Array.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Properties.h>
#include <jive/util/Table.h>
#include <jive/util/TableUtils.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/model/Names.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/PointLoadModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::PointLoadModel );


JIVE_BEGIN_PACKAGE( model )


using jem::newInstance;


//=======================================================================
//   class PointLoadModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  PointLoadModel::TYPE_NAME = "PointLoad";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PointLoadModel::PointLoadModel ( const String& name ) :

  Super ( name )

{}


PointLoadModel::PointLoadModel

  ( const String&         name,
    const Ref<DofSpace>&  dofs,
    const Ref<Table>&     loads ) :

    Super  (  name ),
    dofs_  (  dofs ),
    loads_ ( loads )

{
  JEM_PRECHECK ( dofs  &&
                 loads &&
                 dofs->getItems() == loads->getRowItems() );

  tname_ = loads->getName ();
}


PointLoadModel::~PointLoadModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void PointLoadModel::readFrom ( ObjectInput& in )
{
  decode ( in,
           myName_,
           dofs_,
           loads_,
           tname_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void PointLoadModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           dofs_,
           loads_,
           tname_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void PointLoadModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties  myProps = props.findProps ( myName_ );

  String      tname;

  if ( myProps.find( tname, PropNames::LOAD_TABLE ) )
  {
    String        context = getContext ();

    Ref<ItemSet>  items;

    String        name;
    String        path;


    if ( tname.size() == 0 )
    {
      loads_ = nullptr;
      tname_ = "";
      return;
    }

    idx_t  i = tname.find ( '/' );

    if ( i < 0 )
    {
      dofs_ = DofSpace::get   ( globdat, context );
      items = dofs_->getItems ();
      name  = tname;
    }
    else
    {
      path  = tname[slice(BEGIN,i)];
      name  = tname[slice(i + 1,END)];

      items = ItemSet ::get ( path,  globdat, context );
      dofs_ = DofSpace::get ( items, globdat, context );
    }

    loads_ = Table::get ( name, items, globdat, context );
    tname_ = tname;
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void PointLoadModel::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::LOAD_TABLE, tname_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool PointLoadModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::TableUtils;


  if ( ! loads_ )
  {
    return false;
  }

  if ( action == Actions::GET_EXT_VECTOR )
  {
    const idx_t  tableSize = loads_->size ();

    Vector       loads ( tableSize );
    IdxVector    idofs ( tableSize );

    Vector       fext;
    double       scale;


    scale = 1.0;

    params.find ( scale, ActionParams::SCALE_FACTOR );
    params.get  ( fext,  ActionParams::EXT_VECTOR   );

    TableUtils::toVector ( loads, idofs, *loads_, *dofs_ );

    fext[idofs] += scale * loads;

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> PointLoadModel::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void PointLoadModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( model )
