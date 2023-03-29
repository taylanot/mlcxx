
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


#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/util/Assignable.h>
#include <jive/geom/ShapeTable.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/XBoundarySet.h>
#include <jive/fem/BoundaryManager.h>
#include <jive/fem/BndgenModule.h>


JEM_DEFINE_CLASS( jive::fem::BndgenModule );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class BndgenModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BndgenModule::TYPE_NAME = "Bndgen";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


BndgenModule::BndgenModule ( const String& name ) :

  Super ( name )

{}


BndgenModule::~BndgenModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status BndgenModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::getStorageMode;
  using jive::util::StorageMode;
  using jive::util::Assignable;


  Assignable
    <XBoundarySet>   bounds  = XBoundarySet::find ( globdat );
  Assignable
    <BoundaryGroup>  igroup;
  Assignable
    <BoundaryGroup>  xgroup;
  Assignable
    <ElementSet>     elems;

  const String       context = getContext ();

  Properties         myConf  = conf .makeProps ( myName_ );
  Properties         myProps = props.findProps ( myName_ );

  Ref<ShapeTable>    st;


  bndMan_ = nullptr;

  if ( ! bounds )
  {
    StorageMode  smode = util::COMPACT_STORAGE;

    getStorageMode ( smode,   conf,   props );
    getStorageMode ( smode, myConf, myProps );

    elems  = ElementSet::get ( globdat, context );
    bounds = newXBoundarySet ( elems,   smode   );

    bounds.store ( globdat );
  }
  else
  {
    elems  = bounds.getElements  ();
    igroup = BoundaryGroup::find ( "internal", bounds, globdat );
    xgroup = BoundaryGroup::find ( "external", bounds, globdat );
  }

  if ( igroup == nullptr || xgroup == nullptr )
  {
    st = ShapeTable::get ( elems.getData(), globdat, context );

    Ref<BoundaryManager>  bman =

      newInstance<BoundaryManager> ( bounds, st );

    if ( ! igroup )
    {
      igroup = bman->getIBoundaries ();

      igroup.store ( "internal", globdat );
    }

    if ( ! xgroup )
    {
      xgroup = bman->getXBoundaries ();

      xgroup.store ( "external", globdat );
    }

    bndMan_.swap ( bman );

    return OK;
  }

  return DONE;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status BndgenModule::run ( const Properties& globdat )
{
  if ( ! bndMan_ )
  {
    return DONE;
  }
  else
  {
    bndMan_->update ();

    return OK;
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void BndgenModule::shutdown ( const Properties& globdat )
{
  bndMan_ = nullptr;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> BndgenModule::makeNew

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


void BndgenModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( fem )
