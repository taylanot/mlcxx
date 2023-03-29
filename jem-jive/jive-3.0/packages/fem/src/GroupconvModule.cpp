
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


#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jive/geom/ShapeTable.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/import.h>
#include <jive/fem/BndGroupConverter.h>
#include <jive/fem/ElemGroupConverter.h>
#include <jive/fem/GroupconvModule.h>


JEM_DEFINE_CLASS( jive::fem::GroupconvModule );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class GroupconvModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GroupconvModule::TYPE_NAME = "Groupconv";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GroupconvModule::GroupconvModule ( const String& name ) :

  Super ( name )

{}


GroupconvModule::~GroupconvModule ()
{}


//-----------------------------------------------------------------------
//   newConverter
//-----------------------------------------------------------------------


Ref<app::GroupConverter> GroupconvModule::newConverter

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const String&      context )

{
  using jem::System;


  BoundarySet  bounds = BoundarySet::find ( globdat );

  if ( bounds )
  {
    ElementSet       elems = bounds.getElements ();

    Ref<ShapeTable>  st =

      ShapeTable::find ( elems.getData(), globdat );


    if ( st )
    {
      return newInstance<BndGroupConverter> ( bounds,
                                              st,
                                              globdat,
                                              context );
    }
    else
    {
      print ( System::out(),
              "Not converting boundary groups because "
              "no shape table has been defined\n" );
    }
  }

  ElementSet  elems = ElementSet::find ( globdat );

  if ( elems )
  {
    return newInstance<ElemGroupConverter> ( elems,
                                             globdat,
                                             context );
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> GroupconvModule::makeNew

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


void GroupconvModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   newConverter_
//-----------------------------------------------------------------------


Ref<app::GroupConverter> GroupconvModule::newConverter_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newConverter ( conf, props, globdat, getContext() );
}


JIVE_END_PACKAGE( fem )
