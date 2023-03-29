
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
#include <jive/util/utilities.h>
#include <jive/geom/ShapeTable.h>
#include <jive/geom/ShapeTableFactory.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/declare.h>
#include <jive/fem/Names.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/AutoShapeTable.h>
#include <jive/fem/ShapeModule.h>


JEM_DEFINE_CLASS( jive::fem::ShapeModule );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class ShapeModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ShapeModule::TYPE_NAME = "FEShape";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ShapeModule::ShapeModule ( const String& name ) :

  Super ( name )

{}


ShapeModule::~ShapeModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status ShapeModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;
  using jive::geom::ShapeTableFactory;


  ElementSet  elems = ElementSet::find ( globdat );

  String      name  = joinNames ( myName_, PropNames::SHAPE_TABLE );


  if ( elems )
  {
    Ref<ShapeTable>  st =

      ShapeTable::find ( elems.getData(), globdat );

    if ( ! st )
    {
      declareShapeTables ();

      if ( props.contains( name ) )
      {
        st = ShapeTableFactory::newInstance ( name,
                                              conf,
                                              props,
                                              elems.getData(),
                                              globdat );
      }
      else
      {
        conf.makeProps(name).set ( "type", "Auto" );

        st = AutoShapeTable::makeNew ( name,
                                       conf,
                                       Properties(),
                                       elems.getData(),
                                       globdat );

        // Shape table may be NIL if the rank of the mesh is
        // not supported

        if ( ! st )
        {
          return DONE;
        }
      }

      st->store ( globdat );
    }

    st->configure ( props, globdat );
    st->getConfig ( conf );
  }

  return DONE;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ShapeModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  ElementSet  elems = ElementSet::find ( globdat );

  if ( elems )
  {
    Ref<ShapeTable>  st =

      ShapeTable::find ( elems.getData(), globdat );

    if ( st )
    {
      st->configure ( props, globdat );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ShapeModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  ElementSet  elems = ElementSet::find ( globdat );

  if ( elems )
  {
    Ref<ShapeTable>  st =

      ShapeTable::find ( elems.getData(), globdat );

    if ( st )
    {
      st->getConfig ( conf );
    }
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> ShapeModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return jem::newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void ShapeModule::declare ()
{
  using jive::app::ModuleFactory;


  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( fem )
