
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
#include <jive/util/utilities.h>
#include <jive/app/ModuleFactory.h>
#include <jive/gl/declare.h>
#include <jive/gl/MeshView.h>
#include <jive/gl/MarkerManager.h>
#include <jive/gl/PaletteManager.h>
#include <jive/gl/FemViewModule.h>


JEM_DEFINE_CLASS( jive::gl::FemViewModule );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;


//=======================================================================
//   class FemViewModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  FemViewModule::TYPE_NAME = "FemView";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FemViewModule::FemViewModule

  ( const String&  myName,
    int            mode ) :

    Super ( myName, mode )

{
  using jive::util::joinNames;

  Ref<MeshView>  mview;

  String         name;


  name = joinNames ( myName, "markerStyles" );

  addPlugin ( newInstance<MarkerManager> ( name ) );

  name = joinNames ( myName, "palettes" );

  addPlugin ( newInstance<PaletteManager> ( name ) );

  name  = joinNames ( myName, "mesh" );
  mview = newInstance<MeshView> ( name );

  mview->addUserPlugin ();

  addPlugin       ( mview );
  addUserPlugin   ();
  declareDataSets ();
  declarePlugins  ();
}


FemViewModule::~FemViewModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status FemViewModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return Super::init ( conf, props, globdat );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> FemViewModule::makeNew

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


void FemViewModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( gl )
