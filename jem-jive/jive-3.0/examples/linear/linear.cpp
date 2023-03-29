
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


#include <jive/app/ChainModule.h>
#include <jive/app/CheckpointModule.h>
#include <jive/app/RootOnlyModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/OutputModule.h>
#include <jive/app/Application.h>
#include <jive/app/InfoModule.h>
#include <jive/fem/declare.h>
#include <jive/fem/InputModule.h>
#include <jive/fem/MPInputModule.h>
#include <jive/fem/PartitionModule.h>
#include <jive/fem/ShapeModule.h>
#include <jive/fem/InitModule.h>
#include <jive/mesh/MeshgenModule.h>
#include <jive/model/declare.h>
#include <jive/femodel/declare.h>
#include <jive/implict/LinsolveModule.h>


using namespace jem;

using jive::app::rootOnly;
using jive::app::Application;
using jive::app::Module;
using jive::app::ChainModule;
using jive::app::CheckpointModule;
using jive::app::ControlModule;
using jive::app::ReportModule;
using jive::app::InfoModule;
using jive::app::OutputModule;
using jive::fem::InputModule;
using jive::fem::MPInputModule;
using jive::fem::PartitionModule;
using jive::fem::ShapeModule;
using jive::fem::InitModule;
using jive::mesh::MeshgenModule;
using jive::implict::LinsolveModule;


//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------


Ref<Module> mainModule ()
{
  Ref<ChainModule>  inputChain = newInstance<ChainModule> ();
  Ref<ChainModule>  mainChain  = newInstance<ChainModule> ();


  jive::fem    ::declareMBuilders ();
  jive::model  ::declareModels    ();
  jive::femodel::declareModels    ();

  inputChain->pushBack (
    rootOnly           ( newInstance<MeshgenModule>   () )
  );
  inputChain->pushBack ( newInstance<MPInputModule>   () );
  inputChain->pushBack ( newInstance<PartitionModule> () );
  inputChain->pushBack ( newInstance<InputModule>     () );

  mainChain->pushBack  (
    newInstance<CheckpointModule> (
      "checkpoint",
      inputChain
    )
  );

  mainChain->pushBack ( newInstance<ControlModule>    () );
  mainChain->pushBack ( newInstance<ShapeModule>      ( "" ) );
  mainChain->pushBack ( newInstance<InitModule>       () );
  mainChain->pushBack ( newInstance<InfoModule>       () );
  mainChain->pushBack ( newInstance<LinsolveModule>   () );
  mainChain->pushBack ( newInstance<OutputModule>     () );

  return newInstance<ReportModule> ( "report", mainChain );
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ( int argc, char** argv )
{
  return Application::pexec ( argc, argv, & mainModule );
}
