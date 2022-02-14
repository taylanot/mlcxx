/*
 *  main Implementation
 *
 *  Everything to be done by Jive should be added to the ChainModule
 *
 *  Ozgur Taylan Turan, February 2022
 *
 */


#include <jive/app/ChainModule.h>
#include <jive/app/OutputModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/Application.h>
#include <jive/app/InfoModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/declare.h>
#include <jive/geom/declare.h>
#include <jive/fem/declare.h>
#include <jive/mesh/declare.h>
#include <jive/fem/InitModule.h>
#include <jive/fem/InputModule.h>
#include <jive/fem/ShapeModule.h>
#include <jive/app/UserconfModule.h>
#include <jive/app/SampleModule.h>
#include <jive/model/declare.h>
#include <jive/implict/declare.h>
#include <jive/implict/LinsolveModule.h>
#include <jive/implict/NonlinModule.h>

#include "modules/UserModules.h"
#include "models/UserModels.h"

using namespace jem;

using jive::app::Application;
using jive::app::Module;
using jive::app::ChainModule;
using jive::app::ControlModule;
using jive::app::ReportModule;
using jive::app::InfoModule;
using jive::app::OutputModule;
using jive::app::SampleModule;
using jive::app::UserconfModule;
using jive::fem::InitModule;
using jive::fem::InputModule;
using jive::fem::ShapeModule;
using jive::implict::LinsolveModule;
using jive::implict::NonlinModule;

//------------------------------------------------------------------------------
//    mainModule
//------------------------------------------------------------------------------


Ref<Module> mainModule ()
{
  Ref<ChainModule>  chain = newInstance<ChainModule> ();

  // Register the required models/modules and other classes.

  declareUserModels   ();
  declareUserModules  ();

  jive::fem::declareMBuilders   ();
  jive::model::declareModels    ();
  jive::geom::declareIShapes    ();
  jive::geom::declareShapes     ();

  jive::implict::declareModules ();
  jive::app::declareModules     ();
  jive::mesh::declareModules    ();

  // Define the main module chain.

  chain->pushBack ( newInstance<UserconfModule>     ( "userinput"    ) );
  chain->pushBack ( newInstance<ShapeModule>        ( "shape"        ) );
  chain->pushBack ( newInstance<InitModule>         ( "init"         ) );
  chain->pushBack ( newInstance<InfoModule>         ( "info"         ) );
  chain->pushBack ( newInstance<UserconfModule>     ( "usermodules"  ) );
  chain->pushBack ( newInstance<ControlModule>      ( "control"      ) );

  return newInstance<ReportModule> ( "report", chain );
}


//------------------------------------------------------------------------------
//    main
//------------------------------------------------------------------------------


int main ( int argc, char** argv )
{
  return Application::pexec ( argc, argv, & mainModule );
}
