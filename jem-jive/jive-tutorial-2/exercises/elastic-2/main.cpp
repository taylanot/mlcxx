
#include <jive/model/declare.h>
#include <jive/algebra/declare.h>
#include <jive/fem/declare.h>
#include <jive/fem/InputModule.h>
#include <jive/app/Application.h>
#include <jive/app/InfoModule.h>
#include <jive/app/InitModule.h>
#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/OutputModule.h>
#include <jive/app/ReportModule.h>
#include <jive/implict/LinsolveModule.h>
#include "import.h"
#include "declare.h"

using jive::app::Module;

//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------

// Constructs the main module chain.

Ref<Module> mainModule ()
{
  using jem::newInstance;
  using jive::fem::InputModule;
  using jive::app::ChainModule;
  using jive::app::InitModule;
  using jive::app::InfoModule;
  using jive::app::ControlModule;
  using jive::app::OutputModule;
  using jive::app::ReportModule;
  using jive::implict::LinsolveModule;

  Ref<ChainModule>  chain = newInstance<ChainModule> ();

  jive::fem ::declareMBuilders ();
  jive::model::declareModels   ();
  declareElasticModel          ();

  chain->pushBack ( newInstance<InputModule>    () );

  // Add more modules here ...

  chain->pushBack ( newInstance<ControlModule>  () );

  return newInstance<ReportModule> ( "report", chain );
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

// Program entry point.

int main ( int argc, char** argv )
{
  using jive::app::Application;

  return Application::exec ( argc, argv, mainModule );
}
