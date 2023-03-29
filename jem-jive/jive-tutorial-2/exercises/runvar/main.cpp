
#include <jive/app/Application.h>
#include <jive/app/InitModule.h>
#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/SampleModule.h>
#include <jive/app/ReportModule.h>
#include <jive/gl/GraphModule.h>
#include "StepModule.h"
#include "ExampleModel.h"

using namespace jem;
using namespace jive::app;
using namespace jive::model;
using namespace jive::gl;

//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------

Ref<Module> mainModule ()
{
  Ref<ChainModule>  chain = newInstance<ChainModule> ();

  declareExampleModel ();

  chain->pushBack ( newInstance<InitModule>   () );
  chain->pushBack ( newInstance<StepModule>   () );
  chain->pushBack ( newInstance<ControlModule>() );

  return newInstance<ReportModule> ( "report", chain );
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ( int argc, char** argv )
{
  return Application::exec ( argc, argv, mainModule );
}
