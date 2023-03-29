
#include <jive/fem/InputModule.h>
#include <jive/app/Application.h>
#include <jive/app/InfoModule.h>
#include <jive/app/InitModule.h>
#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/ReportModule.h>
#include <jive/model/declare.h>
#include "ExampleModel.h"

using namespace jem;
using namespace jive::app;
using namespace jive::fem;
using namespace jive::model;

//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------

Ref<Module> mainModule ()
{
  Ref<ChainModule>  chain = newInstance<ChainModule> ();

  declareModels       ();
  declareExampleModel ();

  chain->pushBack ( newInstance<InputModule>  () );
  chain->pushBack ( newInstance<InitModule>   () );
  chain->pushBack ( newInstance<InfoModule>   () );
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
