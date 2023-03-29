#include <jive/fem/InputModule.h>
#include <jive/app/Application.h>
#include <jive/app/InfoModule.h>
#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>

#include <jem/base/System.h>

using namespace jem;
using namespace jive::app;
using namespace jive::fem;

//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------

Ref<Module> mainModule ()
{
  Ref<ChainModule>  chain = newInstance<ChainModule> ();

  chain->pushBack ( newInstance<InputModule>  ("inputs") );
  System::out() << "Hello-1" << "\n";
  chain->pushBack ( newInstance<InfoModule>   () );
  System::out() << "Hello-2" << "\n";
  chain->pushBack ( newInstance<ControlModule>() );
  System::out() << "Hello-3" << "\n";

  return chain;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ( int argc, char** argv )
{
  return Application::exec ( argc, argv, mainModule );
}
