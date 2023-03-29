
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jive/util/Globdat.h>
#include <jive/fem/InputModule.h>
#include <jive/app/ControlModule.h>

using namespace jem;
using namespace jem::util;
using namespace jive;
using namespace jive::util;
using namespace jive::app;
using namespace jive::fem;

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  Ref<Module>  modules[2];
  Properties   globdat;
  Properties   props;
  Properties   conf;
  int          status;


  modules[0] = newInstance<InputModule>   ();
  modules[1] = newInstance<ControlModule> ();

  props.parseFile ( "input.pro" );
  Globdat::init   ( globdat );

  for ( int i = 0; i < 2; i++ )
  {
    modules[i]->configure ( props, globdat );
    modules[i]->getConfig ( conf,  globdat );
    modules[i]->init      ( conf, props, globdat );
  }

  System::out() << "Runtime parameters:\n\n";
  System::out() << conf << "\n\n";

  status = Module::OK;

  for ( int istep = 0; status != Module::EXIT; istep++ )
  {
    System::out() << "Step " << istep << "\n";

    for ( int i = 0; i < 2; i++ )
    {
      if ( modules[i] != NIL )
      {
        status = modules[i]->run ( globdat );

        if      ( status == Module::DONE )
        {
          modules[i] = NIL;
        }
        else if ( status == Module::EXIT )
        {
          break;
        }
      }
    }
  }

  for ( int i = 0; i < 2; i++ )
  {
    if ( modules[i] != NIL )
    {
      modules[i]->shutdown ( globdat );
    }
  }

  System::out() << "\n";
  System::out() << "Global database:\n\n";
  System::out() << globdat << "\n\n";

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}
