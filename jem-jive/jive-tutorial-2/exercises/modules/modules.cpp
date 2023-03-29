
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

  // Continue here ...

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}
