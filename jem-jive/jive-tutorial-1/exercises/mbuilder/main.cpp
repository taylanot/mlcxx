
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jive/algebra/FlexMatrixBuilder.h>

using namespace jem;
using namespace jive;
using namespace jive::algebra;

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  Ref<MatrixBuilder>  mbuilder;

  mbuilder = newInstance<FlexMatrixBuilder> ();

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
