
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
  Matrix              elmat;
  IdxVector           idofs;

  mbuilder = newInstance<FlexMatrixBuilder> ();

  elmat.resize ( 2, 2 );
  idofs.resize ( 2 );

  elmat(0,0) = elmat(1,1) =  1.0;
  elmat(0,1) = elmat(1,0) = -1.0;
  idofs[0]   = 0;
  idofs[1]   = 1;

  mbuilder->clear    ();
  mbuilder->addBlock ( idofs, idofs, elmat );

  idofs += 2;

  mbuilder->addBlock ( idofs, idofs, elmat );

  System::out() << "Matrix = \n";

  for ( idx_t i = 0; i < 4; i++ )
  {
    for ( idx_t j = 0; j < 4; j++ )
    {
      System::out() << "  " << mbuilder->getValue ( i, j );
    }

    System::out() << "\n";
  }

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}
