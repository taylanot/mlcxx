
#include <jem/base/String.h>
#include <jem/base/System.h>

using jem::System;

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  using jem::String;
  using jem::io::endl;

  String  header;
  int     maxIter;
  double  initValue;
  double  tolerance;
  double  value;

  // Read configuration data here ...

  // Set initial values for header, maxIter, initValue
  // and tolerance ...

  System::out() << header << endl;

  value = initValue;

  for ( int i = 0; i < maxIter; i++ )
  {
    System::out() << "iteration " << i
                  << ", value = " << value << endl;

    value *= 0.5;

    if ( value <= tolerance )
    {
      break;
    }
  }

  if ( value <= tolerance )
  {
    System::out() << "Converged.\n";

    return 0;
  }
  else
  {
    System::out() << "No convergence reached.\n";

    return 1;
  }
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}
