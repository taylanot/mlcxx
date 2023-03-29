
#include <jem/base/String.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>

using jem::System;

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  using jem::String;
  using jem::io::endl;
  using jem::util::Properties;

  Properties  conf;
  String      header;
  int         maxIter;
  double      initValue;
  double      tolerance;
  double      value;


  conf.parseFile ( "input.pro" );

  conf.find ( header,    "header" );
  conf.get  ( maxIter,   "solver.maxIter",   1, 1000 );
  conf.get  ( initValue, "solver.initValue" );
  conf.get  ( tolerance, "solver.tolerance", 1e-20, 1e20 );

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
