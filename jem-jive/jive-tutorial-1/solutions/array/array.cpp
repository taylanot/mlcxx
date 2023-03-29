
#include <jem/base/Array.h>
#include <jem/base/System.h>

using namespace jem;

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  Array<int>  a;
  int         i, k;

  k = 0;

  System::in() >> i;

  while ( i > 0 )
  {
    if ( k >= a.size() )
    {
      a.reshape ( 2 * k + 1 );
    }

    a[k++] = i;

    System::in() >> i;
  }

  a.reshape ( k );

  System::out() << a << "\n";

  return 0;
}



//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}

