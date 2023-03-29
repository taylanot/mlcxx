
#include <jem/base/System.h>
#include <jem/base/Array.h>
#include <jem/base/array/tensor.h>
#include <jem/util/Timer.h>


using namespace jem;

using jem::io::endl;
using jem::util::Timer;


//-----------------------------------------------------------------------
//   constants
//-----------------------------------------------------------------------


const int  N = 2;


//-----------------------------------------------------------------------
//   matmul1
//-----------------------------------------------------------------------

/*
  Computes a matrix-matrix multiplication using three nested do-loops.
*/

void matmul1 ()
{
  Array<double,2>   a ( N, N );
  Array<double,2>   b ( N, N );
  Array<double,2>   c ( N, N );

  b = 1.0;
  c = 2.0;

  for ( int i = 0; i < N; i++ )
  {
    for ( int j = 0; j < N; j++ )
    {
      double  x = 0.0;

      for ( int k = 0; k < N; k++ )
      {
	      x += b(i,k) * c(k,j);
      }

      a(i,j) = x;
    }
  }
}


//-----------------------------------------------------------------------
//   matmul2
//-----------------------------------------------------------------------

/*
  Computes a matrix-matrix multiplication using array slices and the
  dot() function.
*/

void matmul2 ()
{
  Array<double,2>   a ( N, N );
  Array<double,2>   b ( N, N );
  Array<double,2>   c ( N, N );

  b = 1.0;
  c = 2.0;

  for ( int i = 0; i < N; i++ )
  {
    for ( int j = 0; j < N; j++ )
    {
      a(i,j) = dot ( b(i,ALL), c(ALL,j) );
    }
  }
}


//-----------------------------------------------------------------------
//   matmul3
//-----------------------------------------------------------------------

/*
  Computes a matrix-matrix multiplication using tensor expressions.
*/

void matmul3 ()
{
  Array<double,2>   a ( N, N );
  Array<double,2>   b ( N, N );
  Array<double,2>   c ( N, N );
  Array<double,1>   d ( N );
  Array<double,1>   e ( N );

  TensorIndex       i, j, k;

  b = 1.0;
  c = 2.0;
  d = 3.0;
  e = 4.0;

  System::out() << b << "\n";
  System::out() << c << "\n";

  a(i,j) = dot ( b(i,k), c(k,j), k );

  a = 1.0;

  e[i] = dot ( a(i,j), d[j], j );

  System::out() << a << "\n";
  System::out() << e << "\n";


}


//-----------------------------------------------------------------------
//   timeFunc
//-----------------------------------------------------------------------

/*
  Measures the execution time of a given function.
*/

void timeFunc ( void (*func)() )
{
  Timer  t;

  t.start ();
  func    ();
  t.stop  ();

  System::out() << "Elapsed time = " << t << endl;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  timeFunc ( matmul1 );
  timeFunc ( matmul2 );
  timeFunc ( matmul3 );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( run );
}
