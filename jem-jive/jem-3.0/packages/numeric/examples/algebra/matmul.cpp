
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */

/************************************************************************
 *
 *  This example demonstrates the matmul function.
 *
 ***********************************************************************/

#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/util/Timer.h>
#include <jem/numeric/algebra/matmul.h>


using namespace jem;
using namespace jem::io;
using namespace jem::numeric;
using namespace jem::util;


//-----------------------------------------------------------------------
//   constants & typedefs
//-----------------------------------------------------------------------


static const int         N = 128;

typedef Array<double>    Vector;
typedef Array<double,2>  Matrix;


//-----------------------------------------------------------------------
//   createVector
//-----------------------------------------------------------------------


Vector createVector ( idx_t n )
{
  Vector  v ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    v[i] = (double) (n - i);
  }

  return v;
}


//-----------------------------------------------------------------------
//   createMatrix
//-----------------------------------------------------------------------


Matrix createMatrix ( idx_t m, idx_t n )
{
  Matrix  A ( shape( m, n ) );

  for ( idx_t i = 0; i < m; i++ )
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      A(i,j) = (double) (i * j);
    }
  }

  return A;
}


//-----------------------------------------------------------------------
//   setMatrix
//-----------------------------------------------------------------------


void setMatrix ( double A[N][N], double value )
{
  for ( idx_t i = 0; i < N; i++ )
  {
    for ( idx_t j = 0; j < N; j++ )
    {
      A[i][j] = value;
    }
  }
}


//-----------------------------------------------------------------------
//   matmul (C-arrays)
//-----------------------------------------------------------------------


void matmul ( double A[][N], double B[][N], double C[][N] )
{
  for ( idx_t i = 0; i < N; i++ )
  {
    for ( idx_t j = 0; j < N; j++ )
    {
      double  x = 0.0;

      for ( idx_t k = 0; k < N; k++ )
      {
        x += B[i][k] * C[k][j];
      }

      A[i][j] = x;
    }
  }
}


//-----------------------------------------------------------------------
//   testSpeed
//-----------------------------------------------------------------------


double testSpeed ()
{
  Writer&  out = System::out ();

  Matrix   A ( shape(N,N) );
  Matrix   B ( shape(N,N) );
  Matrix   C ( shape(N,N) );

  double   D[N][N];
  double   E[N][N];
  double   F[N][N];

  Timer    timer;

  double   flops[2];
  double   dt;
  idx_t    k;

  A = 0.0;
  B = 1.0 / 3.0;
  C = 4.66;

  setMatrix ( D, 0.0 );
  setMatrix ( E, 1.0 / 3.0 );
  setMatrix ( F, 4.66 );

  timer.start ();

  dt = 0.0;
  k  = 1;

  while ( dt < 1.0 )
  {
    timer.reset ();

    k *= 2;

    for ( idx_t i = 0; i < k; i++ )
    {
      matmul ( A, B, C );
    }

    dt = timer.time().toDouble ();
  }

  flops[0] = 2.0 * (double) N * N * N * (double) k / dt;

  print ( out, "  jem    matmul : ", flops[0], " FLOPS\n" );

  dt = 0.0;
  k  = 1;

  while ( dt < 1.0 )
  {
    timer.reset ();

    k *= 2;

    for ( idx_t i = 0; i < k; i++ )
    {
      matmul ( D, E, F );
    }

    dt = timer.time().toDouble ();
  }

  flops[1] = 2.0 * (double) N * N * N * (double) k / dt;

  print ( out, "  native matmul : ", flops[1], " FLOPS\n\n" );

  return flops[1] / flops[0];
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Writer&  out = System::out ();

  Matrix   A   = createMatrix ( 4, 5 );
  Matrix   B   = createMatrix ( 5, 4 );
  Vector   x   = createVector ( 4 );
  Vector   y   = createVector ( 5 );

  double   relTime;

  print ( out, endl );

  print ( out, "Matrix A:\n\n", A, "\n\n" );
  print ( out, "Matrix B:\n\n", B, "\n\n" );

  print ( out, "Vector x:\n\n", x, "\n\n" );
  print ( out, "Vector y:\n\n", y, "\n\n" );

  print ( out, "A * B   :\n\n", matmul( A, B ), "\n\n" );
  print ( out, "A * y   :\n\n", matmul( A, y ), "\n\n" );
  print ( out, "x * A   :\n\n", matmul( x, A ), "\n\n" );
  print ( out, "x * y   :\n\n", matmul( x, y ), "\n\n" );

  print ( out, "Testing speed of matmul ",
          "(matrix size = ", N, ") ... \n\n" );

  relTime = testSpeed ();

  if( relTime > 1.0 )
  {
    print ( out, "Native matmul operation is ", relTime,
            " times faster than jem matmul operation\n\n" );
  }
  else
  {
    print ( out, "Jem matmul operation is ", 1.0 / relTime,
            " times faster than native matmul operation\n\n" );
  }

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main()
{
  return System::exec ( &run );
}
