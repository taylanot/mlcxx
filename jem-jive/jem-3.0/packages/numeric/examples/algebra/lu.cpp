
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
 *  This example demonstrates how to invert matrices by using
 *  the LUSolver class and related functions.
 *
 ***********************************************************************/

#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jem/numeric/algebra/matmul.h>


using namespace jem;
using namespace jem::io;
using namespace jem::numeric;


//-----------------------------------------------------------------------
//   constants & typedefs
//-----------------------------------------------------------------------


static const idx_t        N = 20;

typedef Array<double,2>   Matrix;


//-----------------------------------------------------------------------
//   identityMatrix
//-----------------------------------------------------------------------


Matrix identityMatrix ( idx_t n )
{
  Matrix  id ( n, n );

  id = 0.0;

  for ( idx_t i = 0; i < n; i++ )
  {
    id(i,i) = 1.0;
  }

  return id;
}


//-----------------------------------------------------------------------
//   invertAndPrint
//-----------------------------------------------------------------------


void invertAndPrint ( const Matrix& a )
{
  Writer&  out = System::out ();

  Matrix   b   = inverse        ( a );
  Matrix   id  = identityMatrix ( a.size(0) );

  double   err = sum ( abs( matmul( b, a ) - id ) );


  print ( out, "Matrix A     :\n\n", a, "\n\n" );
  print ( out, "Matrix inv(A):\n\n", b, "\n\n" );
  print ( out, "sum( abs (A * inv(A) - I) ) = ", err, "\n\n" );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Matrix  a ( N, N );

  for ( idx_t i = 0; i < N; i++ )
  {
    for ( idx_t j = 0; j < N; j++ )
    {
      a(i,j) = (1.0 + (double) i) * (double)

        (2.0 * (double) N - std::fabs( (double) (i - j) ));
    }
  }

  print ( System::out(), '\n' );

  invertAndPrint ( a );

  a.resize ( 6, 6 );

  a(0,0) =  0.000878431;
  a(0,1) =  0.000577005;
  a(0,2) =  0.000727718;
  a(0,3) =  0.000131113;
  a(0,4) =  5.03458e-05;
  a(0,5) =  9.07293e-05;

  a(1,0) =  0.000314113;
  a(1,1) =  0.000206328;
  a(1,2) =  0.000260221;
  a(1,3) =  4.68839e-05;
  a(1,4) =  1.80029e-05;
  a(1,5) =  3.24434e-05;

  a(2,0) =  0.000596272;
  a(2,1) =  0.000391666;
  a(2,2) =  0.000493969;
  a(2,3) =  8.89983e-05;
  a(2,4) =  3.41743e-05;
  a(2,5) =  6.15863e-05;

  a(3,0) = -0.000520668;
  a(3,1) = -0.000342005;
  a(3,2) = -0.000431336;
  a(3,3) =  0.000477842;
  a(3,4) =  0.000525714;
  a(3,5) =  0.000501778;

  a(4,0) = -0.000671876;
  a(4,1) = -0.000441328;
  a(4,2) = -0.000556602;
  a(4,3) =  0.000455273;
  a(4,4) =  0.000517048;
  a(4,5) =  0.00048616;

  a(5,0) = -0.000596272;
  a(5,1) = -0.000391666;
  a(5,2) = -0.000493969;
  a(5,3) =  0.000466557;
  a(5,4) =  0.000521381;
  a(5,5) =  0.000493969;

  invertAndPrint ( a );

  a.resize ( 2, 2 );

  a      = 0.0;
  a(0,1) = 1.0;
  a(1,0) = 1.0;

  invertAndPrint ( a );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( &run );
}
