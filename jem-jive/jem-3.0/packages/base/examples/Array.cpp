
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
 *  This example demonstrates how to use the Array class.
 *
 ***********************************************************************/

#include <jem/std/complex.h>
#include <jem/base/System.h>
#include <jem/base/Array.h>
#include <jem/util/Timer.h>

using namespace jem;
using namespace jem::io;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Writer&          out = System::out ();

  Array<int>       a ( 20 );
  Array<int>       b ( 10 );
  Array<int>       c;
  Array<bool>      d ( 5  );
  Array<bool>      e ( 5  );
  Array<int>       f ( 10 );
  Array<double>    g ( 5  );
  Array<double>    h ( 5  );
  Array<double>    q ( 5  );

  Array< std::complex<double> > z ( 10 );


  imag(z) = 0;
  real(z) = 1;

  print ( out, abs( sum( z * conj(z) ) ), endl );
  print ( out, sum( norm(z) ), endl );

  print ( out, abs(z), endl );

  z = zadd( 0.8 + imag(z), -9 );

  print ( out, z, endl );

  a = iarray ( a.size() );

  print ( out, "array a : ", a, endl, endl );

  b = a[slice(0,10)];

  print ( out, "array b : ", b, endl, endl );

  b = a[slice(0,END,2)];

  print ( out, "array b : ", b, endl, endl );

  c.ref ( a[slice(10,20)]);

  c[0] = 55;

  print ( out, "array a : ", a, endl, endl );
  print ( out, "array c : ", c, endl, endl );

  for ( idx_t i = 0; i < b.size(); i++ )
  {
    b[i] = (int) pow ( 2.0, (double) (i + 2) );
  }

  for ( idx_t i = 0; i < h.size(); i++ )
  {
    h[i] = 0.5 * M_PI * (double) i;
  }

  e = false;
  d = where( a[slice(0,5)] % 2 == 0, e[slice(0,5)], false );
  e = ! d;

  print ( out, "array d : ", d, endl, endl );
  print ( out, "array e : ", e, endl, endl );

  f = brshift ( b, 3 );

  print ( out, "array b : ", b, endl, endl );
  print ( out, "array f : ", f, endl, endl );

  g = f[slice(1,END,2)] + 0.1;

  q = sqrt( g ) + 2 * sin( h );

  print ( out, "array g : ", g, endl, endl );
  print ( out, "array h : ", h, endl, endl );
  print ( out, "array q : ", q, endl, endl );

  print ( out, "array  q : ",      q  , endl, endl );
  print ( out, "min of q : ", min( q ), endl, endl );
  print ( out, "max of q : ", max( q ), endl, endl );
  print ( out, "sum of q : ", sum( q ), endl, endl );

  Array<int,3>  f2 = reshape ( f,  2, 2, 2 );
  Array<int,2>  f3 = reshape ( f2, 4, 2 );
  Array<int,2>  f4 = flatten ( f2, 2, 1 );
  Array<int,2>  f5 = flatten ( f2, 1, 2 );
  Array<int,1>  f6 = flatten ( f2 );

  print ( out, "array f2 : ", endl, f2, endl, endl );
  print ( out, "array f3 : ", endl, f3, endl, endl );
  print ( out, "array f4 : ", endl, f4, endl, endl );
  print ( out, "array f5 : ", endl, f5, endl, endl );
  print ( out, "array f6 : ", endl, f6, endl, endl);

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}
