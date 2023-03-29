
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
 *  This example demonstrates how to use the Tuple class.
 *
 ***********************************************************************/

#include <jem/std/complex.h>
#include <jem/base/Tuple.h>
#include <jem/base/System.h>


using namespace jem;
using namespace jem::io;

typedef std::complex<double>  Complex;


//-----------------------------------------------------------------------
//   changeArray
//-----------------------------------------------------------------------


void changeArray ( int* a, int n )
{
  if ( n )
  {
    a[n - 1] = 100;
  }

  return;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Writer&           out = System::out ();

  Tuple<int,3>      a ( 0, 1, 2 );
  Tuple<lint,3>     b;
  Tuple<int,3>      c;
  Tuple<Complex,4>  z;


  z = 1.0;

  print ( out, abs( sum( z * conj(z) ) ), endl );
  print ( out, sum( norm(z) ), endl );

  print ( out, abs(z), endl );

  z = zadd( 0.8 + imag(z), -9 ) + 0.9;

  print ( out, z, endl );

  print ( out, a, endl, endl );

  b      = a;
  b[ 0 ] = 5;

  print ( out, b, endl, endl );
  print ( out, a, endl, endl );

  b += 1;

  print ( out, b, endl, endl );

  c = castTo<int> ( abs( sum( b * a ) * b - 18 ) );

  print ( out, c, endl, endl );

  c = castTo<int> ( where( a > b, a, b ) );

  print ( out, c, endl, endl );

  changeArray ( c.addr(), 3 );

  print ( out, c, endl, endl );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}
