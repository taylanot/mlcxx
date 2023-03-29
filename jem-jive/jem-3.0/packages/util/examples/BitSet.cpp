
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
 *  This example demonstrates the use of the class BitSet.
 *
 ************************************************************************/

#include <cstdlib>
#include <jem/base/System.h>
#include <jem/util/BitSet.h>


using namespace jem;
using namespace jem::io;
using namespace jem::util;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  using std::rand;

  Writer&    out = System::out ();

  const int  N   = 20;

  BitSet     a;
  BitSet     b;
  BitSet     c;

  int        i, j;


  a.reserve ( N );
  b.reserve ( N );

  for ( i = 0; i < N; i++ )
  {
    j = (int) (N * (::rand() / (RAND_MAX + 1.0))); a.set(j);
    j = (int) (N * (::rand() / (RAND_MAX + 1.0))); b.set(j);
    j = (int) (N * (::rand() / (RAND_MAX + 1.0))); c.set(j);
  }

  print ( out, '\n' );
  print ( out, "a           : ", a, '\n' );
  print ( out, "b           : ", b, '\n' );
  print ( out, "c           : ", c, '\n' );

  a.xorWith ( b );

  print ( out, "a xor b = d : ", a, '\n' );

  a.andWith ( b );

  print ( out, "d and b = e : ", a, '\n' );

  a.orWith ( b );

  print ( out, "e or b  = f : ", a, '\n' );

  a.clear ( c );

  print ( out, "a.clear(c)  : ", a, "\n\n" );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}


