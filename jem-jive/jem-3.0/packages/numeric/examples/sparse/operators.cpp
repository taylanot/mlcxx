
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
 *  This example demonstrates various sparse matrix operators.
 *
 ***********************************************************************/

#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/numeric/sparse/operators.h>
#include <jem/numeric/sparse/SparseMatrix.h>

#include "makeMatrix.h"
#include "utilities.h"


using namespace jem;
using namespace jem::io;
using namespace jem::numeric;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Writer&               out = System::out();

  SparseMatrix<double>  a;
  SparseMatrix<double>  b;
  Array<double,2>       c;
  Array<double,2>       d;


  a = makeMatrix ( 8, 2.0 );

  print ( out, '\n' );
  print ( out, "Matrix A:\n\n", a, "\n\n" );

  b = a * a.transpose () + a;

  print ( out, "Matrix B = A * A^(T) + A:\n\n", b, "\n\n" );
  print ( out, "Non-zeroes in B: ", b.nonZeroCount(), "\n\n" );

  c.ref ( toDenseMatrix( b ) );

  print ( out, "Matrix B as a dense matrix:\n\n", c, "\n\n" );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}


