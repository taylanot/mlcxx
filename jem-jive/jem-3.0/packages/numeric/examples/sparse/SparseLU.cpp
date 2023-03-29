
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
 *  This example shows how to use the SparseLU class to factor a
 *  sparse matrix.
 *
 ***********************************************************************/


#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/Throwable.h>
#include <jem/base/ArithmeticException.h>
#include <jem/util/Event.h>
#include <jem/util/Timer.h>
#include <jem/numeric/sparse/matmul.h>
#include <jem/numeric/sparse/SparseLU.h>

#include "makeMatrix.h"


using namespace jem;
using namespace jem::io;
using namespace jem::numeric;
using namespace jem::util;


//-----------------------------------------------------------------------
//   printProgress
//-----------------------------------------------------------------------


void printProgress ( idx_t icol )
{
  print ( System::out(), "  processing column ", icol, '\n' );
}


//-----------------------------------------------------------------------
//   factor
//-----------------------------------------------------------------------


void                            factor

  ( SparseLU&                     lu,
    const SparseMatrix<double>&   a,
    const Array<idx_t>&           jperm )

{
  Array<bool>   mask  ( a.size(1) );
  Array<idx_t>  iperm ( iarray( a.size(0) ) );

  mask = true;

  if ( lu.factor( a, mask, iperm, jperm ) > 0 )
  {
    throw ArithmeticException (
      "function factor",
      "singular sparse matrix"
    );
  }
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  const idx_t           N   = 150;

  Writer&               out = System::out ();

  SparseMatrix<double>  a;
  SparseLU              lu;

  Array<idx_t>          jperm;
  Array<double>         x;
  Array<double>         b;

  Timer                 timer;


  connect ( lu.progressEvent, printProgress );

  print ( out, "\nCreating sparse matrix with size ",
          N, " ...\n\n" );

  a = makeMatrix<double> ( N, 23.0 );

  b    .resize ( a.size(0) );
  x    .resize ( a.size(0) );
  jperm.resize ( a.size(1) );

  for ( idx_t i = 0; i < jperm.size(); i++ )
  {
    jperm[i] = jperm.size() - i - 1;
  }

  print ( out, "Factoring matrix ...\n\n" );

  timer.start ();

  factor ( lu, a, jperm );

  print ( out, "\nReady in ", timer.toDouble(), " seconds.\n\n" );

  b = 2.0;

  timer.reset ();

  print ( out, "Solving A x = b ... " );
  out.flush ();

  lu.solve ( x, b );

  print ( out, "ready in ", timer.toDouble(), " seconds.\n\n" );
  print ( out, "Error = ", max( abs( b - matmul( a, x ) ) ), "\n\n" );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}

