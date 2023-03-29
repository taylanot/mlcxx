
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
 *  This example shows how to modify a sparse matrix using
 *  SparseFilters.
 *
 ***********************************************************************/

#include <jem/base/System.h>
#include <jem/numeric/sparse/matmul.h>
#include <jem/numeric/sparse/utilities.h>
#include <jem/numeric/sparse/SparseFilters.h>
#include <jem/numeric/sparse/SparseMatrix.h>

#include "makeMatrix.h"


using namespace jem;
using namespace jem::io;
using namespace jem::numeric;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  Writer&            out = System::out ();

  SparseMatrix<int>  a   = makeMatrix<int> ( 8, 5 );


  print ( out, "Matrix A:\n\n", a, "\n\n" );

  print ( out, "Matrix A with diagonal filtered print ( out:\n\n"
     , filter ( a, ! SparseFilters<int>::diagonal() ), "\n\n") ;

  print ( out, "Matrix A from which the following elements "
     , "have been removed:\n\n" );
  print ( out, "  all A(i,j) <= 10;\n" );
  print ( out, "  all A(i,j) with j < i - 1 and j > i + 1;\n" );
  print ( out, "  all A(i,i).\n\n" );

  print ( out, filter( a, SparseFilters<int>::greater(10) &&
                       SparseFilters<int>::band(1)        &&
                       SparseFilters<int>::diagonal() ), "\n\n" );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}


