
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
 *  This example shows how to use the SparseArray and SparseMatrix
 *  classes.
 *
 ***********************************************************************/

#include <jem/base/System.h>
#include <jem/numeric/sparse/utilities.h>


using jem::System;
using jem::io::Writer;
using jem::util::SparseArray;
using jem::numeric::SparseMatrix;
using jem::numeric::toMatrix;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Writer&  out = System::out ();

  SparseArray<double,2>  a;
  SparseMatrix<double>   b;


  a(0,0) = 0.0;
  a(3,6) = 3.6;
  a(2,3) = 2.3;
  a(1,5) = 1.5;
  a(4,4) = 4.4;

  print ( out, a, "\n\n" );

  b = toMatrix ( a );

  print ( out, b, "\n\n" );

  toArray ( a, b );

  print ( out, a, "\n\n" );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}

