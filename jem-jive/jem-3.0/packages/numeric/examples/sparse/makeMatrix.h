
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

#ifndef MAKE_MATRIX_H
#define MAKE_MATRIX_H


#include <jem/util/SparseArray.h>
#include <jem/numeric/sparse/utilities.h>
#include <jem/numeric/sparse/SparseMatrix.h>


//-----------------------------------------------------------------------
//   makeMatrix
//-----------------------------------------------------------------------


template <class T>

  jem::numeric::SparseMatrix<T>  makeMatrix

  ( jem::idx_t  size,
    T           value )

{
  jem::util::SparseArray<T,2>  a;

  for ( jem::idx_t i = 0; i < size; i++ )
  {
    a(i,i) = value * (T) i;

    if( i > 0 )
    {
      a(i - 1,i) = value - (T) i;
    }

    if ( i < (size - 1) )
    {
      a (i + 1,i) = value;
    }
  }

  return jem::numeric::toMatrix ( a );
}


#endif
