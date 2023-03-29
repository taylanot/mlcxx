
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

#ifndef JEM_NUMERIC_SPARSE_UTILITIES_H
#define JEM_NUMERIC_SPARSE_UTILITIES_H

#include <jem/util/SparseArray.h>
#include <jem/numeric/sparse/SparseMatrix.h>
#include <jem/numeric/sparse/SparseFilter.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   utility functions
//-----------------------------------------------------------------------


template <class T>

  Array<T>                          diag

  ( const SparseMatrix<T>&            a );


template <class T>

  void                              getDiag

  ( const Array<T>&                   d,
    const SparseMatrix<T>&            a );


template <class T>

  void                              sortRows

  ( SparseMatrix<T>&                  a );


template <class T, class F>

  SparseMatrix<T>                   filter

  ( const SparseMatrix<T>&            a,
    const SparseFilter<T,F>&          f );


template <class T, class A>

  SparseMatrix<T>                   toMatrix

  ( const util::SparseArray<T,2,A>&   a );


template <class T, class A>

  util::SparseArray<T,2,A>&         toArray

  ( util::SparseArray<T,2,A>&         out,
    const SparseMatrix<T>&            in );





//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE( numeric )

#include <jem/numeric/sparse/utilities.tcc>

#endif
