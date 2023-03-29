
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

#ifndef JEM_NUMERIC_SPARSE_MATMUL_TCC
#define JEM_NUMERIC_SPARSE_MATMUL_TCC

#include <jem/base/assert.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/NumberTraits.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   matrix - vector multiply
//-----------------------------------------------------------------------


template <class T>

  void                    matmul

  ( const Array<T>&         r,
    const SparseMatrix<T>&  a,
    const Array<T>&         b )

{
  JEM_ASSERT2 ( a.size(0) == r.size() &&
                a.size(1) == b.size(),
                "sparse matrix/vector shape mismatch" );

  const idx_t          rowCount = a.size(0);

  const Array<idx_t>&  offsets  = a.getRowOffsets    ();
  const Array<idx_t>&  indices  = a.getColumnIndices ();
  const Array<T>&      values   = a.getValues        ();

  T      x;
  idx_t  irow;
  idx_t  i, k;


  i = 0;

  for ( irow = 0; irow < rowCount; irow++ )
  {
    x = NumberTraits<T>::zero ();
    k = offsets[irow + 1];

    for ( ; i < k; i++ )
    {
      NumberTraits<T>::madd ( x, values[i], b[indices[i]] );
    }

    r[irow] = x;
  }
}


template <class T>

  Array<T>                matmul

  ( const SparseMatrix<T>&  a,
    const Array<T>&         b )

{
  Array<T> r ( a.size(0) );

  matmul ( r, a, b );

  return r;
}


//-----------------------------------------------------------------------
//   vector - matrix multiply
//-----------------------------------------------------------------------


template <class T>

  void                    matmul

  ( const Array<T>&         r,
    const Array<T>&         a,
    const SparseMatrix<T>&  b )

{
  JEM_ASSERT2 ( b.size(0) == a.size() &&
                b.size(1) == r.size(),
                "sparse matrix/vector shape mismatch" );

  const idx_t          rowCount = a.size();

  const Array<idx_t>&  offsets  = b.getRowOffsets    ();
  const Array<idx_t>&  indices  = b.getColumnIndices ();
  const Array<T>&      values   = b.getValues        ();

  T      x;
  idx_t  irow;
  idx_t  i, k;


  i = 0;
  r = NumberTraits<T>::zero ();

  for ( irow = 0; irow < rowCount; irow++ )
  {
    x = a[irow];
    k = offsets[irow + 1];

    for ( ; i < k; i++ )
    {
      NumberTraits<T>::madd ( r[indices[i]], values[i], x );
    }
  }
}


template <class T>

  Array<T>                matmul

  ( const Array<T>&         a,
    const SparseMatrix<T>&  b )

{
  Array<T> r ( b.size(1) );

  matmul ( r, a, b );

  return r;
}


//-----------------------------------------------------------------------
//   matrix - matrix multiply
//-----------------------------------------------------------------------


template <class T>

  void                              matmul

  ( const Array<T,2>&                 r,
    const SparseMatrix<T>&            a,
    const Array<T,2>&                 b )

{
  JEM_PRECHECK ( r.size(0) == a.size(0) &&
                 r.size(1) == b.size(1) &&
                 a.size(1) == b.size(0) );

  const idx_t   rowCount = r.size (0);
  const idx_t   colCount = r.size (1);

  Array<idx_t>  offsets  = a.getRowOffsets    ();
  Array<idx_t>  indices  = a.getColumnIndices ();
  Array<T>      values   = a.getValues        ();

  T             t0, t1, t2, t3;
  idx_t         jcol;


  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    const idx_t  istart = offsets[irow];
    const idx_t  iend   = offsets[irow + 1];

    for ( jcol = 0; jcol < (colCount - 3); jcol += 4 )
    {
      t0 = t1 = t2 = t3 = NumberTraits<T>::zero ();

      for ( idx_t j = istart; j < iend; j++ )
      {
        idx_t  jrow = indices[j];
        T      x    = values [j];

        NumberTraits<T>::madd ( t0, x, b(jrow,jcol + 0) );
        NumberTraits<T>::madd ( t1, x, b(jrow,jcol + 1) );
        NumberTraits<T>::madd ( t2, x, b(jrow,jcol + 2) );
        NumberTraits<T>::madd ( t3, x, b(jrow,jcol + 3) );
      }

      r(irow,jcol + 0) = t0;
      r(irow,jcol + 1) = t1;
      r(irow,jcol + 2) = t2;
      r(irow,jcol + 3) = t3;
    }

    for ( ; jcol < colCount; jcol++ )
    {
      t0 = NumberTraits<T>::zero ();

      for ( idx_t j = istart; j < iend; j++ )
      {
        idx_t  jrow = indices[j];
        T      x    = values [j];

        NumberTraits<T>::madd ( t0, x, b(jrow,jcol) );
      }

      r(irow,jcol) = t0;
    }
  }
}


template <class T>

  Array<T,2>                        matmul

  ( const SparseMatrix<T>&            a,
    const Array<T,2>&                 b )

{
  Array<T,2>  r ( a.size(0), b.size(1) );

  matmul ( r, a, b );

  return r;
}


template <class T>

  SparseMatrix<T>                   matmul

  ( const SparseMatrix<T>&            lhs,
    const SparseMatrix<T>&            rhs,
    const SparseFilterInterface<T>&   filter )

{
  const idx_t    rowCount    = lhs.size(0);
  const idx_t    colCount    = rhs.size(1);

  Array<T>       tbuf        ( colCount );
  Array<idx_t>   ibuf        ( colCount * 2 + 1 );

  T*             accu        = tbuf.addr ();
  idx_t*         mask        = ibuf.addr ();
  idx_t*         ilist       = mask + colCount;

  Array<idx_t>   lhsOffsets  = lhs.getRowOffsets    ();
  Array<idx_t>   lhsIndices  = lhs.getColumnIndices ();
  Array<T>       lhsValues   = lhs.getValues        ();

  Array<idx_t>   rhsOffsets  = rhs.getRowOffsets    ();
  Array<idx_t>   rhsIndices  = rhs.getColumnIndices ();
  Array<T>       rhsValues   = rhs.getValues        ();

  Array<idx_t>   offsets     ( rowCount + 1 );
  Array<idx_t>   indices;
  Array<T>       values;

  lint           opCount;
  idx_t          nnz;


  opCount = 0;

  for ( idx_t j = 0; j < colCount; j++ )
  {
    mask[j] = 1;
    accu[j] = NumberTraits<T>::zero ();
  }

  // The multiplication is executed in two phases. In the first phase
  // the number of non-zeroes are counted, and in the second phase the
  // index and value arrays are filled.

  for ( idx_t phase = 0; phase < 2; phase++ )
  {
    nnz = 0;

    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      idx_t  m = lhsOffsets[irow + 1];
      idx_t  k = 0;

      offsets[irow] = nnz;

      for ( idx_t i = lhsOffsets[irow]; i < m; i++ )
      {
        T      x    = lhsValues[i];
        idx_t  icol = lhsIndices[i];
        idx_t  j    = rhsOffsets[icol];
        idx_t  n    = rhsOffsets[icol + 1];

        opCount    += (lint) (n - j);

        for ( ; j < n; j++ )
        {
          idx_t  jcol  = rhsIndices[j];

          ilist[k]   = jcol;
          k         += mask[jcol];
          mask[jcol] = 0;

          NumberTraits<T>::madd ( accu[jcol], x, rhsValues[j] );
        }
      }

      testCancelled ( opCount, JEM_FUNC,
                      "sparse matrix multiply cancelled" );

      for ( idx_t j = 0; j < k; j++ )
      {
        idx_t  jcol = ilist[j];

        if ( filter.accept( accu[jcol], irow, jcol ) )
        {
          if ( phase != 0 )
          {
            indices[nnz] = jcol;
            values [nnz] = accu[jcol];
          }

          nnz++;
        }

        mask[jcol] = 1;
        accu[jcol] = NumberTraits<T>::zero ();
      }
    }

    offsets[rowCount] = nnz;

    if ( phase == 0 )
    {
      indices.resize ( nnz );
      values .resize ( nnz );
    }
  }

  return SparseMatrix<T> ( shape ( rowCount, colCount ),
                           offsets, indices, values );
}


template <class T>

  inline SparseMatrix<T>    matmul

  ( const SparseMatrix<T>&    a,
    const SparseMatrix<T>&    b )

{
  JEM_PRECHECK2 ( a.size(1) == b.size(0),
                  "SparseMatrix shape mismatch" );

  return matmul (
    a, b,
    SparseFilterWrapper<T,PassFilter> (
      SparseFilter<T,PassFilter> ()
    )
  );
}


template <class T, class F>

  inline SparseMatrix<T>    matmul

  ( const SparseMatrix<T>&    a,
    const SparseMatrix<T>&    b,
    const SparseFilter<T,F>&  filter )

{
  JEM_PRECHECK2 ( a.size(1) == b.size(0),
                  "SparseMatrix shape mismatch" );

  return matmul (
    a, b,
    SparseFilterWrapper<T,F> ( filter )
  );
}


JEM_END_PACKAGE( numeric )

#endif
