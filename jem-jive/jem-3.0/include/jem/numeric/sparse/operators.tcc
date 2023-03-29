
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

#ifndef JEM_NUMERIC_SPARSE_OPERATORS_TCC
#define JEM_NUMERIC_SPARSE_OPERATORS_TCC

#include <jem/base/assert.h>
#include <jem/base/tuple/utilities.h>
#include <jem/util/ArrayBuffer.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   internal functions
//=======================================================================

//-----------------------------------------------------------------------
//   unite_
//-----------------------------------------------------------------------


template <class T, class Op>

  static SparseMatrix<T>            unite_

  ( const SparseMatrix<T>&            lhs,
    const SparseMatrix<T>&            rhs,
    const Op&                         op,
    const SparseFilterInterface<T>&   filter )

{
  typedef util::ArrayBuffer<idx_t>  IBuffer;
  typedef util::ArrayBuffer<T>      TBuffer;

  const idx_t   rowCount    = lhs.size(0);
  const idx_t   colCount    = lhs.size(1);

  Array<T>      accu        ( colCount );
  Array<idx_t>  mask        ( colCount );
  Array<idx_t>  queue       ( colCount );

  Array<idx_t>  lhsOffsets  = lhs.getRowOffsets    ();
  Array<idx_t>  lhsIndices  = lhs.getColumnIndices ();
  Array<T>      lhsValues   = lhs.getValues        ();

  Array<idx_t>  rhsOffsets  = rhs.getRowOffsets    ();
  Array<idx_t>  rhsIndices  = rhs.getColumnIndices ();
  Array<T>      rhsValues   = rhs.getValues        ();

  Array<idx_t>  offsets     ( rowCount + 1);
  IBuffer       indices;
  TBuffer       values;

  idx_t         queueEnd;


  indices.reserve ( max( lhs.nonZeroCount(), rhs.nonZeroCount() ) );
  values .reserve ( indices.capacity() );

  mask       = -1_idx;
  queueEnd   = 0;
  offsets[0] = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  i = lhsOffsets[irow];
    idx_t  n = lhsOffsets[irow + 1];

    for ( ; i < n; i++ )
    {
      idx_t  icol       = lhsIndices[i];

      mask[icol]        = irow;
      accu[icol]        = lhsValues[i];
      queue[queueEnd++] = icol;
    }

    i = rhsOffsets[irow];
    n = rhsOffsets[irow + 1];

    for ( ; i < n; i++ )
    {
      idx_t  icol = rhsIndices[i];

      if ( mask[icol] == irow )
      {
        accu[icol] = op ( accu[icol], rhsValues[i] );
      }
      else
      {
        accu[icol]        = op ( T(), rhsValues[i] );
        mask[icol]        = irow;
        queue[queueEnd++] = icol;
      }
    }

    for ( idx_t j = 0; j < queueEnd; j++ )
    {
      idx_t  icol = queue[j];

      if ( filter.accept( accu[icol], irow, icol ) )
      {
        indices.pushBack ( icol );
        values .pushBack ( accu[icol] );
      }
    }

    offsets[irow + 1] = indices.size();
    queueEnd          = 0;
  }

  return SparseMatrix<T> ( shape ( rowCount, colCount ),
                           offsets,
                           indices.toArray(),
                           values .toArray() );
}


//-----------------------------------------------------------------------
//  intersect_
//-----------------------------------------------------------------------


template <class T, class Op>

  static SparseMatrix<T>            intersect_

  ( const SparseMatrix<T>&            lhs,
    const SparseMatrix<T>&            rhs,
    const Op&                         op,
    const SparseFilterInterface<T>&   filter )

{
  typedef util::ArrayBuffer<idx_t>  IBuffer;
  typedef util::ArrayBuffer<T>      TBuffer;

  const idx_t   rowCount    = lhs.size(0);
  const idx_t   colCount    = lhs.size(1);

  Array<T>      accu        ( colCount );
  Array<idx_t>  mask        ( colCount );

  Array<idx_t>  lhsOffsets  = lhs.getRowOffsets    ();
  Array<idx_t>  lhsIndices  = lhs.getColumnIndices ();
  Array<T>      lhsValues   = lhs.getValues        ();

  Array<idx_t>  rhsOffsets  = rhs.getRowOffsets    ();
  Array<idx_t>  rhsIndices  = rhs.getColumnIndices ();
  Array<T>      rhsValues   = rhs.getValues        ();

  Array<idx_t>  offsets     ( rowCount + 1);
  IBuffer       indices;
  TBuffer       values;

  T             x;


  indices.reserve ( min( lhs.nonZeroCount(),
                         rhs.nonZeroCount() ) );

  values .reserve ( indices.capacity() );

  mask       = -1_idx;
  offsets[0] = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  i = lhsOffsets[irow];
    idx_t  n = lhsOffsets[irow + 1];

    for ( ; i < n; i++ )
    {
      idx_t  icol = lhsIndices[i];

      mask[icol]  = irow;
      accu[icol]  = lhsValues[i];
    }

    i = rhsOffsets[irow];
    n = rhsOffsets[irow + 1];

    for ( ; i < n; i++ )
    {
      idx_t  icol = rhsIndices[i];

      if ( mask[icol] == irow )
      {
        x = op ( accu[icol], rhsValues[i] );

        if ( filter.accept( x, irow, icol ) )
        {
          indices.pushBack ( icol );
          values .pushBack ( x );
        }
      }
    }

    offsets[irow + 1] = indices.size();
  }

  return SparseMatrix<T> ( shape( rowCount, colCount ),
                           offsets,
                           indices.toArray(),
                           values .toArray() );
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   operator +
//-----------------------------------------------------------------------


template <class T>

  inline SparseMatrix<T>  operator +

  ( const SparseMatrix<T>&  a,
    const SparseMatrix<T>&  b )

{
  return unite ( a, b, Plus<T>() );
}


//-----------------------------------------------------------------------
//   operator -
//-----------------------------------------------------------------------


template <class T>

  inline SparseMatrix<T>  operator -

  ( const SparseMatrix<T>&  a,
    const SparseMatrix<T>&  b )

{
  return unite ( a, b, Minus<T>() );
}


//-----------------------------------------------------------------------
//   operator *
//-----------------------------------------------------------------------


template <class T>

  inline SparseMatrix<T>  operator *

  ( const SparseMatrix<T>&  a,
    const SparseMatrix<T>&  b )

{
  return intersect ( a, b, Multiplies<T>() );
}


//-----------------------------------------------------------------------
//   operator ||
//-----------------------------------------------------------------------


template <class T>

  inline SparseMatrix<T>  operator ||

  ( const SparseMatrix<T>&  a,
    const SparseMatrix<T>&  b )

{
  return unite ( a, b, LogicalOr<T>() );
}


//-----------------------------------------------------------------------
//   operator &&
//-----------------------------------------------------------------------


template <class T>

  inline SparseMatrix<T>  operator &&

  ( const SparseMatrix<T>&  a,
    const SparseMatrix<T>&  b )

{
  return intersect ( a, b, LogicalAnd<T>() );
}


//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


template <class T>

  bool                    operator ==

  ( const SparseMatrix<T>&  lhs,
    const SparseMatrix<T>&  rhs )

{
  if ( ! equal( lhs.shape(), rhs.shape() ) )
  {
    return false;
  }

  const idx_t   rowCount    = lhs.size(0);
  const idx_t   colCount    = lhs.size(1);

  Array<idx_t>  mask        ( colCount );
  Array<T>      accu        ( colCount );

  Array<idx_t>  lhsOffsets  = lhs.getRowIndices    ();
  Array<idx_t>  lhsIndices  = lhs.getColumnIndices ();
  Array<T>      lhsValues   = lhs.getValues        ();

  Array<idx_t>  rhsOffsets  = rhs.getRowIndices    ();
  Array<idx_t>  rhsIndices  = rhs.getColumnIndices ();
  Array<T>      rhsValues   = rhs.getValues        ();


  mask = -1_idx;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  i = lhsOffsets[irow];
    idx_t  n = lhsOffsets[irow + 1];

    for ( ; i < n; i++ )
    {
      idx_t  icol  = lhsIndices[i];

      mask[icol] = irow;
      accu[icol] = lhsValues[i];
    }

    i = rhsOffsets[irow];
    n = rhsOffsets[irow + 1];

    for ( ; i < n; i++ )
    {
      idx_t  icol = rhsIndices[i];

      if ( mask[icol] != irow || rhsValues[i] != accu[icol] )
      {
        return false;
      }
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   operator !=
//-----------------------------------------------------------------------


template <class T>

  inline bool             operator !=

  ( const SparseMatrix<T>&  a,
    const SparseMatrix<T>&  b )

{
  return ! ( a == b );
}


//-----------------------------------------------------------------------
//   operator +=
//-----------------------------------------------------------------------


template <class T>

  inline SparseMatrix<T>&   operator +=

  ( SparseMatrix<T>&          a,
    const SparseMatrix<T>&    b )

{
  a = unite ( a, b, Plus<T>() );

  return a;
}


//-----------------------------------------------------------------------
//   operator -=
//-----------------------------------------------------------------------


template <class T>

  inline SparseMatrix<T>&   operator -=

  ( SparseMatrix<T>&          a,
    const SparseMatrix<T>&    b )

{
  a = unite ( a, b, Minus<T>() );

  return a;
}


//-----------------------------------------------------------------------
//   operator *=
//-----------------------------------------------------------------------


template <class T>

  inline
  const SparseMatrix<T>&    operator *=

  ( const SparseMatrix<T>&    a,
    const T&                  b )

{
  a.getValues().assign ( b, Multiplies<T>() );

  return a;
}


template <class T>

  inline SparseMatrix<T>&   operator *=

  ( SparseMatrix<T>&          a,
    const SparseMatrix<T>&    b )

{
  a = intersect ( a, b, Multiplies<T>() );

  return a;
}


//-----------------------------------------------------------------------
//   operator /=
//-----------------------------------------------------------------------


template <class T>

  inline
  const SparseMatrix<T>&    operator /=

  ( const SparseMatrix<T>&    a,
    const T&                  b )

{
  a.getValues().assign ( b, Divides<T>() );

  return a;
}


//-----------------------------------------------------------------------
//   operator %=
//-----------------------------------------------------------------------


template <class T>

  inline
  const SparseMatrix<T>&    operator %=

  ( const SparseMatrix<T>&    a,
    const T&                  b )

{
  a.getValues().assign ( b, Modulus<T>() );

  return a;
}


//-----------------------------------------------------------------------
//   unite
//-----------------------------------------------------------------------


template <class T, class Op>

  SparseMatrix<T>         unite

  ( const SparseMatrix<T>&  a,
    const SparseMatrix<T>&  b,
    const Op&               op )

{
  JEM_PRECHECK2 ( equal( a.shape(), b.shape() ),
                  "SparseMatrix shape mismatch" );

  return unite_ (
    a, b, op,
    SparseFilterWrapper<T,PassFilter> ()
  );
}


template <class T, class Op, class F>

  SparseMatrix<T>           unite

  ( const SparseMatrix<T>&    a,
    const SparseMatrix<T>&    b,
    const Op&                 op,
    const SparseFilter<T,F>&  filter )

{
  JEM_PRECHECK2 ( equal( a.shape(), b.shape() ),
                  "SparseMatrix shape mismatch" );

  return unite_ (
    a, b, op,
    SparseFilterWrapper<T,F> ( filter )
  );
}


//-----------------------------------------------------------------------
//   intersect
//-----------------------------------------------------------------------


template <class T, class Op>

  SparseMatrix<T>         intersect

  ( const SparseMatrix<T>&  a,
    const SparseMatrix<T>&  b,
    const Op&               op )

{
  JEM_PRECHECK2 ( equal( a.shape(), b.shape() ),
                  "SparseMatrix shape mismatch" );

  return intersect_ (
    a, b, op,
    SparseFilterWrapper<T,PassFilter> ()
  );
}


template <class T, class Op, class F>

  SparseMatrix<T>           intersect

  ( const SparseMatrix<T>&    a,
    const SparseMatrix<T>&    b,
    const Op&                 op,
    const SparseFilter<T,F>&  filter )

{
  JEM_PRECHECK2 ( equal( a.shape(), b.shape() ),
                  "SparseMatrix shape mismatch" );

  return intersect_ (
    a, b, op,
    SparseFilterWrapper<T,F> ( filter )
  );
}


JEM_END_PACKAGE( numeric )

#endif
