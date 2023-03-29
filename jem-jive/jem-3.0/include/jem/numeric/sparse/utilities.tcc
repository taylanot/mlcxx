
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

#ifndef JEM_NUMERIC_SPARSE_UTILITIES_TCC
#define JEM_NUMERIC_SPARSE_UTILITIES_TCC

#include <utility>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/numeric/NumberTraits.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   internal functions
//=======================================================================

//-----------------------------------------------------------------------
//   filter_
//-----------------------------------------------------------------------


template <class T>

  static SparseMatrix<T>            filter_

  ( const SparseMatrix<T>&            a,
    const SparseFilterInterface<T>&   f )

{
  typedef util::ArrayBuffer<idx_t>  IBuffer;
  typedef util::ArrayBuffer<T>      TBuffer;

  const idx_t          rowCount   = a.size(0);

  const Array<idx_t>&  inOffsets  = a.getRowOffsets    ();
  const Array<idx_t>&  inIndices  = a.getColumnIndices ();
  const Array<T>&      inValues   = a.getValues        ();

  Array<idx_t>         outOffsets ( rowCount + 1 );
  IBuffer              outIndices;
  TBuffer              outValues;


  outIndices.reserve ( a.nonZeroCount() );
  outValues. reserve ( outIndices.capacity() );

  outOffsets[0] = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  i = inOffsets[irow];
    idx_t  n = inOffsets[irow + 1];

    for ( ; i < n; i++ )
    {
      idx_t  jcol = inIndices[i];

      if ( f.accept( inValues[i], irow, jcol ) )
      {
        outIndices.pushBack ( jcol );
        outValues .pushBack ( inValues[i] );
      }
    }

    outOffsets[irow + 1] = outIndices.size();
  }

  return SparseMatrix<T> ( a.shape (),
                           std::move ( outOffsets ),
                           outIndices.toArray (),
                           outValues .toArray () );
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   diag
//-----------------------------------------------------------------------


template <class T>

  Array<T>  diag ( const SparseMatrix<T>& a )

{
  Array<T>  d ( min( a.size(0), a.size(1) ) );

  getDiag ( d, a );

  return d;
}


//-----------------------------------------------------------------------
//   getDiag
//-----------------------------------------------------------------------


template <class T>

  void                    getDiag

  ( const Array<T>&         d,
    const SparseMatrix<T>&  a )

{
  JEM_PRECHECK2 ( d.size() == min( a.size(0), a.size(1) ),
                  "diagional array has wrong size" );

  const idx_t          diagLen = d.size             ();

  const Array<idx_t>&  offsets = a.getRowOffsets    ();
  const Array<idx_t>&  indices = a.getColumnIndices ();
  const Array<T>&      values  = a.getValues        ();


  for ( idx_t irow = 0; irow < diagLen; irow++ )
  {
    idx_t  i  = offsets[irow];
    idx_t  n  = offsets[irow + 1];

    d[irow] = NumberTraits<T>::zero ();

    for ( ; i < n; i++ )
    {
      if ( indices[i] == irow )
      {
        d[irow] = values[i];

        break;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   sortRows
//-----------------------------------------------------------------------


template <class T>

  void sortRows ( SparseMatrix<T>& a )

{
  Array<idx_t>  offsets ( a.getRowOffsets()    );
  Array<idx_t>  indices ( a.getColumnIndices() );
  Array<T>      values  ( a.getValues()        );

  const idx_t   rowCount     = a.size(0);
  const idx_t   colCount     = a.size(1);

  const idx_t   maxRowLength = (idx_t)

    max ( offsets[slice(1,END)] -
          offsets[slice(BEGIN,rowCount)] );

  Array<idx_t>  ibuf ( maxRowLength );
  Array<T>      vbuf ( maxRowLength );
  Array<idx_t>  perm ( maxRowLength );

  idx_t         pos  = 0;


  a = SparseMatrix<T> ();

  for ( idx_t i = 0; i < rowCount; i++ )
  {
    idx_t  first = offsets[i];
    idx_t  last  = offsets[i + 1];
    idx_t  n     = last - first;

    offsets[i] = pos;

    if ( n == 0 )
    {
      continue;
    }

    for ( idx_t j = 0; j < n; j++ )
    {
      perm[j] = j;
    }

    sort ( perm[slice(BEGIN,n)], indices[slice(first,last)] );

    for ( idx_t j = 0; j < n; j++ )
    {
      ibuf[j] = indices [ first + perm[j] ];
      vbuf[j] = values  [ first + perm[j] ];
    }

    idx_t  jcol = -1;

    for ( idx_t j = 0; j < n; j++ )
    {
      if ( ibuf[j] != jcol )
      {
        jcol         = ibuf[j];
        indices[pos] = jcol;
        values [pos] = vbuf[j];
        pos++;
      }
    }
  }

  offsets[rowCount] = pos;

  a = SparseMatrix<T> ( shape ( rowCount, colCount ),
                        offsets,
                        indices[slice(BEGIN,pos)],
                        values[slice(BEGIN,pos)] );

}


//-----------------------------------------------------------------------
//   filter
//-----------------------------------------------------------------------


template <class T, class F>

  inline SparseMatrix<T>    filter

  ( const SparseMatrix<T>&    a,
    const SparseFilter<T,F>&  f )

{
  return filter_ ( a, SparseFilterWrapper<T,F>( f ) );
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


template <class T, class A>

  SparseMatrix<T> toMatrix ( const util::SparseArray<T,2,A>& a )

{
  typename util::SparseArray<T,2,A>::ConstIterator  first;
  typename util::SparseArray<T,2,A>::ConstIterator  last;

  const idx_t   rowCount = a.size(0);
  const idx_t   colCount = a.size(1);
  const idx_t   nzCount  = a.nonZeroCount ();

  Array<idx_t>  offsets ( rowCount + 1 );
  Array<idx_t>  indices ( nzCount );
  Array<T>      values  ( nzCount );


  offsets = 0;
  last    = a.end ();

  for ( first = a.begin(); first != last; ++first )
  {
    idx_t  irow = first->index[0];

    offsets[irow]++;
  }

  SparseUtils::sumOffsets ( offsets );

  for ( first = a.begin(); first != last; ++first )
  {
    idx_t  irow   = first->index[0];
    idx_t  jcol   = first->index[1];
    idx_t  i      = offsets[irow];

    indices[i]    = jcol;
    values [i]    = first->value;

    offsets[irow] = i + 1;
  }

  SparseUtils::shiftOffsets ( offsets );

  return SparseMatrix<T> ( shape     ( rowCount, colCount ),
                           std::move ( offsets ),
                           std::move ( indices ),
                           std::move ( values ) );
}


//-----------------------------------------------------------------------
//   toArray
//-----------------------------------------------------------------------


template <class T, class A>

  util::SparseArray<T,2,A>&   toArray

  ( util::SparseArray<T,2,A>&   out,
    const SparseMatrix<T>&      in )

{
  const Array<idx_t>&  offsets  = in.getRowOffsets    ();
  const Array<idx_t>&  indices  = in.getColumnIndices ();
  const Array<T>&      values   = in.getValues        ();

  const idx_t          rowCount = in.size(0);


  out.clear   ();
  out.reserve ( in.nonZeroCount() );
  out.reshape ( in.shape() );

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  i = offsets[irow];
    idx_t  n = offsets[irow + 1];

    for ( ; i < n; i++ )
    {
      idx_t  jcol    = indices[i];

      out(irow,jcol) = values[i];
    }
  }

  return out;
}


JEM_END_PACKAGE( numeric )

#endif
