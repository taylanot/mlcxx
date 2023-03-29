
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


#include <jem/base/assert.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/base/ParseException.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/io/TextOutput.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/numeric/sparse/SparseStructure.h>


JEM_BEGIN_PACKAGE( numeric )


using jem::io::DataInput;
using jem::io::DataOutput;
using jem::io::TextOutput;


//=======================================================================
//   class SparseStructure
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


SparseStructure::SparseStructure () : offsets_ ( 1 )
{
  shape_      = 0;
  offsets_[0] = 0;
}


SparseStructure::SparseStructure

  ( const Shape&         sh,
    const Array<idx_t>&  offsets,
    const Array<idx_t>&  indices ) :

    shape_   (      sh ),
    offsets_ ( offsets ),
    indices_ ( indices )

{
  JEM_PRECHECK2 ( sh[0] >= 0 && sh[1] >= 0,
                  "invalid SparseStructure shape" );
  JEM_PRECHECK2 ( sh[0] == offsets_.size() - 1,
                  "SparseStructure offset array has wrong length" );
}


SparseStructure::SparseStructure

  ( const Shape&    sh,
    Array<idx_t>&&  offsets,
    Array<idx_t>&&  indices ) :

    shape_   ( sh ),
    offsets_ ( std::move( offsets ) ),
    indices_ ( std::move( indices ) )

{
  JEM_PRECHECK2 ( sh[0] >= 0 && sh[1] >= 0,
                  "invalid SparseStructure shape" );
  JEM_PRECHECK2 ( sh[0] == offsets_.size() - 1,
                  "SparseStructure offset array has wrong length" );
}


SparseStructure::SparseStructure ( const Self& rhs ) noexcept :

  shape_   (   rhs.shape_ ),
  offsets_ ( rhs.offsets_ ),
  indices_ ( rhs.indices_ )

{}


SparseStructure::SparseStructure ( Self&& rhs ) noexcept :

  shape_   ( rhs.shape_ ),
  offsets_ ( std::move( rhs.offsets_ ) ),
  indices_ ( std::move( rhs.indices_ ) )

{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void SparseStructure::readFrom ( DataInput& in )
{
  decode ( in, shape_, offsets_, indices_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void SparseStructure::writeTo ( DataOutput& out ) const
{
  encode ( out, shape_, offsets_, indices_ );
}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


SparseStructure&
  SparseStructure::operator = ( const Self& rhs ) noexcept
{
  if ( this != &rhs )
  {
    shape_ = rhs.shape_;

    offsets_.ref ( rhs.offsets_ );
    indices_.ref ( rhs.indices_ );
  }

  return *this;
}


SparseStructure&
  SparseStructure::operator = ( Self&& rhs ) noexcept
{
  shape_ = rhs.shape_;

  offsets_.ref ( std::move( rhs.offsets_ ) );
  indices_.ref ( std::move( rhs.indices_ ) );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void SparseStructure::swap ( SparseStructure& rhs ) noexcept
{
  shape_  .swap (   rhs.shape_ );
  offsets_.swap ( rhs.offsets_ );
  indices_.swap ( rhs.indices_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


SparseStructure SparseStructure::clone () const
{
  Self tmp;

  tmp.shape_ = shape_;

  tmp.offsets_.ref ( offsets_.clone() );
  tmp.indices_.ref ( indices_.clone() );

  return tmp;
}


//-----------------------------------------------------------------------
//   transpose
//-----------------------------------------------------------------------


SparseStructure SparseStructure::transpose () const
{
  const idx_t   rowCount = shape_[0];
  const idx_t   colCount = shape_[1];
  const idx_t   nzCount  = nonZeroCount ();

  Array<idx_t>  trOffsets ( colCount + 1 );
  Array<idx_t>  trIndices ( nzCount );

  idx_t         first = 0;
  idx_t         last  = 0;


  // Count the number of entries in each row of the transposed matrix

  trOffsets = 0;
  first     = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    last = offsets_[irow + 1];

    for ( idx_t i = first; i < last; i++ )
    {
      trOffsets[indices_[i]]++;
    }

    first = last;
  }

  SparseUtils::sumOffsets ( trOffsets );

  first = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    last = offsets_[irow + 1];

    for ( idx_t i = first; i < last; i++ )
    {
      idx_t  icol = indices_[i];

      trIndices[trOffsets[icol]] = irow;
      trOffsets[icol]++;
    }

    first = last;
  }

  SparseUtils::shiftOffsets ( trOffsets );

  return SparseStructure ( Shape ( colCount, rowCount ),
                           trOffsets, trIndices );
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


bool SparseStructure::isValid () const noexcept
{
  const idx_t  nzCount = nonZeroCount ();

  if ( offsets_.size() == 0 )
  {
    return false;
  }

  if ( offsets_.front() != 0 || offsets_.back() != nzCount )
  {
    return false;
  }

  if ( testany( (offsets_ < 0) || (offsets_ > nzCount) ) )
  {
    return false;
  }

  if ( nzCount > 0 )
  {
    return testall ( (indices_ >= 0) && (indices_ < shape_[1]) );
  }
  else
  {
    return true;
  }
}


//-----------------------------------------------------------------------
//   isSorted
//-----------------------------------------------------------------------


bool SparseStructure::isSorted () const noexcept
{
  const idx_t  rowCount = shape_[0];

  if ( isContiguous() )
  {
    const idx_t* JEM_RESTRICT  offsets = offsets_.addr ();
    const idx_t* JEM_RESTRICT  indices = indices_.addr ();

    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      const idx_t  n = offsets[irow + 1] - 1;

      for ( idx_t i = offsets[irow]; i < n; i++ )
      {
        if ( indices[i] > indices[i + 1] )
        {
          return false;
        }
      }
    }
  }
  else
  {
    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      const idx_t  n = offsets_[irow + 1] - 1;

      for ( idx_t i = offsets_[irow]; i < n; i++ )
      {
        if ( indices_[i] > indices_[i + 1] )
        {
          return false;
        }
      }
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   isSymmetric
//-----------------------------------------------------------------------


bool SparseStructure::isSymmetric () const
{
  bool  result = false;

  if ( shape_[0] == shape_[1] )
  {
    const idx_t  rowCount = shape_[0];
    idx_t        upCount  = 0;
    idx_t        loCount  = 0;

    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      idx_t  n = offsets_[irow + 1];

      for ( idx_t i = offsets_[irow]; i < n; i++ )
      {
        idx_t  jcol = indices_[i];

        if      ( jcol > irow )
        {
          upCount++;
        }
        else if ( jcol < irow )
        {
          loCount++;
        }
      }
    }

    if ( loCount == upCount )
    {
      SparseStructure  trans     = transpose ();

      Array<idx_t>     trOffsets = trans.getRowOffsets    ();
      Array<idx_t>     trIndices = trans.getColumnIndices ();

      Array<idx_t>     mask      ( rowCount );


      result = true;
      mask   = -1_idx;

      for ( idx_t irow = 0; irow < rowCount; irow++ )
      {
        idx_t  n = offsets_[irow + 1];

        if ( n != trOffsets[irow + 1] )
        {
          return false;
        }

        for ( idx_t i = offsets_[irow]; i < n; i++ )
        {
          mask[indices_[i]] = irow;
        }

        for ( idx_t i = trOffsets[irow]; i < n; i++ )
        {
          if ( mask[trIndices[i]] != irow )
          {
            return false;
          }
        }
      }
    }
  }

  return result;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


SparseStructure             matmul

  ( const SparseStructure&    lhs,
    const SparseStructure&    rhs )

{
  JEM_PRECHECK2 ( lhs.size(1) == rhs.size(0),
                  "SparseStructure shape mismatch" );

  const idx_t   rowCount   = lhs.size(0);
  const idx_t   colCount   = rhs.size(1);

  Array<idx_t>  lhsOffsets = lhs.getRowOffsets    ();
  Array<idx_t>  lhsIndices = lhs.getColumnIndices ();

  Array<idx_t>  rhsOffsets = rhs.getRowOffsets    ();
  Array<idx_t>  rhsIndices = rhs.getColumnIndices ();

  Array<idx_t>  offsets    ( rowCount + 1 );
  Array<idx_t>  indices;

  Array<idx_t>  scratch    ( colCount * 2 + 1 );

  idx_t*        mask       = scratch.addr ();
  idx_t*        ilist      = mask + colCount;

  const idx_t*  rhsIdx     = nullptr;

  lint          opCount    = 0;
  idx_t         nnz        = 0;


  if ( ! rhsIndices.isContiguous() )
  {
    rhsIndices.ref ( rhsIndices.clone() );
  }

  for ( idx_t j = 0; j < colCount; j++ )
  {
    mask[j] = 1;
  }

  rhsIdx = rhsIndices.addr ();

  // Count the number of non-zeroes.

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  m = lhsOffsets[irow + 1];
    idx_t  k = 0;

    for ( idx_t i = lhsOffsets[irow]; i < m; i++ )
    {
      idx_t  icol = lhsIndices[i];
      idx_t  j    = rhsOffsets[icol];
      idx_t  n    = rhsOffsets[icol + 1];

      opCount    += (lint) (n - j);

      for ( ; j < n; j++ )
      {
        idx_t  jcol = rhsIdx[j];

        ilist[k]    = jcol;
        k          += mask[jcol];
        mask[jcol]  = 0;
      }
    }

    testCancelled ( opCount, JEM_FUNC,
                    "sparse structure multiply cancelled" );

    nnz += k;

    for ( idx_t j = 0; j < k; j++ )
    {
      mask[ilist[j]] = 1;
    }
  }

  // Allocate and fill the index array.

  indices.resize ( nnz );

  nnz = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  m = lhsOffsets[irow + 1];
    idx_t  k = 0;

    offsets[irow] = nnz;

    for ( idx_t i = lhsOffsets[irow]; i < m; i++ )
    {
      idx_t  icol = lhsIndices[i];
      idx_t  j    = rhsOffsets[icol];
      idx_t  n    = rhsOffsets[icol + 1];

      opCount    += (lint) (n - j);

      for ( ; j < n; j++ )
      {
        idx_t  jcol = rhsIdx[j];

        ilist[k]    = jcol;
        k          += mask[jcol];
        mask[jcol]  = 0;
      }
    }

    testCancelled ( opCount, JEM_FUNC,
                    "sparse structure multiply cancelled" );

    for ( idx_t j = 0; j < k; j++, nnz++ )
    {
      idx_t  jcol  = ilist[j];

      mask[jcol]   = 1;
      indices[nnz] = jcol;
    }
  }

  offsets[rowCount] = nnz;

  return SparseStructure ( shape( rowCount, colCount ),
                           offsets, indices );
}


//-----------------------------------------------------------------------
//   sortRows
//-----------------------------------------------------------------------


void sortRows ( SparseStructure& s )
{
  const idx_t   rowCount = s.size (0);
  const idx_t   colCount = s.size (1);

  Array<idx_t>  offsets  = s.getRowOffsets    ();
  Array<idx_t>  indices  = s.getColumnIndices ();

  idx_t         pos      = 0;


  s = SparseStructure ();

  for ( idx_t i = 0; i < rowCount; i++ )
  {
    idx_t  first = offsets[i];
    idx_t  last  = offsets[i + 1];

    offsets[i]  = pos;

    sort ( indices[slice(first,last)] );

    idx_t  jcol = -1;

    for ( idx_t j = first; j < last; j++ )
    {
      if ( indices[j] != jcol )
      {
        jcol         = indices[j];
        indices[pos] = jcol;
        pos++;
      }
    }
  }

  offsets[rowCount] = pos;

  if ( pos < indices.size() )
  {
    indices.reshape ( pos );
  }

  s = SparseStructure ( shape( rowCount, colCount ),
                        offsets, indices );
}


//-----------------------------------------------------------------------
//   findEqualRows
//-----------------------------------------------------------------------


void                        findEqualRows

  ( const Array<idx_t>&       rmap,
    const SparseStructure&    s )

{
  JEM_PRECHECK2 ( rmap.size() == s.size(0),
                  "row map array has wrong length" );

  SparseStructure       t = s.transpose ();

  Array<idx_t>  rowOffsets = s.getRowOffsets    ();
  Array<idx_t>  colIndices = s.getColumnIndices ();
  Array<idx_t>  colOffsets = t.getRowOffsets    ();
  Array<idx_t>  rowIndices = t.getColumnIndices ();

  const idx_t   rowCount   = s.size(0);
  const idx_t*  colIdx     = 0;


  if ( ! colIndices.isContiguous() )
  {
    colIndices.ref ( colIndices.clone() );
  }

  rmap   = -1;
  colIdx =  colIndices.addr ();

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    if ( rmap[irow] >= 0 )
    {
      continue;
    }

    idx_t  ifirst = rowOffsets[irow];
    idx_t  ilast  = rowOffsets[irow + 1];
    idx_t  icount = ilast - ifirst;

    if ( icount <= 0 )
    {
      continue;
    }

    idx_t  jcol   = colIndices[ifirst];
    idx_t  jcount = colOffsets[jcol + 1] - colOffsets[jcol];

    for ( idx_t i = ifirst + 1; i < ilast; i++ )
    {
      idx_t  icol = colIndices[i];
      idx_t  k    = colOffsets[icol + 1] - colOffsets[icol];

      if ( k < jcount )
      {
        jcount = k;
        jcol   = icol;
      }
    }

    idx_t  n = colOffsets[jcol + 1];

    for ( idx_t j = colOffsets[jcol]; j < n; j++ )
    {
      idx_t  jrow = rowIndices[j];

      if ( jrow <= irow )
      {
        continue;
      }

      idx_t  jfirst = rowOffsets[jrow];
      idx_t  jlast  = rowOffsets[jrow + 1];
      idx_t  jcount = jlast - jfirst;

      if ( jcount != icount )
      {
        continue;
      }

      for ( idx_t i = 0; i < icount; i++ )
      {
        if ( colIdx[i + ifirst] != colIdx[i + jfirst] )
        {
          icount = 0;
          break;
        }
      }

      if ( icount == jcount )
      {
        rmap[irow] = jrow;
        rmap[jrow] = irow;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   findSimilarRows
//-----------------------------------------------------------------------


void                        findSimilarRows

  ( const Array<idx_t>&       rmap,
    const SparseStructure&    s )

{
  JEM_PRECHECK2 ( rmap.size() == s.size(0),
                  "row map array has wrong length" );

  SparseStructure       t = s.transpose ();

  Array<idx_t>  rowOffsets = s.getRowOffsets    ();
  Array<idx_t>  colIndices = s.getColumnIndices ();
  Array<idx_t>  colOffsets = t.getRowOffsets    ();
  Array<idx_t>  rowIndices = t.getColumnIndices ();

  const idx_t   rowCount   = s.size(0);
  const idx_t   colCount   = s.size(1);

  Array<idx_t>  ibuf       ( colCount );

  idx_t*        colMask    = ibuf.addr ();
  const idx_t*  colIdx     = 0;


  if ( ! colIndices.isContiguous() )
  {
    colIndices.ref ( colIndices.clone() );
  }

  rmap   = -1;
  colIdx =  colIndices.addr ();

  for ( idx_t j = 0; j < colCount; j++ )
  {
    colMask[j] = -1;
  }

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    if ( rmap[irow] >= 0 )
    {
      continue;
    }

    idx_t  i      = rowOffsets[irow];
    idx_t  m      = rowOffsets[irow + 1];
    idx_t  icount = m - i;

    if ( icount <= 0 )
    {
      continue;
    }

    idx_t  jcol   = colIndices[i];
    idx_t  jcount = colOffsets[jcol + 1] - colOffsets[jcol];

    colMask[jcol] = irow + 1;

    for ( i++; i < m; i++ )
    {
      idx_t  icol = colIndices[i];
      idx_t  k    = colOffsets[icol + 1] - colOffsets[icol];

      colMask[icol] = irow + 1;

      if ( k < jcount )
      {
        jcount = k;
        jcol   = icol;
      }
    }

    idx_t  j = colOffsets[jcol];
    idx_t  n = colOffsets[jcol + 1];

    for ( ; j < n; j++ )
    {
      idx_t  jrow = rowIndices[j];

      if ( jrow <= irow )
      {
        continue;
      }

      i      = rowOffsets[jrow];
      m      = rowOffsets[jrow + 1];
      jcount = m - i;

      if ( jcount != icount )
      {
        continue;
      }

      for ( ; i < m; i++ )
      {
        jcount -= (colMask[colIdx[i]] - irow);
      }

      if ( jcount == 0 )
      {
        rmap[irow] = jrow;
        rmap[jrow] = irow;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


void                      print

  ( TextOutput&             out,
    const SparseStructure&  s )

{
  const idx_t   rowCount = s.size(0);

  Array<idx_t>  offsets  = s.getRowOffsets    ();
  Array<idx_t>  indices  = s.getColumnIndices ();


  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  i = offsets[irow];
    idx_t  n = offsets[irow + 1];

    if ( irow > 0 )
    {
      print ( out, '\n' );
    }

    print ( out, "row ", irow, ": " );

    for ( ; i < n; i++ )
    {
      idx_t  icol = indices[i];

      print ( out, icol );

      if ( i < n - 1 )
      {
        print ( out, ", " );
      }
    }
  }
}


JEM_END_PACKAGE( numeric )
