
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <cmath>
#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Error.h>
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/util/Event.h>
#include <jem/util/SparseArray.h>
#include <jive/util/error.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/algebra/MBuilderParams.h>
#include <jive/algebra/MBuilderFactory.h>
#include <jive/algebra/FlexMatrixBuilder.h>
#include "private/search.h"


JEM_DEFINE_CLASS( jive::algebra::FlexMatrixBuilder );


JIVE_BEGIN_PACKAGE( algebra )


using jem::min;
using jem::max;
using jem::shape;
using jem::newInstance;
using jem::util::SparseArray;


//=======================================================================
//   class FlexMatrixBuilder::Data_
//=======================================================================


class FlexMatrixBuilder::Data_ : public jem::Collectable
{
 public:

  typedef Data_             Self;

  typedef
    SparseArray<double,2>   MBuffer;


  explicit                  Data_

    ( Ref<SparseMatrixObj>    mat );

  inline void               checkRow

    ( idx_t                   irow )           const;

  inline void               checkCol

    ( idx_t                   jcol )           const;

  inline void               checkRowStrict

    ( idx_t                   irow );

  inline void               checkColStrict

    ( idx_t                   jcol );

  inline void               clear           ();

  inline void               scale

    ( double                  s );

  inline void               setValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline void               addValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline double             getValue

    ( idx_t                   irow,
      idx_t                   jcol  )          const;

  inline bool               eraseValue

    ( idx_t                   irow,
      idx_t                   jcol  );

  void                      reshape

    ( idx_t                   m,
      idx_t                   n );

  void                      syncShape       ();
  void                      eraseZeroes     ();
  void                      updateMatrix    ();


 public:

  Ref<DofSpace>             rowSpace;
  Ref<DofSpace>             colSpace;
  Ref<SparseMatrixObj>      output;
  MBuffer                   mbuffer;
  Options                   options;
  double                    multiplier;
  idx_t                     rowCount;
  idx_t                     colCount;
  IdxVector                 rowOffsets;
  IdxVector                 colIndices;
  Vector                    mvalues;
  bool                      newValues;
  bool                      newStruct;


 private:

  inline void               updateOutput_   ();
  void                      mergeBuffer_    ();
  void                      sortRows_       ();

  void                      shrinkMatrix_

    ( idx_t                   m,
      idx_t                   n );

  void                      reallocMatrix_

    ( idx_t                   nnz );

  void                      valuesChanged_  ();
  void                      structChanged_  ();

  void                      checkRowStrict_

    ( idx_t                   irow );

  void                      checkColStrict_

    ( idx_t                   jcol );

  void                      rowIndexError_

    ( idx_t                   irow )           const;

  void                      colIndexError_

    ( idx_t                   jcol )           const;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


FlexMatrixBuilder::Data_::Data_ ( Ref<SparseMatrixObj> mat ) :

  output     ( mat ),
  rowOffsets (   1 )

{
  using jem::util::connect;

  options       = AUTO_RESHAPE;
  multiplier    = 1.0;
  rowCount      = 0;
  colCount      = 0;
  newValues     = true;
  newStruct     = true;
  rowOffsets[0] = 0;

  connect ( output->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( output->newStructEvent, this, & Self::structChanged_ );
}


//-----------------------------------------------------------------------
//   checkRow & checkCol
//-----------------------------------------------------------------------


inline void FlexMatrixBuilder::Data_::checkRow ( idx_t irow ) const
{
  if ( irow < 0 )
  {
    rowIndexError_ ( irow );
  }
}


inline void FlexMatrixBuilder::Data_::checkCol ( idx_t jcol ) const
{
  if ( jcol < 0 )
  {
    colIndexError_ ( jcol );
  }
}


//-----------------------------------------------------------------------
//   checkRowStrict & checkColStrict
//-----------------------------------------------------------------------


inline void FlexMatrixBuilder::Data_::checkRowStrict ( idx_t irow )
{
  if ( irow < 0 || irow >= rowCount )
  {
    checkRowStrict_ ( irow );
  }
}


inline void FlexMatrixBuilder::Data_::checkColStrict ( idx_t jcol )
{
  if ( jcol < 0 || jcol >= colCount )
  {
    checkColStrict_ ( jcol );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void FlexMatrixBuilder::Data_::clear ()
{
  mbuffer   .clear   ();
  colIndices.reshape ( 0 );
  mvalues   .reshape ( 0 );

  rowOffsets = 0;

  if ( options & AUTO_SHAPE0 )
  {
    rowOffsets.reshape ( 1 );

    rowCount = 0;
  }

  if ( options & AUTO_SHAPE1 )
  {
    colCount = 0;
  }

  syncShape ();

  mbuffer.reshape ( shape( rowCount, colCount ) );

  updateOutput_ ();

  newStruct = newValues = false;
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


inline void FlexMatrixBuilder::Data_::scale ( double s )
{
  MBuffer::Iterator  it  = mbuffer.begin ();
  MBuffer::Iterator  end = mbuffer.end   ();

  for ( ; it != end; ++it )
  {
    it->value *= s;
  }

  mvalues  *= s;
  newValues = true;
}


//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


inline void FlexMatrixBuilder::Data_::setValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  if ( irow < rowCount && jcol < colCount )
  {
    idx_t  k = binarySearch ( jcol,
                              colIndices.addr(),
                              rowOffsets[irow],
                              rowOffsets[irow + 1] );

    if ( k >= 0 )
    {
      mvalues[k] = multiplier * value;
      newValues  = true;

      return;
    }
  }

  if ( ! (options & DROP_ZEROES) || std::fabs( value ) > 0.0 )
  {
    mbuffer(irow,jcol) = multiplier * value;
  }
}


//-----------------------------------------------------------------------
//   addValue
//-----------------------------------------------------------------------


inline void FlexMatrixBuilder::Data_::addValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  if ( irow < rowCount && jcol < colCount )
  {
    idx_t  k = binarySearch ( jcol,
                              colIndices.addr(),
                              rowOffsets[irow],
                              rowOffsets[irow + 1] );

    if ( k >= 0 )
    {
      mvalues[k] += multiplier * value;
      newValues   = true;

      return;
    }
  }

  if ( ! (options & DROP_ZEROES) || std::fabs( value ) > 0.0 )
  {
    mbuffer(irow,jcol) += multiplier * value;
  }
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


inline double FlexMatrixBuilder::Data_::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  if ( irow < rowCount && jcol < colCount )
  {
    idx_t  k = binarySearch ( jcol,
                              colIndices.addr(),
                              rowOffsets[irow],
                              rowOffsets[irow + 1] );

    if ( k >= 0 )
    {
      return mvalues[k];
    }
  }

  const double*  xptr = mbuffer.find ( irow, jcol );

  return ( xptr ? *xptr : 0.0 );
}


//-----------------------------------------------------------------------
//   eraseValue
//-----------------------------------------------------------------------


inline bool FlexMatrixBuilder::Data_::eraseValue

  ( idx_t  irow,
    idx_t  jcol )

{
  if ( mbuffer.erase( irow, jcol ) )
  {
    return true;
  }

  if ( irow < rowCount && jcol < colCount )
  {
    idx_t  k = binarySearch ( jcol,
                              colIndices.addr(),
                              rowOffsets[irow],
                              rowOffsets[irow + 1] );

    if ( k >= 0 )
    {
      mvalues[k] = 0.0;
      newValues  = true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   reshape
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::reshape ( idx_t m, idx_t n )
{
  if ( m == rowCount && n == colCount )
  {
    return;
  }

  if ( m < rowCount || n < colCount )
  {
    shrinkMatrix_ ( m, n );
  }

  if ( m > rowCount )
  {
    rowOffsets.reshape ( m + 1 );

    rowOffsets[slice(rowCount,END)] = rowOffsets[rowCount];
  }

  if ( mbuffer.size(0) != m || mbuffer.size(1) != n )
  {
    idx_t  k = max( m, n ) - rowOffsets[rowCount];

    mbuffer.reshape ( shape( m, n ) );

    if ( k > 0 )
    {
      mbuffer.reserve ( k );
    }
  }

  rowCount  = m;
  colCount  = n;
  newStruct = true;
}


//-----------------------------------------------------------------------
//   syncShape
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::syncShape ()
{
  idx_t  m = rowCount;
  idx_t  n = colCount;

  if ( rowSpace )
  {
    m = rowSpace->dofCount ();
  }

  if ( colSpace )
  {
    n = colSpace->dofCount ();
  }

  if ( m != rowCount || n != colCount )
  {
    reshape ( m, n );
  }
}


//-----------------------------------------------------------------------
//   eraseZeroes
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::eraseZeroes ()
{
  idx_t*   jcols = colIndices.addr ();
  double*  mvals = mvalues   .addr ();

  idx_t    irow;
  idx_t    i, j, k;


  k = 0;

  for ( irow = 0; irow < rowCount; irow++ )
  {
    i = rowOffsets[irow];
    j = rowOffsets[irow + 1];

    rowOffsets[irow] = k;

    for ( ; i < j; i++ )
    {
      if ( std::fabs( mvals[i] ) > 0.0 )
      {
        jcols[k] = jcols[i];
        mvals[k] = mvals[i];
        k++;
      }
    }
  }

  if ( k > rowOffsets[rowCount] )
  {
    throw jem::Error (
      output->getContext (),
      "oops, more non-zeroes in matrix after erasing zeroes"
    );
  }

  if ( rowOffsets[rowCount] != k )
  {
    rowOffsets[rowCount] = k;
    newStruct            = true;

    reallocMatrix_ ( k );
  }
}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::updateMatrix ()
{
  using jem::System;

  syncShape ();

  if ( mbuffer.nonZeroCount() )
  {
    mergeBuffer_ ();
  }

  if ( ! (newStruct || newValues) )
  {
    return;
  }

  if ( mvalues.size() > 0 )
  {
    double  s = jem::sum ( mvalues );

    if ( jem::Float::isNaN( s ) )
    {
      throw jem::IllegalInputException (
        output->getContext (),
        "invalid matrix element(s): NaN"
      );
    }
  }

  if      ( newStruct )
  {
    String  matName = output->getName ();

    print ( System::debug( matName ), matName,
            " : updating sparse matrix structure ...\n" );

    updateOutput_ ();
  }
  else if ( newValues )
  {
    const idx_t  nnz     = rowOffsets[rowCount];
    String       matName = output->getName ();

    print ( System::debug( matName ), matName,
            " : updating sparse matrix values ...\n" );

    output->setValues ( mvalues[slice(BEGIN,nnz)] );
  }

  newStruct = newValues = false;

  output->resetEvents ();
}


//-----------------------------------------------------------------------
//   updateOutput_
//-----------------------------------------------------------------------


inline void FlexMatrixBuilder::Data_::updateOutput_ ()
{
  const idx_t  nnz = rowOffsets[rowCount];

  SparseMatrix  tmp ( shape ( rowCount, colCount ),
                      rowOffsets,
                      colIndices[slice(BEGIN,nnz)],
                      mvalues   [slice(BEGIN,nnz)] );

  output->setMatrix ( tmp );
}


//-----------------------------------------------------------------------
//   mergeBuffer_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::mergeBuffer_ ()
{
  using jem::numeric::SparseUtils;

  MBuffer::Iterator  end = mbuffer.end ();
  MBuffer::Iterator  it;

  const idx_t        m = max ( rowCount, mbuffer.size(0) );
  const idx_t        n = max ( colCount, mbuffer.size(1) );

  IdxVector          newOffsets ( m + 1 );

  idx_t*             jcols;
  double*            mvals;
  idx_t              irow;
  idx_t              i, j, k;


  newOffsets = 0_idx;

  for ( irow = 0; irow < rowCount; irow++ )
  {
    newOffsets[irow] = rowOffsets[irow + 1] - rowOffsets[irow];
  }

  for ( it = mbuffer.begin(); it != end; ++it )
  {
    irow = it->index[0];

    newOffsets[irow]++;
  }

  SparseUtils::sumOffsets ( newOffsets );

  k = rowOffsets[rowCount] + mbuffer.nonZeroCount ();

  if ( k != newOffsets[m] )
  {
    throw jem::Error (
      output->getContext (),
      "inconsistent number of non-zeroes in matrix"
    );
  }

  reallocMatrix_ ( newOffsets[m] );

  jcols = colIndices.addr ();
  mvals = mvalues   .addr ();

  for ( irow = rowCount - 1; irow >= 0; irow-- )
  {
    i = rowOffsets[irow];
    j = rowOffsets[irow + 1] - 1;
    k = newOffsets[irow] + (j - i);

    newOffsets[irow] = k + 1;

    for ( ; j >= i; j--, k-- )
    {
      jcols[k] = jcols[j];
      mvals[k] = mvals[j];
    }
  }

  for ( it = mbuffer.begin(); it != end; ++it )
  {
    irow     = it->index[0];
    k        = newOffsets[irow];

    jcols[k] = it->index[1];
    mvals[k] = it->value;

    newOffsets[irow] = k + 1;
  }

  SparseUtils::shiftOffsets ( newOffsets );

  rowOffsets.swap ( newOffsets  );
  newOffsets.ref  ( IdxVector() );

  rowCount = m;
  colCount = n;

  // Make sure the current matrix buffer frees all memory

  MBuffer  newBuffer;

  newBuffer.reshape ( jem::shape( m, n ) );
  newBuffer.swap    ( mbuffer );

  sortRows_ ();

  newStruct = true;
}


//-----------------------------------------------------------------------
//   sortRows_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::sortRows_ ()
{
  idx_t*     jcols = colIndices.addr ();
  double*    mvals = mvalues   .addr ();

  IdxVector  iperm;
  Vector     rbuf;

  bool       sorted;
  idx_t      irow;
  idx_t      i, j, k;


  for ( irow = 0; irow < rowCount; irow++ )
  {
    i = rowOffsets[irow];
    j = rowOffsets[irow + 1];
    k = j - i;

    if ( k <= 1 )
    {
      continue;
    }

    if ( k != iperm.size() )
    {
      iperm.resize ( k );
      rbuf .resize ( k );
    }

    sorted   = true;
    iperm[0] = i;

    for ( k = 1, i++; i < j; i++, k++ )
    {
      iperm[k] = i;

      if ( jcols[i] < jcols[i - 1] )
      {
        sorted = false;
      }
    }

    if ( sorted )
    {
      continue;
    }

    jem::sort ( iperm, colIndices );

    for ( i = 0; i < k; i++ )
    {
      j        = iperm[i];
      iperm[i] = jcols[j];
      rbuf[i]  = mvals[j];
    }

    i = rowOffsets[irow];

    std::memcpy ( & jcols[i], iperm.addr(),
                  (size_t) k * sizeof(idx_t)  );
    std::memcpy ( & mvals[i], rbuf .addr(),
                  (size_t) k * sizeof(double) );

#ifndef NDEBUG

    for ( i++; i < j; i++ )
    {
      if ( jcols[i] < 0 || jcols[i] <= jcols[i - 1] )
      {
        throw jem::Error (
          output->getContext (),
          "invalid column indices in matrix row " + String ( irow )
        );
      }
    }

#endif

  }
}


//-----------------------------------------------------------------------
//   shrinkMatrix_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::shrinkMatrix_ ( idx_t m, idx_t n )
{
  if ( m < rowCount && n >= colCount )
  {
    rowOffsets.reshape ( m + 1 );

    reallocMatrix_ ( rowOffsets[m] );
  }
  else
  {
    idx_t*   jcols = colIndices.addr ();
    double*  mvals = mvalues   .addr ();

    idx_t    irow;
    idx_t    jcol;
    idx_t    i, j, k;


    m = min ( m, rowCount );
    k = 0;

    for ( irow = 0; irow < m; irow++ )
    {
      i = rowOffsets[irow];
      j = rowOffsets[irow + 1];

      rowOffsets[irow] = k;

      for ( ; i < j; i++ )
      {
        jcol = jcols[i];

        if ( jcol < n && std::fabs( mvals[i] ) > 0.0 )
        {
          jcols[k] = jcol;
          mvals[k] = mvals[i];
          k++;
        }
      }
    }

    rowOffsets[m] = k;

    rowOffsets.reshape ( m + 1 );

    reallocMatrix_ ( k );
  }

  rowCount = m;
  colCount = n;
}


//-----------------------------------------------------------------------
//   reallocMatrix_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::reallocMatrix_ ( idx_t nnz )
{
  using jem::minOf;
  using jem::maxOf;

  idx_t  k = (idx_t) (0.2 * (double) nnz);

  if ( nnz > maxOf( nnz ) - k )
  {
    k = maxOf ( nnz );
  }
  else
  {
    k = nnz + k;
  }

  if ( nnz <= mvalues.size() )
  {
    k = jem::min ( k, mvalues.size() );

    if ( k != mvalues.size() )
    {
      colIndices.reshape ( k );
      mvalues   .reshape ( k );
    }
  }
  else
  {
    idx_t  j = mvalues.size ();

    colIndices.reshape ( k );
    mvalues   .reshape ( k );

    colIndices[slice(j,END)] = minOf<idx_t> ();
    mvalues   [slice(j,END)] = 0.0;
  }
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::valuesChanged_ ()
{
  newValues = true;
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::structChanged_ ()
{
  newStruct = true;
}


//-----------------------------------------------------------------------
//   checkRowStrict_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::checkRowStrict_ ( idx_t irow )
{
  if ( rowSpace )
  {
    syncShape ();
  }

  if ( irow < 0 || irow >= rowCount )
  {
    util::indexError ( output->getContext (),
                       "row", irow, rowCount );
  }
}


//-----------------------------------------------------------------------
//   checkColStrict_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::checkColStrict_ ( idx_t jcol )
{
  if ( colSpace )
  {
    syncShape ();
  }

  if ( jcol < 0 || jcol >= colCount )
  {
    util::indexError ( output->getContext(),
                       "column", jcol, colCount );
  }
}


//-----------------------------------------------------------------------
//   rowIndexError_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::rowIndexError_ ( idx_t irow ) const
{
  throw jem::IllegalIndexException (
    output->getContext (),
    "negative row index: " + String ( irow )
  );
}


//-----------------------------------------------------------------------
//   colIndexError_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::Data_::colIndexError_ ( idx_t jcol ) const
{
  throw jem::IllegalIndexException (
    output->getContext (),
    "negative column index: " + String ( jcol )
  );
}


//=======================================================================
//   class FlexMatrixBuilder
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  FlexMatrixBuilder::TYPE_NAME = "Sparse";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FlexMatrixBuilder::FlexMatrixBuilder

  ( const String&         name,
    Ref<SparseMatrixObj>  mat ) :

    Super ( name )

{
  if ( ! mat )
  {
    mat = newInstance<SparseMatrixObj> ( myName_ );
  }

  data_ = newInstance<Data_> ( mat );
}


FlexMatrixBuilder::~FlexMatrixBuilder ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void FlexMatrixBuilder::clear ()
{
  data_->clear ();
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void FlexMatrixBuilder::scale ( double s )
{
  data_->scale ( s );
}


//-----------------------------------------------------------------------
//   setToZero
//-----------------------------------------------------------------------


void FlexMatrixBuilder::setToZero ()
{
  data_->mbuffer   = 0.0;
  data_->mvalues   = 0.0;
  data_->newValues = true;

  data_->syncShape ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void FlexMatrixBuilder::reserve ( idx_t n )
{
  Data_&  d = * data_;

  n -= d.rowOffsets[d.rowCount];

  if ( n > 0 )
  {
    d.mbuffer.reserve ( n );
  }
}


//-----------------------------------------------------------------------
//   shapeHint
//-----------------------------------------------------------------------


void FlexMatrixBuilder::shapeHint

  ( idx_t  rowCount,
    idx_t  colCount )

{
  JEM_PRECHECK2 ( rowCount >= 0 && colCount >= 0,
                  "invalid matrix shape" );

  Data_&  d = * data_;

  d.syncShape ();

  if ( ! (d.options & AUTO_RESHAPE) )
  {
    return;
  }

  idx_t  m = d.rowCount;
  idx_t  n = d.colCount;

  if ( d.options & AUTO_SHAPE0 )
  {
    m = max ( m, rowCount );
  }

  if ( d.options & AUTO_SHAPE1 )
  {
    n = max ( n, colCount );
  }

  if ( m > d.rowCount || n > d.colCount )
  {
    d.reshape ( m, n );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void FlexMatrixBuilder::trimToSize ()
{
  data_->syncShape          ();
  data_->eraseZeroes        ();
  data_->mbuffer.trimToSize ();
}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void FlexMatrixBuilder::updateMatrix ()
{
  data_->updateMatrix ();
}


//-----------------------------------------------------------------------
//   setMultiplier
//-----------------------------------------------------------------------


void FlexMatrixBuilder::setMultiplier ( double x )
{
  data_->multiplier = x;
}


//-----------------------------------------------------------------------
//   getMultiplier
//-----------------------------------------------------------------------


double FlexMatrixBuilder::getMultiplier () const
{
  return data_->multiplier;
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void FlexMatrixBuilder::setData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  values )

{
  Data_&  d = * data_;

  double  tmp;
  idx_t   first, last;
  idx_t   irow,  jcol;
  idx_t   i, j, k;


  if ( icount * jcount == 1 )
  {
    irow = irows[0];
    jcol = jcols[0];

    if ( d.options & AUTO_SHAPE0 )
    {
      d.checkRow       ( irow );
    }
    else
    {
      d.checkRowStrict ( irow );
    }

    if ( d.options & AUTO_SHAPE1 )
    {
      d.checkCol       ( jcol );
    }
    else
    {
      d.checkColStrict ( jcol );
    }

    d.setValue ( irow, jcol, values[0] );
  }
  else
  {
    const idx_t* JEM_RESTRICT  colIndices = d.colIndices.addr ();

    const bool   drop0s = ((d.options & DROP_ZEROES) != 0);


    if ( d.options & AUTO_SHAPE0 )
    {
      for ( i = 0; i < icount; i++ )
      {
        d.checkRow ( irows[i] );
      }
    }
    else
    {
      for ( i = 0; i < icount; i++ )
      {
        d.checkRowStrict ( irows[i] );
      }
    }

    if ( d.options & AUTO_SHAPE1 )
    {
      for ( j = 0; j < jcount; j++ )
      {
        d.checkCol ( jcols[j] );
      }
    }
    else
    {
      for ( j = 0; j < jcount; j++ )
      {
        d.checkColStrict ( jcols[j] );
      }
    }

    for ( i = 0; i < icount; i++, values++ )
    {
      irow = irows[i];

      if ( irow >= d.rowCount )
      {
        for ( j = 0; j < jcount; j++ )
        {
          tmp = values[j * icount];

          if ( ! drop0s || std::fabs( tmp ) > 0.0 )
          {
            d.mbuffer(irow,jcols[j]) = d.multiplier * tmp;
          }
        }

        continue;
      }

      first = d.rowOffsets[irow];
      last  = d.rowOffsets[irow + 1];

      for ( j = 0; j < jcount; )
      {
        tmp = values[j * icount];
        k   = binarySearch ( jcols[j], colIndices, first, last );

        if ( k >= 0 )
        {
          d.mvalues[k] = d.multiplier * tmp;
          d.newValues  = true;

          for ( j++, k++;
                k < last && j < jcount && jcols[j] == colIndices[k];
                j++, k++ )
          {
            d.mvalues[k] = d.multiplier * values[j * icount];
          }
        }
        else
        {
          if ( ! drop0s || std::fabs( tmp ) > 0.0 )
          {
            d.mbuffer(irow,jcols[j]) = d.multiplier * tmp;
          }

          j++;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------


void FlexMatrixBuilder::addData

  ( const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount,
    const double*  values )

{
  Data_&  d = * data_;

  double  tmp;
  idx_t   first, last;
  idx_t   irow,  jcol;
  idx_t   i, j, k;


  if ( icount * jcount == 1 )
  {
    irow = irows[0];
    jcol = jcols[0];

    if ( d.options & AUTO_SHAPE0 )
    {
      d.checkRow       ( irow );
    }
    else
    {
      d.checkRowStrict ( irow );
    }

    if ( d.options & AUTO_SHAPE1 )
    {
      d.checkCol       ( jcol );
    }
    else
    {
      d.checkColStrict ( jcol );
    }

    d.addValue ( irow, jcol, values[0] );
  }
  else
  {
    const idx_t* JEM_RESTRICT  colIndices = d.colIndices.addr ();

    const bool   drop0s = ((d.options & DROP_ZEROES) != 0);


    if ( d.options & AUTO_SHAPE0 )
    {
      for ( i = 0; i < icount; i++ )
      {
        d.checkRow ( irows[i] );
      }
    }
    else
    {
      for ( i = 0; i < icount; i++ )
      {
        d.checkRowStrict ( irows[i] );
      }
    }

    if ( d.options & AUTO_SHAPE1 )
    {
      for ( j = 0; j < jcount; j++ )
      {
        d.checkCol ( jcols[j] );
      }
    }
    else
    {
      for ( j = 0; j < jcount; j++ )
      {
        d.checkColStrict ( jcols[j] );
      }
    }

    for ( i = 0; i < icount; i++, values++ )
    {
      irow = irows[i];

      if ( irow >= d.rowCount )
      {
        for ( j = 0; j < jcount; j++ )
        {
          tmp = values[j * icount];

          if ( ! drop0s || std::fabs( tmp ) > 0.0 )
          {
            d.mbuffer(irow,jcols[j]) += d.multiplier * tmp;
          }
        }

        continue;
      }

      first = d.rowOffsets[irow];
      last  = d.rowOffsets[irow + 1];

      for ( j = 0; j < jcount; )
      {
        tmp = values[j * icount];
        k   = binarySearch ( jcols[j], colIndices, first, last );

        if ( k >= 0 )
        {
          d.mvalues[k] += d.multiplier * tmp;
          d.newValues   = true;

          for ( j++, k++;
                k < last && j < jcount && jcols[j] == colIndices[k];
                j++, k++ )
          {
            d.mvalues[k] += d.multiplier * values[j * icount];
          }
        }
        else
        {
          if ( ! drop0s || std::fabs( tmp ) > 0.0 )
          {
            d.mbuffer(irow,jcols[j]) += d.multiplier * tmp;
          }

          j++;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   eraseData
//-----------------------------------------------------------------------


idx_t FlexMatrixBuilder::eraseData

  ( const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount )

{
  Data_&  d = * data_;

  idx_t   first, last;
  idx_t   irow,  jcol;
  idx_t   erased;
  idx_t   i, j, k;


  if ( icount * jcount == 1 )
  {
    irow = irows[0];
    jcol = jcols[0];

    d.checkRow ( irow );
    d.checkCol ( jcol );

    erased = d.eraseValue ( irow, jcol );
  }
  else
  {
    const idx_t* JEM_RESTRICT  colIndices = d.colIndices.addr ();

    erased = 0;

    for ( j = 0; j < jcount; j++ )
    {
      d.checkCol ( jcols[j] );
    }

    for ( i = 0; i < icount; i++ )
    {
      irow = irows[i];

      d.checkRow ( irow );

      if ( irow >= d.rowCount )
      {
        for ( j = 0; j < jcount; j++ )
        {
          if ( d.mbuffer.erase( irow, jcols[j] ) )
          {
            erased++;
          }
        }

        continue;
      }

      first = d.rowOffsets[irow];
      last  = d.rowOffsets[irow + 1];

      for ( j =0 ; j < jcount; )
      {
        k = binarySearch ( jcols[j], colIndices, first, last );

        if ( k >= 0 )
        {
          d.mvalues[k] = 0.0;
          d.newValues  = true;

          for ( j++, k++;
                k < last && j < jcount && jcols[j] == colIndices[k];
                j++, k++ )
          {
            d.mvalues[k] = 0.0;
          }
        }
        else
        {
          if ( d.mbuffer.erase( irow, jcols[j] ) )
          {
            erased++;
          }

          j++;
        }
      }
    }
  }

  return erased;
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void FlexMatrixBuilder::getData

  ( double*       buf,
    const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount ) const

{
  const Data_&   d = * data_;

  const double*  xptr;

  idx_t          first, last;
  idx_t          irow,  jcol;
  idx_t          i, j, k;


  if ( icount * jcount == 1 )
  {
    irow = irows[0];
    jcol = jcols[0];

    d.checkRow ( irow );
    d.checkCol ( jcol );

    buf[0] = d.getValue ( irow, jcol );
  }
  else
  {
    const idx_t* JEM_RESTRICT  colIndices = d.colIndices.addr ();

    for ( j = 0; j < jcount; j++ )
    {
      d.checkCol ( jcols[j] );
    }

    for ( i = 0; i < icount; i++, buf++ )
    {
      irow = irows[i];

      d.checkRow ( irow );

      if ( irow >= d.rowCount )
      {
        for ( j = 0; j < jcount; j++ )
        {
          xptr            = d.mbuffer.find ( irow, jcols[j] );
          buf[j * icount] = xptr ? *xptr : 0.0;
        }

        continue;
      }

      first = d.rowOffsets[irow];
      last  = d.rowOffsets[irow + 1];

      for ( j = 0; j < jcount; )
      {
        k = binarySearch ( jcols[j], colIndices, first, last );

        if ( k >= 0 )
        {
          buf[j * icount] = d.mvalues[k];

          for ( j++, k++;
                k < last && j < jcount && jcols[j] == colIndices[k];
                j++, k++ )
          {
            buf[j * icount] = d.mvalues[k];
          }
        }
        else
        {
          xptr            = d.mbuffer.find ( irow, jcols[j] );
          buf[j * icount] = xptr ? *xptr : 0.0;

          j++;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* FlexMatrixBuilder::getMatrix () const
{
  return data_->output.get ();
}


//-----------------------------------------------------------------------
//   getSparseMatrix
//-----------------------------------------------------------------------


SparseMatrixObj* FlexMatrixBuilder::getSparseMatrix () const
{
  return data_->output.get ();
}


//-----------------------------------------------------------------------
//   setShape
//-----------------------------------------------------------------------


void FlexMatrixBuilder::setShape ( idx_t rowCount, idx_t colCount )
{
  JEM_PRECHECK2 ( rowCount >= 0 && colCount >= 0,
                  "invalid matrix shape" );

  Data_&  d = * data_;

  if ( d.rowSpace )
  {
    rowCount = d.rowSpace->dofCount ();
  }

  if ( d.colSpace )
  {
    colCount = d.colSpace->dofCount ();
  }

  data_->reshape ( rowCount, colCount );
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void FlexMatrixBuilder::setOption

  ( Option  option,
    bool    yesno )

{
  data_->options.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void FlexMatrixBuilder::setOptions ( Options options )
{
  Data_&  d = * data_;

  if ( d.rowSpace )
  {
    options &= (~AUTO_SHAPE0);
  }

  if ( d.colSpace )
  {
    options &= (~AUTO_SHAPE1);
  }

  d.options = options;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


FlexMatrixBuilder::Options
  FlexMatrixBuilder::getOptions () const
{
  return data_->options;
}


//-----------------------------------------------------------------------
//   trackDofSpaces
//-----------------------------------------------------------------------


void FlexMatrixBuilder::trackDofSpaces

  ( const Ref<DofSpace>&  rowSpace,
    const Ref<DofSpace>&  colSpace )

{
  using jem::util::connect;
  using jem::util::disconnect;

  Data_&   d  = * data_;

  if ( d.rowSpace )
  {
    disconnect ( d.rowSpace->newSizeEvent,
                 this, & Self::newDofCount_ );
  }

  if ( d.colSpace )
  {
    disconnect ( d.colSpace->newSizeEvent,
                 this, & Self::newDofCount_ );
  }

  if ( rowSpace )
  {
    d.options &= (~AUTO_SHAPE0);
    d.rowSpace = rowSpace;

    connect ( rowSpace->newSizeEvent,
              this, & Self::newDofCount_ );
  }

  if ( colSpace )
  {
    d.options &= (~AUTO_SHAPE1);
    d.colSpace = colSpace;

    connect ( colSpace->newSizeEvent,
              this, & Self::newDofCount_ );
  }

  d.syncShape ();
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<MBuilder> FlexMatrixBuilder::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<DofSpace>  dofs;
  Ref<Self>      mbuilder =

    newInstance<Self> (
      myName,
      newSparseMatrix ( myName, conf, props, params )
    );

  if ( params.find( dofs, MBuilderParams::DOF_SPACE ) )
  {
    mbuilder->trackDofSpace ( dofs );
  }

  return mbuilder;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void FlexMatrixBuilder::declare ()
{
  MBuilderFactory::declare ( TYPE_NAME,  & makeNew );
  MBuilderFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   newDofCount_
//-----------------------------------------------------------------------


void FlexMatrixBuilder::newDofCount_ ()
{
  data_->syncShape ();
}


JIVE_END_PACKAGE( algebra )
