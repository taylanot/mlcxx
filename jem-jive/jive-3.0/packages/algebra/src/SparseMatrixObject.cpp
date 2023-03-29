
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
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/OutOfMemoryException.h>
#include <jem/base/tuple/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/error.h>
#include <jive/algebra/SparseMatrixObject.h>
#include "private/search.h"


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::SparseMatrixObject );


JIVE_BEGIN_PACKAGE( algebra )


using jive::util::sizeError;
using jive::util::indexError;


//=======================================================================
//   class SparseMatrixObject
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  SparseMatrixObject::SORTED_  = 1 << 0;
const int  SparseMatrixObject::BLOCKED_ = 1 << 1;
const int  SparseMatrixObject::PACKED_  = 1 << 2;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SparseMatrixObject::SparseMatrixObject

  ( const String&  name,
    Traits         traits ) :

    Super ( name )

{
  init_ ();

  traits_ = traits;
}


SparseMatrixObject::SparseMatrixObject

  ( const String&        name,
    const SparseMatrix&  matrix,
    Traits               traits ) :

    Super ( name )

{
  init_ ();

  traits_ = traits;

  setMatrix ( matrix );
}


SparseMatrixObject::~SparseMatrixObject ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void SparseMatrixObject::readFrom ( ObjectInput& in )
{
  SparseMatrix  matrix;

  decode    ( in, myName_, traits_, options_, matrix );
  setMatrix ( matrix );
}


void SparseMatrixObject::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, traits_, options_, toSparseMatrix() );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> SparseMatrixObject::clone () const
{
  return jem::newInstance<Self>

    ( myName_, cloneSparseMatrix(), traits_ );
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape SparseMatrixObject::shape () const
{
  return shape_;
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void SparseMatrixObject::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  JEM_PRECHECK2 ( lhs.size() == this->size(0) &&
                  rhs.size() == this->size(1),
                  "Array size mismatch" );

  if ( (status_ & BLOCKED_) && ! (status_ & PACKED_) )
  {
    Self*  self = const_cast<Self*> ( this );

    self->mcounter_--;

    if ( mcounter_ <= 0 )
    {
      self->packValues_ ();
    }
  }

  if ( status_ & PACKED_ )
  {
    packedMatmul_ ( lhs, rhs );
  }
  else
  {
    normalMatmul_ ( lhs, rhs );
  }
}


//-----------------------------------------------------------------------
//   multiMatmul
//-----------------------------------------------------------------------


void SparseMatrixObject::multiMatmul

  ( Matrix&           lhsVecs,
    IdxVector&        lhsTags,
    const Matrix&     rhsVecs,
    const IdxVector&  rhsTags ) const

{
  JEM_PRECHECK2 ( rhsVecs.size(0) == this->size(1) &&
                  rhsVecs.size(1) == rhsTags.size(),
                  "Array shape mismatch" );

  const idx_t  rhsCount = rhsTags.size ();

  idx_t        jcol;


  lhsVecs.resize ( shape_[0], rhsCount );
  lhsTags.resize ( rhsCount );

  lhsTags = rhsTags;

  if ( (status_ & BLOCKED_) && ! (status_ & PACKED_) )
  {
    Self*  self = const_cast<Self*> ( this );

    self->mcounter_ -= rhsCount;

    if ( mcounter_ <= 0 )
    {
      self->packValues_ ();
    }
  }

  // Multiply blocks of four columns at a time.

  for ( jcol = 0; jcol < rhsCount - 3; jcol += 4 )
  {
    Matrix  lhs = lhsVecs[slice(jcol,jcol + 4)];
    Matrix  rhs = rhsVecs[slice(jcol,jcol + 4)];

    if ( status_ & PACKED_ )
    {
      packedMatmul4_ ( lhs, rhs );
    }
    else
    {
      normalMatmul4_ ( lhs, rhs );
    }
  }

  // Multiply the remaining columns.

  for ( ; jcol < rhsCount; jcol++ )
  {
    Vector  lhs = lhsVecs[jcol];
    Vector  rhs = rhsVecs[jcol];

    if ( status_ & PACKED_ )
    {
      packedMatmul_ ( lhs, rhs );
    }
    else
    {
      normalMatmul_ ( lhs, rhs );
    }
  }
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool SparseMatrixObject::hasTrait ( const String& trait ) const
{
  if ( trait == MatrixTraits::SYMMETRIC )
  {
    return (traits_ & SYMMETRIC);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* SparseMatrixObject::getExtByID ( ExtensionID extID ) const
{
  if ( extID == DiagMatrixExt::ID )
  {
    DiagMatrixExt*    ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == MultiMatmulExt::ID )
  {
    MultiMatmulExt*   ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == DirectMatrixExt::ID )
  {
    DirectMatrixExt*  ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == SparseMatrixExt::ID )
  {
    SparseMatrixExt*  ext = const_cast<Self*> ( this );

    return ext;
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


double SparseMatrixObject::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  if ( irow < 0 || irow >= shape_[0] )
  {
    indexError ( getContext(), "row", irow, shape_[0] );
  }

  if ( jcol < 0 || jcol >= shape_[1] )
  {
    indexError ( getContext(), "column", jcol, shape_[1] );
  }

  const idx_t*  rowOffsets = rowOffsets_.addr ();
  const idx_t*  colIndices = colIndices_.addr ();

  idx_t         k;

  if ( status_ & SORTED_ )
  {
    k = binarySearch ( jcol, colIndices,
                       rowOffsets[irow], rowOffsets[irow + 1] );
  }
  else
  {
    k = linearSearch ( jcol, colIndices,
                       rowOffsets[irow], rowOffsets[irow + 1] );
  }

  if ( k < 0 )
  {
    return 0.0;
  }
  else
  {
    return matValues_[k];
  }
}


//-----------------------------------------------------------------------
//   getBlock
//-----------------------------------------------------------------------


void SparseMatrixObject::getBlock

  ( const Matrix&     block,
    const IdxVector&  irows,
    const IdxVector&  jcols ) const

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array shape mismatch" );

  const idx_t*   rowOffsets = rowOffsets_.addr ();
  const idx_t*   colIndices = colIndices_.addr ();
  const double*  matValues  = matValues_. addr ();

  const idx_t    rowCount   = shape_[0];
  const idx_t    colCount   = shape_[1];

  const idx_t    icount     = irows.size ();
  const idx_t    jcount     = jcols.size ();

  idx_t          k;


  for ( idx_t j = 0; j < jcount; j++ )
  {
    const idx_t  jcol = jcols[j];

    if ( jcol < 0 || jcol >= colCount )
    {
      indexError ( getContext(), "column", jcol, colCount );
    }
  }

  for ( idx_t i = 0; i < icount; i++ )
  {
    const idx_t  irow = irows[i];

    if ( irow < 0 || irow >= rowCount )
    {
      indexError ( getContext(), "row", irow, rowCount );
    }

    const idx_t  rbegin = rowOffsets[irow];
    const idx_t  rend   = rowOffsets[irow + 1];

    for ( idx_t j = 0; j < jcount; )
    {
      if ( status_ & SORTED_ )
      {
        k = binarySearch ( jcols[j], colIndices, rbegin, rend );
      }
      else
      {
        k = linearSearch ( jcols[j], colIndices, rbegin, rend );
      }

      if ( k < 0 )
      {
        block(i,j) = 0.0;

        j++;
      }
      else
      {
        block(i,j) = matValues[k];

        for ( j++, k++;
              j < jcount && k < rend && jcols[j] == colIndices[k];
              j++, k++ )
        {
          block(i,j) = matValues[k];
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getDiagonal
//-----------------------------------------------------------------------


void SparseMatrixObject::getDiagonal ( const Vector& diag ) const
{
  const idx_t*  JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  colIndices = colIndices_.addr ();
  const double* JEM_RESTRICT  matValues  = matValues_ .addr ();

  const idx_t   diagSize = min ( shape_ );


  if ( diag.size() != diagSize )
  {
    sizeError ( getContext(), "diagonal vector",
                diag.size(), diagSize );
  }

  for ( idx_t irow = 0; irow < diagSize; irow++ )
  {
    idx_t  rend  = rowOffsets[irow + 1];

    diag[irow] = 0.0;

    for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
    {
      if ( colIndices[j] == irow )
      {
        diag[irow] = matValues[j];

        break;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getRowScales
//-----------------------------------------------------------------------


void SparseMatrixObject::getRowScales ( const Vector& rscales ) const
{
  const idx_t*  JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const double* JEM_RESTRICT  matValues  = matValues_ .addr ();

  const idx_t   rowCount = shape_[0];


  if ( rscales.size() != rowCount )
  {
    sizeError ( getContext(), "row scale vector",
                rscales.size(), rowCount );
  }

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t   rend = rowOffsets[irow + 1];
    double  s    = 0.0;

    for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
    {
      double  t = std::fabs ( matValues[j] );

      if ( t > s )
      {
        s = t;
      }
    }

    rscales[irow] = s;
  }
}


//-----------------------------------------------------------------------
//   getStructure
//-----------------------------------------------------------------------


SparseStruct SparseMatrixObject::getStructure () const
{
  return SparseStruct ( shape_, rowOffsets_, colIndices_ );
}


//-----------------------------------------------------------------------
//   toSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix SparseMatrixObject::toSparseMatrix () const
{
  return SparseMatrix ( shape_,
                        rowOffsets_,
                        colIndices_,
                        matValues_ );
}


//-----------------------------------------------------------------------
//   cloneSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix SparseMatrixObject::cloneSparseMatrix () const
{
  return SparseMatrix ( shape_,
                        rowOffsets_.clone(),
                        colIndices_.clone(),
                        matValues_ .clone() );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void SparseMatrixObject::printTo ( PrintWriter& out ) const
{
  using jem::io::endl;
  using jem::io::space;

  const idx_t*  JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  colIndices = colIndices_.addr ();
  const double* JEM_RESTRICT  matValues  = matValues_ .addr ();

  const idx_t   rowCount = shape_[0];
  const idx_t   colCount = shape_[1];


  print ( out, rowCount, space, colCount, space, nonZeroCount() );

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  rend = rowOffsets[irow + 1];

    for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
    {
      print ( out, endl, irow + 1, space,
              colIndices[j] + 1, space, matValues[j] );
    }
  }
}


//-----------------------------------------------------------------------
//   setToZero
//-----------------------------------------------------------------------


void SparseMatrixObject::setToZero ()
{
  matValues_ = 0.0;

  if ( status_ & PACKED_ )
  {
    supValues_ = 0.0;
  }
  else
  {
    resetMCounter_ ();
  }

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   setMatrix
//-----------------------------------------------------------------------


void SparseMatrixObject::setMatrix ( const SparseMatrix&  matrix )
{
  using jem::makeContiguous;
  using jem::System;

  JEM_PRECHECK2 ( matrix.isValid(), "invalid sparse matrix" );

  shape_ = matrix.shape ();

  if ( ! matrix.isContiguous() )
  {
    print ( System::debug( myName_ ), myName_,
            " : non-contiguous matrix; creating a private copy\n" );
  }

  rowOffsets_.ref ( makeContiguous( matrix.getRowOffsets() ) );
  colIndices_.ref ( makeContiguous( matrix.getColumnIndices() ) );
  matValues_ .ref ( makeContiguous( matrix.getValues() ) );
  resetMCounter_  ();

  status_ = 0;

  if ( matrix.isSorted() )
  {
    status_ |= SORTED_;
  }

  supRows_  .ref ( IdxVector() );
  supValues_.ref ( Vector() );

  supOffsets_ = 0;

  if ( options_ & ENABLE_BLOCKING )
  {
    initSuperRows_ ();
  }

  newStructEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   addMatrix
//-----------------------------------------------------------------------


void SparseMatrixObject::addMatrix

  ( const SparseMatrix&  rhs,
    double               scale )

{
  using jem::equal;

  JEM_PRECHECK2 ( equal( this->shape(), rhs.shape() ),
                  "sparse matrix shape mismatch" );

  const idx_t  rowCount  = shape_[0];

  IdxVector    rhOffsets = rhs.getRowOffsets    ();
  IdxVector    rhIndices = rhs.getColumnIndices ();
  Vector       rhValues  = rhs.getValues        ();

  IdxVector    myOffsets = rowOffsets_;
  IdxVector    myIndices = colIndices_;
  Vector       myValues  = matValues_;

  IdxVector    mask;
  IdxVector    count;
  Vector       accu;

  bool         newStruct;

  idx_t        k;

  // Common case: identical structures.

  if ( equal( rhOffsets, myOffsets ) &&
       equal( rhIndices, myIndices ) )
  {
    myValues += scale * rhValues;

    setValues ( myValues );

    return;
  }

  // Count the number of new entries in this matrix.

  mask .resize ( rowCount );
  count.resize ( rowCount + 1 );

  mask = 1_idx;
  k    = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  rend = myOffsets[irow + 1];

    count[irow] = k;

    for ( idx_t j = myOffsets[irow]; j < rend; j++ )
    {
      mask[myIndices[j]] = 0;
    }

    rend = rhOffsets[irow + 1];

    for ( idx_t j = rhOffsets[irow]; j < rend; j++ )
    {
      k += mask[rhIndices[j]];
    }

    rend = myOffsets[irow + 1];

    for ( idx_t j = myOffsets[irow]; j < rend; j++ )
    {
      mask[myIndices[j]] = 1;
    }
  }

  count[rowCount] =  k;
  newStruct       = (k > 0 );

  if ( newStruct )
  {
    idx_t  n = k + myValues.size ();

    // Make sure that the index and value arrays are not shared.

    colIndices_.ref ( IdxVector() );
    matValues_ .ref ( Vector()    );

    myIndices.reshape ( n );
    myValues .reshape ( n );
  }

  // Merge the right-hand matrix with my matrix

  accu.resize ( rowCount );

  mask = rowCount;
  accu = 0.0;
  k   += myOffsets[rowCount];

  for ( idx_t irow = rowCount - 1; irow >= 0; irow-- )
  {
    idx_t  j = myOffsets[irow + 1] - 1;
    idx_t  n = myOffsets[irow];

    myOffsets[irow + 1] = k;

    for ( ; j >= n; j-- )
    {
      idx_t  jcol = myIndices[j];

      accu[jcol]     = myValues[j];
      mask[jcol]     = irow;
      myIndices[--k] = jcol;
    }

    n = rhOffsets[irow + 1];

    for ( j = rhOffsets[irow]; j < n; j++ )
    {
      idx_t  jcol = rhIndices[j];

      accu[jcol] += scale * rhValues[j];

      if ( mask[jcol] != irow )
      {
        mask[jcol]     = irow;
        myIndices[--k] = jcol;
      }
    }

    n = myOffsets[irow + 1];

    for ( j = k; j < n; j++ )
    {
      idx_t  jcol = myIndices[j];

      myValues[j] = accu[jcol];
      accu[jcol]  = 0.0;
    }
  }

  if ( k != 0 )
  {
    throw jem::Error ( getContext(),
                       "oops, lost a few matrix entries" );
  }

  if ( newStruct )
  {
    setMatrix ( SparseMatrix( rhs.shape (),
                              myOffsets, myIndices, myValues ) );
  }
  else
  {
    setValues ( myValues );
  }
}


//-----------------------------------------------------------------------
//   setValues
//-----------------------------------------------------------------------


void SparseMatrixObject::setValues ( const Vector& values )
{
  JEM_PRECHECK2 ( values.size() == nonZeroCount(),
                  "Array size mismatch" );

  matValues_.ref ( makeContiguous( values ) );
  resetMCounter_ ();

  status_ &= ~PACKED_;

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void SparseMatrixObject::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void SparseMatrixObject::setOptions ( Options options )
{
  if ( options & ENABLE_BLOCKING )
  {
    if ( ! (options_ & ENABLE_BLOCKING) )
    {
      initSuperRows_ ();
    }
  }
  else
  {
    status_ &= ~BLOCKED_;
    status_ &= ~PACKED_;

    supRows_  .ref ( IdxVector() );
    supValues_.ref ( Vector() );
  }

  options_ = options;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void SparseMatrixObject::init_ ()
{
  traits_     = 0;
  status_     = 0;
  options_    = 0;
  mcounter_   = 0;
  shape_      = 0;
  supOffsets_ = 0;

  rowOffsets_.resize ( 1 );

  rowOffsets_[0] = 0;
}


//-----------------------------------------------------------------------
//   packValues_
//-----------------------------------------------------------------------


void SparseMatrixObject::packValues_ ()
{
  using jem::System;
  using jem::OutOfMemoryException;

  JEM_PRECHECK2 ( status_ & BLOCKED_,
                  "invalid sparse matrix storage mode" );

  const idx_t*  JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  supRows    = supRows_   .addr ();
  const double* JEM_RESTRICT  matValues  = matValues_ .addr ();
  double*       JEM_RESTRICT  supValues  = nullptr;

  const double* JEM_RESTRICT  mp         = nullptr;
  double*       JEM_RESTRICT  sp         = nullptr;

  idx_t         ssize;
  idx_t         iend;


  print ( System::debug( myName_ ), myName_,
          " : packing matrix for better cache utilization ...\n" );

  try
  {
    supValues_.resize ( nonZeroCount() );
  }
  catch ( const OutOfMemoryException& )
  {
    print ( System::debug( myName_ ), myName_,
            " : not enough memory available\n" );

    supValues_.ref ( Vector() );

    status_ &= ~BLOCKED_;

    return;
  }

  supValues = supValues_.addr ();

  // Pack all super rows with size 1.

  ssize = 1;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];

    for ( idx_t j = rbegin; j < rend; j++ )
    {
      supValues[j] = matValues[j];
    }
  }

  // Pack all rows with size 2.

  ssize = 2;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];
    idx_t  rsize  = rend - rbegin;

    mp = matValues + rbegin;
    sp = supValues + rbegin;

    for ( idx_t j = 0; j < rsize; j++ )
    {
      sp[j * 2 + 0] = mp[j];
      sp[j * 2 + 1] = mp[j + rsize];
    }
  }

  // Pack all rows with size 3.

  ssize = 3;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];
    idx_t  rsize  = rend - rbegin;

    mp = matValues + rbegin;
    sp = supValues + rbegin;

    for ( idx_t j = 0; j < rsize; j++ )
    {
      sp[j * 3 + 0] = mp[j];
      sp[j * 3 + 1] = mp[j + rsize];
      sp[j * 3 + 2] = mp[j + rsize * 2];
    }
  }

  // Pack all rows with size 4.

  ssize = 4;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];
    idx_t  rsize  = rend - rbegin;

    mp = matValues + rbegin;
    sp = supValues + rbegin;

    for ( idx_t j = 0; j < rsize; j++ )
    {
      sp[j * 4 + 0] = mp[j];
      sp[j * 4 + 1] = mp[j + rsize];
      sp[j * 4 + 2] = mp[j + rsize * 2];
      sp[j * 4 + 3] = mp[j + rsize * 3];
    }
  }

  status_ |= PACKED_;
}


//-----------------------------------------------------------------------
//   initSuperRows_
//-----------------------------------------------------------------------


void SparseMatrixObject::initSuperRows_ ()
{
  using jem::Array;
  using jem::Tuple;
  using jem::System;

  const idx_t*    JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const idx_t*    JEM_RESTRICT  colIndices = colIndices_.addr ();

  const idx_t*    JEM_RESTRICT  iptr = nullptr;
  const idx_t*    JEM_RESTRICT  jptr = nullptr;

  const idx_t     rowCount = shape_[0];

  Array<byte>     rowSpan  ( rowCount );

  Tuple<idx_t,6>  supOffsets;

  bool            equal;
  idx_t           supCount;
  idx_t           ssize;
  idx_t           isize;
  idx_t           jsize;
  idx_t           jrow;


  if ( rowCount <= 0 )
  {
    return;
  }

  print ( System::debug( myName_ ), myName_,
          " : scanning matrix for super rows ...\n" );

  // Scan the matrix structure for super rows.

  jrow       = 0;
  jptr       = colIndices;
  jsize      = rowOffsets[1] - rowOffsets[0];
  ssize      = 1;
  supOffsets = 0_idx;
  rowSpan    = (byte) 0;

  for ( idx_t irow = 1; irow < rowCount; irow++ )
  {
    iptr  = colIndices + rowOffsets[irow];
    isize = rowOffsets[irow + 1] - rowOffsets[irow];
    equal = (isize == jsize);

    for ( idx_t j = 0; j < isize && equal; j++ )
    {
      equal = equal && (iptr[j] == jptr[j]);
    }

    if ( equal && (ssize < MAX_BLOCK_SIZE_) )
    {
      ssize++;
    }
    else
    {
      rowSpan[jrow] = (byte) ssize;

      supOffsets[ssize]++;

      jrow  = irow;
      jptr  = iptr;
      jsize = isize;
      ssize = 1;
    }
  }

  rowSpan[jrow] = (byte) ssize;

  supOffsets[ssize]++;

  // Count the total number of super rows and compute the super row
  // offsets.

  supCount = 0;

  for ( idx_t i = 0; i < (MAX_BLOCK_SIZE_ + 2); i++ )
  {
    idx_t  itmp = supOffsets[i];

    supOffsets[i] = supCount;
    supCount     += itmp;
  }

  print ( System::debug( myName_ ), myName_,
          " : done; found ", supCount, " super rows in ",
          rowCount, " matrix rows.\n" );

  // Only continue if the number of super rows is significantly less
  // than the total number of rows.

  if ( (double) supCount > (0.55 * (double) rowCount) )
  {
    print ( System::debug( myName_ ), myName_,
            " : not enough super rows to enable blocking mode.\n" );
    return;
  }

  // Allocate and initialize the super rows.

  supRows_.resize ( supCount );

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    ssize = (idx_t) rowSpan[irow];

    if ( ssize )
    {
      idx_t  k = supOffsets[ssize];

      supRows_[k]       = irow;
      supOffsets[ssize] = k + 1;
    }
  }

  // Initialize the super row offsets.

  supOffsets_[0] = 0;

  for ( idx_t i = 1; i < (MAX_BLOCK_SIZE_ + 2); i++ )
  {
    supOffsets_[i] = supOffsets[i - 1];
  }

  status_ |= BLOCKED_;
}


//-----------------------------------------------------------------------
//   resetMCounter_
//-----------------------------------------------------------------------


void SparseMatrixObject::resetMCounter_ ()
{
  // Pack the matrix sooner when it already has been packed before.

  if ( status_ & PACKED_ )
  {
    mcounter_ = 4;
  }
  else
  {
    mcounter_ = 8;
  }
}


//-----------------------------------------------------------------------
//   normalMatmul_
//-----------------------------------------------------------------------


void SparseMatrixObject::normalMatmul_

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  const idx_t*  JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  colIndices = colIndices_.addr ();
  const double* JEM_RESTRICT  matValues  = matValues_ .addr ();
  const double* JEM_RESTRICT  rhsValues  = rhs        .addr ();

  const idx_t   rowCount = shape_     [0];
  const idx_t   rst      = rhs.stride ();

  double        t;


  if ( rst == 1_idx )
  {
    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      idx_t  rend = rowOffsets[irow + 1];

      t = 0.0;

      for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
      {
        t += matValues[j] * rhsValues[colIndices[j]];
      }

      lhs[irow] = t;
    }
  }
  else
  {
    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      idx_t  rend = rowOffsets[irow + 1];

      t = 0.0;

      for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
      {
        t += matValues[j] * rhsValues[rst * colIndices[j]];
      }

      lhs[irow] = t;
    }
  }
}


//-----------------------------------------------------------------------
//   normalMatmul4_
//-----------------------------------------------------------------------


void SparseMatrixObject::normalMatmul4_

  ( const Matrix&  lhs,
    const Matrix&  rhs ) const

{
  JEM_ASSERT2 ( rhs.size(1) == 4,
                "right-hand side matrix must have 4 columns" );

  const idx_t*  JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  colIndices = colIndices_.addr ();
  const double* JEM_RESTRICT  matValues  = matValues_ .addr ();

  const double* JEM_RESTRICT  rcol0 =         rhs.addr   ();
  const double* JEM_RESTRICT  rcol1 = rcol0 + rhs.stride (1);
  const double* JEM_RESTRICT  rcol2 = rcol1 + rhs.stride (1);
  const double* JEM_RESTRICT  rcol3 = rcol2 + rhs.stride (1);

  const idx_t   rowCount = shape_     [0];
  const idx_t   rst0     = rhs.stride (0);

  double        t0, t1, t2, t3;


  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  rend = rowOffsets[irow + 1];

    t0 = t1 = t2 = t3 = 0.0;

    for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
    {
      double  a = matValues [j];
      idx_t    i = colIndices[j] * rst0;

      t0 += a * rcol0[i];
      t1 += a * rcol1[i];
      t2 += a * rcol2[i];
      t3 += a * rcol3[i];
    }

    lhs(irow,0) = t0;
    lhs(irow,1) = t1;
    lhs(irow,2) = t2;
    lhs(irow,3) = t3;
  }
}


//-----------------------------------------------------------------------
//   packedMatmul_
//-----------------------------------------------------------------------


void SparseMatrixObject::packedMatmul_

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  const idx_t*  JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  colIndices = colIndices_.addr ();
  const idx_t*  JEM_RESTRICT  supRows    = supRows_   .addr ();
  const double* JEM_RESTRICT  supValues  = supValues_ .addr ();
  const double* JEM_RESTRICT  rhsValues  = rhs        .addr ();

  const double* JEM_RESTRICT  sp = nullptr;

  const idx_t   rst = rhs.stride ();

  double        t0, t1, t2, t3;
  idx_t         ssize;
  idx_t         iend;


  // Multiply all super rows with size 1.

  ssize = 1;
  iend  = supOffsets_[ssize + 1];

  if ( rst == 1L )
  {
    for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
    {
      idx_t  irow = supRows   [isup];
      idx_t  rend = rowOffsets[irow + 1];

      t0 = 0.0;

      for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
      {
        t0 += supValues[j] * rhsValues[colIndices[j]];
      }

      lhs[irow] = t0;
    }
  }
  else
  {
    for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
    {
      idx_t  irow = supRows   [isup];
      idx_t  rend = rowOffsets[irow + 1];

      t0 = 0.0;

      for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
      {
        t0 += supValues[j] * rhsValues[rst * colIndices[j]];
      }

      lhs[irow] = t0;
    }
  }

  // Multiply all super rows with size 2.

  ssize = 2;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];

    t0 = t1 = 0.0;
    sp = supValues + rbegin;

    for ( idx_t j = rbegin; j < rend; j++, sp += 2 )
    {
      double  b = rhsValues[rst * colIndices[j]];

      t0 += sp[0] * b;
      t1 += sp[1] * b;
    }

    lhs[irow + 0] = t0;
    lhs[irow + 1] = t1;
  }

  // Multiply all super rows with size 3.

  ssize = 3;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];

    t0 = t1 = t2 = 0.0;
    sp = supValues + rbegin;

    for ( idx_t j = rbegin; j < rend; j++, sp += 3 )
    {
      double  b = rhsValues[rst * colIndices[j]];

      t0 += sp[0] * b;
      t1 += sp[1] * b;
      t2 += sp[2] * b;
    }

    lhs[irow + 0] = t0;
    lhs[irow + 1] = t1;
    lhs[irow + 2] = t2;
  }

  // Multiply all super rows with size 4.

  ssize = 4;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];

    t0 = t1 = t2 = t3 = 0.0;
    sp = supValues + rbegin;

    for ( idx_t j = rbegin; j < rend; j++, sp += 4 )
    {
      double  b = rhsValues[rst * colIndices[j]];

      t0 += sp[0] * b;
      t1 += sp[1] * b;
      t2 += sp[2] * b;
      t3 += sp[3] * b;
    }

    lhs[irow + 0] = t0;
    lhs[irow + 1] = t1;
    lhs[irow + 2] = t2;
    lhs[irow + 3] = t3;
  }
}


//-----------------------------------------------------------------------
//   packedMatmul4_
//-----------------------------------------------------------------------


void SparseMatrixObject::packedMatmul4_

  ( const Matrix&  lhs,
    const Matrix&  rhs ) const

{
  JEM_ASSERT2 ( rhs.size(1) == 4,
                "right-hand side matrix must have 4 columns" );

  const idx_t*  JEM_RESTRICT  rowOffsets = rowOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  colIndices = colIndices_.addr ();
  const idx_t*  JEM_RESTRICT  supRows    = supRows_   .addr ();
  const double* JEM_RESTRICT  supValues  = supValues_ .addr ();

  const double* JEM_RESTRICT  rcol0 = rhs.addr ();
  const double* JEM_RESTRICT  rcol1 = rcol0 + rhs.stride (1);
  const double* JEM_RESTRICT  rcol2 = rcol1 + rhs.stride (1);
  const double* JEM_RESTRICT  rcol3 = rcol2 + rhs.stride (1);
  const double* JEM_RESTRICT  sp    = nullptr;

  const idx_t   rst0 = rhs.stride (0);

  double        t00, t01, t02, t03;
  double        t10, t11, t12, t13;
  double        t20, t21, t22, t23;
  double        t30, t31, t32, t33;

  idx_t         ssize;
  idx_t         iend;


  // Multiply all super rows with size 1.

  ssize = 1;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow = supRows   [isup];
    idx_t  rend = rowOffsets[irow + 1];

    t00 = t01 = t02 = t03 = 0.0;

    for ( idx_t j = rowOffsets[irow]; j < rend; j++ )
    {
      double  a = supValues [j];
      idx_t   i = colIndices[j] * rst0;

      t00 += a * rcol0[i];
      t01 += a * rcol1[i];
      t02 += a * rcol2[i];
      t03 += a * rcol3[i];
    }

    lhs(irow,0) = t00;
    lhs(irow,1) = t01;
    lhs(irow,2) = t02;
    lhs(irow,3) = t03;
  }

  // Multiply all super rows with size 2.

  ssize = 2;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];

    t00 = t01 = t02 = t03 = 0.0;
    t10 = t11 = t12 = t13 = 0.0;

    sp  = supValues + rbegin;

    for ( idx_t j = rbegin; j < rend; j++, sp += 2 )
    {
      idx_t   i  = colIndices[j] * rst0;
      double  b0 = rcol0[i];
      double  b1 = rcol1[i];
      double  b2 = rcol2[i];
      double  b3 = rcol3[i];

      t00 += sp[0] * b0;
      t01 += sp[0] * b1;
      t02 += sp[0] * b2;
      t03 += sp[0] * b3;

      t10 += sp[1] * b0;
      t11 += sp[1] * b1;
      t12 += sp[1] * b2;
      t13 += sp[1] * b3;
    }

    lhs(irow + 0,0) = t00;
    lhs(irow + 1,0) = t10;
    lhs(irow + 0,1) = t01;
    lhs(irow + 1,1) = t11;
    lhs(irow + 0,2) = t02;
    lhs(irow + 1,2) = t12;
    lhs(irow + 0,3) = t03;
    lhs(irow + 1,3) = t13;
  }

  // Multiply all super rows with size 3.

  ssize = 3;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];

    t00 = t01 = t02 = t03 = 0.0;
    t10 = t11 = t12 = t13 = 0.0;
    t20 = t21 = t22 = t23 = 0.0;

    sp  = supValues + rbegin;

    for ( idx_t j = rbegin; j < rend; j++, sp += 3 )
    {
      idx_t   i  = colIndices[j] * rst0;
      double  b0 = rcol0[i];
      double  b1 = rcol1[i];
      double  b2 = rcol2[i];
      double  b3 = rcol3[i];

      t00 += sp[0] * b0;
      t01 += sp[0] * b1;
      t02 += sp[0] * b2;
      t03 += sp[0] * b3;

      t10 += sp[1] * b0;
      t11 += sp[1] * b1;
      t12 += sp[1] * b2;
      t13 += sp[1] * b3;

      t20 += sp[2] * b0;
      t21 += sp[2] * b1;
      t22 += sp[2] * b2;
      t23 += sp[2] * b3;
    }

    lhs(irow + 0,0) = t00;
    lhs(irow + 1,0) = t10;
    lhs(irow + 2,0) = t20;
    lhs(irow + 0,1) = t01;
    lhs(irow + 1,1) = t11;
    lhs(irow + 2,1) = t21;
    lhs(irow + 0,2) = t02;
    lhs(irow + 1,2) = t12;
    lhs(irow + 2,2) = t22;
    lhs(irow + 0,3) = t03;
    lhs(irow + 1,3) = t13;
    lhs(irow + 2,3) = t23;
  }

  // Multiply all super rows with size 4.

  ssize = 4;
  iend  = supOffsets_[ssize + 1];

  for ( idx_t isup = supOffsets_[ssize]; isup < iend; isup++ )
  {
    idx_t  irow   = supRows   [isup];
    idx_t  rbegin = rowOffsets[irow];
    idx_t  rend   = rowOffsets[irow + 1];

    t00 = t01 = t02 = t03 = 0.0;
    t10 = t11 = t12 = t13 = 0.0;
    t20 = t21 = t22 = t23 = 0.0;
    t30 = t31 = t32 = t33 = 0.0;

    sp  = supValues + rbegin;

    for ( idx_t j = rbegin; j < rend; j++, sp += 4 )
    {
      idx_t   i  = colIndices[j] * rst0;
      double  b0 = rcol0[i];
      double  b1 = rcol1[i];
      double  b2 = rcol2[i];
      double  b3 = rcol3[i];

      t00 += sp[0] * b0;
      t01 += sp[0] * b1;
      t02 += sp[0] * b2;
      t03 += sp[0] * b3;

      t10 += sp[1] * b0;
      t11 += sp[1] * b1;
      t12 += sp[1] * b2;
      t13 += sp[1] * b3;

      t20 += sp[2] * b0;
      t21 += sp[2] * b1;
      t22 += sp[2] * b2;
      t23 += sp[2] * b3;

      t30 += sp[3] * b0;
      t31 += sp[3] * b1;
      t32 += sp[3] * b2;
      t33 += sp[3] * b3;
    }

    lhs(irow + 0,0) = t00;
    lhs(irow + 1,0) = t10;
    lhs(irow + 2,0) = t20;
    lhs(irow + 3,0) = t30;
    lhs(irow + 0,1) = t01;
    lhs(irow + 1,1) = t11;
    lhs(irow + 2,1) = t21;
    lhs(irow + 3,1) = t31;
    lhs(irow + 0,2) = t02;
    lhs(irow + 1,2) = t12;
    lhs(irow + 2,2) = t22;
    lhs(irow + 3,2) = t32;
    lhs(irow + 0,3) = t03;
    lhs(irow + 1,3) = t13;
    lhs(irow + 2,3) = t23;
    lhs(irow + 3,3) = t33;
  }
}


JIVE_END_PACKAGE( algebra )
