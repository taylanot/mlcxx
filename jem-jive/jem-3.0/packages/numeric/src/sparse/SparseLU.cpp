
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


#include <cmath>
#include <jem/pragmas.h>
#include <jem/std/complex.h>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/base/Thread.h>
#include <jem/base/MemoryError.h>
#include <jem/base/CancelledException.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/numeric/sparse/SparseLU.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class SparseLU::Work_
//=======================================================================


class SparseLU::Work_
{
 public:

                              Work_

    ( idx_t                     msize,
      SparseLU&                 self );


 public:

  util::Event<idx_t,double>*  pivotEvent;
  util::Event<idx_t,double>*  zeroPivotEvent;

  const idx_t                 matrixSize;
  const double                zeroThreshold;
  const double                pivotThreshold;

  Array<bool>                 mask;
  Array<double>               scales;
  Array<double>               accu;
  Array<double>               scratch;
  Array<idx_t>                endPoints;
  Array<idx_t>                rowIndices;
  Array<idx_t>                rowPerm;
  Array<idx_t>                usrPerm;
  Array<idx_t>                stack;

  idx_t                       lastLowerIndex;
  idx_t                       firstUpperIndex;
  idx_t                       zeroCount;
  lint                        flopCount;
  idx_t                       type;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


SparseLU::Work_::Work_

  ( idx_t      msize,
    SparseLU&  self ) :

    pivotEvent     (     0 ),
    zeroPivotEvent (     0 ),
    matrixSize     ( msize ),

    zeroThreshold  ( self.getZeroThreshold () ),
    pivotThreshold ( self.getPivotThreshold() )

{
  lastLowerIndex  = 0;
  firstUpperIndex = msize;
  flopCount       = 0;
  zeroCount       = 0;
  type            = NONE;

  if ( self.pivotEvent.isConnected() )
  {
    pivotEvent = & self.pivotEvent;
  }

  if ( self.zeroPivotEvent.isConnected() )
  {
    zeroPivotEvent = & self.zeroPivotEvent;
  }
}


//=======================================================================
//   class SparseLU:Matrix_
//=======================================================================


class SparseLU::Matrix_
{
 public:

  inline                      Matrix_       ();

  explicit                    Matrix_

    ( const Matrix&             rhs );

  explicit                    Matrix_

    ( const ZMatrix&            rhs );

  inline idx_t                size          () const;
  inline idx_t                nonZeroCount  () const;

  inline Array<idx_t>         getIndices

    ( idx_t                     irow )         const;

  inline Array<double>        getValues

    ( idx_t                      irow )        const;


 public:

  Array<idx_t>                offsets;
  Array<idx_t>                indices;
  Array<double>               values;
  idx_t                       type;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline SparseLU::Matrix_::Matrix_ () :

  offsets ( 1 )

{
  offsets[0] = 0;
  type       = NONE;
}


SparseLU::Matrix_::Matrix_ ( const Matrix& rhs ) :

  offsets ( rhs.getRowOffsets() ),
  indices ( rhs.getColumnIndices() ),
  values  ( rhs.getValues() ),
  type    ( DOUBLE )

{}


SparseLU::Matrix_::Matrix_ ( const ZMatrix& rhs ) :

  offsets ( rhs.getRowOffsets() ),
  indices ( rhs.getColumnIndices() ),
  values  ( rhs.nonZeroCount() * 2 ),
  type    ( COMPLEX )

{
  Array<Complex>  zvals = rhs.getValues ();

  const idx_t     nnz   = zvals.size    ();

  for ( idx_t i = 0; i < nnz; i++ )
  {
    values[2 * i + 0] = zvals[i].real ();
    values[2 * i + 1] = zvals[i].imag ();
  }
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t SparseLU::Matrix_::size () const
{
  return (offsets.size() - 1);
}


//-----------------------------------------------------------------------
//   noneZeroCount
//-----------------------------------------------------------------------


inline idx_t SparseLU::Matrix_::nonZeroCount () const
{
  return indices.size ();
}


//-----------------------------------------------------------------------
//   getIndices
//-----------------------------------------------------------------------


inline Array<idx_t> SparseLU::Matrix_::getIndices ( idx_t irow ) const
{
  idx_t  i = offsets[irow];
  idx_t  j = offsets[irow + 1];

  return indices[slice(i,j)];
}


//-----------------------------------------------------------------------
//   getValues
//-----------------------------------------------------------------------


inline Array<double> SparseLU::Matrix_::getValues ( idx_t irow ) const
{
  idx_t  i = type * offsets[irow];
  idx_t  j = type * offsets[irow + 1];

  return values[slice(i,j)];
}


//=======================================================================
//   class SparseLU::Upper_
//=======================================================================


void SparseLU::Upper_::clear ()
{
  colOffsets.resize ( 0 );
  rowIndices.resize ( 0 );
  colPerm   .resize ( 0 );
  values    .clear  ();
}


double SparseLU::Upper_::getMemUsage () const
{
  const double  isize = sizeof(idx_t);
  const double  rsize = sizeof(double);

  return ((double) colOffsets.size() * isize +
          (double) rowIndices.size() * isize +
          (double) colPerm   .size() * isize +
          (double) values    .size() * rsize);
}


//=======================================================================
//   class SparseLU::Lower_
//=======================================================================


void SparseLU::Lower_::clear ()
{
  colOffsets.resize ( 0 );
  rowIndices.resize ( 0 );
  rowPerm   .resize ( 0 );
  values    .clear  ();
}


double SparseLU::Lower_::getMemUsage () const
{
  const double  isize = sizeof(idx_t);
  const double  rsize = sizeof(double);

  return ((double) colOffsets.size() * isize +
          (double) rowIndices.size() * isize +
          (double) rowPerm   .size() * isize +
          (double) values    .size() * rsize);
}


//=======================================================================
//   class SparseLU
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

// Note that these three values are significant as they indicate
// the number of doubles that make up a single number (double or
// complex).

const idx_t  SparseLU::NONE            = 0;
const idx_t  SparseLU::DOUBLE          = 1;
const idx_t  SparseLU::COMPLEX         = 2;

const idx_t  SparseLU::CANARY_VALUE_   = 0xABC;
const lint   SparseLU::MAX_FLOP_COUNT_ = 128 * 1024;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


SparseLU::SparseLU ()
{
  maxZeroPivots_  = 0;
  pivotThreshold_ = 1.0;
  size_           = 0;
  type_           = NONE;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SparseLU::clear ()
{
  upper_  .clear  ();
  lower_  .clear  ();
  scratch_.resize ( 0 );

  size_ = 0;
  type_ = NONE;
}


//-----------------------------------------------------------------------
//   factor ( matrix )
//-----------------------------------------------------------------------


idx_t SparseLU::factor ( const Matrix& matrix )
{
  JEM_PRECHECK2 ( matrix.size(0) == matrix.size(1),
                  "non-square matrix" );
  JEM_PRECHECK2 ( matrix.isValid(),
                  "invalid matrix" );

  return factor_ ( Matrix_( matrix ) );
}


idx_t SparseLU::factor ( const ZMatrix& matrix )
{
  JEM_PRECHECK2 ( matrix.size(0) == matrix.size(1),
                  "non-square complex matrix" );
  JEM_PRECHECK2 ( matrix.isValid(),
                  "invalid complex matrix" );

  return factor_ ( Matrix_( matrix ) );
}


//-----------------------------------------------------------------------
//   factor ( matrix, mask )
//-----------------------------------------------------------------------


idx_t SparseLU::factor

  ( const Matrix&       matrix,
    const Array<bool>&  mask )

{
  JEM_PRECHECK2 ( matrix.size(0) == matrix.size(1),
                  "non-square matrix" );
  JEM_PRECHECK2 ( matrix.isValid(),
                  "invalid matrix" );
  JEM_PRECHECK2 ( matrix.size(0) == mask.size(),
                  "mask array size mismatch" );

  return factor_ ( Matrix_( matrix ), mask );
}


idx_t SparseLU::factor

  ( const ZMatrix&      matrix,
    const Array<bool>&  mask )

{
  JEM_PRECHECK2 ( matrix.size(0) == matrix.size(1),
                  "non-square complex matrix" );
  JEM_PRECHECK2 ( matrix.isValid(),
                  "invalid complex matrix" );
  JEM_PRECHECK2 ( matrix.size(0) == mask.size(),
                  "mask array size mismatch" );

  return factor_ ( Matrix_( matrix ), mask );
}


//-----------------------------------------------------------------------
//   factor ( matrix, mask, rowPerm, colPerm )
//-----------------------------------------------------------------------


idx_t SparseLU::factor

  ( const Matrix&        matrix,
    const Array<bool>&   mask,
    const Array<idx_t>&  rowPerm,
    const Array<idx_t>&  colPerm )

{
  JEM_PRECHECK2 ( matrix.size(0) == matrix.size(1),
                  "non-square matrix" );
  JEM_PRECHECK2 ( matrix.isValid(),
                  "invalid matrix" );
  JEM_PRECHECK2 ( matrix.size(0) == mask.size(),
                  "mask array size mismatch" );
  JEM_PRECHECK2 ( matrix.size(0) == rowPerm.size(),
                  "row permutation array size mismatch" );
  JEM_PRECHECK2 ( matrix.size(0) == colPerm.size(),
                  "column permutation array size mismatch" );
  JEM_PRECHECK2 ( isValidPerm( rowPerm ),
                  "invalid row permutation array" );
  JEM_PRECHECK2 ( isValidPerm( colPerm ),
                  "invalid column permutation array" );

  return factor_ ( Matrix_( matrix ), mask, rowPerm, colPerm );
}


idx_t SparseLU::factor

  ( const ZMatrix&       matrix,
    const Array<bool>&   mask,
    const Array<idx_t>&  rowPerm,
    const Array<idx_t>&  colPerm )

{
  JEM_PRECHECK2 ( matrix.size(0) == matrix.size(1),
                  "non-square complex matrix" );
  JEM_PRECHECK2 ( matrix.isValid(),
                  "invalid complex matrix" );
  JEM_PRECHECK2 ( matrix.size(0) == mask.size(),
                  "mask array size mismatch" );
  JEM_PRECHECK2 ( matrix.size(0) == rowPerm.size(),
                  "row permutation array size mismatch" );
  JEM_PRECHECK2 ( matrix.size(0) == colPerm.size(),
                  "column permutation array size mismatch" );
  JEM_PRECHECK2 ( isValidPerm( rowPerm ),
                  "invalid row permutation array" );
  JEM_PRECHECK2 ( isValidPerm( colPerm ),
                  "invalid column permutation array" );

  return factor_ ( Matrix_( matrix ), mask, rowPerm, colPerm );
}


//-----------------------------------------------------------------------
//   solve ( double )
//-----------------------------------------------------------------------


void SparseLU::solve

  ( const Array<double>&  lhs,
    const Array<double>&  rhs )

{
  JEM_PRECHECK2 ( lhs.size() == size() &&
                  rhs.size() == size(),
                  "Array size mismatch" );

  double*       JEM_RESTRICT  scratch = scratch_      .addr ();
  const idx_t*  JEM_RESTRICT  iperm   = lower_.rowPerm.addr ();
  const idx_t                 msize   = size_;


  if       ( type_ == DOUBLE )
  {
JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      scratch[iperm[irow]] = rhs[irow];
    }

    solve_ ();

    iperm = upper_.colPerm.addr ();

JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      lhs[iperm[irow]] = scratch[irow];
    }
  }
  else if ( type_ == COMPLEX )
  {
    // Store the rhs vector as a complex vector with a zero
    // imaginary part.

JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      idx_t  jrow = iperm[irow];

      scratch[2 * jrow + 0] = rhs[irow];
      scratch[2 * jrow + 1] = 0.0;
    }

    solve_ ();

    iperm = upper_.colPerm.addr ();

    // Retrieve the real part of the complex solution vector.

JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      lhs[iperm[irow]] = scratch[2 * irow];
    }
  }

  if ( ! isTiny( scratch[type_ * msize] ) )
  {
    throw MemoryError ( JEM_FUNC );
  }
}


//-----------------------------------------------------------------------
//   solve ( Complex )
//-----------------------------------------------------------------------


void SparseLU::solve

  ( const Array<Complex>&  lhs,
    const Array<Complex>&  rhs )

{
  JEM_PRECHECK2 ( lhs.size() == size() &&
                  rhs.size() == size(),
                  "Array size mismatch" );

  double*       JEM_RESTRICT  scratch = scratch_      .addr ();
  const idx_t*  JEM_RESTRICT  iperm   = lower_.rowPerm.addr ();
  const idx_t                 msize   = size_;


  if       ( type_ == DOUBLE )
  {
    // First, calculate the solution for the real part of
    // the rhs vector.

JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      scratch[iperm[irow]] = rhs[irow].real ();
    }

    solve_ ();

    iperm = upper_.colPerm.addr ();

JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      lhs[iperm[irow]].real ( scratch[irow] );
    }

    // Next, calculate the solution for the imaginary part of
    // the rhs vector.

    iperm = lower_.rowPerm.addr ();

JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      scratch[iperm[irow]] = rhs[irow].imag ();
    }

    solve_ ();

    iperm = upper_.colPerm.addr ();

JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      lhs[iperm[irow]].imag ( scratch[irow] );
    }
  }
  else if ( type_ == COMPLEX )
  {
JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      idx_t  jrow = iperm[irow];

      scratch[2 * jrow + 0] = rhs[irow].real ();
      scratch[2 * jrow + 1] = rhs[irow].imag ();
    }

    solve_ ();

    iperm = upper_.colPerm.addr ();

JEM_IVDEP

    for ( idx_t irow = 0; irow < msize; irow++ )
    {
      idx_t  jrow = iperm[irow];

      lhs[jrow].real ( scratch[2 * irow + 0] );
      lhs[jrow].imag ( scratch[2 * irow + 1] );
    }
  }

  if ( ! isTiny( scratch[type_ * msize] ) )
  {
    throw MemoryError ( JEM_FUNC );
  }
}


//-----------------------------------------------------------------------
//   getMemUsage
//-----------------------------------------------------------------------


double SparseLU::getMemUsage () const noexcept
{
  const double  rsize = sizeof(double);

  return (upper_.getMemUsage() +
          lower_.getMemUsage() +
          (double) scratch_.size() * rsize);
}


//-----------------------------------------------------------------------
//   setMaxZeroPivots
//-----------------------------------------------------------------------


void SparseLU::setMaxZeroPivots ( idx_t maxZeroes )
{
  maxZeroPivots_ = maxZeroes;
}


//-----------------------------------------------------------------------
//   setPivotThreshold
//-----------------------------------------------------------------------


void SparseLU::setPivotThreshold ( double alpha )
{
  JEM_PRECHECK2 ( alpha >= 0.0 && alpha <= 1.0,
                  "invalid pivot threshold" );

  pivotThreshold_ = alpha;
}


//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------


void SparseLU::solve_ ()
{
  double*       JEM_RESTRICT  scratch = scratch_.addr ();

  const idx_t*  JEM_RESTRICT  colOffsets;
  const idx_t*  JEM_RESTRICT  rowIndices;
  const double* JEM_RESTRICT  values;

  const idx_t   msize = size_;

  idx_t         irow, jcol;
  idx_t         i,    n;


  // Forward substitution with the lower factor

  colOffsets = lower_.colOffsets.addr ();
  rowIndices = lower_.rowIndices.addr ();
  values     = lower_.values.addr     ();

  if      ( type_ == DOUBLE )
  {
    double  t;

    for ( jcol = 0; jcol < msize; jcol++ )
    {
      n =  colOffsets[jcol + 1];
      t = -scratch   [jcol];

JEM_NOPREFETCH( scratch )
JEM_IVDEP

      for ( i = colOffsets[jcol]; i < n; i++ )
      {
        scratch[rowIndices[i]] += t * values[i];
      }
    }
  }
  else if ( type_ == COMPLEX )
  {
    double  tr, ti;
    double  vr, vi;

    for ( jcol = 0; jcol < msize; jcol++ )
    {
      n  =  colOffsets [jcol + 1];
      tr = -scratch[2 * jcol + 0];
      ti = -scratch[2 * jcol + 1];

JEM_NOPREFETCH( scratch )
JEM_IVDEP

      for ( i = colOffsets[jcol]; i < n; i++ )
      {
        irow                   = rowIndices[i];
        vr                     = values[2 * i + 0];
        vi                     = values[2 * i + 1];
        scratch[2 * irow + 0] += tr * vr - ti * vi;
        scratch[2 * irow + 1] += tr * vi + ti * vr;
      }
    }
  }

  // Backward substitution with the upper factor

  colOffsets = upper_.colOffsets.addr ();
  rowIndices = upper_.rowIndices.addr ();
  values     = upper_.values    .addr ();

  if      ( type_ == DOUBLE )
  {
    double  t;

    for ( jcol = msize - 1; jcol >= 0; jcol-- )
    {
      i =  colOffsets[jcol + 1] - 1;
      n =  colOffsets[jcol];
      t = -scratch   [jcol] * values[i];

JEM_NOPREFETCH( scratch )
JEM_IVDEP

      for ( i--; i >= n; i-- )
      {
        scratch[rowIndices[i]] += t * values[i];
      }

      scratch[jcol] = -t;
    }
  }
  else if ( type_ == COMPLEX )
  {
    double  sr, si;
    double  tr, ti;
    double  vr, vi;

    for ( jcol = msize - 1; jcol >= 0; jcol-- )
    {
      i  =  colOffsets [jcol + 1] - 1;
      n  =  colOffsets [jcol];
      sr = -scratch[2 * jcol + 0];
      si = -scratch[2 * jcol + 1];
      vr =  values [2 * i    + 0];
      vi =  values [2 * i    + 1];
      tr =  sr * vr - si * vi;
      ti =  sr * vi + si * vr;

JEM_NOPREFETCH( scratch )
JEM_IVDEP

      for ( i--; i >= n; i-- )
      {
        irow                   = rowIndices[i];
        vr                     = values[2 * i + 0];
        vi                     = values[2 * i + 1];
        scratch[2 * irow + 0] += tr * vr - ti * vi;
        scratch[2 * irow + 1] += tr * vi + ti * vr;
      }

      scratch[2 * jcol + 0] = -tr;
      scratch[2 * jcol + 1] = -ti;
    }
  }
}


//-----------------------------------------------------------------------
//   factor_
//-----------------------------------------------------------------------


idx_t SparseLU::factor_ ( const Matrix_& matrix )
{
  const idx_t   n     = matrix.size ();

  Array<bool>   mask  ( n );
  Array<idx_t>  iperm ( n );

  mask  = true;
  iperm = iarray ( n );

  return factor_ ( matrix, mask, iperm, iperm );
}


idx_t SparseLU::factor_

  ( const Matrix_&      matrix,
    const Array<bool>&  mask )

{
  const idx_t   n     = matrix.size ();

  Array<idx_t>  iperm ( n );

  iperm = iarray ( n );

  return factor_ ( matrix, mask, iperm, iperm );
}


idx_t SparseLU::factor_

  ( const Matrix_&       matrix,
    const Array<bool>&   mask,
    const Array<idx_t>&  rowPerm,
    const Array<idx_t>&  colPerm )

{
  const idx_t  msize = matrix.size ();

  Work_        work  ( msize, *this );

  Matrix_      trans;
  idx_t        maxZeroes;


  clear ();
  init_ ( work, upper_, lower_, matrix, mask, rowPerm, colPerm );

  if ( msize == 0 )
  {
    return 0;
  }

  if ( maxZeroPivots_ < 0 )
  {
    maxZeroes = msize;
  }
  else
  {
    maxZeroes = maxZeroPivots_;
  }

  if ( work.zeroCount > maxZeroes )
  {
    clear ();

    return work.zeroCount;
  }

  work.zeroCount = 0;

  transpose_ ( trans, matrix, mask );

  for ( idx_t jcol = 0; jcol < msize; jcol++ )
  {
    idx_t          jperm = upper_.colPerm[jcol];
    Array<idx_t>   irows = trans.getIndices ( jperm );
    Array<double>  mvals = trans.getValues  ( jperm );

    getColStruct_ ( work, upper_, lower_, irows );
    loadColumn_   ( work, irows,  mvals         );
    factorColumn_ ( work, upper_, lower_, jcol  );
    pivotColumn_  ( work, lower_,         jcol  );
    pruneColumn_  ( work, lower_,         jcol  );
    storeColumn_  ( work, upper_, lower_, jcol  );

    if ( work.zeroCount > maxZeroes )
    {
      clear ();

      return work.zeroCount;
    }

    if ( work.flopCount > MAX_FLOP_COUNT_ )
    {
      if ( Thread::cancelled() )
      {
        clear ();

        throw CancelledException (
          JEM_FUNC,
          "sparse LU factorization cancelled"
        );
      }

      progressEvent.emit ( jcol );

      work.flopCount = 0;
    }
  }

  applyRowPerm_   (                 lower_, msize  );
  applyUserPerm_  (           work, lower_         );
  checkData_      ( JEM_FUNC, work, upper_, lower_ );
  scratch_.ref    ( work.scratch );

  size_ = msize;
  type_ = matrix.type;

  return work.zeroCount;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void SparseLU::init_

  ( Work_&                work,
    Upper_&               upper,
    Lower_&               lower,
    const Matrix_&        matrix,
    const Array<bool>&    mask,
    const Array<idx_t>&   rowPerm,
    const Array<idx_t>&   colPerm )

{
  const idx_t    msize   = matrix.size             ();
  const idx_t    nzCount = matrix.nonZeroCount     ();
  const idx_t    bufsz   = 2 * nzCount;
  const idx_t    type    = matrix.type;

  Array<idx_t>   offsets = matrix.offsets;
  Array<idx_t>   indices = matrix.indices;
  Array<double>  values  = matrix.values;


  upper.colOffsets.resize  ( msize + 2 );
  upper.rowIndices.reserve ( bufsz );
  upper.values    .reserve ( bufsz * type );
  upper.colPerm   .resize  ( msize );

  upper.colPerm            =  colPerm;
  upper.colOffsets[0]      =  0;
  upper.colOffsets.back()  =  CANARY_VALUE_;

  lower.colOffsets.resize  ( msize + 2 );
  lower.rowPerm   .resize  ( msize + 1 );
  lower.rowIndices.reserve ( bufsz );
  lower.values    .reserve ( bufsz * type );

  lower.colOffsets[0]      =  0;
  lower.colOffsets.back()  =  CANARY_VALUE_;
  lower.rowPerm            =  minOf<idx_t> ();
  lower.rowPerm.back()     =  CANARY_VALUE_;

  work.endPoints .resize   ( msize );
  work.rowIndices.resize   ( msize );
  work.rowPerm   .resize   ( msize );
  work.mask      .resize   ( msize );
  work.scales    .resize   ( msize );
  work.accu      .resize   ( msize * type );
  work.scratch   .resize   ( msize * type + 1 );
  work.stack     .resize   ( msize * 2 );

  work.mask                = false;
  work.accu                = 0.0;
  work.scratch             = 0.0;
  work.type                = type;

  work.usrPerm.ref ( rowPerm );

  for ( idx_t i = 0; i < msize; i++ )
  {
    work.rowPerm[rowPerm[i]] = i;
  }

  // Initialize the scale factors.

  for ( idx_t irow = 0; irow < msize; irow++ )
  {
    idx_t  jrow = rowPerm[irow];

    if ( ! mask[jrow] )
    {
      work.scales[irow] = 1.0;
      continue;
    }

    double  scale = 0.0;
    idx_t   n     = offsets[jrow + 1];

    for ( idx_t i = offsets[jrow]; i < n; i++ )
    {
      if ( mask[indices[i]] )
      {
        double  val = 0.0;

        if      ( type == DOUBLE )
        {
          val = std::fabs ( values[i] );
        }
        else if ( type == COMPLEX )
        {
          double  vr = values[2 * i + 0];
          double  vi = values[2 * i + 1];

          val = vr * vr + vi * vi;
        }

        if ( val > scale )
        {
          scale = val;
        }
      }
    }

    if ( type == COMPLEX )
    {
      scale = std::sqrt ( scale );
    }

    if ( isTiny( scale ) )
    {
      work.scales[irow] = 1.0;
      work.zeroCount++;
    }
    else
    {
      work.scales[irow] = 1.0 / scale;
    }
  }
}


//-----------------------------------------------------------------------
//   transpose_
//-----------------------------------------------------------------------


void SparseLU::transpose_

  ( Matrix_&            tr,
    const Matrix_&      mat,
    const Array<bool>&  mask )

{
  const idx_t    rowCount   = mat.size ();
  const idx_t    type       = mat.type;

  Array<idx_t>   rowOffsets = mat.offsets;
  Array<idx_t>   colIndices = mat.indices;
  Array<double>  matValues  = mat.values;

  Array<idx_t>   colOffsets ( rowCount + 1 );
  Array<idx_t>   rowIndices;
  Array<double>  trValues;


  colOffsets = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    if ( ! mask[irow] )
    {
      colOffsets[irow] = 1;
      continue;
    }

    idx_t  n = rowOffsets[irow + 1];

    for ( idx_t i = rowOffsets[irow]; i < n; i++ )
    {
      idx_t  jcol = colIndices[i];

      if ( mask[jcol] )
      {
        colOffsets[jcol]++;
      }
    }
  }

  SparseUtils::sumOffsets ( colOffsets );

  rowIndices.resize ( colOffsets[rowCount] );
  trValues  .resize ( colOffsets[rowCount] * type );

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    if ( ! mask[irow] )
    {
      idx_t  k = colOffsets[irow]++;

      rowIndices[k] = irow;

      if      ( type == DOUBLE )
      {
        trValues[k] = 1.0;
      }
      else if ( type == COMPLEX )
      {
        trValues[2 * k + 0] = 1.0;
        trValues[2 * k + 1] = 0.0;
      }

      continue;
    }

    idx_t  n = rowOffsets[irow + 1];

    for ( idx_t i = rowOffsets[irow]; i < n; i++ )
    {
      idx_t  jcol = colIndices[i];

      if ( mask[jcol] )
      {
        idx_t  k = colOffsets[jcol]++;

        rowIndices[k] = irow;

        if      ( type == DOUBLE )
        {
          trValues[k] = matValues[i];
        }
        else if ( type == COMPLEX )
        {
          trValues[2 * k + 0] = matValues[2 * i + 0];
          trValues[2 * k + 1] = matValues[2 * i + 1];
        }
      }
    }
  }

  SparseUtils::shiftOffsets ( colOffsets );

  tr.type = type;

  tr.offsets.ref ( colOffsets );
  tr.indices.ref ( rowIndices );
  tr.values .ref ( trValues );
}


//-----------------------------------------------------------------------
//   getColStruct_
//-----------------------------------------------------------------------


void SparseLU::getColStruct_

  ( Work_&               work,
    Upper_&              upper,
    Lower_&              lower,
    const Array<idx_t>&  irows )

{
  const idx_t* JEM_RESTRICT  rowPerm      = lower.rowPerm   .addr ();
  const idx_t* JEM_RESTRICT  lowerIndices = lower.rowIndices.addr ();
  const idx_t* JEM_RESTRICT  lowerOffsets = lower.colOffsets.addr ();
  const idx_t* JEM_RESTRICT  endPoints    = work.endPoints  .addr ();

  idx_t*       JEM_RESTRICT  rowIndices   = work.rowIndices .addr ();
  idx_t*       JEM_RESTRICT  stack        = work.stack      .addr ();
  bool*        JEM_RESTRICT  mask         = work.mask       .addr ();

  const idx_t  icount  = irows.size ();
  const idx_t  msize   = work.matrixSize;

  idx_t        ifirst  = msize;
  idx_t        ilast   = 0;
  idx_t        itop    = 0;

  idx_t        irow,  jrow;
  idx_t        iperm;
  idx_t        i, k;
  idx_t        j, n;


  for ( i = 0; i < icount; i++ )
  {
    irow = work.rowPerm[irows[i]];

    if ( mask[irow] )
    {
      continue;
    }

    mask[irow] = true;
    iperm      = rowPerm[irow];

    if ( iperm < 0 )
    {
      rowIndices[ilast++] = irow;
    }
    else
    {
      j = lowerOffsets[iperm];

    next_row:

      n = endPoints[iperm];

      for ( ; j < n; j++ )
      {
        jrow = lowerIndices[j];

        if ( mask[jrow] )
        {
          continue;
        }

        mask[jrow] = true;
        iperm      = rowPerm[jrow];

        if ( iperm < 0 )
        {
          rowIndices[ilast++] = jrow;
        }
        else
        {
          stack[itop++] = irow;
          stack[itop++] = j;

          irow = jrow;
          j    = lowerOffsets[iperm];

          goto next_row;
        }
      }

      rowIndices[--ifirst] = irow;

      if ( itop > 0 )
      {
        j     = stack[--itop];
        irow  = stack[--itop];
        iperm = rowPerm[irow];

        goto next_row;
      }
    }
  }

  // Reset the mask array.

  for ( i = 0; i < ilast; i++ )
  {
    mask[rowIndices[i]] = false;
  }

  for ( i = msize - 1; i >= ifirst; i-- )
  {
    mask[rowIndices[i]] = false;
  }

  if ( ilast == 0 )
  {
    // Add an arbitrary free row so that the lower part of the
    // new column contains at least one entry.

    k = minOf ( k );

    for ( i = 0; i < msize; i++ )
    {
      if ( rowPerm[i] < 0 )
      {
        k = i;
        break;
      }
    }

    JEM_ASSERT ( k >= 0 );

    rowIndices[ilast++] = k;
  }

  work.lastLowerIndex  = ilast;
  work.firstUpperIndex = ifirst;
}


//-----------------------------------------------------------------------
//   loadColumn_
//-----------------------------------------------------------------------


void SparseLU::loadColumn_

  ( Work_&                work,
    const Array<idx_t>&   mrows,
    const Array<double>&  mvals )

{
  const idx_t* JEM_RESTRICT  irows = work.rowIndices .addr ();
  const idx_t* JEM_RESTRICT  iperm = work.rowPerm    .addr ();
  double*      JEM_RESTRICT  accu  = work.accu       .addr ();

  const idx_t  type = work.type;

  idx_t        n;


  // Zero the relevant entries in the accumulator array.

  n = work.lastLowerIndex;

  if      ( type == DOUBLE )
  {
JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = 0; i < n; i++ )
    {
      accu[irows[i]] = 0.0;
    }
  }
  else if ( type == COMPLEX )
  {
JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  irow = irows[i];

      accu[2 * irow + 0] = 0.0;
      accu[2 * irow + 1] = 0.0;
    }
  }

  n = work.matrixSize;

  if      ( type == DOUBLE )
  {
JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = work.firstUpperIndex; i < n; i++ )
    {
      accu[irows[i]] = 0.0;
    }
  }
  else if ( type == COMPLEX )
  {
JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = work.firstUpperIndex; i < n; i++ )
    {
      idx_t  irow = irows[i];

      accu[2 * irow + 0] = 0.0;
      accu[2 * irow + 1] = 0.0;
    }
  }

  // Load the matrix values into the accumulator array.

  n = mrows.size ();

  if      ( type == DOUBLE )
  {
JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = 0; i < n; i++ )
    {
      accu[iperm[mrows[i]]] = mvals[i];
    }
  }
  else if ( type == COMPLEX )
  {
JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  irow = iperm[mrows[i]];

      accu[2 * irow + 0] = mvals[2 * i + 0];
      accu[2 * irow + 1] = mvals[2 * i + 1];
    }
  }
}


//-----------------------------------------------------------------------
//   factorColumn_
//-----------------------------------------------------------------------


void SparseLU::factorColumn_

  ( Work_&   work,
    Upper_&  upper,
    Lower_&  lower,
    idx_t    jcol )

{
  const idx_t*  JEM_RESTRICT  rowPerm      = lower.rowPerm   .addr ();
  const idx_t*  JEM_RESTRICT  lowerOffsets = lower.colOffsets.addr ();
  const idx_t*  JEM_RESTRICT  lowerIndices = lower.rowIndices.addr ();
  const double* JEM_RESTRICT  lowerValues  = lower.values    .addr ();
  const idx_t*  JEM_RESTRICT  irows        = work.rowIndices .addr ();
  double*       JEM_RESTRICT  accu         = work.accu       .addr ();
  double*       JEM_RESTRICT  scratch      = work.scratch    .addr ();

  const idx_t   msize = work.matrixSize;
  const idx_t   ilast = work.lastLowerIndex;
  const idx_t   type  = work.type;

  idx_t         irow, jrow, iperm;
  idx_t         j,    n;


  if      ( type == DOUBLE )
  {
    double  t;

    for ( idx_t i = work.firstUpperIndex; i < msize; i++ )
    {
      irow  =  irows[i];
      iperm =  rowPerm[irow];
      t     = -accu[irow];
      j     =  lowerOffsets[iperm];
      n     =  lowerOffsets[iperm + 1];

      work.flopCount += (lint) (n - j);

JEM_NOPREFETCH( accu )
JEM_IVDEP

      for ( ; j < n; j++ )
      {
        accu[lowerIndices[j]] += t * lowerValues[j];
      }
    }
  }
  else if ( type == COMPLEX )
  {
    double  tr, ti;
    double  vr, vi;

    for ( idx_t i = work.firstUpperIndex; i < msize; i++ )
    {
      irow  =  irows[i];
      iperm =  rowPerm [irow];
      tr    = -accu[2 * irow + 0];
      ti    = -accu[2 * irow + 1];
      j     =  lowerOffsets[iperm];
      n     =  lowerOffsets[iperm + 1];

      work.flopCount += 2_lint * (lint) (n - j);

JEM_NOPREFETCH( accu )
JEM_IVDEP

      for ( ; j < n; j++ )
      {
        jrow                = lowerIndices[j];
        vr                  = lowerValues[2 * j + 0];
        vi                  = lowerValues[2 * j + 1];
        accu[2 * jrow + 0] += tr * vr - ti * vi;
        accu[2 * jrow + 1] += tr * vi + ti * vr;
      }
    }
  }

  // Copy the lower & upper values into the scratch array.

  if      ( type == DOUBLE )
  {
JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = 0; i < ilast; i++ )
    {
      irow       = irows[i];
      scratch[i] = accu[irow];
      accu[irow] = 0.0;
    }

JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = work.firstUpperIndex; i < msize; i++ )
    {
      irow       = irows[i];
      scratch[i] = accu[irow];
      accu[irow] = 0.0;
    }
  }
  else if ( type == COMPLEX )
  {
JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = 0; i < ilast; i++ )
    {
      irow               = irows[i];
      scratch[2 * i + 0] = accu[2 * irow + 0];
      scratch[2 * i + 1] = accu[2 * irow + 1];
      accu[2 * irow + 0] = 0.0;
      accu[2 * irow + 1] = 0.0;
    }

JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t i = work.firstUpperIndex; i < msize; i++ )
    {
      irow               = irows[i];
      scratch[2 * i + 0] = accu[2 * irow + 0];
      scratch[2 * i + 1] = accu[2 * irow + 1];
      accu[2 * irow + 0] = 0.0;
      accu[2 * irow + 1] = 0.0;
    }
  }
}


//-----------------------------------------------------------------------
//   pivotColumn_
//-----------------------------------------------------------------------


void SparseLU::pivotColumn_

  ( Work_&   work,
    Lower_&  lower,
    idx_t    jcol )

{
  const double* JEM_RESTRICT  scale   = work.scales    .addr ();
  double*       JEM_RESTRICT  scratch = work.scratch   .addr ();
  idx_t*        JEM_RESTRICT  irows   = work.rowIndices.addr ();
  idx_t*        JEM_RESTRICT  rowPerm = lower.rowPerm  .addr ();

  const idx_t   ilast = work.lastLowerIndex;
  const idx_t   type  = work.type;

  double        xpiv;
  double        xtmp;
  idx_t         ipiv;
  idx_t         irow;
  idx_t         dist;
  idx_t         itmp;


  JEM_ASSERT ( ilast > 0 );

  // Find the largest (scaled) pivot

  ipiv = 0;
  xpiv = 0.0;
  irow = irows[ipiv];

  if     ( type == DOUBLE )
  {
    xpiv = scale[irow] * std::fabs ( scratch[ipiv] );

    for ( idx_t i = 1; i < ilast; i++ )
    {
      irow = irows[i];
      xtmp = scale[irow] * std::fabs ( scratch[i] );

      if ( xtmp > xpiv )
      {
        xpiv = xtmp;
        ipiv = i;
      }
    }
  }
  else if ( type == COMPLEX )
  {
    double  sr = scratch[2 * ipiv + 0];
    double  si = scratch[2 * ipiv + 1];

    xpiv = scale[irow] * scale[irow] * (sr * sr + si * si);

    for ( idx_t i = 1; i < ilast; i++ )
    {
      irow = irows[i];
      sr   = scratch[2 * i + 0];
      si   = scratch[2 * i + 1];
      xtmp = scale[irow] * scale[irow] * (sr * sr + si * si);

      if ( xtmp > xpiv )
      {
        xpiv = xtmp;
        ipiv = i;
      }
    }

    xpiv = std::sqrt ( xpiv );
  }

  // Find the pivot closest to the diagonal that is larger than the
  // pivot threshold times the largest pivot. If the largest pivot is
  // zero, then just find the pivot that is closest to the diagonal.

  irow = irows[ipiv];
  dist = abs ( jcol - irow );

  if ( dist > 0 )
  {
    if ( xpiv <= work.zeroThreshold )
    {
      xpiv  = -1.0;
    }
    else
    {
      xpiv *= work.pivotThreshold;
    }

    if      ( type == DOUBLE )
    {
      for ( idx_t i = 0; i < ilast; i++ )
      {
        irow = irows[i];
        itmp = abs ( jcol - irow );
        xtmp = scale[irow] * std::fabs ( scratch[i] );

        if ( xtmp >= xpiv && itmp < dist )
        {
          dist = itmp;
          ipiv = i;
        }
      }
    }
    else if ( type == COMPLEX )
    {
      xpiv = xpiv * std::fabs ( xpiv );

      for ( idx_t i = 0; i < ilast; i++ )
      {
        double  sr = scratch[2 * i + 0];
        double  si = scratch[2 * i + 1];

        irow = irows[i];
        itmp = abs ( jcol - irow );
        xtmp = scale[irow] * scale[irow] * (sr * sr + si * si);

        if ( xtmp >= xpiv && itmp < dist )
        {
          dist = itmp;
          ipiv = i;
        }
      }
    }

    irow = irows[ipiv];

    if      ( type == DOUBLE )
    {
      xpiv = scale[irow] * std::fabs ( scratch[ipiv] );
    }
    else if ( type == COMPLEX )
    {
      double  sr = scratch[2 * ipiv + 0];
      double  si = scratch[2 * ipiv + 1];

      xpiv = scale[irow] * std::sqrt ( sr * sr + si * si );
    }
  }

  JEM_ASSERT ( rowPerm[irow] < 0 );

  // Update the row permutation array.

  rowPerm[irow] = jcol;

  // Swap the row indices and the row values.

  irows[ipiv] = irows[0];
  irows[0]    = irow;

  if      ( type == DOUBLE )
  {
    xtmp          = scratch[0];
    scratch[0]    = scratch[ipiv];
    scratch[ipiv] = xtmp;
  }
  else if ( type == COMPLEX )
  {
    xtmp                  = scratch[0];
    scratch[0]            = scratch[2 * ipiv + 0];
    scratch[2 * ipiv + 0] = xtmp;
    xtmp                  = scratch[1];
    scratch[1]            = scratch[2 * ipiv + 1];
    scratch[2 * ipiv + 1] = xtmp;
  }

  if ( xpiv <= work.zeroThreshold )
  {
    work.zeroCount++;

    if ( isTiny( xpiv ) )
    {
      xtmp       = std::sqrt ( work.zeroThreshold ) / scale[irow];
      scratch[0] = 1.0;

      if ( type == COMPLEX )
      {
        scratch[1] = 0.0;
      }
    }
    else
    {
      xtmp = std::sqrt ( work.zeroThreshold ) / xpiv;
    }

    scratch[0] *= xtmp;

    if ( type == COMPLEX )
    {
      scratch[1] *= xtmp;
    }

    if ( work.zeroPivotEvent )
    {
      work.zeroPivotEvent->emit ( work.usrPerm[irow], xpiv );
    }
  }
  else
  {
    if ( work.pivotEvent )
    {
      work.pivotEvent->emit ( work.usrPerm[irow], scratch[0] );
    }
  }

  if      ( type == DOUBLE )
  {
    xtmp = scratch[0] = 1.0 / scratch[0];

JEM_IVDEP

    for ( idx_t i = 1; i < ilast; i++ )
    {
      scratch[i] *= xtmp;
    }
  }
  else if ( type == COMPLEX )
  {
    double  xr, xi;
    double  sr, si;

    xtmp       =  scratch[0] * scratch[0] + scratch[1] * scratch[1];
    xr         =  scratch[0] / xtmp;
    xi         = -scratch[1] / xtmp;
    scratch[0] =  xr;
    scratch[1] =  xi;

JEM_IVDEP

    for ( idx_t i = 1; i < ilast; i++ )
    {
      sr                 = scratch[2 * i + 0];
      si                 = scratch[2 * i + 1];
      scratch[2 * i + 0] = xr * sr - xi * si;
      scratch[2 * i + 1] = xr * si + xi * sr;
    }
  }
}


//-----------------------------------------------------------------------
//   pruneColumn_
//-----------------------------------------------------------------------


void SparseLU::pruneColumn_

  ( Work_&   work,
    Lower_&  lower,
    idx_t    jcol )

{
  const idx_t* JEM_RESTRICT  irows        = work.rowIndices .addr ();
  idx_t*       JEM_RESTRICT  endPoints    = work.endPoints  .addr ();
  const idx_t* JEM_RESTRICT  rowPerm      = lower.rowPerm   .addr ();
  const idx_t* JEM_RESTRICT  lowerOffsets = lower.colOffsets.addr ();
  idx_t*       JEM_RESTRICT  lowerIndices = lower.rowIndices.addr ();
  double*      JEM_RESTRICT  lowerValues  = lower.values    .addr ();

  const idx_t  msize = work.matrixSize;
  const idx_t  type  = work.type;

  idx_t        i, icol;
  idx_t        j, n;


  for ( i = work.firstUpperIndex; i < msize; i++ )
  {
    icol = rowPerm[irows[i]];
    j    = lowerOffsets[icol];
    n    = endPoints[icol];

    // Skip this column if it has already been pruned

    if ( lowerOffsets[icol + 1] != n )
    {
      continue;
    }

JEM_NOPREFETCH( rowPerm )

    for ( ; j < n; j++ )
    {
      if ( rowPerm[lowerIndices[j]] == jcol )
      {
        break;
      }
    }

    if ( j == n )
    {
      continue;
    }

    j = lowerOffsets[icol];

    if      ( type == DOUBLE )
    {
      while ( j < n )
      {
        if ( rowPerm[lowerIndices[j]] < 0 )
        {
          n--;
          jem::swap ( lowerIndices[j], lowerIndices[n] );
          jem::swap ( lowerValues [j], lowerValues [n] );
        }
        else
        {
          j++;
        }
      }
    }
    else if ( type == COMPLEX )
    {
      while ( j < n )
      {
        if ( rowPerm[lowerIndices[j]] < 0 )
        {
          n--;
          jem::swap ( lowerIndices[j], lowerIndices[n] );
          jem::swap ( lowerValues [2 * j + 0],
                      lowerValues [2 * n + 0] );
          jem::swap ( lowerValues [2 * j + 1],
                      lowerValues [2 * n + 1] );
        }
        else
        {
          j++;
        }
      }

    }

    endPoints[icol] = n;
  }
}


//-----------------------------------------------------------------------
//   storeColumn_
//-----------------------------------------------------------------------


void SparseLU::storeColumn_

  ( Work_&   work,
    Upper_&  upper,
    Lower_&  lower,
    idx_t    jcol )

{
  const idx_t*  JEM_RESTRICT  rowPerm = lower.rowPerm  .addr ();
  const idx_t*  JEM_RESTRICT  irows   = work.rowIndices.addr ();
  const double* JEM_RESTRICT  scratch = work.scratch   .addr ();

  const idx_t   type = work.type;

  idx_t         irow;
  idx_t         i, n;


  JEM_ASSERT ( work.lastLowerIndex > 0 );

  n = work.lastLowerIndex;

  lower.rowIndices.pushBack ( irows   + 1, irows   + n );

  if      ( type == DOUBLE )
  {
    lower.values.pushBack ( scratch + 1, scratch + n );
  }
  else if ( type == COMPLEX )
  {
    lower.values.pushBack ( scratch + 2, scratch + 2 * n );
  }

  n = work.matrixSize;

  if      ( type == DOUBLE )
  {
    for ( i = work.firstUpperIndex; i < n; i++ )
    {
      irow = irows[i];

      upper.rowIndices.pushBack ( rowPerm[irow] );
      upper.values    .pushBack ( scratch[i] );
    }
  }
  else if ( type == COMPLEX )
  {
    for ( i = work.firstUpperIndex; i < n; i++ )
    {
      irow = irows[i];

      upper.rowIndices.pushBack ( rowPerm[irow] );
      upper.values    .pushBack ( scratch[2 * i + 0] );
      upper.values    .pushBack ( scratch[2 * i + 1] );
    }
  }

  lower.colOffsets[jcol + 1] = lower.rowIndices.size ();
  work .endPoints [jcol]     = lower.rowIndices.size ();

  upper.rowIndices.pushBack ( jcol );
  upper.values    .pushBack ( scratch[0] );

  if ( type == COMPLEX )
  {
    upper.values  .pushBack ( scratch[1] );
  }

  upper.colOffsets[jcol + 1] = upper.rowIndices.size ();
}


//-----------------------------------------------------------------------
//   applyRowPerm_
//-----------------------------------------------------------------------


void SparseLU::applyRowPerm_

  ( Lower_&  lower,
    idx_t    msize )

{
  const idx_t*  rowPerm = lower.rowPerm   .addr ();
  const idx_t*  offsets = lower.colOffsets.addr ();
  idx_t*        indices = lower.rowIndices.addr ();

  idx_t         j, jcol;
  idx_t         i, n;


  for ( jcol = 0; jcol < msize; jcol++ )
  {
    n = offsets[jcol + 1];

    for ( i = offsets[jcol]; i < n; i++ )
    {
      j = rowPerm[indices[i]];

      if ( j < 0 )
      {
        throw Error ( JEM_FUNC, "oops, invalid row pivot sequence" );
      }

      indices[i] = j;
    }
  }
}


//-----------------------------------------------------------------------
//   applyUserPerm_
//-----------------------------------------------------------------------


void SparseLU::applyUserPerm_

  ( Work_&   work,
    Lower_&  lower )

{
  const idx_t  msize = work.matrixSize;


  for ( idx_t i = 0; i < msize; i++ )
  {
    work.rowPerm[work.usrPerm[i]] = lower.rowPerm[i];
  }

  for ( idx_t i = 0; i < msize; i++ )
  {
    lower.rowPerm[i] = work.rowPerm[i];
  }
}


//-----------------------------------------------------------------------
//   checkData_
//-----------------------------------------------------------------------


void SparseLU::checkData_

  ( const char*    where,
    const Work_&   work,
    const Upper_&  upper,
    const Lower_&  lower )
{
  bool  ok = true;

  ok = ok && ( work .matrixSize        >= 0             );
  ok = ok && ( upper.colOffsets.back() == CANARY_VALUE_ );
  ok = ok && ( lower.colOffsets.back() == CANARY_VALUE_ );
  ok = ok && ( lower.rowPerm   .back() == CANARY_VALUE_ );

  if ( ! ok )
  {
    throw MemoryError ( where );
  }
}


JEM_END_PACKAGE( numeric )
