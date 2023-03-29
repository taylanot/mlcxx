
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
#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/base/MemoryError.h>
#include <jem/base/CancelledException.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/numeric/sparse/SparseILU.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class SparseILU::Work_
//=======================================================================


class SparseILU::Work_
{
 public:

  Array<double>   accu;

  idx_t           rowCount;
  idx_t           maxZeroes;
  idx_t           zeroCount;

  bool            relaxed;
  bool            cancelled;

};


//=======================================================================
//   class SparseILU
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


SparseILU::SparseILU ()
{
  size_ = 0;
}


//-----------------------------------------------------------------------
//   factor ( matrix )
//-----------------------------------------------------------------------


idx_t SparseILU::factor ( const Matrix& matrix )
{
  const idx_t   n = matrix.size (0);

  Array<bool>   mask  ( n );
  Array<idx_t>  iperm ( n );

  mask  = true;
  iperm = iarray ( n );

  return factor ( matrix, mask, iperm );
}


//-----------------------------------------------------------------------
//   factor ( matrix, mask )
//-----------------------------------------------------------------------


idx_t SparseILU::factor

  ( const Matrix&       matrix,
    const Array<bool>&  mask )

{
  const idx_t   n = matrix.size (0);

  Array<idx_t>  iperm ( n );

  iperm = iarray ( n );

  return factor ( matrix, mask, iperm );
}


//-----------------------------------------------------------------------
//   factor ( matrix, mask, iperm )
//-----------------------------------------------------------------------


idx_t SparseILU::factor

  ( const Matrix&        matrix,
    const Array<bool>&   mask,
    const Array<idx_t>&  iperm )

{
  JEM_PRECHECK2 ( matrix.size(0) == matrix.size(1) &&
                  matrix.size(0) == mask  .size()  &&
                  matrix.size(0) == iperm .size(),
                  "Array size mismatch" );

  JEM_PRECHECK2 ( isValidPerm ( iperm ),
                  "invalid permutation array" );

  Work_  work;


  size_ = 0;

  init_   ( work, matrix, mask, iperm );
  factor_ ( work );

  if      ( work.zeroCount > work.maxZeroes )
  {
    clear_ ();
  }
  else if ( work.cancelled )
  {
    clear_ ();

    throw CancelledException (
      JEM_FUNC,
      "sparse, incomplete LU factorization cancelled"
    );
  }
  else
  {
    size_ = work.rowCount;

    scratch_.resize ( size_ );
  }

  return work.zeroCount;
}


//-----------------------------------------------------------------------
//   solve
//-----------------------------------------------------------------------


void SparseILU::solve

  ( const Array<double>&  lhs,
    const Array<double>&  rhs )

{
  JEM_PRECHECK2 ( lhs.size() == size() && rhs.size() == size(),
                  "Array size mismatch" );

  double*       JEM_RESTRICT  scratch = scratch_.addr ();
  const double* JEM_RESTRICT  pivots  = pivots_ .addr ();
  const double* JEM_RESTRICT  scale   = scale_  .addr ();
  const idx_t*  JEM_RESTRICT  permi   = permi_  .addr ();

  const idx_t*  JEM_RESTRICT  offsets;
  const idx_t*  JEM_RESTRICT  indices;
  const double* JEM_RESTRICT  values;

  const idx_t   rowCount = size_;


JEM_NOPREFETCH( scratch )
JEM_IVDEP

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    scratch[permi[irow]] = scale[irow] * rhs[irow];
  }

  offsets = lowerOffsets_.addr ();
  indices = lowerIndices_.addr ();
  values  = lowerValues_ .addr ();

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    double  t = 0.0;
    idx_t   n = offsets[irow + 1];

JEM_NOPREFETCH( scratch )

    for ( idx_t j = offsets[irow]; j < n; j++ )
    {
      t += scratch[indices[j]] * values[j];
    }

    scratch[irow] -= t;
  }

  offsets = upperOffsets_.addr ();
  indices = upperIndices_.addr ();
  values  = upperValues_ .addr ();

  for ( idx_t irow = rowCount - 1; irow >= 0; irow-- )
  {
    double  t = 0.0;
    idx_t   n = offsets[irow + 1];

JEM_NOPREFETCH( scratch )

    for ( idx_t j = offsets[irow]; j < n; j++ )
    {
      t += scratch[indices[j]] * values[j];
    }

    scratch[irow] = (scratch[irow] - t) * pivots[irow];
  }

JEM_NOPREFETCH( scratch )

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    lhs[irow] = scale[irow] * scratch[permi[irow]];
  }
}


//-----------------------------------------------------------------------
//   symsolve
//-----------------------------------------------------------------------


void SparseILU::symsolve

  ( const Array<double>&  lhs,
    const Array<double>&  rhs )

{
  JEM_PRECHECK2 ( lhs.size() == size() && rhs.size() == size(),
                  "Array size mismatch" );

  const idx_t*  JEM_RESTRICT  permi   = permi_       .addr ();
  const idx_t*  JEM_RESTRICT  offsets = lowerOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  indices = lowerIndices_.addr ();
  const double* JEM_RESTRICT  values  = lowerValues_ .addr ();
  const double* JEM_RESTRICT  pivots  = pivots_      .addr ();
  const double* JEM_RESTRICT  scale   = scale_       .addr ();
  double*       JEM_RESTRICT  scratch = scratch_     .addr ();

  const idx_t   rowCount = size_;


JEM_NOPREFETCH( scratch )
JEM_IVDEP

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    scratch[permi[irow]] = scale[irow] * rhs[irow];
  }

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    double  t = 0.0;
    idx_t   n = offsets[irow + 1];

JEM_NOPREFETCH( scratch )

    for ( idx_t i = offsets[irow]; i < n; i++ )
    {
      t += scratch[indices[i]] * values[i];
    }

    scratch[irow] -= t;
  }

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    scratch[irow] *= pivots[irow];
  }

  for ( idx_t jcol = rowCount - 1; jcol >= 0; jcol-- )
  {
    double  t = -scratch[jcol];
    idx_t   n =  offsets[jcol + 1];

JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t i = offsets[jcol]; i < n; i++ )
    {
      scratch[indices[i]] += t * values[i];
    }
  }

JEM_NOPREFETCH( scratch )

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    lhs[irow] = scale[irow] * scratch[permi[irow]];
  }
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


void SparseILU::clear_ ()
{
  upperOffsets_.resize ( 0 );
  upperIndices_.resize ( 0 );
  upperValues_ .resize ( 0 );
  lowerOffsets_.resize ( 0 );
  lowerIndices_.resize ( 0 );
  lowerValues_ .resize ( 0 );

  permi_       .resize ( 0 );
  scale_       .resize ( 0 );
  pivots_      .resize ( 0 );
  scratch_     .resize ( 0 );

  size_ = 0;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void SparseILU::init_

  ( Work_&               work,
    const Matrix&        matrix,
    const Array<bool>&   mask,
    const Array<idx_t>&  iperm )

{
  const idx_t    rowCount   = matrix.size            (0);

  Array<idx_t>   matOffsets = matrix.getRowOffsets    ();
  Array<idx_t>   matIndices = matrix.getColumnIndices ();
  Array<double>  matValues  = matrix.getValues        ();

  Array<idx_t>   jcols      ( rowCount );

  idx_t*         upperOffsets;
  idx_t*         upperIndices;
  double*        upperValues;

  idx_t*         lowerOffsets;
  idx_t*         lowerIndices;
  double*        lowerValues;

  idx_t*         permi;
  double*        scale;
  double*        pivots;
  double*        accu;


  work.accu.resize ( rowCount + 1 );

  accu           = work.accu.addr ();
  work.rowCount  = rowCount;
  work.zeroCount = 0;
  work.cancelled = false;

  /*
  if ( maxZeroPivots_ < 0 )
  {
    work.maxZeroes = work.rowCount;
  }
  else
  {
    work.maxZeroes = maxZeroPivots_;
  }
*/

  permi_       .resize ( rowCount );
  scale_       .resize ( rowCount );
  upperOffsets_.resize ( rowCount + 1 );
  lowerOffsets_.resize ( rowCount + 1 );

  upperOffsets_  = 0_idx;
  lowerOffsets_  = 0_idx;

  permi          = permi_       .addr ();
  scale          = scale_       .addr ();
  upperOffsets   = upperOffsets_.addr ();
  lowerOffsets   = lowerOffsets_.addr ();

  accu[rowCount] = 0.0;

  // Initialize the inverse row permutation array.

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    permi[iperm[irow]] = irow;
  }

  // Initialize the scale factors.

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    if ( ! mask[irow] )
    {
      scale[irow] = 1.0;
      continue;
    }

    double  s =  0.0;
    double  t = -1.0;
    idx_t   n = matOffsets[irow + 1];

    for ( idx_t j = matOffsets[irow]; j < n; j++ )
    {
      s = max ( s, std::fabs( matValues[j] ) );

      if ( irow == matIndices[j] )
      {
        t = std::fabs ( matValues[j] );
      }
    }

    // Use the diagonal value if it is non-zero, and the largest row
    // value otherwise.

    if ( t > 0.0 )
    {
      s = t;
    }

    if ( isTiny( s ) )
    {
      scale[irow] = 1.0;
    }
    else
    {
      scale[irow] = 1.0 / std::sqrt ( s );
    }
  }

  // Initialize the lower and upper offset arrays.

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  jrow = iperm[irow];

    if ( ! mask[jrow] )
    {
      continue;
    }

    idx_t  n = matOffsets[jrow + 1];

    for ( idx_t j = matOffsets[jrow]; j < n; j++ )
    {
      idx_t  jcol = matIndices[j];

      if ( ! mask[jcol] )
      {
        continue;
      }

      jcol = permi[jcol];

      if      ( jcol > irow )
      {
        upperOffsets[irow]++;
      }
      else if ( jcol < irow )
      {
        lowerOffsets[irow]++;
      }
    }
  }

  SparseUtils::sumOffsets ( upperOffsets, rowCount + 1 );
  SparseUtils::sumOffsets ( lowerOffsets, rowCount + 1 );

  upperIndices_.resize ( upperOffsets[rowCount] );
  upperValues_ .resize ( upperOffsets[rowCount] );
  lowerIndices_.resize ( lowerOffsets[rowCount] );
  lowerValues_ .resize ( lowerOffsets[rowCount] );
  pivots_      .resize ( rowCount );

  upperIndices = upperIndices_.addr ();
  upperValues  = upperValues_ .addr ();
  lowerIndices = lowerIndices_.addr ();
  lowerValues  = lowerValues_ .addr ();
  pivots       = pivots_      .addr ();

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  jrow = iperm[irow];

    if ( ! mask[jrow] )
    {
      pivots[irow] = 1.0;
      continue;
    }

    double  s = scale[jrow];
    idx_t   k = 0;
    idx_t   n = matOffsets[jrow + 1];

    for ( idx_t j = matOffsets[jrow]; j < n; j++ )
    {
      idx_t  jcol = matIndices[j];

      if ( ! mask[jcol] )
      {
        continue;
      }

      double  t = scale[jcol];

      jcol       = permi[jcol];
      accu[jcol] = matValues[j] * s * t;
      jcols[k++] = jcol;
    }

    if ( k > 1 )
    {
      jem::sort ( jcols[slice(BEGIN,k)] );
    }

    pivots[irow] = 0.0;

    for ( idx_t j = 0; j < k; j++ )
    {
      idx_t  jcol = jcols[j];

      if      ( jcol > irow )
      {
        *upperIndices = jcol;       upperIndices++;
        *upperValues  = accu[jcol]; upperValues++;
      }
      else if ( jcol < irow )
      {
        *lowerIndices = jcol;       lowerIndices++;
        *lowerValues  = accu[jcol]; lowerValues++;
      }
      else
      {
        pivots[irow]  = accu[jcol];
      }
    }
  }

  if ( ! isTiny( accu[rowCount] ) )
  {
    throw MemoryError ( JEM_FUNC );
  }

  work.accu = 0.0;
}


//-----------------------------------------------------------------------
//   factor_
//-----------------------------------------------------------------------


void SparseILU::factor_ ( Work_& work )
{
  const idx_t* JEM_RESTRICT  upperOffsets = upperOffsets_.addr ();
  const idx_t* JEM_RESTRICT  upperIndices = upperIndices_.addr ();
  const idx_t* JEM_RESTRICT  lowerOffsets = lowerOffsets_.addr ();
  const idx_t* JEM_RESTRICT  lowerIndices = lowerIndices_.addr ();

  double*      JEM_RESTRICT  upperValues  = upperValues_ .addr ();
  double*      JEM_RESTRICT  lowerValues  = lowerValues_ .addr ();
  double*      JEM_RESTRICT  pivots       = pivots_      .addr ();
  double*      JEM_RESTRICT  accu         = work.accu    .addr ();

  const idx_t  rowCount  = work.rowCount;

  lint         flopCount = 0;
  double       pivot;
  double       xtmp;


  for ( idx_t  irow = 0; irow < rowCount; irow++ )
  {
    accu[irow] = pivots[irow];

    // Load the upper values into the accumulation buffer.

    idx_t  n = upperOffsets[irow + 1];

JEM_NOPREFETCH( accu )

    for ( idx_t j = upperOffsets[irow]; j < n; j++ )
    {
      accu[upperIndices[j]] = upperValues[j];
    }

    // Load the lower values into the accumulation buffer.

    n = lowerOffsets[irow + 1];

JEM_NOPREFETCH( accu )

    for ( idx_t j = lowerOffsets[irow]; j < n; j++ )
    {
      accu[lowerIndices[j]] = lowerValues[j];
    }

    // Compute the ILU values on the current row.

    for ( idx_t j = lowerOffsets[irow]; j < n; j++ )
    {
      idx_t   jcol =  lowerIndices[j];
      double  t    = -accu[jcol] * pivots[jcol];
      idx_t   m    =  upperOffsets[jcol + 1];

JEM_NOPREFETCH( accu )
JEM_IVDEP

      for ( idx_t i = upperOffsets[jcol]; i < m; i++ )
      {
        accu[upperIndices[i]] += t * upperValues[i];
      }

      lowerValues[j] = -t;
      accu[jcol]     =  0.0;
      flopCount     += (lint) (m - upperOffsets[jcol]);
    }

    // Collect the upper ILU values.

    n = upperOffsets[irow + 1];

JEM_NOPREFETCH( accu )
JEM_IVDEP

    for ( idx_t j = upperOffsets[irow]; j < n; j++ )
    {
      idx_t  jcol      = upperIndices[j];
      upperValues[j] = accu[jcol];
      accu[jcol]     = 0.0;
    }

    pivot      = accu[irow];
    accu[irow] = 0.0;

    if ( work.relaxed )
    {
      xtmp = 0.0;

      // Collect the dropped fill-ins and zero the accumulation
      // buffer.

      for ( idx_t j = lowerOffsets[irow]; j < n; j++ )
      {
        idx_t  jcol = lowerIndices[j];
        idx_t  m    = upperOffsets[jcol + 1];

JEM_NOPREFETCH( accu )
JEM_IVDEP

        for ( idx_t i = upperOffsets[jcol]; i < m; i++ )
        {
          idx_t  icol = upperIndices[i];
          xtmp       += accu[icol];
          accu[icol]  = 0.0;
        }
      }
    }

    xtmp = std::fabs ( pivot );

    if ( xtmp <= zeroThreshold_ )
    {
      work.zeroCount++;

      if ( work.zeroCount > work.maxZeroes )
      {
        return;
      }

      if ( isTiny( xtmp ) )
      {
        pivot = 1.0;
        xtmp  = std::sqrt ( zeroThreshold_ );
      }
      else
      {
        xtmp  = std::sqrt ( zeroThreshold_ ) / xtmp;
      }

      pivot *= xtmp;
    }

    pivots[irow] = 1.0 / pivot;

    if ( testCancelled( flopCount ) )
    {
      work.cancelled = true;
      return;
    }
  }

  if ( ! isTiny( accu[rowCount] ) )
  {
    throw MemoryError ( JEM_FUNC );
  }
}


JEM_END_PACKAGE( numeric )
