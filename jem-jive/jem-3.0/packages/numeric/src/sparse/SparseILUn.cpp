
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
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Flex.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/numeric/sparse/SparseILUn.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class SparseILUn
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


SparseILUn::SparseILUn ()
{
  size_    = 0;
  maxFill_ = 0;
  dshift_  = 0.0;
  droptol_ = 0.0;
}


//-----------------------------------------------------------------------
//   factor ( matrix )
//-----------------------------------------------------------------------


void SparseILUn::factor ( const Matrix& matrix )
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


void SparseILUn::factor

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


void SparseILUn::factor

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

  bool  result = false;

  // Invalidate the current factors.

  size_ = 0;

  init_ ( matrix, mask, iperm );

  for ( int iiter = 0; iiter < 20; iiter++ )
  {
    result = factor_ ( matrix, mask, iperm );

    if ( result )
    {
      commit_ ();
      break;
    }

    if ( dshift_ < 1.0e-6 )
    {
      dshift_ = 1.0e-6;
    }
    else
    {
      dshift_ *= 10.0;
    }
  }

  if ( ! result )
  {
    throw ArithmeticException (
      JEM_FUNC,
      "failed to compute an incomplete LU decomposition; "
      "maximum diagonal shift has been applied"
    );
  }
}


//-----------------------------------------------------------------------
//   solve
//-----------------------------------------------------------------------


void SparseILUn::solve

  ( const Array<double>&  lhs,
    const Array<double>&  rhs )

{
  JEM_PRECHECK2 ( lhs.size() == size() &&
                  rhs.size() == size(),
                  "Array size mismatch" );

  const double* JEM_RESTRICT  scale    = scale_  .addr ();
  double*       JEM_RESTRICT  scratch  = scratch_.addr ();
  const double* JEM_RESTRICT  pivots   = pivots_ .addr ();
  const idx_t*  JEM_RESTRICT  permi    = permi_  .addr ();

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

  offsets = loOffsets_.addr ();
  indices = loIndices_.addr ();
  values  = loValues_ .addr ();

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t   jend = offsets[irow + 1];
    double  temp = 0.0;

JEM_NOPREFETCH( scratch )

    for ( idx_t j = offsets[irow]; j < jend; j++ )
    {
      temp += scratch[indices[j]] * values[j];
    }

    scratch[irow] -= temp;
  }

  offsets = upOffsets_.addr ();
  indices = upIndices_.addr ();
  values  = upValues_ .addr ();

  for ( idx_t irow = rowCount - 1; irow >= 0; irow-- )
  {
    idx_t   jend = offsets[irow + 1];
    double  temp = 0.0;

JEM_NOPREFETCH( scratch )

    for ( idx_t j = offsets[irow]; j < jend; j++ )
    {
      temp += scratch[indices[j]] * values[j];
    }

    scratch[irow] = (scratch[irow] - temp) * pivots[irow];
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


void SparseILUn::symsolve

  ( const Array<double>&  lhs,
    const Array<double>&  rhs )

{
  JEM_PRECHECK2 ( lhs.size() == size() &&
                  rhs.size() == size(),
                  "Array size mismatch" );

  const idx_t*  JEM_RESTRICT  permi   = permi_    .addr ();
  const double* JEM_RESTRICT  scale   = scale_    .addr ();
  const idx_t*  JEM_RESTRICT  offsets = loOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  indices = loIndices_.addr ();
  const double* JEM_RESTRICT  values  = loValues_ .addr ();
  const double* JEM_RESTRICT  pivots  = pivots_   .addr ();
  double*       JEM_RESTRICT  scratch = scratch_  .addr ();

  const idx_t   rowCount = size_;


JEM_NOPREFETCH( scratch )
JEM_IVDEP

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    scratch[permi[irow]] = scale[irow] * rhs[irow];
  }

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t   jend = offsets[irow + 1];
    double  temp = 0.0;

JEM_NOPREFETCH( scratch )

    for ( idx_t j = offsets[irow]; j < jend; j++ )
    {
      temp += scratch[indices[j]] * values[j];
    }

    scratch[irow] -= temp;
  }

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    scratch[irow] *= pivots[irow];
  }

  for ( idx_t jcol = rowCount - 1; jcol >= 0; jcol-- )
  {
    idx_t   jend =  offsets[jcol + 1];
    double  temp = -scratch[jcol];

JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t j = offsets[jcol]; j < jend; j++ )
    {
      scratch[indices[j]] += temp * values[j];
    }
  }

JEM_NOPREFETCH( scratch )

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    lhs[irow] = scale[irow] * scratch[permi[irow]];
  }
}


//-----------------------------------------------------------------------
//   getMemUsage
//-----------------------------------------------------------------------


double SparseILUn::getMemUsage () const noexcept
{
  const double  isize = sizeof(idx_t);
  const double  rsize = sizeof(double);

  return ((double) permi_    .size() * isize +
          (double) scale_    .size() * rsize +
          (double) pivots_   .size() * rsize +
          (double) scratch_  .size() * rsize +
          (double) upOffsets_.size() * isize +
          (double) upIndices_.size() * isize +
          (double) upValues_ .size() * rsize +
          (double) loOffsets_.size() * isize +
          (double) loOffsets_.size() * isize +
          (double) loValues_ .size() * rsize);
}


//-----------------------------------------------------------------------
//   setMaxFill
//-----------------------------------------------------------------------


void SparseILUn::setMaxFill ( idx_t mfill )
{
  JEM_PRECHECK2 ( mfill >= 0, "invalid maximum fill-in level" );

  maxFill_ = mfill;
}


//-----------------------------------------------------------------------
//   setDropTol
//-----------------------------------------------------------------------


void SparseILUn::setDropTol ( double tol )
{
  JEM_PRECHECK2( tol >= 0.0, "invalid drop tolerance" );

  droptol_ = tol;
}


//-----------------------------------------------------------------------
//   setDiagShift
//-----------------------------------------------------------------------


void SparseILUn::setDiagShift ( double dshift )
{
  JEM_PRECHECK2 ( dshift >= 0, "invalid diagonal shift factor" );

  dshift_ = dshift;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void SparseILUn::init_

  ( const Matrix&        matrix,
    const Array<bool>&   mask,
    const Array<idx_t>&  iperm )

{
  const idx_t    rowCount   = matrix.size            (0);

  Array<idx_t>   matOffsets = matrix.getRowOffsets    ();
  Array<idx_t>   matIndices = matrix.getColumnIndices ();
  Array<double>  matValues  = matrix.getValues        ();

  double*        scale;


  // Allocate memory.

  permi_    .resize ( rowCount );
  pivots_   .resize ( rowCount );
  scratch_  .resize ( rowCount );
  scale_    .resize ( rowCount );
  upOffsets_.resize ( rowCount + 1 );
  loOffsets_.resize ( rowCount + 1 );

  // Initialize the inverse row permutation array.

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    permi_[iperm[irow]] = irow;
  }

  // Initialize the scale factors.

  scale = scale_.addr ();

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    if ( ! mask[irow] )
    {
      scale[irow] = 1.0;
      continue;
    }

    double  s =  0.0;
    double  d = -1.0;
    idx_t   n = matOffsets[irow + 1];

    for ( idx_t j = matOffsets[irow]; j < n; j++ )
    {
      idx_t  jcol = matIndices[j];

      if ( ! mask[jcol] )
      {
        continue;
      }

      s = max ( s, std::fabs( matValues[j] ) );

      if ( irow == jcol )
      {
        d = std::fabs ( matValues[j] );
      }
    }

    // Use the diagonal value if it is non-zero, and the largest row
    // value otherwise.

    if ( d > (1.0e-3 * s) )
    {
      s = d;
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
}


//-----------------------------------------------------------------------
//   factor_
//-----------------------------------------------------------------------


bool SparseILUn::factor_

  ( const Matrix&        matrix,
    const Array<bool>&   mask,
    const Array<idx_t>&  iperm )

{
  using jem::util::Flex;

  const double*  JEM_RESTRICT  scale     = scale_    .addr ();
  const idx_t*   JEM_RESTRICT  permi     = permi_    .addr ();
  double*        JEM_RESTRICT  accu      = scratch_  .addr ();
  double*        JEM_RESTRICT  pivots    = pivots_   .addr ();
  idx_t*         JEM_RESTRICT  upOffsets = upOffsets_.addr ();
  idx_t*         JEM_RESTRICT  loOffsets = loOffsets_.addr ();

  idx_t*         JEM_RESTRICT  indexPtr  = nullptr;
  idx_t*         JEM_RESTRICT  levelPtr  = nullptr;
  double*        JEM_RESTRICT  valuePtr  = nullptr;
  idx_t*         JEM_RESTRICT  levls     = nullptr;
  idx_t*         JEM_RESTRICT  jcols     = nullptr;
  idx_t*         JEM_RESTRICT  jlist     = nullptr;
  idx_t*         JEM_RESTRICT  itags     = nullptr;

  const idx_t    size       = matrix.size (0);
  const idx_t    maxLevel   = maxFill_;
  const double   eps        = zeroThreshold_;
  const double   droptol    = droptol_;
  const double   dshift     = dshift_ + droptol_;

  Array<idx_t>   matOffsets = matrix.getRowOffsets    ();
  Array<idx_t>   matIndices = matrix.getColumnIndices ();
  Array<double>  matValues  = matrix.getValues        ();

  Array<idx_t>   istore     ( 4 * size );
  Flex<idx_t>    upLevels;

  lint           flops      = 0;

  double         pivot;
  double         diagv;
  double         value;
  idx_t          ipiv;
  idx_t          jpiv;
  idx_t          jmin;
  idx_t          jcol;
  idx_t          jend;
  idx_t          incl;
  idx_t          incu;
  idx_t          ilev;
  idx_t          jlev;
  idx_t          j, n;


  if ( upIndices_.capacity() == 0 )
  {
    // Reserve space for te upper and lower factors.

    incu = matrix.nonZeroCount () / 2;

    if ( maxLevel > 0 )
    {
      incu *= 2;
    }

    upIndices_.reserve ( incu );
    upValues_ .reserve ( incu );
    loIndices_.reserve ( incu );
    loValues_ .reserve ( incu );
  }

  upIndices_.clear ();
  upValues_ .clear ();
  loIndices_.clear ();
  loValues_ .clear ();

  upLevels.reserve ( upIndices_.capacity() );

  levls = istore.addr ();
  jcols = levls + size;
  jlist = jcols + size;
  itags = jlist + size;

  for ( j = 0; j < size; j++ )
  {
    accu [j] = 0.0;
    itags[j] = -1_idx;
  }

  loOffsets[0] = 0;
  upOffsets[0] = 0;

  for ( ipiv = 0; ipiv < size; ipiv++ )
  {
    jpiv = iperm[ipiv];

    if ( ! mask[jpiv] )
    {
      pivots   [ipiv]     = 0.0;
      loOffsets[ipiv + 1] = loIndices_.size ();
      upOffsets[ipiv + 1] = upIndices_.size ();

      continue;
    }

    // Fetch the current row from the matrix.

    jend = matOffsets[jpiv + 1];
    incl = 0;
    incu = ipiv;

    for ( j = matOffsets[jpiv]; j < jend; j++ )
    {
      jcol = matIndices[j];

      if ( ! mask[jcol] )
      {
        continue;
      }

      value = matValues[j] * scale[jcol] * scale[jpiv];
      jcol  = permi[jcol];

      if ( itags[jcol] == ipiv )
      {
        throw IllegalArgumentException (
          JEM_FUNC,
          String::format (
            "invalid matrix; element (%d,%d) appears twice",
            jpiv,
            matIndices[j]
          )
        );
      }

      accu [jcol] = value;
      itags[jcol] = ipiv;
      levls[jcol] = 0;

      if      ( jcol < ipiv )
      {
        jcols[incl++] = jcol;
      }
      else if ( jcol > ipiv )
      {
        jcols[incu++] = jcol;
      }
    }

    // Make sure that the diagonal element is always included.

    if ( itags[ipiv] != ipiv )
    {
      accu [ipiv] = 0.0;
      itags[ipiv] = ipiv;
    }

    // Apply a diagonal shift to increase the stability of the
    // incomplete factorization procedure.

    diagv = accu[ipiv] * (1.0 + dshift);

    if ( (std::fabs( diagv ) <= eps) ||
         (std::fabs( diagv ) <= droptol) )
    {
      if ( dshift <= eps )
      {
        // Try again with a larger diagonal shift.
        return false;
      }

      if ( diagv < 0.0 )
      {
        diagv -= dshift;
      }
      else
      {
        diagv += dshift;
      }
    }

    accu[ipiv] = diagv;

    // Execute a sparse Gaussian elimination procedure. Small values
    // in the lower factor are dropped. The remaining entries are
    // stored in the array 'jlist'.

    indexPtr = upIndices_.addr ();
    valuePtr = upValues_ .addr ();
    levelPtr = upLevels  .addr ();

    jpiv = -1;
    n    =  0;

    while ( ++jpiv < incl )
    {
      // Find the element with the smallest column index.

      jmin = jpiv;
      jcol = jcols[jpiv];

      for ( j = jpiv + 1; j < incl; j++ )
      {
        if ( jcols[j] < jcol )
        {
          jcol = jcols[j];
          jmin = j;
        }
      }

      if ( jmin != jpiv )
      {
        jcols[jmin] = jcols[jpiv];
        jcols[jpiv] = jcol;
      }

      pivot = accu[jcol] * pivots[jcol];

      if ( std::fabs( pivot ) <= droptol )
      {
        accu[jcol] = 0.0;
        continue;
      }

      JEM_ASSERT ( n < ipiv );

      jlist[n++] = jcol;
      accu[jcol] = pivot;
      ilev       = levls[jcol];
      j          = upOffsets[jcol];
      jend       = upOffsets[jcol + 1];
      flops     += (lint) (jend - j);

      for ( ; j < jend; j++ )
      {
        jcol  = indexPtr[j];

        JEM_ASSERT ( jcol >= 0 && jcol < size );

        jlev  = levelPtr[j] + ilev + 1;
        value = pivot * valuePtr[j];

        if ( itags[jcol] == ipiv )
        {
          accu[jcol] -= value;

          if ( jlev < levls[jcol] )
          {
            levls[jcol] = jlev;
          }
        }
        else if ( jlev <= maxLevel )
        {
          accu [jcol] = -value;
          levls[jcol] =  jlev;
          itags[jcol] =  ipiv;

          if ( jcol < ipiv )
          {
            jcols[incl++] = jcol;
          }
          else
          {
            jcols[incu++] = jcol;
          }
        }
      }
    }

    incl = n;

    // Drop small values from the upper factor.

    for ( j = n = ipiv; j < incu; j++ )
    {
      jcol  = jcols[j];
      value = accu [jcol];

      if ( std::fabs( value ) > droptol )
      {
        jcols[n++] = jcol;
      }
    }

    incu = n;

    // Compute the inverted diagonal value.

    pivot = accu[ipiv];

    if ( std::fabs( pivot ) > 1.0e8 )
    {
      // Unstable decomposition process; try again with a larger diagonal
      // shift, or give up.

      return false;
    }

    if ( std::fabs( pivot ) <= eps * std::fabs( diagv ) )
    {
      if ( dshift < 1.0 )
      {
        // Try again with a larger diagional shift.

        return false;
      }

      pivot = eps * diagv;
    }

    if ( std::fabs( pivot ) > 0.0 )
    {
      pivot = 1.0 / pivot;
    }
    else
    {
      pivot = 1.0;
    }

    pivots[ipiv] = pivot;

    // Store the new entries in the lower factor.

    indexPtr = loIndices_.extend ( incl );
    valuePtr = loValues_ .extend ( incl );

    for ( j = 0; j < incl; j++ )
    {
      jcol = jlist[j];

      indexPtr[j] = jcol;
      valuePtr[j] = accu[jcol];
    }

    // Store the new entries in the upper factor.

    incu    -= ipiv;
    indexPtr = upIndices_.extend ( incu );
    valuePtr = upValues_ .extend ( incu );
    levelPtr = upLevels  .extend ( incu );

    for ( j = 0; j < incu; j++ )
    {
      jcol = jcols[j + ipiv];

      indexPtr[j] = jcol;
      levelPtr[j] = levls[jcol];
      valuePtr[j] = accu [jcol];
    }

    loOffsets[ipiv + 1] = loIndices_.size ();
    upOffsets[ipiv + 1] = upIndices_.size ();

    if ( testCancelled( flops ) )
    {
      throw CancelledException (
        "jem::numeric::SparseILUn::factor",
        "incomplete LU factorization cancelled"
      );
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   commit_
//-----------------------------------------------------------------------


void SparseILUn::commit_ ()
{
  const double  margin = 1.1;

  size_ = pivots_.size ();

  if ( (double) upIndices_.capacity() >
       margin * (double) upIndices_.size() )
  {
    upIndices_.trimToSize ();
  }

  if ( (double) upValues_.capacity()  >
       margin * (double) upValues_.size() )
  {
    upValues_ .trimToSize ();
  }

  if ( (double) loIndices_.capacity() >
       margin * (double) loIndices_.size() )
  {
    loIndices_.trimToSize ();
  }

  if ( (double) loValues_.capacity()  >
       margin * (double) loValues_.size() )
  {
    loValues_ .trimToSize ();
  }
}


JEM_END_PACKAGE( numeric )
