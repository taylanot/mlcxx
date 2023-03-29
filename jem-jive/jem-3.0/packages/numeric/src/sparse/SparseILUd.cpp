
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
#include <jem/numeric/sparse/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/numeric/sparse/SparseILUd.h>


/* ---------------------------------------------------------------------
 * Implementation Details
 *
 * The SparseILUd preconditioner is based on the following
 * decomposition of a matrix A:
 *
 *   A = L D U
 *
 * with L a lower triangular sparse matrix, D a diagonal matrix,
 * and U an upper triangular sparse matrix. The matrix L is stored
 * column wise and the matrix U is stored row wise. They are
 * computed with a Crout version of ILU, described in the article:
 *
 *  "Crout versions of ILU for general sparse matrices"
 *
 * by Na Li, Yousef Saad and Edmond Chow. This algorithm is combined
 * with the drop-tolerance strategy described in the article:
 *
 *  "Incomplete Cholesky Factorization in Fixed Memory with
 *   Flexible Drop-Tolerance Strategy"
 *
 * by Sergey Saukh from G.Y. Pukhov's Institute of Modelling
 * Problems in Power Engineering, NAS of Ukraine.
 * ------------------------------------------------------------------ */


JEM_BEGIN_PACKAGE( numeric )


using jem::util::Flex;


//=======================================================================
//   class SparseILUd::Node_
//=======================================================================


class SparseILUd::Node_
{
 public:

  union
  {
    idx_t         irow;
    idx_t         jcol;
  };

  idx_t           next;

};


//=======================================================================
//   class SparseILUd::Work_
//=======================================================================


class SparseILUd::Work_
{
 public:

  Flex<Node_>     upNodes;
  Flex<Node_>     loNodes;
  Array<idx_t>    rowLists;
  Array<idx_t>    colLists;
  Array<idx_t>    rowStart;
  Array<idx_t>    colStart;

};


//=======================================================================
//   class SparseILUd
//=======================================================================
//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  SparseILUd::MIN_DSHIFT  = 1.0e-5;
const double  SparseILUd::MIN_DROPTOL = 1.0e-6;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


SparseILUd::SparseILUd ()
{
  size_    = 0;
  minSize_ = 0;
  maxFill_ = 1.0;
  dshift_  = 0.0;
  droptol_ = MIN_DROPTOL;
  quality_ = 1.0;
}


//-----------------------------------------------------------------------
//   factor ( matrix )
//-----------------------------------------------------------------------


void SparseILUd::factor ( const Matrix& matrix )
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


void SparseILUd::factor

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


void SparseILUd::factor

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

  Matrix  lowmat = lower_ ( matrix, iperm );

  Work_   work;

  bool    result = false;


  // Adjust the drop tolerance.

  if ( size_    > 0 &&
       droptol_ > MIN_DROPTOL )
  {
    droptol_ *= 1.0 / (1.0 + quality_);
  }

  // Invalidate the current factors.

  size_ = 0;

  init_ ( work, matrix, mask, iperm );

  for ( int iiter = 0; iiter < 20; iiter++ )
  {
    result = factor_ ( work, matrix, lowmat, mask, iperm );

    if ( result )
    {
      commit_ ();
      break;
    }

    if ( dshift_ < MIN_DSHIFT )
    {
      dshift_ = MIN_DSHIFT;
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


void SparseILUd::solve

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
    double  temp = scratch[irow];

JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t j = offsets[irow]; j < jend; j++ )
    {
      scratch[indices[j]] -= temp * values[j];
    }
  }

  offsets = upOffsets_.addr ();
  indices = upIndices_.addr ();
  values  = upValues_ .addr ();

  for ( idx_t irow = rowCount - 1; irow >= 0; irow-- )
  {
    idx_t   jend = offsets[irow + 1];
    double  temp = scratch[irow] = scratch[irow] * pivots[irow];

JEM_NOPREFETCH( scratch )

    for ( idx_t j = offsets[irow]; j < jend; j++ )
    {
      temp -= scratch[indices[j]] * values[j];
    }

    scratch[irow] = temp;
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


void SparseILUd::symsolve

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
    double  temp = scratch[irow];

JEM_NOPREFETCH( scratch )
JEM_IVDEP

    for ( idx_t j = offsets[irow]; j < jend; j++ )
    {
      scratch[indices[j]] -= temp * values[j];
    }
  }

  for ( idx_t irow = rowCount - 1; irow >= 0; irow-- )
  {
    idx_t   jend = offsets[irow + 1];
    double  temp = scratch[irow] = scratch[irow] * pivots[irow];

JEM_NOPREFETCH( scratch )

    for ( idx_t j = offsets[irow]; j < jend; j++ )
    {
      temp -= scratch[indices[j]] * values[j];
    }

    scratch[irow] = temp;
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


double SparseILUd::getMemUsage () const noexcept
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
//   setMinSize
//-----------------------------------------------------------------------


void SparseILUd::setMinSize ( idx_t msize )
{
  JEM_PRECHECK2 ( msize >= 0, "invalid minimum size" );

  minSize_ = msize;
}


//-----------------------------------------------------------------------
//   setMaxFill
//-----------------------------------------------------------------------


void SparseILUd::setMaxFill ( double mfill )
{
  JEM_PRECHECK2 ( mfill >= 0.0, "invalid maximum fill-in factor" );

  maxFill_ = mfill;
}


//-----------------------------------------------------------------------
//   setDropTol
//-----------------------------------------------------------------------


void SparseILUd::setDropTol ( double tol )
{
  JEM_PRECHECK2( tol >= 0.0, "invalid drop tolerance" );

  droptol_ = tol;
}


//-----------------------------------------------------------------------
//   setDiagShift
//-----------------------------------------------------------------------


void SparseILUd::setDiagShift ( double dshift )
{
  JEM_PRECHECK2 ( dshift >= 0, "invalid diagonal shift factor" );

  dshift_ = dshift;
}


//-----------------------------------------------------------------------
//   setQuality
//-----------------------------------------------------------------------


void SparseILUd::setQuality ( double qlty )
{
  JEM_PRECHECK2 ( qlty >= 0.0, "invalid quality factor" );

  quality_ = qlty;
}

//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void SparseILUd::init_

  ( Work_&               work,
    const Matrix&        matrix,
    const Array<bool>&   mask,
    const Array<idx_t>&  iperm )

{
  const idx_t    rowCount   = matrix.size            (0);

  Array<idx_t>   matOffsets = matrix.getRowOffsets    ();
  Array<idx_t>   matIndices = matrix.getColumnIndices ();
  Array<double>  matValues  = matrix.getValues        ();

  double*        scale;


  // Allocate memory.

  permi_       .resize ( rowCount );
  pivots_      .resize ( rowCount );
  scratch_     .resize ( rowCount );
  scale_       .resize ( rowCount );
  upOffsets_   .resize ( rowCount + 1 );
  loOffsets_   .resize ( rowCount + 1 );
  work.rowLists.resize ( rowCount );
  work.colLists.resize ( rowCount );
  work.rowStart.resize ( rowCount );
  work.colStart.resize ( rowCount );

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


bool SparseILUd::factor_

  ( Work_&               work,
    const Matrix&        matrix,
    const Matrix&        lowmat,
    const Array<bool>&   mask,
    const Array<idx_t>&  iperm )

{
  const double*  JEM_RESTRICT  scale     = scale_       .addr ();
  const idx_t*   JEM_RESTRICT  permi     = permi_       .addr ();
  double*        JEM_RESTRICT  scratch   = scratch_     .addr ();
  double*        JEM_RESTRICT  pivots    = pivots_      .addr ();
  idx_t*         JEM_RESTRICT  upOffsets = upOffsets_   .addr ();
  idx_t*         JEM_RESTRICT  loOffsets = loOffsets_   .addr ();
  idx_t*         JEM_RESTRICT  rowStart  = work.rowStart.addr ();
  idx_t*         JEM_RESTRICT  colStart  = work.colStart.addr ();
  idx_t*         JEM_RESTRICT  rowLists  = work.rowLists.addr ();
  idx_t*         JEM_RESTRICT  colLists  = work.colLists.addr ();

  double*        JEM_RESTRICT  loAccu    = nullptr;
  double*        JEM_RESTRICT  upAccu    = nullptr;
  idx_t*         JEM_RESTRICT  loTags    = nullptr;
  idx_t*         JEM_RESTRICT  upTags    = nullptr;
  idx_t*         JEM_RESTRICT  irows     = nullptr;
  idx_t*         JEM_RESTRICT  jcols     = nullptr;

  const idx_t    size       = matrix.size (0);
  const double   eps        = zeroThreshold_;

  Array<idx_t>   matOffsets = matrix.getRowOffsets    ();
  Array<idx_t>   matIndices = matrix.getColumnIndices ();
  Array<double>  matValues  = matrix.getValues        ();

  Array<idx_t>   lwmOffsets = lowmat.getRowOffsets    ();
  Array<idx_t>   lwmIndices = lowmat.getColumnIndices ();
  Array<double>  lwmValues  = lowmat.getValues        ();

  Array<idx_t>   istore     ( 4 * size );
  Array<double>  rstore     ( 2 * size );

  lint           flops      = 0;

  double         pivot;
  double         diagv;
  double         value;
  double         tiny;
  idx_t          colLen;
  idx_t          rowLen;
  idx_t          maxNnz;
  idx_t          nfill;
  idx_t          inode;
  idx_t          ipiv;
  idx_t          jpiv;
  idx_t          irow;
  idx_t          jrow;
  idx_t          icol;
  idx_t          jcol;
  idx_t          istart;
  idx_t          iend;
  idx_t          incl;
  idx_t          incu;
  idx_t          i, n;


  loAccu = rstore.addr ();
  upAccu = loAccu + size;
  loTags = istore.addr ();
  upTags = loTags + size;
  irows  = upTags + size;
  jcols  = irows  + size;

  for ( i = 0; i < size; i++ )
  {
    upOffsets[i] = 0;
    loOffsets[i] = 0;
    rowLists [i] = -1_idx;
    colLists [i] = -1_idx;
    rowStart [i] = 0;
    colStart [i] = 0;
    loTags   [i] = -1_idx;
    upTags   [i] = -1_idx;
    loAccu   [i] = 0.0;
    upAccu   [i] = 0.0;
  }

  upOffsets[size] = 0;
  loOffsets[size] = 0;

  upIndices_  .clear ();
  upValues_   .clear ();
  loIndices_  .clear ();
  loValues_   .clear ();
  work.upNodes.clear ();
  work.loNodes.clear ();

  // Get the scaled diagonal values and add the shift factor. Also
  // apply the column/row permutation array.

  getDiag ( scratch_, matrix );

  for ( i = 0; i < size; i++ )
  {
    pivot = scratch[i] * scale[i] * scale[i];

    if ( pivot >= 0.0 )
    {
      pivot += dshift_;
    }
    else
    {
      pivot -= dshift_;
    }

    pivots[permi[i]] = pivot;
  }

  // Determine the maximum number of non-zeroes in the ILU factors.
  // Beware of integer overflow.

  maxNnz = n = matrix.nonZeroCount ();
  value  = maxFill_;

  for ( i = 1; i < 10; i++ )
  {
    maxNnz = (idx_t) ((value + 1.0) * (double) n);

    if ( maxNnz >= 0 )
    {
      break;
    }

    value /= 2.0;
  }

  maxNnz = max ( maxNnz, n );

  if ( (maxNnz + minSize_) > maxNnz )
  {
    maxNnz += minSize_;
  }

  // Determine the number of allowed fill-ins per column/row.

  if ( size == 0 )
  {
    nfill = 0;
  }
  else
  {
    nfill = (maxNnz - n) / size;

    if ( quality_ > 0.0 )
    {
      nfill += (idx_t) (quality_ * (double) n / (double) size);
    }
  }

  // Reserve space for storing the upper and lower factors, and the
  // temporary data.

  n = maxNnz / 2;

  upIndices_  .reserve ( n );
  upValues_   .reserve ( n );
  loIndices_  .reserve ( n );
  loValues_   .reserve ( n );
  work.upNodes.reserve ( n );
  work.loNodes.reserve ( n );

  // Execute the main loop over the rows/columns of the matrix.

  for ( ipiv = 0; ipiv < size; ipiv++ )
  {
    Node_*  JEM_RESTRICT  loNodes   = work.loNodes.addr ();
    idx_t*  JEM_RESTRICT  loIndices = loIndices_  .addr ();
    double* JEM_RESTRICT  loValues  = loValues_   .addr ();
    Node_*  JEM_RESTRICT  upNodes   = work.upNodes.addr ();
    idx_t*  JEM_RESTRICT  upIndices = upIndices_  .addr ();
    double* JEM_RESTRICT  upValues  = upValues_   .addr ();

    // Update the column/row start indices.

    for ( inode = work.rowLists[ipiv]; inode >= 0;
          inode = loNodes[inode].next )
    {
      work.colStart[loNodes[inode].jcol] = inode;
    }

    for ( inode = work.colLists[ipiv]; inode >= 0;
          inode = upNodes[inode].next )
    {
      work.rowStart[upNodes[inode].irow] = inode;
    }

    colStart[ipiv] = loValues_.size ();
    rowStart[ipiv] = upValues_.size ();
    jpiv           = iperm[ipiv];

    if ( ! mask[jpiv] )
    {
      loOffsets[ipiv + 1] = loValues_.size ();
      upOffsets[ipiv + 1] = upValues_.size ();

      continue;
    }

    // Fetch the lower column from the matrix. Note that the
    // permutation array has already been applied to the lower
    // matrix.

    iend = lwmOffsets[ipiv + 1];
    incl = 0;

    for ( i = lwmOffsets[ipiv]; i < iend; i++ )
    {
      irow = lwmIndices[i];
      jrow = iperm[irow];

      if ( ! mask[jrow] || irow <= ipiv )
      {
        continue;
      }

      if ( loTags[irow] == ipiv )
      {
        matrixError_ ( JEM_FUNC, jrow, jpiv );
      }

      value = lwmValues[i] * scale[jpiv] * scale[jrow];

      loTags[irow]  = ipiv;
      loAccu[irow]  = value;
      irows[incl++] = irow;
    }

    // Fetch the upper row from the matrix.

    diagv = 0.0;
    iend  = matOffsets[jpiv + 1];
    incu  = 0;

    for ( i = matOffsets[jpiv]; i < iend; i++ )
    {
      icol = matIndices[i];

      if ( ! mask[icol] )
      {
        continue;
      }

      jcol = permi[icol];

      if ( jcol < ipiv )
      {
        continue;
      }

      if ( upTags[jcol] == ipiv )
      {
        matrixError_ ( JEM_FUNC, jpiv, icol );
      }

      value = matValues[i] * scale[jpiv] * scale[icol];

      if ( jcol == ipiv )
      {
        diagv = value;
      }
      else
      {
        upTags[jcol]  = ipiv;
        upAccu[jcol]  = value;
        jcols[incu++] = jcol;
      }
    }

    colLen = incl;
    rowLen = incu;

    // Check the current pivot.

    pivot = pivots[ipiv];

    if ( (std::fabs( pivot ) > 1.0e8) ||
         (std::fabs( pivot ) <= eps  * std::fabs( diagv )) )
    {
      return false;
    }

    // Determine the new column in the lower factor.

    for ( inode = colLists[ipiv]; inode >= 0;
          inode = upNodes[inode].next )
    {
      irow   = upNodes  [inode].irow;
      istart = colStart [irow] + 1;
      iend   = loOffsets[irow  + 1];
      pivot  = upValues [inode] * pivots[irow];
      flops += (lint) (iend - istart);

      JEM_ASSERT ( irow < ipiv );

      for ( i = istart; i < iend; i++ )
      {
        irow  = loIndices[i];
        value = loValues [i] * pivot;

        if ( loTags[irow] != ipiv )
        {
          loTags[irow]  =  ipiv;
          loAccu[irow]  = -value;
          irows[incl++] =  irow;
        }
        else
        {
          loAccu[irow] -= value;
        }
      }
    }

    // Determine the new row in the upper factor.

    for ( inode = rowLists[ipiv]; inode >= 0;
          inode = loNodes[inode].next )
    {
      jcol   = loNodes  [inode].jcol;
      istart = rowStart [jcol] + 1;
      iend   = upOffsets[jcol  + 1];
      pivot  = loValues [inode] * pivots[jcol];
      flops += (lint) (iend - istart);

      JEM_ASSERT ( jcol < ipiv );

      for ( i = istart; i < iend; i++ )
      {
        jcol  = upIndices[i];
        value = upValues [i] * pivot;

        if ( upTags[jcol] != ipiv )
        {
          upTags[jcol]  =  ipiv;
          upAccu[jcol]  = -value;
          jcols[incu++] =  jcol;
        }
        else
        {
          upAccu[jcol] -= value;
        }
      }
    }

    // Divide the new column/row by the pivot.

    pivot = 1.0 / pivots[ipiv];

JEM_IVDEP

    for ( i = 0; i < incl; i++ )
    {
      loAccu[irows[i]] *= pivot;
    }

JEM_IVDEP

    for ( i = 0; i < incu; i++ )
    {
      upAccu[jcols[i]] *= pivot;
    }

    // Drop tiny elements from the lower and upper factors.

    tiny  = 0.5 * droptol_;
    tiny *= tiny;
    n     = 0;

    for ( i = 0; i < incl; i++ )
    {
      irow  = irows[i];
      value = loAccu[irow];

      if ( (value * value) > tiny )
      {
        irows[n++] = irow;
      }
    }

    incl = n;
    n    = 0;

    for ( i = 0; i < incu; i++ )
    {
      jcol  = jcols[i];
      value = upAccu[jcol];

      if ( (value * value) > tiny )
      {
        jcols[n++] = jcol;
      }
    }

    incu = n;
    tiny = droptol_ * droptol_;

    for ( inode = rowLists[ipiv]; inode >= 0;
          inode = loNodes[inode].next )
    {
      jcol  = loNodes[inode].jcol;
      value = loValues[inode];

      if ( std::fabs( pivots[jcol] * value * value ) <=
           std::fabs( pivots[ipiv] * tiny ) )
      {
        loValues[inode] = 0.0;
      }
    }

    for ( inode = colLists[ipiv]; inode >= 0;
          inode = upNodes[inode].next )
    {
      irow  = upNodes[inode].irow;
      value = upValues[inode];

      if ( std::fabs( pivots[irow] * value * value ) <=
           std::fabs( pivots[ipiv] * tiny ) )
      {
        upValues[inode] = 0.0;
      }
    }

    // Determine which entries to keep.

    n = colLen + nfill / 2;

    if ( n < incl )
    {
      asort_ ( loAccu, irows, scratch, incl );

      incl = n;
    }

    n = rowLen + nfill / 2;

    if ( n < incu )
    {
      asort_ ( upAccu, jcols, scratch, incu );

      incu = n;
    }

    // Sort the entries according to their column/row indices.

    isort_ ( irows, incl );
    isort_ ( jcols, incu );

    for ( i = 1; (loValues_.size() + incl +
                  upValues_.size() + incu) > maxNnz; i++ )
    {
      // Increase the drop tolerance until enough entries in the
      // current factors have been dropped.

      if ( droptol_ < MIN_DROPTOL )
      {
        droptol_  = MIN_DROPTOL;
      }
      else
      {
        value     = (double) i * (double) (size - ipiv) /
                    (0.5 * (double) size);
        droptol_ += droptol_ * max ( 0.1, value );
      }

      flops += (lint) (upValues_.size() + loValues_.size());

      prune_ ( work, ipiv );
    }

    // Store the new column in the lower factor.

    n = loValues_.size ();

    loNodes   = work.loNodes.extend ( incl );
    loIndices = loIndices_  .extend ( incl );
    loValues  = loValues_   .extend ( incl );

    for ( i = 0; i < incl; i++ )
    {
      irow            = irows[i];
      loValues [i]    = loAccu[irow];
      loIndices[i]    = irow;
      loNodes[i].jcol = ipiv;
      loNodes[i].next = rowLists[irow];
      rowLists[irow]  = n + i;
    }

    loOffsets[ipiv + 1] = loValues_.size ();

    // Store the new row in the upper factor.

    n = upValues_.size ();

    upNodes   = work.upNodes.extend ( incu );
    upIndices = upIndices_  .extend ( incu );
    upValues  = upValues_   .extend ( incu );

    for ( i = 0; i < incu; i++ )
    {
      jcol            = jcols[i];
      upValues [i]    = upAccu[jcol];
      upIndices[i]    = jcol;
      upNodes[i].irow = ipiv;
      upNodes[i].next = colLists[jcol];
      colLists[jcol]  = n + i;
    }

    upOffsets[ipiv + 1] = upValues_.size ();

    // Update the diagonal values.

    pivot = pivots[ipiv];

    for ( i = 0; i < incl; i++ )
    {
      irow = irows[i];

      if ( upTags[irow] == ipiv )
      {
        pivots[irow] -= pivot * loAccu[irow] * upAccu[irow];
      }
    }

    if ( testCancelled( flops ) )
    {
      throw CancelledException (
        "jem::numeric::SparseILUd::factor",
        "incomplete LU factorization cancelled"
      );
    }
  }

  // Invert the diagonal values.

  for ( i = 0; i < size; i++ )
  {
    if ( mask[iperm[i]] )
    {
      pivots[i] = 1.0 / pivots[i];
    }
    else
    {
      pivots[i] = 0.0;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   commit_
//-----------------------------------------------------------------------


void SparseILUd::commit_ ()
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


//-----------------------------------------------------------------------
//   prune_
//-----------------------------------------------------------------------


void SparseILUd::prune_

  ( Work_&  work,
    idx_t   ipiv )

{
  const double* JEM_RESTRICT  pivots  = pivots_.addr ();
  idx_t*        JEM_RESTRICT  offsets = nullptr;
  idx_t*        JEM_RESTRICT  indices = nullptr;
  double*       JEM_RESTRICT  values  = nullptr;
  Node_*        JEM_RESTRICT  nodes   = nullptr;
  idx_t*        JEM_RESTRICT  lists   = nullptr;
  idx_t*        JEM_RESTRICT  start   = nullptr;

  const idx_t   size = pivots_.size ();
  const double  tiny = droptol_ * droptol_;

  idx_t         n, nnz;


  // Prune the upper factor.

  offsets = upOffsets_   .addr ();
  indices = upIndices_   .addr ();
  values  = upValues_    .addr ();
  nodes   = work.upNodes .addr ();
  lists   = work.colLists.addr ();
  start   = work.rowStart.addr ();

  nnz     = 0;

  for ( idx_t jcol = 0; jcol < size; jcol++ )
  {
    lists[jcol] = -1_idx;
  }

  for ( idx_t irow = 0; irow < ipiv; irow++ )
  {
    idx_t  j    = offsets[irow];
    idx_t  jend = offsets[irow + 1];

    start  [irow] = nnz;
    offsets[irow] = nnz;

    for ( ; j < jend; j++ )
    {
      idx_t   jcol  = indices[j];
      double  value = values [j];

      if ( (value * value) <= tiny )
      {
        continue;
      }

      if ( jcol < ipiv )
      {
        if ( std::fabs( pivots[irow] * value * value ) <=
             std::fabs( pivots[jcol] * tiny ) )
        {
          continue;
        }
      }

      if ( jcol <= ipiv )
      {
        start[irow] = nnz;
      }

      // Keep this entry.

      values [nnz]    = value;
      indices[nnz]    = jcol;
      nodes[nnz].irow = irow;
      nodes[nnz].next = lists[jcol];
      lists[jcol]     = nnz++;
    }
  }

  offsets[ipiv] = nnz;

  n = upIndices_.size() - nnz;

  upIndices_  .popBack ( n );
  upValues_   .popBack ( n );
  work.upNodes.popBack ( n );

  // Prune the lower factor.

  offsets = loOffsets_   .addr ();
  indices = loIndices_   .addr ();
  values  = loValues_    .addr ();
  nodes   = work.loNodes .addr ();
  lists   = work.rowLists.addr ();
  start   = work.colStart.addr ();

  nnz     = 0;

  for ( idx_t irow = 0; irow < size; irow++ )
  {
    lists[irow] = -1_idx;
  }

  for ( idx_t jcol = 0; jcol < ipiv; jcol++ )
  {
    idx_t  i    = offsets[jcol];
    idx_t  iend = offsets[jcol + 1];

    start  [jcol] = nnz;
    offsets[jcol] = nnz;

    for ( ; i < iend; i++ )
    {
      idx_t   irow  = indices[i];
      double  value = values [i];

      if ( (value * value) <= tiny )
      {
        continue;
      }

      if ( irow < ipiv )
      {
        if ( std::fabs( pivots[jcol] * value * value ) <=
             std::fabs( pivots[irow] * tiny ) )
        {
          continue;
        }
      }

      if ( irow <= ipiv )
      {
        start[jcol] = nnz;
      }

      // Keep this entry.

      values [nnz]    = value;
      indices[nnz]    = irow;
      nodes[nnz].jcol = jcol;
      nodes[nnz].next = lists[irow];
      lists[irow]     = nnz++;
    }
  }

  offsets[ipiv] = nnz;

  n = loIndices_.size() - nnz;

  loIndices_  .popBack ( n );
  loValues_   .popBack ( n );
  work.loNodes.popBack ( n );
}


//-----------------------------------------------------------------------
//   lower_
//-----------------------------------------------------------------------


SparseILUd::Matrix SparseILUd::lower_

  ( const Matrix&        matrix,
    const Array<idx_t>&  iperm )

{
  const idx_t    size       = matrix.size (0);

  Array<idx_t>   matOffsets = matrix.getRowOffsets    ();
  Array<idx_t>   matIndices = matrix.getColumnIndices ();
  Array<double>  matValues  = matrix.getValues        ();

  Array<idx_t>   loOffsets  ( size + 1 );
  Array<idx_t>   loIndices;
  Array<double>  loValues;

  Array<idx_t>   permi      ( size );

  idx_t          nnz;


  for ( idx_t irow = 0; irow < size; irow++ )
  {
    permi[iperm[irow]] = irow;
  }

  loOffsets = 0_idx;

  // Count the number of non-zeroes in the lower triangle.

  for ( idx_t irow = 0; irow < size; irow++ )
  {
    idx_t  jrow = iperm[irow];
    idx_t  jend = matOffsets[jrow + 1];

    for ( idx_t j = matOffsets[jrow]; j < jend; j++ )
    {
      idx_t  jcol = permi[matIndices[j]];

      if ( jcol < irow )
      {
        loOffsets[jcol]++;
      }
    }
  }

  nnz = SparseUtils::sumOffsets ( loOffsets );

  loIndices.resize ( nnz );
  loValues .resize ( nnz );

  // Fill the lower triangle column wise.

  for ( idx_t irow = 0; irow < size; irow++ )
  {
    idx_t  jrow = iperm[irow];
    idx_t  jend = matOffsets[jrow + 1];

    for ( idx_t j = matOffsets[jrow]; j < jend; j++ )
    {
      idx_t  jcol = permi[matIndices[j]];

      if ( jcol < irow )
      {
        idx_t     k  = loOffsets[jcol]++;
        loIndices[k] = irow;
        loValues [k] = matValues[j];
      }
    }
  }

  SparseUtils::shiftOffsets ( loOffsets );

  return Matrix ( shape( size, size ),
                  loOffsets, loIndices, loValues );
}


//-----------------------------------------------------------------------
//   isort_
//-----------------------------------------------------------------------


void SparseILUd::isort_

  ( idx_t*  key,
    idx_t   len )

{
  idx_t  i, j, k, m, n;

  m = len / 2;
  n = len - 1;

  while ( n > 0 )
  {
    if ( m > 0 )
    {
      m -= 1;
      k  = key[m];
    }
    else
    {
      k      = key[n];
      key[n] = key[0];
      n     -= 1;
    }

    i = m;
    j = 2 * m + 1;

    while ( j <= n )
    {
      if ( (j < n) && (key[j] < key[j + 1]) )
      {
        j++;
      }

      if ( k < key[j] )
      {
        key[i] = key[j];
        i      = j;
        j      = j * 2 + 1;
      }
      else
      {
        break;
      }
    }

    key[i] = k;
  }
}


//-----------------------------------------------------------------------
//   asort_
//-----------------------------------------------------------------------


void SparseILUd::asort_

  ( const double*  key,
    idx_t*         index,
    double*        work,
    idx_t          len )

{
  idx_t   i, j, k, m, n;
  double  w;

  m = len / 2;
  n = len - 1;

  for ( i = 0; i < len; i++ )
  {
    work[i] = std::fabs( key[index[i]] );
  }

  while ( n > 0 )
  {
    if ( m > 0 )
    {
      m -= 1;
      w  = work [m];
      k  = index[m];
    }
    else
    {
      w        = work [n];
      work [n] = work [0];
      k        = index[n];
      index[n] = index[0];
      n       -= 1;
    }

    i = m;
    j = 2 * m + 1;

    while ( j <= n )
    {
      if ( (j < n) && (work[j] > work[j + 1]) )
      {
        j++;
      }

      if ( w > work[j] )
      {
        work [i] = work [j];
        index[i] = index[j];
        i        = j;
        j        = j * 2 + 1;
      }
      else
      {
        break;
      }
    }

    work [i] = w;
    index[i] = k;
  }
}


//-----------------------------------------------------------------------
//   matrixError_
//-----------------------------------------------------------------------


void SparseILUd::matrixError_

  ( const char*  where,
    idx_t        irow,
    idx_t        jcol )

{
  throw IllegalArgumentException (
    where,
    String::format (
      "invalid matrix; element (%d,%d) appears twice",
      irow,
      jcol
    )
  );
}


JEM_END_PACKAGE( numeric )
