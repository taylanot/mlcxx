
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

#ifndef JEM_NUMERIC_SPARSE_SPARSEILUD_H
#define JEM_NUMERIC_SPARSE_SPARSEILUD_H

#include <jem/util/Flex.h>
#include <jem/numeric/sparse/SparseMatrix.h>
#include <jem/numeric/sparse/SparseSolver.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class SparseILUd
//-----------------------------------------------------------------------


class SparseILUd : public SparseSolver
{
 public:

  typedef SparseILUd        Self;
  typedef SparseSolver      Super;
  typedef
    SparseMatrix<double>    Matrix;

  static const double       MIN_DSHIFT;
  static const double       MIN_DROPTOL;


                            SparseILUd    ();

  void                      factor

    ( const Matrix&           matrix );

  void                      factor

    ( const Matrix&           matrix,
      const Array<bool>&      mask );

  void                      factor

    ( const Matrix&           matrix,
      const Array<bool>&      mask,
      const Array<idx_t>&     iperm );

  void                      solve

    ( const Array<double>&    lhs,
      const Array<double>&    rhs );

  void                      symsolve

    ( const Array<double>&    lhs,
      const Array<double>&    rhs );

  inline idx_t               size         () const noexcept;
  inline idx_t               nonZeroCount () const noexcept;
  double                     getMemUsage  () const noexcept;

  void                       setMinSize

    ( idx_t                    msize );

  inline idx_t               getMinSize   () const noexcept;

  void                       setMaxFill

    ( double                   mfill );

  inline double              getMaxFill   () const noexcept;

  void                       setDropTol

    ( double                   tol );

  inline double              getDropTol   () const noexcept;

  void                       setDiagShift

    ( double                   dshift );

  inline double              getDiagShift () const noexcept;

  void                       setQuality

    ( double                   qlty );

  inline double              getQuality   () const noexcept;


 private:

  class                     Node_;
  class                     Work_;
  friend class              Work_;

  void                      init_

    ( Work_&                  work,
      const Matrix&           matrix,
      const Array<bool>&      mask,
      const Array<idx_t>&     iperm );

  bool                      factor_

    ( Work_&                  work,
      const Matrix&           matrix,
      const Matrix&           lowmat,
      const Array<bool>&      mask,
      const Array<idx_t>&     iperm );

  void                      commit_       ();

  void                      prune_

    ( Work_&                  work,
      idx_t                   ipiv );

  static Matrix             lower_

    ( const Matrix&           matrix,
      const Array<idx_t>&     iperm );

  static void               isort_

    ( idx_t*                  key,
      idx_t                   len );

  static void               asort_

    ( const double*           key,
      idx_t*                  index,
      double*                 work,
      idx_t                   len );

  static void               matrixError_

    ( const char*             where,
      idx_t                   irow,
      idx_t                   jcol );


 private:

  idx_t                     size_;
  idx_t                     minSize_;
  double                    maxFill_;
  double                    dshift_;
  double                    droptol_;
  double                    quality_;

  Array<idx_t>              permi_;
  Array<double>             scale_;
  Array<double>             pivots_;
  Array<double>             scratch_;
  Array<idx_t>              upOffsets_;
  util::Flex<idx_t>         upIndices_;
  util::Flex<double>        upValues_;
  Array<idx_t>              loOffsets_;
  util::Flex<idx_t>         loIndices_;
  util::Flex<double>        loValues_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t SparseILUd::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   nonZeroCount
//-----------------------------------------------------------------------


inline idx_t SparseILUd::nonZeroCount () const noexcept
{
  return (size_ + upIndices_.size() + loIndices_.size());
}


//-----------------------------------------------------------------------
//   getMinSize
//-----------------------------------------------------------------------


inline idx_t SparseILUd::getMinSize () const noexcept
{
  return minSize_;
}


//-----------------------------------------------------------------------
//   getMaxFill
//-----------------------------------------------------------------------


inline double SparseILUd::getMaxFill () const noexcept
{
  return maxFill_;
}


//-----------------------------------------------------------------------
//   getDropTol
//-----------------------------------------------------------------------


inline double SparseILUd::getDropTol () const noexcept
{
  return droptol_;
}


//-----------------------------------------------------------------------
//   getDiagShift
//-----------------------------------------------------------------------


inline double SparseILUd::getDiagShift () const noexcept
{
  return dshift_;
}


//-----------------------------------------------------------------------
//   getDiagShift
//-----------------------------------------------------------------------


inline double SparseILUd::getQuality () const noexcept
{
  return quality_;
}


JEM_END_PACKAGE( numeric )

#endif
