
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

#ifndef JEM_NUMERIC_SPARSE_SPARSEILUN_H
#define JEM_NUMERIC_SPARSE_SPARSEILUN_H

#include <jem/util/Flex.h>
#include <jem/numeric/sparse/SparseMatrix.h>
#include <jem/numeric/sparse/SparseSolver.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class SparseILUn
//-----------------------------------------------------------------------


class SparseILUn : public SparseSolver
{
 public:

  typedef SparseILUn        Self;
  typedef SparseSolver      Super;
  typedef
    SparseMatrix<double>    Matrix;


                            SparseILUn    ();

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

  void                       setMaxFill

    ( idx_t                    mfill );

  inline idx_t               getMaxFill   () const noexcept;

  void                       setDropTol

    ( double                   tol );

  inline double              getDropTol   () const noexcept;

  void                       setDiagShift

    ( double                   dshift );

  inline double              getDiagShift () const noexcept;


 private:

  void                      init_

    ( const Matrix&           matrix,
      const Array<bool>&      mask,
      const Array<idx_t>&     iperm );

  bool                      factor_

    ( const Matrix&           matrix,
      const Array<bool>&      mask,
      const Array<idx_t>&     iperm );

  void                      commit_       ();


 private:

  idx_t                     size_;
  idx_t                     maxFill_;
  double                    dshift_;
  double                    droptol_;

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


inline idx_t SparseILUn::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   nonZeroCount
//-----------------------------------------------------------------------


inline idx_t SparseILUn::nonZeroCount () const noexcept
{
  return (size_ + upIndices_.size() + loIndices_.size());
}


//-----------------------------------------------------------------------
//   getMaxFill
//-----------------------------------------------------------------------


inline idx_t SparseILUn::getMaxFill () const noexcept
{
  return maxFill_;
}


//-----------------------------------------------------------------------
//   getDropTol
//-----------------------------------------------------------------------


inline double SparseILUn::getDropTol () const noexcept
{
  return droptol_;
}


//-----------------------------------------------------------------------
//   getDiagShift
//-----------------------------------------------------------------------


inline double SparseILUn::getDiagShift () const noexcept
{
  return dshift_;
}


JEM_END_PACKAGE( numeric )

#endif
