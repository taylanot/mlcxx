
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

#ifndef JEM_NUMERIC_SPARSE_SPARSEILU_H
#define JEM_NUMERIC_SPARSE_SPARSEILU_H

#include <jem/numeric/sparse/SparseMatrix.h>
#include <jem/numeric/sparse/SparseSolver.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class SparseILU
//-----------------------------------------------------------------------


class SparseILU : public SparseSolver
{
 public:

  typedef SparseILU         Self;
  typedef SparseSolver      Super;
  typedef
    SparseMatrix<double>    Matrix;


                            SparseILU   ();

  idx_t                     factor

    ( const Matrix&           matrix );

  idx_t                     factor

    ( const Matrix&           matrix,
      const Array<bool>&      mask );

  idx_t                     factor

    ( const Matrix&           matrix,
      const Array<bool>&      mask,
      const Array<idx_t>&     iperm );

  void                      solve

    ( const Array<double>&    lhs,
      const Array<double>&    rhs );

  void                      symsolve

    ( const Array<double>&    lhs,
      const Array<double>&    rhs );

  inline idx_t               size        () const noexcept;


 private:

  class                     Work_;


  void                      clear_      ();

  void                      init_

    ( Work_&                  work,
      const Matrix&           matrix,
      const Array<bool>&      mask,
      const Array<idx_t>&     iperm );

  void                      factor_

    ( Work_&                  work );


 private:

  Array<idx_t>              upperOffsets_;
  Array<idx_t>              upperIndices_;
  Array<double>             upperValues_;

  Array<idx_t>              lowerOffsets_;
  Array<idx_t>              lowerIndices_;
  Array<double>             lowerValues_;

  Array<idx_t>              permi_;
  Array<double>             scale_;
  Array<double>             pivots_;
  Array<double>             scratch_;

  idx_t                     size_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t SparseILU::size () const noexcept
{
  return size_;
}


JEM_END_PACKAGE( numeric )

#endif
