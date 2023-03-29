
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

#ifndef JEM_NUMERIC_SPARSE_SPARSELU_H
#define JEM_NUMERIC_SPARSE_SPARSELU_H

#include <jem/util/Flex.h>
#include <jem/util/event/Event.h>
#include <jem/numeric/sparse/SparseMatrix.h>
#include <jem/numeric/sparse/SparseSolver.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class SparseLU
//-----------------------------------------------------------------------


class SparseLU : public SparseSolver
{
 public:

  typedef SparseLU            Self;
  typedef SparseSolver        Super;
  typedef
    SparseMatrix<double>      Matrix;

#ifdef JEM_STD_COMPLEX

  typedef
    std::complex<double>      Complex;
  typedef
    SparseMatrix<Complex>     ZMatrix;

#endif

  static const idx_t          NONE;
  static const idx_t          DOUBLE;
  static const idx_t          COMPLEX;

  util::Event<idx_t>          progressEvent;
  util::Event<idx_t,double>   pivotEvent;
  util::Event<idx_t,double>   zeroPivotEvent;


                              SparseLU          ();

  void                        clear             ();

  idx_t                       factor

    ( const Matrix&             matrix );

  idx_t                       factor

    ( const Matrix&             matrix,
      const Array<bool>&        mask );

  idx_t                       factor

    ( const Matrix&             matrix,
      const Array<bool>&        mask,
      const Array<idx_t>&       rowPerm,
      const Array<idx_t>&       colPerm );

  void                        solve

    ( const Array<double>&      lhs,
      const Array<double>&      rhs );

#ifdef JEM_STD_COMPLEX

  idx_t                       factor

    ( const ZMatrix&            matrix );

  idx_t                       factor

    ( const ZMatrix&            matrix,
      const Array<bool>&        mask );

  idx_t                       factor

    ( const ZMatrix&            matrix,
      const Array<bool>&        mask,
      const Array<idx_t>&       rowPerm,
      const Array<idx_t>&       colPerm );

  void                        solve

    ( const Array<Complex>&     lhs,
      const Array<Complex>&     rhs );

#endif

  inline idx_t                size              () const noexcept;
  inline idx_t                nonZeroCount      () const noexcept;
  double                      getMemUsage       () const noexcept;

  void                        setMaxZeroPivots

    ( idx_t                     maxZeroes );

  inline idx_t                getMaxZeroPivots  () const noexcept;

  void                        setPivotThreshold

    ( double                    alpha );

  inline double               getPivotThreshold () const noexcept;


 private:

  class                       Work_;
  class                       Matrix_;

  class                       Upper_
  {
   public:

    void                        clear           ();
    double                      getMemUsage     () const;

   public:

    Array<idx_t>                colOffsets;
    util::Flex<idx_t>           rowIndices;
    util::Flex<double>          values;
    Array<idx_t>                colPerm;

  };


  class                       Lower_
  {
   public:

    void                        clear           ();
    double                      getMemUsage     () const;

   public:

    Array<idx_t>                colOffsets;
    util::Flex<idx_t>           rowIndices;
    util::Flex<double>          values;
    Array<idx_t>                rowPerm;

  };


 private:

  void                        solve_            ();

  idx_t                       factor_

    ( const Matrix_&            matrix );

  idx_t                       factor_

    ( const Matrix_&            matrix,
      const Array<bool>&        mask );

  idx_t                       factor_

    ( const Matrix_&            matrix,
      const Array<bool>&        mask,
      const Array<idx_t>&       rowPerm,
      const Array<idx_t>&       colPerm );

  static void                 init_

    ( Work_&                    work,
      Upper_&                   upper,
      Lower_&                   lower,
      const Matrix_&            matrix,
      const Array<bool>&        mask,
      const Array<idx_t>&       rowPerm,
      const Array<idx_t>&       colPerm );

  static void                 transpose_

    ( Matrix_&                  trans,
      const Matrix_&            matrix,
      const Array<bool>&        mask );

  static void                 getColStruct_

    ( Work_&                    work,
      Upper_&                   upper,
      Lower_&                   lower,
      const Array<idx_t>&       irows );

  static void                 loadColumn_

    ( Work_&                    work,
      const Array<idx_t>&       irows,
      const Array<double>&      mvals );

  static void                 factorColumn_

    ( Work_&                    work,
      Upper_&                   upper,
      Lower_&                   lower,
      idx_t                     jcol );

  static void                 pivotColumn_

    ( Work_&                    work,
      Lower_&                   lower,
      idx_t                     jcol );

  static void                 pruneColumn_

    ( Work_&                    work,
      Lower_&                   lower,
      idx_t                     jcol );

  static void                 storeColumn_

    ( Work_&                    work,
      Upper_&                   upper,
      Lower_&                   lower,
      idx_t                     jcol );

  static void                 applyRowPerm_

    ( Lower_&                   lower,
      idx_t                     msize );

  static void                 applyUserPerm_

    ( Work_&                    work,
      Lower_&                   lower );

  static void                 checkData_

    ( const char*               where,
      const Work_&              work,
      const Upper_&             upper,
      const Lower_&             lower );


 private:

  static const idx_t          CANARY_VALUE_;
  static const lint           MAX_FLOP_COUNT_;

  Upper_                      upper_;
  Lower_                      lower_;
  Array<double>               scratch_;
  idx_t                       maxZeroPivots_;
  double                      pivotThreshold_;
  idx_t                       size_;
  idx_t                       type_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t SparseLU::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   nonZeroCount
//-----------------------------------------------------------------------


inline idx_t SparseLU::nonZeroCount () const noexcept
{
  return ( upper_.rowIndices.size() + lower_.rowIndices.size() );
}


//-----------------------------------------------------------------------
//   getMaxZeroPivots
//-----------------------------------------------------------------------


inline idx_t SparseLU::getMaxZeroPivots () const noexcept
{
  return maxZeroPivots_;
}


//-----------------------------------------------------------------------
//   getPivotThreshold
//-----------------------------------------------------------------------


inline double SparseLU::getPivotThreshold () const noexcept
{
  return pivotThreshold_;
}


JEM_END_PACKAGE( numeric )

#endif
