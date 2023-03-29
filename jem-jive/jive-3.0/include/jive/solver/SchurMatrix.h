
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

#ifndef JIVE_SOLVER_SCHURMATRIX_H
#define JIVE_SOLVER_SCHURMATRIX_H

#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/MPMatrixExtension.h>
#include <jive/algebra/MultiMatmulExtension.h>
#include <jive/solver/typedefs.h>
#include <jive/solver/SchurDofSpace.h>


JIVE_BEGIN_PACKAGE( solver )


class Preconditioner;


//-----------------------------------------------------------------------
//   class SchurMatrix
//-----------------------------------------------------------------------


class SchurMatrix : public AbstractMatrix,
                    public MPMatrixExt,
                    public MultiMatmulExt
{
 public:

  JEM_DECLARE_CLASS       ( SchurMatrix, AbstractMatrix );


                            SchurMatrix

    ( const String&           name,
      Ref<MPContext>          mpx,
      Ref<AbstractMatrix>     inner,
      Ref<Preconditioner>     precon,
      Ref<SchurDofSpace>      dofs,
      Ref<Constraints>        cons = nullptr );

  void                      start             ();
  void                      finish            ();
  void                      update            ();
  bool                      updated           () const;

  virtual void              resetEvents       ()       override;
  virtual Shape             shape             () const override;

  virtual void              matmul

    ( const Vector&           schurLhs,
      const Vector&           schurRhs )         const override;

  virtual void              multiMatmul

    ( Matrix&                 lhsVecs,
      IdxVector&              lhsTags,
      const Matrix&           rhsVecs,
      const IdxVector&        rhsTags )          const override;

  virtual bool              hasTrait

    ( const String&           trait )            const override;

  virtual void*             getExtByID

    ( ExtensionID             extID )            const override;

  virtual VectorExchanger*  getExchanger      () const override;
  virtual int               getExchangeMode   () const override;

  void                      setExchangeMode

    ( int                     mode );

  idx_t                     schurSize         () const;
  idx_t                     totalSize         () const;

  void                      getSchurRhs

    ( const Vector&           schurRhs,
      const Vector&           totalRhs )         const;

  void                      getTotalLhs

    ( const Vector&           totalLhs,
      const Vector&           schurLhs,
      const Vector&           totalRhs )         const;

  void                      configure

    ( const Properties&       props );

  void                      getConfig

    ( const Properties&       conf )             const;

  inline AbstractMatrix*    getTotalMatrix    () const;
  inline Constraints*       getTotalCons      () const;
  inline Constraints*       getSchurCons      () const;
  inline DofSpace*          getTotalDofSpace  () const;
  inline SchurDofSpace*     getSchurDofSpace  () const;


 protected:

  virtual                  ~SchurMatrix       ();


 private:

  void                      connect_          ();
  void                      update_           ();
  void                      updateCons_       ();

  void                      localMatmul_

    ( const Vector&           schurLhs,
      const Vector&           schurRhs )         const;

  void                      consChanged_      ();
  void                      valuesChanged_    ();
  void                      structChanged_    ();

  void                      setEvents_

    ( int                     events );


 private:

  static const int          NEW_CONS_;
  static const int          NEW_STRUCT_;

  Ref<AbstractMatrix>       inner_;
  Ref<Preconditioner>       precon_;
  Ref<SchurDofSpace>        schurDofs_;
  Ref<VectorExchanger>      exchanger_;
  Ref<Constraints>          innerCons_;
  Ref<Constraints>          schurCons_;
  Ref<Constraints>          totalCons_;

  Matrix                    vbuf_;

  int                       xmode_;
  int                       events_;
  idx_t                     started_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getTotalMatrix
//-----------------------------------------------------------------------


inline AbstractMatrix* SchurMatrix::getTotalMatrix () const
{
  return inner_.get ();
}


//-----------------------------------------------------------------------
//   getTotalCons
//-----------------------------------------------------------------------


inline Constraints* SchurMatrix::getTotalCons () const
{
  return totalCons_.get ();
}


//-----------------------------------------------------------------------
//   getSchurCons
//-----------------------------------------------------------------------


inline Constraints* SchurMatrix::getSchurCons () const
{
  return schurCons_.get ();
}


//-----------------------------------------------------------------------
//   getTotalDofSpace
//-----------------------------------------------------------------------


inline DofSpace* SchurMatrix::getTotalDofSpace () const
{
  return schurDofs_->getTotalDofSpace ();
}


//-----------------------------------------------------------------------
//   getSchurDofSpace
//-----------------------------------------------------------------------


inline SchurDofSpace* SchurMatrix::getSchurDofSpace () const
{
  return schurDofs_.get ();
}


JIVE_END_PACKAGE( solver )

#endif
