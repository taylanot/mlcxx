
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

#ifndef JIVE_SOLVER_COARSEMATRIX_H
#define JIVE_SOLVER_COARSEMATRIX_H

#include <jive/algebra/AbstractMatrix.h>


JIVE_BEGIN_PACKAGE( solver )


class Restrictor;
class CoarseDofSpace;


//-----------------------------------------------------------------------
//   class CoarseMatrix
//-----------------------------------------------------------------------


class CoarseMatrix : public AbstractMatrix
{
 public:

  JEM_DECLARE_CLASS       ( CoarseMatrix, AbstractMatrix );


                            CoarseMatrix

    ( const String&           name,
      Ref<AbstractMatrix>     inner,
      Ref<CoarseDofSpace>     dofs );

  void                      start             ();
  void                      finish            ();
  void                      update            ();

  virtual void              resetEvents       ()       override;

  virtual Shape             shape             () const override;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs )              const override;

  virtual bool              hasTrait

    ( const String&           trait )            const override;

  inline AbstractMatrix*    getInner          () const;
  inline Restrictor*        getRestrictor     () const;
  inline CoarseDofSpace*    getCoarseDofs     () const;

  idx_t                     assemble

    ( Matrix&                 cmat )             const;


 protected:

  virtual                  ~CoarseMatrix      ();


 private:

  void                      connect_          ();

  idx_t                     assemble_

    ( Matrix&                 cmat )             const;

  idx_t                     assembleMMX_

    ( Matrix&                 cmat )             const;

  idx_t                     assembleMPX_

    ( Matrix&                 cmat )             const;

  void                      valuesChanged_    ();
  void                      structChanged_    ();


 private:

  Ref<Restrictor>           rtor_;
  Ref<AbstractMatrix>       inner_;
  Ref<CoarseDofSpace>       coarseDofs_;

  Vector                    totalRhs_;
  Vector                    totalLhs_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getInner
//-----------------------------------------------------------------------


inline AbstractMatrix* CoarseMatrix::getInner () const
{
  return inner_.get ();
}


//-----------------------------------------------------------------------
//   getRestrictor
//-----------------------------------------------------------------------


inline Restrictor* CoarseMatrix::getRestrictor () const
{
  return rtor_.get ();
}


//-----------------------------------------------------------------------
//   getCoarseDofs
//-----------------------------------------------------------------------


inline CoarseDofSpace* CoarseMatrix::getCoarseDofs () const
{
  return coarseDofs_.get ();
}


JIVE_END_PACKAGE( solver )

#endif
