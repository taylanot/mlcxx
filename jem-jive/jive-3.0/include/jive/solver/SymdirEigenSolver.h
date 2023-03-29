
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

#ifndef JIVE_SOLVER_SYMDIREIGENSOLVER_H
#define JIVE_SOLVER_SYMDIREIGENSOLVER_H

#include <jive/solver/import.h>
#include <jive/solver/EigenSolver.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class SymdirEigenSolver
//-----------------------------------------------------------------------


class SymdirEigenSolver : public EigenSolver
{
 public:

  JEM_DECLARE_CLASS       ( SymdirEigenSolver, EigenSolver );

  static const char*        TYPE_NAME;


                            SymdirEigenSolver

    ( const String&           name,
      Ref<AbstractMatrix>     lmat,
      Ref<AbstractMatrix>     rmat = nullptr );

  virtual void              getInfo

    ( const Properties&       info )             const override;

  virtual void              getEigenValues

    ( const Matrix&           evals,
      idx_t                   smallCount )             override;

  virtual void              getEigenVectors

    ( const Matrix&           evals,
      const Cubix&            evecs,
      idx_t                   smallCount )             override;

  virtual void              getRealEigenValues

    ( const Vector&           evals,
      idx_t                   smallCount )             override;

  virtual void              getRealEigenVectors

    ( const Vector&           evals,
      const Matrix&           evecs,
      idx_t                   smallCount )             override;

  virtual void              setPrecision

    ( double                  eps )                    override;

  virtual double            getPrecision      () const override;

  static Ref<EigenSolver>   makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       params );

  static void               declare           ();


 protected:

  virtual                  ~SymdirEigenSolver ();


 private:

  void                      getEigenValues_

    ( const Vector&           e,
      const Matrix&           lmat,
      const Matrix&           rmat )             const;

  void                      getEigenVectors_

    ( const Vector&           e,
      const Matrix&           v,
      const Matrix&           lmat,
      const Matrix&           rmat )             const;

  Matrix                    toDenseMatrix_

    ( AbstractMatrix&         amat )             const;

  String                    getMatrixName_

    ( AbstractMatrix&         amat )             const;

  inline void               checkShapes_

    ( const Matrix&           lmat,
      const Matrix&           rmat )             const;

  void                      choleskyError_    () const;


 private:

  Ref<AbstractMatrix>       lhMatrix_;
  Ref<AbstractMatrix>       rhMatrix_;

  double                    precision_;

};



JIVE_END_PACKAGE( solver )

#endif
