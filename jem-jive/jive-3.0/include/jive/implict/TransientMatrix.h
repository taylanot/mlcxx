
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

#ifndef JIVE_IMPLICT_TRANSIENTMATRIX_H
#define JIVE_IMPLICT_TRANSIENTMATRIX_H

#include <jive/algebra/AbstractMatrix.h>
#include <jive/implict/import.h>


JIVE_BEGIN_PACKAGE( implict )


//-----------------------------------------------------------------------
//   class TransientMatrix
//-----------------------------------------------------------------------


class TransientMatrix : public AbstractMatrix
{
 public:

  JEM_DECLARE_CLASS       ( TransientMatrix, AbstractMatrix );


                            TransientMatrix

    ( const String&           name,
      int                     order,
      const Ref<Model>&       model,
      const Properties&       globdat );

  virtual void              resetEvents       ()       override;

  virtual Shape             shape             () const override;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs  )             const override;

  virtual bool              hasTrait

    ( const String&           trait )            const override;

  virtual void*             getExtByID

    ( ExtensionID             extID )            const override;

  Ref<AbstractMatrix>       getMatrix         () const;

  AbstractMatrix*           getMatrix

    ( int                     imat )             const;

  inline void               invalidate        ();

  void                      update

    ( const Vector&           fint,
      const Properties&       globdat );

  AbstractMatrix*           updateMatrix0

    ( const Vector&           fint,
      const Properties&       globdat );

  AbstractMatrix*           updateMatrix1

    ( const Properties&       globdat );

  AbstractMatrix*           updateMatrix2

    ( const Properties&       globdat );

  void                      setCoeffs

    ( double                  a = 1.0,
      double                  b = 0.0,
      double                  c = 0.0 );


 protected:

  virtual                  ~TransientMatrix   ();


 private:

  void                      shapeError_

    ( int                     imat )             const;

  void                      valuesChanged_    ();
  void                      structChanged_    ();


 private:

  static const int          U_MATRIX_;
  static const int          U_MATRIX0_;
  static const int          U_MATRIX1_;
  static const int          U_MATRIX2_;


  Ref<Model>                model_;
  Ref<SparseMatrixObj>      transmat_;
  Ref<AbstractMatrix>       matrices_[3];

  double                    coeffs_[3];
  int                       updated_;

  Vector                    vbuf_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   invalidate
//-----------------------------------------------------------------------


inline void TransientMatrix::invalidate ()
{
  // The U_MATRIX_ bit is left in its original state; it will be
  // cleared when one of the other matrices is updated. In this way
  // the transient matrix is only updated when really necessary.

  updated_ &= U_MATRIX_;
}


JIVE_END_PACKAGE( implict )

#endif
