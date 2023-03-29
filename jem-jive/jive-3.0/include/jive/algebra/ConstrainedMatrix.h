
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

#ifndef JIVE_ALGEBRA_CONSTRAINEDMATRIX_H
#define JIVE_ALGEBRA_CONSTRAINEDMATRIX_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/MultiMatmulExtension.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   class ConstrainedMatrix
//-----------------------------------------------------------------------


class ConstrainedMatrix : public  AbstractMatrix,
                          public  Clonable,
                          public  Serializable,
                          private MultiMatmulExt
{
 public:

  JEM_DECLARE_CLASS       ( ConstrainedMatrix, AbstractMatrix );


                            ConstrainedMatrix   ();

                            ConstrainedMatrix

    ( Ref<AbstractMatrix>     inner,
      Ref<Constraints>        cons );

  virtual void              resetEvents         ()       override;

  virtual void              readFrom

    ( ObjectInput&            in )                       override;

  virtual void              writeTo

    ( ObjectOutput&           out )                const override;

  virtual Ref<Object>       clone               () const override;
  virtual Shape             shape               () const override;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs )                const override;

  virtual void*             getExtByID

    ( ExtensionID             extID )              const override;

  virtual bool              hasTrait

    ( const String&           trait )              const override;

  inline idx_t              size                () const;
  void                      update              ();
  inline bool               updated             () const;

  void                      getRhs

    ( const Vector&           g,
      const Vector&           f )                  const;

  void                      getLhs

    ( const Vector&           x,
      const Vector&           y )                  const;

  void                      initLhs

    ( const Vector&           x,
      const Vector&           y )                  const;

  inline AbstractMatrix*    getInner            () const;
  inline Constraints*       getConstraints      () const;


 protected:

  virtual                  ~ConstrainedMatrix   ();


 private:

  virtual void              multiMatmul

    ( Matrix&                 lhsVecs,
      IdxVector&              lhsTags,
      const Matrix&           rhsVecs,
      const IdxVector&        rhsTags )            const override;


 private:

  void                      connect_            ();
  void                      structChanged_      ();
  void                      valuesChanged_      ();


 private:

  Ref<AbstractMatrix>       inner_;
  Ref<Constraints>          cons_;
  Vector                    tmpRhs_;
  bool                      updated_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t ConstrainedMatrix::size () const
{
  return inner_->shape()[0];
}


//-----------------------------------------------------------------------
//   updated
//-----------------------------------------------------------------------


inline bool ConstrainedMatrix::updated () const
{
  return updated_;
}


//-----------------------------------------------------------------------
//   getInner
//-----------------------------------------------------------------------


inline AbstractMatrix* ConstrainedMatrix::getInner () const
{
  return inner_.get ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


inline Constraints* ConstrainedMatrix::getConstraints () const
{
  return cons_.get ();
}


JIVE_END_PACKAGE( algebra )

#endif
