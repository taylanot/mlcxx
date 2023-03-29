
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

#ifndef JIVE_ALGEBRA_DIAGMATRIXOBJECT_H
#define JIVE_ALGEBRA_DIAGMATRIXOBJECT_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/algebra/DiagMatrixExtension.h>
#include <jive/algebra/MultiMatmulExtension.h>
#include <jive/algebra/DirectMatrixExtension.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/algebra/AbstractMatrix.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   class DiagMatrixObject
//-----------------------------------------------------------------------


class DiagMatrixObject : public AbstractMatrix,
                         public DiagMatrixExt,
                         public MultiMatmulExt,
                         public DirectMatrixExt,
                         public SparseMatrixExt,
                         public Clonable,
                         public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( DiagMatrixObject, AbstractMatrix );


  explicit                  DiagMatrixObject

    ( const String&           name = "" );

                            DiagMatrixObject

    ( const String&           name,
      const Vector&           diag );

  virtual void              readFrom

    ( ObjectInput&            in  )                      override;

  virtual void              writeTo

    ( ObjectOutput&           out )                const override;

  virtual Ref<Object>       clone               () const override;
  virtual Shape             shape               () const override;
  inline idx_t              size                () const;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs )                const override;

  virtual void              multiMatmul

    ( Matrix&                 lhsVecs,
      IdxVector&              lhsTags,
      const Matrix&           rhsVecs,
      const IdxVector&        rhsTags )            const override;

  virtual bool              hasTrait

    ( const String&           trait )              const override;

  virtual void*             getExtByID

    ( ExtensionID             extID )              const override;

  virtual double            getValue

    ( idx_t                   irow,
      idx_t                   jcol )               const override;

  virtual void              getBlock

    ( const Matrix&           block,
      const IdxVector&        irows,
      const IdxVector&        jcols )              const override;

  virtual void              getDiagonal

    ( const Vector&           diag )               const override;

  virtual void              getRowScales

    ( const Vector&           rscales )            const override;

  virtual SparseStruct      getStructure        () const override;
  virtual SparseMatrix      toSparseMatrix      () const override;
  virtual SparseMatrix      cloneSparseMatrix   () const override;

  virtual void              setValues

    ( const Vector&           diag );

  virtual void              handleNewValues     ();

  virtual void              printTo

    ( PrintWriter&            out )                const override;

  inline const Vector&      getValues           () const;


 protected:

  virtual                  ~DiagMatrixObject    ();


 protected:

  Vector                    values_;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef DiagMatrixObject    DiagMatrixObj;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t DiagMatrixObject::size () const
{
  return values_.size ();
}


//-----------------------------------------------------------------------
//   getValues
//-----------------------------------------------------------------------


inline const Vector& DiagMatrixObject::getValues () const
{
  return values_;
}


JIVE_END_PACKAGE( algebra )

#endif
