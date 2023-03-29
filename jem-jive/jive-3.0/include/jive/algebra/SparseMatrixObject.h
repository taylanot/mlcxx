
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

#ifndef JEM_ALGEBRA_SPARSEMATRIXOBJECT_H
#define JEM_ALGEBRA_SPARSEMATRIXOBJECT_H

#include <jem/base/Flags.h>
#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/SparseMatrix.h>
#include <jive/algebra/import.h>
#include <jive/algebra/DiagMatrixExtension.h>
#include <jive/algebra/MultiMatmulExtension.h>
#include <jive/algebra/DirectMatrixExtension.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/algebra/AbstractMatrix.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   class SparseMatrixObject
//-----------------------------------------------------------------------


class SparseMatrixObject : public AbstractMatrix,
                           public DiagMatrixExt,
                           public MultiMatmulExt,
                           public DirectMatrixExt,
                           public SparseMatrixExt,
                           public Clonable,
                           public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( SparseMatrixObject, AbstractMatrix );

  enum                      Trait
  {
                              SYMMETRIC       = 1 << 0
  };

  enum                      Option
  {
                              ENABLE_BLOCKING = 1 << 0
  };

  typedef jem::
    Flags<Trait>            Traits;
  typedef jem::
    Flags<Option>           Options;


  explicit                  SparseMatrixObject

    ( const String&           name   = "",
      Traits                  traits = 0 );

  explicit                  SparseMatrixObject

    ( const String&           name,
      const SparseMatrix&     matrix,
      Traits                  traits = 0 );

  virtual void              readFrom

    ( ObjectInput&            in )                       override;

  virtual void              writeTo

    ( ObjectOutput&           out )                const override;

  virtual Ref<Object>       clone               () const override;
  virtual Shape             shape               () const override;

  inline idx_t                size

    ( int                     idim )               const;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs  )               const override;

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

  virtual void              printTo

    ( PrintWriter&            out )                const override;

  void                      setToZero           ();

  void                      setMatrix

    ( const SparseMatrix&     matrix );

  void                      addMatrix

    ( const SparseMatrix&     rhs,
      double                  scale = 1.0 );

  void                      setValues

    ( const Vector&           values );

  inline idx_t              nonZeroCount        () const;
  inline IdxVector          getRowOffsets       () const;
  inline IdxVector          getColumnIndices    () const;
  inline Vector             getValues           () const;

  void                      setOption

    ( Option                  option,
      bool                    yesno = true );

  void                      setOptions

    ( Options                 options );

  inline Options            getOptions          () const noexcept;
  inline Traits             getTraits           () const noexcept;
  inline bool               isSymmetric         () const noexcept;


 protected:

  virtual                  ~SparseMatrixObject  ();


 private:

  void                      init_               ();
  void                      packValues_         ();
  void                      initSuperRows_      ();
  void                      resetMCounter_      ();

  void                      normalMatmul_

    ( const Vector&           lhs,
      const Vector&           rhs  )               const;

  void                      normalMatmul4_

    ( const Matrix&           lhs,
      const Matrix&           rhs )                const;

  void                      packedMatmul_

    ( const Vector&           lhs,
      const Vector&           rhs  )               const;

  void                      packedMatmul4_

    ( const Matrix&           lhs,
      const Matrix&           rhs )                const;


 private:

  enum                    { MAX_BLOCK_SIZE_ = 4 };

  static const int          SORTED_;
  static const int          BLOCKED_;
  static const int          PACKED_;

  Traits                    traits_;
  int                       status_;
  Options                   options_;
  idx_t                     mcounter_;

  Shape                     shape_;
  IdxVector                 rowOffsets_;
  IdxVector                 colIndices_;
  Vector                    matValues_;

  IdxVector                 supRows_;
  Vector                    supValues_;
  jem::Tuple<idx_t,6>       supOffsets_;

};


JEM_DEFINE_FLAG_OPS( SparseMatrixObject::Traits )
JEM_DEFINE_FLAG_OPS( SparseMatrixObject::Options )


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef SparseMatrixObject  SparseMatrixObj;


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<SparseMatrixObj>        newSparseMatrix

  ( const String&             name,
    const Properties&         conf,
    const Properties&         props,
    const Properties&         params );






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t SparseMatrixObject::size ( int idim ) const
{
  return shape_[idim];
}


//-----------------------------------------------------------------------
//   nonZeroCount
//-----------------------------------------------------------------------


inline idx_t SparseMatrixObject::nonZeroCount () const
{
  return matValues_.size ();
}


//-----------------------------------------------------------------------
//   getRowOffsets
//-----------------------------------------------------------------------


inline IdxVector SparseMatrixObject::getRowOffsets () const
{
  return rowOffsets_;
}


//-----------------------------------------------------------------------
//   getColumnIndices
//-----------------------------------------------------------------------


inline IdxVector SparseMatrixObject::getColumnIndices () const
{
  return colIndices_;
}


//-----------------------------------------------------------------------
//   getValues
//-----------------------------------------------------------------------


inline Vector SparseMatrixObject::getValues () const
{
  return matValues_;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


inline SparseMatrixObject::Options
  SparseMatrixObject::getOptions () const noexcept
{
  return options_;
}



//-----------------------------------------------------------------------
//   getTraits
//-----------------------------------------------------------------------


inline SparseMatrixObject::Traits
  SparseMatrixObject::getTraits () const noexcept
{
  return traits_;
}


//-----------------------------------------------------------------------
//   isSymmetric
//-----------------------------------------------------------------------


inline bool SparseMatrixObject::isSymmetric () const noexcept
{
  return ((traits_ & SYMMETRIC) != 0);
}


JIVE_END_PACKAGE( algebra )

#endif
