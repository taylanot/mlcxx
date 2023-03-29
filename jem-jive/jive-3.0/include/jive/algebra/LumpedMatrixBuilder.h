
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

#ifndef JIVE_ALGEBRA_LUMPEDMATRIXBUILDER_H
#define JIVE_ALGEBRA_LUMPEDMATRIXBUILDER_H

#include <jem/base/Flags.h>
#include <jive/algebra/import.h>
#include <jive/algebra/typedefs.h>
#include <jive/algebra/MatrixBuilder.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   class LumpedMatrixBuilder
//-----------------------------------------------------------------------


class LumpedMatrixBuilder : public MatrixBuilder
{
 public:

  JEM_DECLARE_CLASS       ( LumpedMatrixBuilder, MatrixBuilder );

  static const char*        TYPE_NAME;

  enum                      Option
  {
                              AUTO_RESHAPE = 1 << 0
  };

  typedef
    jem::Flags<Option>      Options;


  explicit                  LumpedMatrixBuilder

    ( const String&           name = "" );

                            LumpedMatrixBuilder

    ( const String&           name,
      Ref<DiagMatrixObj>      mat );

  virtual void              clear               ()       override;

  virtual void              scale

    ( double                  factor )                   override;

  virtual void              setToZero           ()       override;

  virtual void              reserve

    ( idx_t                   nzCount )                  override;

  virtual void              shapeHint

    ( idx_t                   rowCount,
      idx_t                   colCount )                 override;

  virtual void              trimToSize          ()       override;
  virtual void              updateMatrix        ()       override;

  virtual void              setMultiplier

    ( double                  x )                        override;

  virtual double            getMultiplier       () const override;

  virtual void              setData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount,
      const double*           values )                   override;

  virtual void              addData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount,
      const double*           values )                   override;

  virtual idx_t             eraseData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount )                   override;

  virtual void              getData

    ( double*                 buf,
      const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount )             const override;

  virtual AbstractMatrix*   getMatrix           () const override;
  DiagMatrixObj*            getDiagMatrix       () const;

  void                      setSize

    ( idx_t                   size );

  void                      setOption

    ( Option                  option,
      bool                    yesno = true );

  void                      setOptions

    ( Options                 options );

  Options                   getOptions        () const;

  void                      trackDofSpace

    ( const Ref<DofSpace>&    dofs );

  static Ref<MBuilder>      makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       params,
      const Properties&       globdat );

  static void               declare             ();


 protected:

  virtual                  ~LumpedMatrixBuilder ();


 private:

  void                      newDofCount_        ();


 private:

  class                     Data_;

  Ref<Data_>                data_;

};


JEM_DEFINE_FLAG_OPS( LumpedMatrixBuilder::Options )


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef
  LumpedMatrixBuilder       LumpedMBuilder;


JIVE_END_PACKAGE( algebra )

#endif
