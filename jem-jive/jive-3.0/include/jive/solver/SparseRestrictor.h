
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

#ifndef JIVE_SOLVER_SPARSERESTRICTOR_H
#define JIVE_SOLVER_SPARSERESTRICTOR_H

#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/solver/Restrictor.h>
#include <jive/solver/LocalRestrictor.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class SparseRestrictor
//-----------------------------------------------------------------------


class SparseRestrictor : public Restrictor,
                         public SparseMatrixExt,
                         public LocalRestrictor
{
 public:

  JEM_DECLARE_CLASS       ( SparseRestrictor, Restrictor );


  virtual Shape             shape               () const override;

  virtual void              start               ()       override;
  virtual void              finish              ()       override;
  virtual void              update              ()       override;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs )                const override;

  virtual void              transmul

    ( const Vector&           lhs,
      const Vector&           rhs )                const override;

  virtual void              getBasis

    ( const Vector&           v,
      idx_t                   j )                  const override;

  virtual void*             getExtByID

    ( ExtensionID             extID )              const override;

  virtual SparseStruct      getStructure        () const override;
  virtual SparseMatrix      toSparseMatrix      () const override;
  virtual SparseMatrix      cloneSparseMatrix   () const override;


 protected:

                            SparseRestrictor

    ( const String&           name = "" );

  virtual                  ~SparseRestrictor    ();

  virtual void              update_

    ( int                     events )                   = 0;

  void                      setEvents_

    ( int                     events );


 protected:

  static const int          NEW_VALUES_;
  static const int          NEW_STRUCT_;

  Shape                     shape_;
  IdxVector                 rowOffsets_;
  IdxVector                 colIndices_;
  Vector                    rtValues_;


 private:

  int                       events_;
  idx_t                     started_;

};


JIVE_END_PACKAGE( solver )

#endif
