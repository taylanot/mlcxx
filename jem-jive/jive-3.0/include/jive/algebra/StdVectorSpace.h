
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

#ifndef JIVE_ALGEBRA_STDVECTORSPACE_H
#define JIVE_ALGEBRA_STDVECTORSPACE_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/algebra/import.h>
#include <jive/algebra/VectorSpace.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   class StdVectorSpace
//-----------------------------------------------------------------------


class StdVectorSpace : public VectorSpace,
                       public Clonable,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( StdVectorSpace, VectorSpace );


                            StdVectorSpace  ();

  explicit                  StdVectorSpace

    ( const Ref<DofSpace>&    dofs );

  virtual void              readFrom

    ( ObjectInput&            in  )                  override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  virtual Ref<Object>       clone           () const override;
  virtual idx_t             globalSize      () const override;

  virtual double            norm2

    ( const Vector&           x )              const override;

  virtual double            product

    ( const Vector&           x,
      const Vector&           y )              const override;

  virtual void              products

    ( const Vector&           a,
      const Matrix&           x,
      const Matrix&           y )              const override;

  virtual void              project

    ( const Vector&           a,
      const Vector&           x,
      const Matrix&           y )              const override;

  virtual DofSpace*         getDofSpace     () const override;


 protected:

  virtual                  ~StdVectorSpace  ();


 private:

  Ref<DofSpace>             dofs_;

};


JIVE_END_PACKAGE( algebra )

#endif
