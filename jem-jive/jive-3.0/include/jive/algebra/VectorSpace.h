
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

#ifndef JIVE_ALGEBRA_VECTORSPACE_H
#define JIVE_ALGEBRA_VECTORSPACE_H

#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/algebra/import.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   class VectorSpace
//-----------------------------------------------------------------------


class VectorSpace : public Object
{
 public:

  JEM_DECLARE_CLASS       ( VectorSpace, Object );


  idx_t                     size            () const;
  virtual idx_t             globalSize      () const = 0;
  virtual bool              isDistributed   () const;

  virtual double            norm2

    ( const Vector&           x )              const;

  virtual double            product

    ( const Vector&           x,
      const Vector&           y )              const = 0;

  virtual void              products

    ( const Vector&           a,
      const Matrix&           x,
      const Matrix&           y )              const;

  virtual void              project

    ( const Vector&           a,
      const Vector&           x,
      const Matrix&           y )              const;

  virtual DofSpace*         getDofSpace     () const = 0;

  void                      store

    ( const Ref<DofSpace>&    dofs,
      const Properties&       globdat )        const;

  static Ref<VectorSpace>   find

    ( const Ref<DofSpace>&    dofs,
      const Properties&       globdat );

  static Ref<VectorSpace>   get

    ( const Ref<DofSpace>&    dofs,
      const Properties&       globdat );


 protected:

  virtual                  ~VectorSpace     ();

};


JIVE_END_PACKAGE( algebra )

#endif
