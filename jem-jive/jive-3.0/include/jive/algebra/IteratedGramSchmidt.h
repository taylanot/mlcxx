
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

#ifndef JIVE_ALGEBRA_ITERATEDGRAMSCHMIDT_H
#define JIVE_ALGEBRA_ITERATEDGRAMSCHMIDT_H

#include <jive/algebra/GramSchmidt.h>


JIVE_BEGIN_PACKAGE( algebra )


class VectorSpace;


//-----------------------------------------------------------------------
//   class IteratedGramSchmidt
//-----------------------------------------------------------------------


class IteratedGramSchmidt : public GramSchmidt
{
 public:

  JEM_DECLARE_CLASS       ( IteratedGramSchmidt, GramSchmidt );

  static const char*        TYPE_NAME;


                            IteratedGramSchmidt

    ( const String&           name,
      Ref<GramSchmidt>        gs );

  virtual void              orthogonalize

    ( const Vector&           a,
      const Vector&           x,
      const Matrix&           y )                  const override;

  virtual void              configure

    ( const Properties&       props )                    override;

  virtual void              getConfig

    ( const Properties&       props )              const override;

  void                      setMaxIterCount

    ( idx_t                   count );

  idx_t                     getMaxIterCount     () const;


  static Ref<GramSchmidt>   makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      Ref<VectorSpace>        vspace );

  static void               declare             ();


 protected:

  virtual                  ~IteratedGramSchmidt ();


 private:

  Ref<GramSchmidt>          gs_;
  Vector                    scratch_;
  idx_t                     maxIter_;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef
  IteratedGramSchmidt       IteratedGS;



JIVE_END_PACKAGE( algebra )

#endif
