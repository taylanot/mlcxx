
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

#ifndef JIVE_SOLVER_GENERICCONSTRAINER_H
#define JIVE_SOLVER_GENERICCONSTRAINER_H

#include <jive/solver/import.h>
#include <jive/solver/Constrainer.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class GenericConstrainer
//-----------------------------------------------------------------------


class GenericConstrainer : public Constrainer
{
 public:

  JEM_DECLARE_CLASS       ( GenericConstrainer, Constrainer );

  static const char*        TYPE_NAME;


                            GenericConstrainer

    ( const String&           name,
      Ref<ConstrainedMatrix>  matrix );

                            GenericConstrainer

    ( const String&           name,
      Ref<Constraints>        cons,
      Ref<AbstractMatrix>     matrix );

  virtual void              update              ()       override;
  virtual bool              updated             () const override;

  virtual void              getRhs

    ( const Vector&           g,
      const Vector&           f )                  const override;

  virtual void              getLhs

    ( const Vector&           x,
      const Vector&           y )                  const override;

  virtual void              initLhs

    ( const Vector&           x,
      const Vector&           y )                  const override;

  virtual Constraints*      getConstraints      () const override;
  virtual AbstractMatrix*   getInputMatrix      () const override;
  virtual AbstractMatrix*   getOutputMatrix     () const override;

  static Ref<Constrainer>   makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      Ref<Constraints>        cons,
      Ref<AbstractMatrix>     matrix );


  static void               declare             ();


 protected:

  virtual                  ~GenericConstrainer  ();


 private:

  Ref<ConstrainedMatrix>    output_;

};


JIVE_END_PACKAGE( solver )

#endif
