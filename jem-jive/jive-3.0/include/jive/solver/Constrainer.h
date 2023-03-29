
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

#ifndef JIVE_SOLVER_CONSTRAINER_H
#define JIVE_SOLVER_CONSTRAINER_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/solver/import.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class Constrainer
//-----------------------------------------------------------------------


class Constrainer : public NamedObject
{
 public:

  JEM_DECLARE_CLASS       ( Constrainer, NamedObject );


  explicit                  Constrainer

    ( const String&           name = "" );

  virtual void              update            ()       = 0;
  virtual bool              updated           () const = 0;

  virtual void              getRhs

    ( const Vector&           g,
      const Vector&           f )                const = 0;

  virtual void              getLhs

    ( const Vector&           x,
      const Vector&           y )                const = 0;

  virtual void              initLhs

    ( const Vector&           x,
      const Vector&           y )                const = 0;

  virtual void              configure

    ( const Properties&       props );

  virtual void              getConfig

    ( const Properties&       props )            const;

  virtual Constraints*      getConstraints    () const = 0;
  virtual AbstractMatrix*   getInputMatrix    () const = 0;
  virtual AbstractMatrix*   getOutputMatrix   () const = 0;
  virtual String            getContext        () const override;


 protected:

  virtual                  ~Constrainer       ();

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<Constrainer>            newConstrainer

  ( const String&             name,
    Ref<Constraints>          cons,
    Ref<AbstractMatrix>       matrix );


JIVE_END_PACKAGE( solver )

#endif
