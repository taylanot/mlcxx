
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

#ifndef JIVE_SOLVER_SOLVER_H
#define JIVE_SOLVER_SOLVER_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/solver/import.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class Solver
//-----------------------------------------------------------------------


class Solver : public NamedObject
{
 public:

  JEM_DECLARE_CLASS       ( Solver, NamedObject );

  static const double       PRECISION;
  static const int          LENIENT_MODE;
  static const int          PRECON_MODE;


  explicit                  Solver

    ( const String&           name = "" );

  virtual void              start           ()       = 0;
  virtual void              finish          ()       = 0;

  void                      solve

    ( const Vector&           lhs,
      const Vector&           rhs );

  virtual void              improve

   ( const Vector&            lhs,
     const Vector&            rhs )                  = 0;

  virtual void              getInfo

    ( const Properties&       info )           const;

  virtual void              configure

    ( const Properties&       props );

  virtual void              getConfig

    ( const Properties&       props )          const;

  virtual void              setMode

    ( int                     mode )                 = 0;

  virtual int               getMode         () const = 0;

  virtual void              setPrecision

    ( double                  eps )                  = 0;

  virtual double            getPrecision    () const = 0;
  virtual AbstractMatrix*   getMatrix       () const = 0;
  virtual Constraints*      getConstraints  () const;
  virtual String            getContext      () const override;


 protected:

  virtual                  ~Solver ();

};


//-----------------------------------------------------------------------
//   class SolverScope
//-----------------------------------------------------------------------


class SolverScope
{
 public:

                            SolverScope  ();

  explicit                  SolverScope

    ( Solver&                 solver );

                           ~SolverScope  ();

  void                      start

    ( Solver&                 solver );

  void                      finish       ();


 private:

  Ref<Solver>               solver_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<Solver>                 newSolver

  ( const String&             name,
    const Properties&         conf,
    const Properties&         params,
    const Properties&         globdat );

Ref<Solver>                 newSolver

  ( const String&             name,
    const Properties&         conf,
    const Properties&         props,
    const Properties&         params,
    const Properties&         globdat );


JIVE_END_PACKAGE( solver )

#endif
