
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

#ifndef JIVE_SOLVER_DIRECTSOLVER_H
#define JIVE_SOLVER_DIRECTSOLVER_H

#include <jem/util/Event.h>
#include <jive/solver/Solver.h>


JIVE_BEGIN_PACKAGE( solver )


using jem::util::Event;


//-----------------------------------------------------------------------
//   class DirectSolver
//-----------------------------------------------------------------------


class DirectSolver : public Solver
{
 public:

  JEM_DECLARE_CLASS       ( DirectSolver, Solver );

  static const double       ZERO_THRESHOLD;

  Event<idx_t, Self&>       factorEvent;
  Event<double,Self&>       zeroPivotEvent;
  Event<double,Self&>       solveEvent;


  explicit                  DirectSolver

    ( const String&           name = "" );

  virtual void              clear             ()       = 0;

  virtual void              configure

    ( const Properties&       props )                  override;

  virtual void              getConfig

    ( const Properties&       props )            const override;

  virtual void              getNullSpace

    ( Matrix&                 cmat )                   = 0;

  virtual void              setZeroThreshold

    ( double                  eps )                    = 0;

  virtual double            getZeroThreshold  () const = 0;

  virtual void              setMaxZeroPivots

    ( idx_t                   maxPivots )              = 0;

  virtual idx_t             getMaxZeroPivots  () const = 0;


 protected:

  virtual                  ~DirectSolver      ();

};


JIVE_END_PACKAGE( solver )

#endif
