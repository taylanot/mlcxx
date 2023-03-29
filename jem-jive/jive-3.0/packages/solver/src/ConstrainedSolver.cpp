
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


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/solver/Constrainer.h>
#include <jive/solver/ConstrainedSolver.h>


JEM_DEFINE_CLASS( jive::solver::ConstrainedSolver );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class ConstrainedSolver
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ConstrainedSolver::ConstrainedSolver

  ( const String&       name,
    const Ref<Solver>&  solver,
    Ref<Constrainer>    conman ) :

    Super   (   name ),
    solver_ ( solver ),
    conman_ ( conman )

{
  JEM_PRECHECK ( solver && conman );

  if ( myName_.size() == 0 )
  {
    myName_ = solver_->getName ();
  }
}


ConstrainedSolver::~ConstrainedSolver()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void ConstrainedSolver::start ()
{
  conman_->update ();
  solver_->start  ();
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void ConstrainedSolver::finish ()
{
  solver_->finish ();
}


//-----------------------------------------------------------------------
//   improve
//-----------------------------------------------------------------------


void ConstrainedSolver::improve

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  typedef AbstractMatrix::Shape Shape;

  SolverScope  scope ( *this );

  const idx_t  mode  = solver_->getMode ();

  if ( mode & PRECON_MODE )
  {
    solver_->improve ( lhs, rhs );
  }
  else
  {
    const Shape  sh   = conman_->getOutputMatrix()->shape ();
    const idx_t  m    = sh[1];
    const idx_t  n    = sh[1] + sh[0];

    Vector       vbuf ( n );
    Vector       u    ( vbuf[slice(BEGIN,m)] );
    Vector       f    ( vbuf[slice(m,END)] );

    conman_->getRhs  ( f, rhs );
    solver_->improve ( u, f );
    conman_->getLhs  ( lhs, u );
  }
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void ConstrainedSolver::getInfo ( const Properties& info ) const
{
  solver_->getInfo ( info );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ConstrainedSolver::configure ( const Properties& props )
{
  solver_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ConstrainedSolver::getConfig ( const Properties& props ) const
{
  solver_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   setMode
//-----------------------------------------------------------------------


void ConstrainedSolver::setMode ( int mode )
{
  solver_->setMode ( mode );
}


//-----------------------------------------------------------------------
//   isMode
//-----------------------------------------------------------------------


int ConstrainedSolver::getMode () const
{
  return solver_->getMode ();
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void ConstrainedSolver::setPrecision ( double eps )
{
  solver_->setPrecision ( eps );
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double ConstrainedSolver::getPrecision () const
{
  return solver_->getPrecision ();
}

//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* ConstrainedSolver::getMatrix () const
{
  return conman_->getInputMatrix ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* ConstrainedSolver::getConstraints () const
{
  return conman_->getConstraints ();
}


JIVE_END_PACKAGE( solver )
