
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


#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/util/Event.h>
#include <jive/util/utilities.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/utilities.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/MPVectorSpace.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/LocalSolver.h>


JEM_DEFINE_CLASS( jive::solver::LocalSolver );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jive::mp::EXCHANGE;
using jive::mp::SCATTER;


//=======================================================================
//   class LocalSolver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  LocalSolver::TYPE_NAME   = "Local";

const int    LocalSolver::NEW_DOFS_   = 1 << 0;
const int    LocalSolver::NEW_CONS_   = 1 << 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LocalSolver::LocalSolver

  ( const String&         name,
    Ref<Solver>           inner,
    Ref<VectorExchanger>  vex,
    Ref<Constraints>      cons ) :

    Super      ( name  ),
    solver_    ( inner ),
    inCons_    ( cons  ),
    exchanger_ ( vex   )

{
  using jem::IllegalArgumentException;

  JEM_PRECHECK ( inner );

  AbstractMatrix*  matrix = inner->getMatrix ();

  if ( matrix->isDistributed() )
  {
    throw IllegalArgumentException (
      getContext (),
      "distributed input matrix"
    );
  }

  outCons_ = inner->getConstraints ();

  if ( inCons_ == nullptr || outCons_ == nullptr )
  {
    inCons_ = outCons_;
  }

  if ( inCons_ != outCons_ )
  {
    if ( inCons_->getDofSpace() != outCons_->getDofSpace() )
    {
      throw IllegalArgumentException (
        getContext (),
        "input and output constraints are associated with "
        "different DOF spaces"
      );
    }
  }

  if ( inCons_ && vex )
  {
    if ( inCons_->getDofSpace() != vex->getDofSpace() )
    {
      throw IllegalArgumentException (
        getContext (),
        "constraints and vector exchanger are associated with "
        "different DOF spaces"
      );
    }
  }

  symmetric_ =  false;
  xmode_     =  EXCHANGE;
  events_    = ~0x0;
  started_   =  0;

  connect_ ();
}


LocalSolver::~LocalSolver ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void LocalSolver::start ()
{
  solver_->start ();

  if ( ! started_ )
  {
    AbstractMatrix*  matrix = solver_->getMatrix ();
    Constraints*     cons   = inCons_ .get       ();

    matrix->resetEvents ();

    if ( cons )
    {
      cons->resetEvents ();
    }
  }

  if ( events_ )
  {
    update_ ();
  }

  started_++;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void LocalSolver::finish ()
{
  solver_->finish ();

  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   improve
//-----------------------------------------------------------------------


void LocalSolver::improve

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  SolverScope  scope ( *this );

  const int    mode  = solver_->getMode ();


  if ( exchanger_ )
  {
    if ( ! (mode & PRECON_MODE) )
    {
      throw jem::IllegalOperationException (
        getContext(),
        "parallel mode of operation is only possible when this "
        "solver is used as preconditioner"
      );
    }

    idx_t  bdofCount = borderDofs_.size ();
    int    xmode     = xmode_;

    if ( xmode == EXCHANGE && symmetric_ )
    {
      xmode = SCATTER;
    }

    // Zero the border DOFs to make sure that the constraints are
    // handles correctly by the solver (that is operating in
    // PRECON_MODE).

    for ( idx_t i = 0; i < bdofCount; i++ )
    {
      idx_t  idof = borderDofs_[i];

      tmpbuf_[i]  = rhs[idof];
      rhs[idof]   = 0.0;
    }

    try
    {
      exchanger_->startOne ( xmode );
      solver_   ->improve  ( lhs, rhs );
    }
    catch ( ... )
    {
      rhs[borderDofs_] = tmpbuf_;
      throw;
    }

    // Restore the right-hand side vector.

    rhs[borderDofs_] = tmpbuf_;

    if ( exchanger_->hasOverlap() )
    {
      lhs[borderDofs_] = 0.0;
    }
    else
    {
      lhs[borderDofs_] = rhs[borderDofs_];
    }

    exchanger_->endOne ( lhs );
  }
  else
  {
    solver_->improve ( lhs, rhs );
  }
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void LocalSolver::getInfo ( const Properties& info ) const
{
  solver_->getInfo ( info );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LocalSolver::configure ( const Properties& props )
{
  solver_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LocalSolver::getConfig ( const Properties& props ) const
{
  solver_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   setMode
//-----------------------------------------------------------------------


void LocalSolver::setMode ( int mode )
{
  solver_->setMode ( mode );
}


//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


int LocalSolver::getMode () const
{
  return solver_->getMode ();
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void LocalSolver::setPrecision ( double eps )
{
  solver_->setPrecision ( eps );
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double LocalSolver::getPrecision () const
{
  return solver_->getPrecision ();
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* LocalSolver::getMatrix () const
{
  return solver_->getMatrix ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* LocalSolver::getConstraints () const
{
  return inCons_.get ();
}


//-----------------------------------------------------------------------
//   setExchangeMode
//-----------------------------------------------------------------------


void LocalSolver::setExchangeMode ( int xmode )
{
  xmode_ = xmode;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> LocalSolver::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::dynamicCast;
  using jive::util::joinNames;
  using jive::algebra::MPMatrixObj;
  using jive::algebra::MPVectorSpace;

  Ref<AbstractMatrix>  matrix;

  params.find ( matrix, SolverParams::MATRIX );

  if ( ! matrix )
  {
    return nullptr;
  }

  MPMatrixObj*  mpMat = dynamicCast<MPMatrixObj*> ( matrix );

  String    innerName =

    joinNames ( name, PropNames::SOLVER );

  // Handle the trivial case:

  if ( ! matrix->isDistributed() )
  {
    return newSolver ( innerName, conf, props, params, globdat );
  }

  // If the matrix is distributed, then it must have type MPMatrixObj.

  if ( ! mpMat )
  {
    return nullptr;
  }

  Properties  innerParams ( joinNames( innerName, "params" ) );

  Ref<Constraints>   cons;
  Ref<VectorSpace>   vspace;


  params.find ( vspace, SolverParams::VECTOR_SPACE );

  if ( vspace )
  {
    Ref<MPVectorSpace>  mpVSpace =

      dynamicCast<MPVectorSpace> ( vspace );

    if ( mpVSpace )
    {
      vspace = mpVSpace->getInner ();
    }

    // If the vector space is distributed, then it must have type
    // MPVectorSpace.

    if ( vspace->isDistributed() )
    {
      return nullptr;
    }

    innerParams.set ( SolverParams::VECTOR_SPACE, vspace );
  }

  innerParams.set ( SolverParams::MATRIX, mpMat->getInner() );
  params.find     ( cons, SolverParams::CONSTRAINTS );

  if ( cons )
  {
    Ref<Constraints>  innerCons =

      newInstance<Constraints> ( cons->getDofSpace() );

    innerParams.set ( SolverParams::CONSTRAINTS, innerCons );
  }

  Ref<Object>  extra;

  if ( params.find( extra, SolverParams::EXTRA ) )
  {
    innerParams.set ( SolverParams::EXTRA, extra );
  }

  Ref<Solver>  innerSolver = newSolver

    ( innerName, conf, props, innerParams, globdat );

  if ( ! innerSolver )
  {
    return nullptr;
  }

  Ref<Self>  localSolver = newInstance<Self>

    ( name, innerSolver, mpMat->getExchanger(), cons );

  localSolver->xmode_ = mpMat->getExchangeMode ();

  return localSolver;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void LocalSolver::declare ()
{
  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void LocalSolver::connect_ ()
{
  using jem::util::connect;

  AbstractMatrix*  matrix = solver_->getMatrix ();
  Constraints*     cons   = inCons_ .get       ();

  connect ( matrix->newStructEvent, this, & Self::dofsChanged_ );

  if ( cons )
  {
    connect ( cons->newStructEvent,  this, & Self::consChanged_ );
    connect ( cons->newRvaluesEvent, this, & Self::consChanged_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void LocalSolver::update_ ()
{
  using jem::count;
  using jive::algebra::markBorderRows;

  AbstractMatrix*  matrix   = solver_->getMatrix ();
  const idx_t      dofCount = matrix ->size      (0);


  if ( events_ & NEW_DOFS_ )
  {
    if ( ! exchanger_ )
    {
      tmpbuf_    .resize ( 0 );
      borderDofs_.resize ( 0 );
    }
    else
    {
      BoolVector  mask ( dofCount );

      idx_t       i, j;

      mask = false;

      markBorderRows ( mask, *exchanger_, *matrix );

      borderDofs_.resize ( count( mask ) );

      for ( i = j = 0; i < dofCount; i++ )
      {
        if ( mask[i] )
        {
          borderDofs_[j++] = i;
        }
      }

      tmpbuf_.resize ( borderDofs_.size() );
    }

    events_ |= NEW_CONS_;
  }

  matrix->resetEvents ();

  if ( (events_ & NEW_CONS_) && inCons_ )
  {
    if ( inCons_ != outCons_ )
    {
      outCons_->copy ( *inCons_ );
    }

    for ( idx_t i = 0; i < borderDofs_.size(); i++ )
    {
      idx_t  idof = borderDofs_[i];

      if ( ! outCons_->isSlaveDof( idof ) )
      {
        outCons_->addConstraint ( idof );
      }
    }

    inCons_->resetEvents ();
  }

  symmetric_ = matrix->isSymmetric ();
  events_    = 0;
}


//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------


void LocalSolver::dofsChanged_ ()
{
  borderDofs_.ref ( IdxVector() );

  setEvents_ ( NEW_DOFS_ );
}


//-----------------------------------------------------------------------
//   consChanged_
//-----------------------------------------------------------------------


void LocalSolver::consChanged_ ()
{
  setEvents_ ( NEW_CONS_ );
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void LocalSolver::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "system of equations changed while solving it"
    );
  }

  events_ |= events;
}


JIVE_END_PACKAGE( solver )
