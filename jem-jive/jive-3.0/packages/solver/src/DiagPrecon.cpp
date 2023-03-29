
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
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/util/Event.h>
#include <jem/mp/utilities.h>
#include <jive/util/error.h>
#include <jive/util/Constraints.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MPMatrixExtension.h>
#include <jive/algebra/DiagMatrixExtension.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/PreconFactory.h>
#include <jive/solver/DiagPrecon.h>


JEM_DEFINE_CLASS( jive::solver::DiagPrecon );


JIVE_BEGIN_PACKAGE( solver )


using jive::algebra::DiagMatrixExt;


//=======================================================================
//   class DiagPrecon
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DiagPrecon::TYPE_NAME   = "Diag";

const int    DiagPrecon::NEW_VALUES_ = 1 << 0;
const int    DiagPrecon::NEW_STRUCT_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DiagPrecon::DiagPrecon

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<Constraints>     cons ) :

    Super   (   name ),
    matrix_ ( matrix ),
    cons_   (   cons )

{
  JEM_PRECHECK ( matrix &&
                 matrix->hasExtension<DiagMatrixExt>() );

  events_  = ~0x0;
  started_ =  0;

  connect_ ();
}


DiagPrecon::~DiagPrecon ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void DiagPrecon::resetEvents ()
{
  matrix_->resetEvents ();

  if ( cons_ )
  {
    cons_->resetEvents ();
  }
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


algebra::AbstractMatrix::Shape DiagPrecon::shape () const
{
  return matrix_->shape ();
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void DiagPrecon::start ()
{
  if ( ! started_ )
  {
    syncEvents_ ();
  }

  if ( events_ & NEW_STRUCT_ )
  {
    update_ ();
  }

  started_++;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void DiagPrecon::finish ()
{
  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void DiagPrecon::update ()
{
  if ( ! started_ )
  {
    syncEvents_ ();
  }

  if ( events_ )
  {
    update_ ();
  }
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void DiagPrecon::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  JEM_PRECHECK ( ! (events_ & NEW_STRUCT_) );

  if ( rhs.size() != rscales_.size() )
  {
    util::sizeError ( JEM_FUNC, "rhs vector",
                      rhs.size(), rscales_.size() );
  }

  lhs = rscales_ * rhs;
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool DiagPrecon::hasTrait ( const String& trait ) const
{
  using jive::algebra::MatrixTraits;

  if      ( trait == MatrixTraits::SYMMETRIC )
  {
    return true;
  }
  else if ( trait == MatrixTraits::DISTRIBUTED )
  {
    return matrix_->hasTrait ( trait );
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* DiagPrecon::getConstraints () const
{
  return cons_.get ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Preconditioner> DiagPrecon::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<AbstractMatrix>  mat;
  Ref<Constraints>     cons;

  params.find ( mat, SolverParams::MATRIX );

  if ( mat == nullptr || ! mat->hasExtension<DiagMatrixExt>() )
  {
    return nullptr;
  }

  params.find ( cons, SolverParams::CONSTRAINTS );

  return jem::newInstance<Self> ( name, mat, cons );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DiagPrecon::declare ()
{
  PreconFactory::declare ( TYPE_NAME,  & makeNew );
  PreconFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void DiagPrecon::connect_ ()
{
  using jem::util::connect;

  connect ( matrix_->newValuesEvent, this, &Self::valuesChanged_ );
  connect ( matrix_->newStructEvent, this, &Self::structChanged_ );

  if ( cons_ )
  {
    connect ( cons_->newStructEvent, this, &Self::valuesChanged_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void DiagPrecon::update_ ()
{
  using jem::isTiny;
  using jive::util::zeroSlaveDofs;

  const Shape     sh = matrix_->shape ();
  const idx_t     n  = sh[0];

  DiagMatrixExt*  dx = matrix_->getExtension<DiagMatrixExt> ();


  if ( sh[0] != sh[1] )
  {
    util::nonSquareMatrixError ( getContext(), sh );
  }

  rscales_.resize  ( n );
  dx->getDiagonal  ( rscales_ );

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( isTiny( rscales_[i] ) )
    {
      rscales_[i] = 1.0;
    }
    else
    {
      rscales_[i] = 1.0 / rscales_[i];
    }
  }

  if ( cons_ )
  {
    zeroSlaveDofs ( rscales_, *cons_ );
  }

  events_ = 0;

  Self::resetEvents   ();

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void DiagPrecon::valuesChanged_ ()
{
  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void DiagPrecon::structChanged_ ()
{
  setEvents_          ( NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   syncEvents_
//-----------------------------------------------------------------------


void DiagPrecon::syncEvents_ ()
{
  using jive::algebra::MPMatrixExt;

  MPMatrixExt*  ext = matrix_->getExtension<MPMatrixExt> ();

  if ( ext )
  {
    MPContext*  mpx = ext->getExchanger()->getMPContext ();

    events_ = allreduce ( *mpx, events_, jem::mp::BOR );
  }

  Self::resetEvents ();
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void DiagPrecon::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "diagonal preconditioner changed while "
      "solving a linear system of equations"
    );
  }

  events_ |= events;
}


JIVE_END_PACKAGE( solver )
