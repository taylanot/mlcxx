
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/Float.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/base/array/operators.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ImplicitMatrix.h>


JEM_DEFINE_CLASS( jive::model::ImplicitMatrix );


JIVE_BEGIN_PACKAGE( model )


using jem::Float;
using jive::algebra::MatrixTraits;


//=======================================================================
//   class ImplicitMatrix
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  ImplicitMatrix::SYMMETRIC = 1 << 0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ImplicitMatrix::ImplicitMatrix

  ( const String&         name,
    const Ref<Model>&     model,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat,
    int                   traits ) :

    Super    ( name     ),
    model_   ( model    ),
    dofs_    ( dofs     ),
    globdat_ ( globdat  ),
    params_  ( "params" )

{
  using jem::util::connect;

  JEM_PRECHECK ( model && dofs );

  traits_  =  traits;
  vspace_  =  VectorSpace::get ( dofs_, globdat );
  eps_     =  Float::EPSILON;
  scale_   = -1.0;
  updated_ =  false;

  connect ( dofs_->newSizeEvent,  this, & Self::sizeChanged_ );
  connect ( dofs_->newOrderEvent, this, & Self::invalidate_  );
}


ImplicitMatrix::~ImplicitMatrix ()
{}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


ImplicitMatrix::Shape ImplicitMatrix::shape () const
{
  const idx_t  size = dofs_->dofCount ();

  return Shape ( size, size );
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void ImplicitMatrix::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  using jive::util::sizeError;

  const idx_t  size = fint_.size ();

  Vector       u;

  double       rnorm;
  double       eps;


  if ( ! updated_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "`matmul\' method invoked on an invalid matrix"
    );
  }

  if ( lhs.size() != size )
  {
    sizeError ( getContext(), "lhs vector", lhs.size(), size );
  }
  if ( rhs.size() != size )
  {
    sizeError ( getContext(), "rhs vector", lhs.size(), size );
  }

  rnorm = std::sqrt ( vspace_->product( rhs, rhs ) );

  if ( Float::isNaN( rnorm ) )
  {
    throw jem::ArithmeticException (
      getContext (),
      "invalid norm of right-hand vector: NaN"
    );
  }

  if ( Float::isTiny( rnorm ) )
  {
    // Zero in, zero out

    lhs = 0.0;

    return;
  }

  StateVector::get ( u, dofs_, globdat_ );

  eps   = scale_ / rnorm;
  lhs   = u;
  u     = state0_ + eps * rhs;

  model_->takeAction ( Actions::UPDATE,         params_, globdat_ );

  fint_ = 0.0;

  model_->takeAction ( Actions::GET_INT_VECTOR, params_, globdat_ );

  u     = lhs;
  lhs   = (1.0 / eps) * (fint_ - fint0_);
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool ImplicitMatrix::hasTrait ( const String& trait ) const
{
  if ( trait == MatrixTraits::SYMMETRIC )
  {
    return (traits_ & SYMMETRIC);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool ImplicitMatrix::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  if ( action == Actions::UPDATE )
  {
    model_->takeAction ( action, params, globdat );

    update_ ();

    return true;
  }

  if ( action == Actions::GET_INT_VECTOR && updated_ )
  {
    Vector  fint;

    params.get ( fint, ActionParams::INT_VECTOR );

    fint = fint0_;

    return true;
  }

  if ( action == Actions::UPD_MATRIX0 )
  {
    Vector  fint;

    if ( ! updated_ )
    {
      update_ ();
    }

    params.get ( fint, ActionParams::INT_VECTOR );

    fint = fint0_;

    return true;
  }

  if ( action == Actions::NEW_MATRIX0 )
  {
    params.set ( ActionParams::MATRIX0, this );

    return true;
  }

  return model_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   setEpsilon
//-----------------------------------------------------------------------


void ImplicitMatrix::setEpsilon ( double eps )
{
  JEM_PRECHECK ( eps > 0.0 );

  eps_ = eps;

  if ( scale_ > 0.0 && scale_ < eps_ )
  {
    scale_ = eps_;
  }
}


//-----------------------------------------------------------------------
//   getEpsilon
//-----------------------------------------------------------------------


double ImplicitMatrix::getEpsilon () const
{
  return eps_;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void ImplicitMatrix::update_ ()
{
  const idx_t  dofCount = dofs_->dofCount ();

  Vector       u;

  if ( dofCount != fint_.size() )
  {
    params_.erase  ( ActionParams::INT_VECTOR );

    fint_  .resize ( dofCount );
    fint0_ .resize ( dofCount );
    state0_.resize ( dofCount );

    params_.set    ( ActionParams::INT_VECTOR, fint_ );
  }

  StateVector::get  ( u, dofs_, globdat_ );

  scale_ = std::sqrt ( Float::EPSILON * vspace_->product( u, u ) );

  if ( scale_ < eps_ )
  {
    scale_ = eps_;
  }

  fint_ = 0.0;

  model_->takeAction ( Actions::GET_INT_VECTOR, params_, globdat_ );

  fint0_   = fint_;
  state0_  = u;
  updated_ = true;

  dofs_->resetEvents  ();
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void ImplicitMatrix::invalidate_ ()
{
  updated_ = false;
}


//-----------------------------------------------------------------------
//   sizeChanged_
//-----------------------------------------------------------------------


void ImplicitMatrix::sizeChanged_ ()
{
  updated_ = false;

  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( model )
