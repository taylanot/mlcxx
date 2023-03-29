
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


#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/base/array/select.h>
#include <jem/base/array/operators.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/implict/Names.h>
#include <jive/implict/TransientMatrix.h>
#include <jive/implict/Park3Model.h>


JEM_DEFINE_CLASS( jive::implict::Park3Model );


JIVE_BEGIN_PACKAGE( implict )


using jem::minOf;
using jem::maxOf;
using jem::newInstance;
using jem::System;
using jem::IllegalOperationException;
using jem::io::endl;
using jem::numeric::axpy;
using jive::util::joinNames;
using jive::util::Globdat;
using jive::model::Actions;
using jive::model::ActionParams;


//=======================================================================
//   class Park3Model
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Park3Model::TYPE_NAME  = "Park3";

const int    Park3Model::STARTED_   = 1 << 0;
const int    Park3Model::ADVANCING_ = 1 << 1;

const int    Park3Model::U_EXT_VEC_ = 1 << 0;
const int    Park3Model::U_INT_VEC_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Park3Model::Park3Model ()
{
  init_ ();
}


Park3Model::Park3Model

  ( const String&      name,
    const Ref<Model>&  child,
    Options            options ) :

    Super  ( name ),
    child_ ( child )

{
  JEM_PRECHECK ( child );

  init_ ();

  options_ = options;
}


Park3Model::Park3Model

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Super ( name )

{
  using jive::model::newModel;

  init_ ();

  child_ = newModel ( joinNames( name, PropNames::MODEL ),
                      conf, props, globdat );

  init  ( globdat );
}


Park3Model::~Park3Model ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void Park3Model::init ( const Properties& globdat )
{
  using jem::util::connect;
  using jive::util::Globdat;

  dofs_ = DofSpace   ::get ( globdat, getContext() );
  cons_ = Constraints::get ( dofs_,   globdat );

  transmat_ = newInstance<TransientMatrix> (
    joinNames ( myName_, PropNames::MATRIX ),
    2,
    child_,
    globdat
  );

  istep_   = 0;
  istep0_  = 0;
  status_  = 0;
  updated_ = 0;

  Globdat::initTime ( globdat );

  AbstractMatrix*  m2 = transmat_->getMatrix ( 2 );

  connect ( dofs_->newSizeEvent,   this, & Self::dofsChanged_ );
  connect ( dofs_->newOrderEvent,  this, & Self::dofsChanged_ );
  connect ( cons_->newStructEvent, this, & Self::consChanged_ );
  connect ( m2   ->newValuesEvent, this, & Self::mat2Changed_ );
  connect ( m2   ->newStructEvent, this, & Self::mat2Changed_ );
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void Park3Model::readFrom ( ObjectInput& in )
{
  init_  ();

  decode ( in,
           myName_,
           context_,
           child_ );

  decode ( in,
           dtime_,
           alpha_,
           beta_,
           options_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void Park3Model::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           context_,
           child_ );

  encode ( out,
           dtime_,
           alpha_,
           beta_,
           options_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* Park3Model::findModel ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else
  {
    return child_->findModel ( name );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Park3Model::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    if ( myProps.find( dtime_,
                       PropNames::DELTA_TIME,
                       minOf( dtime_ ), maxOf( dtime_ ) ) )
    {
      status_ = 0;
    }

    if ( myProps.find( alpha_, PropNames::RAYLEIGH1 ) )
    {
      status_ = 0;
    }

    if ( myProps.find( beta_,  PropNames::RAYLEIGH2 ) )
    {
      status_ = 0;
    }
  }

  if ( ! (options_ & MANAGED) )
  {
    child_->configure ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Park3Model::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::DELTA_TIME, dtime_ );
  myConf.set ( PropNames::RAYLEIGH1,  alpha_ );
  myConf.set ( PropNames::RAYLEIGH2,  beta_  );

  if ( ! (options_ & MANAGED) )
  {
    child_->getConfig ( conf, globdat );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool Park3Model::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  if ( ! dofs_ )
  {
    if ( action == Actions::INIT )
    {
      child_->takeAction ( action, params, globdat );

      init ( globdat );

      return true;
    }
    else
    {
      goto run_child;
    }
  }

  if ( ! (status_ & STARTED_) )
  {
    restart_ ();
  }

  if ( action == Actions::UPDATE )
  {
    updated_ &= ~U_INT_VEC_;

    transmat_->invalidate ();

    if ( status_ & ADVANCING_ )
    {
      // Update the velocity vector.

      globvecs_.udot = (1.0 / coeffs_.hb) * (globvecs_.u - uhist_);
    }

    goto run_child;
  }

  if ( action == Actions::UPD_MATRIX0 )
  {
    Vector  fint;

    params.get ( fint, ActionParams::INT_VECTOR );

    transmat_->update ( fint, globdat );

    fint_     = fint;
    updated_ |= U_INT_VEC_;

    getIdxVector_ ( fint, globdat );

    return true;
  }

  if ( action == Actions::GET_INT_VECTOR )
  {
    Vector  fint;

    params.get ( fint, ActionParams::INT_VECTOR );

    if ( ! (updated_ & U_INT_VEC_) )
    {
      child_->takeAction ( action, params, globdat );

      fint_ = fint;
    }

    updated_ |= U_INT_VEC_;

    getIdxVector_ ( fint, globdat );

    return true;
  }

  if ( action == Actions::GET_EXT_VECTOR )
  {
    Vector  fext;

    params.get ( fext, ActionParams::EXT_VECTOR );

    if ( ! (updated_ & U_EXT_VEC_) )
    {
      child_->takeAction ( action, params, globdat );

      fext_ = fext;
    }

    updated_ |= U_EXT_VEC_;

    getExtVector_ ( fext, globdat );

    return true;
  }

  if ( action == Actions::GET_CONSTRAINTS )
  {
    if ( status_ & ADVANCING_ )
    {
      Ref<Constraints>  cons;

      params.get ( cons, ActionParams::CONSTRAINTS );

      if ( cons == cons_ )
      {
        // The constraints have already been updated when the
        // ADVANCE actions was requested.

        return true;
      }
    }

    goto run_child;
  }

  if ( action == Actions::NEW_MATRIX0 )
  {
    params.set ( ActionParams::MATRIX0, transmat_->getMatrix() );

    return true;
  }

  if ( action == Actions::ADVANCE )
  {
    if ( ! (status_ & ADVANCING_) )
    {
      Globdat::advanceTime ( dtime_, globdat );
      child_ ->takeAction  ( action, params, globdat );
      advance_             ( globdat );

      return true;
    }

    return false;
  }

  if ( action == Actions::COMMIT )
  {
    if ( status_ & ADVANCING_ )
    {
      commit_ ( globdat );
    }

    goto run_child;
  }

  if ( action == Actions::CANCEL )
  {
    if ( status_ & ADVANCING_ )
    {
      cancel_ ( globdat );
    }

    goto run_child;
  }

 run_child:

  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Park3Model::getContext () const
{
  if ( context_.size() )
  {
    return context_;
  }
  else
  {
    return Super::getContext ();
  }
}


//-----------------------------------------------------------------------
//   setContext
//-----------------------------------------------------------------------


void Park3Model::setContext ( const String& context )
{
  context_ = context;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> Park3Model::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Park3Model::declare ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void Park3Model::init_ ()
{
  dtime_   = 1.0;
  alpha_   = 0.0;
  beta_    = 0.0;
  istep_   = 0;
  istep0_  = 0;
  status_  = 0;
  updated_ = 0;
  options_ = 0;
}


//-----------------------------------------------------------------------
//   restart_
//-----------------------------------------------------------------------


void Park3Model::restart_ ()
{
  print ( System::info( myName_ ),
          endl, myName_, " : restarting\n\n" );

  transmat_->invalidate ();

  // Resize all vectors.

  const idx_t  dofCount = dofs_->dofCount ();

  if ( mbuf_.size(0) != dofCount )
  {
    mbuf_.resize ( dofCount, 6 );

    idx_t  j = 0;

    vbuf_ .ref ( mbuf_[j++] );
    fext_ .ref ( mbuf_[j++] );
    fint_ .ref ( mbuf_[j++] );
    vdot_ .ref ( mbuf_[j++] );
    uhist_.ref ( mbuf_[j++] );
    vhist_.ref ( mbuf_[j++] );
  }

  mbuf_ = 0.0;

  for ( idx_t j = 0; j < 3; j++ )
  {
    uold_[j].resize ( dofCount );
    vold_[j].resize ( dofCount );

    uold_[j] = 0.0;
    vold_[j] = 0.0;
  }

  istep_   = 0;
  istep0_  = 0;
  status_  = STARTED_;
  updated_ = 0;

  slaveDofs_.resize   ( cons_->slaveDofCount() );
  cons_->getSlaveDofs ( slaveDofs_ );

  updateCoeffs_ ();

  AbstractMatrix*  m2 = transmat_->getMatrix ( 2 );

  dofs_->resetEvents ();
  cons_->resetEvents ();
  m2   ->resetEvents ();
}


//-----------------------------------------------------------------------
//   advance_
//-----------------------------------------------------------------------


void Park3Model::advance_ ( const Properties& globdat )
{
  using jive::model::STATE0;
  using jive::model::STATE1;
  using jive::model::StateVector;

  Vector&     v0   = vold_[0];
  Vector&     u    = globvecs_.u;
  Vector&     udot = globvecs_.udot;

  Properties  params;


  // Get the constraints now, as the structure of the constraints
  // may not change during the execution of the time step. Note
  // that this may trigger a restart.

  params .set        ( ActionParams::CONSTRAINTS, cons_ );
  child_->takeAction ( Actions::GET_CONSTRAINTS,  params, globdat );
  cons_ ->compress   ();
  params .clear      ();

  // Update the mass matrix as this can trigger a restart.

  transmat_->updateMatrix2 ( globdat );

  if ( ! (status_ & STARTED_) )
  {
    restart_ ();
  }

  StateVector::get ( u,    STATE0, dofs_, globdat );
  StateVector::get ( udot, STATE1, dofs_, globdat );

  v0.resize ( u.size() );

  status_ |= ADVANCING_;
  updated_ = 0;
  istep_   = istep0_;

  if ( istep_ < maxOf( istep_ ) )
  {
    istep_++;
  }

  if ( istep_ <= 1 )
  {
    fint_ = 0.0;
    fext_ = 0.0;

    transmat_->updateMatrix0 ( fint_, globdat );

    params.set ( ActionParams::EXT_VECTOR, fext_ );

    child_->takeAction ( Actions::GET_EXT_VECTOR,
                         params, globdat );

    updated_ = (U_EXT_VEC_ | U_INT_VEC_);

    axpy   ( vdot_, fext_, -1.0, fint_ );

    vdot_[slaveDofs_] = 0.0;
  }

  updateCoeffs_ ();

  if ( istep_ <= 1 )
  {
    AbstractMatrix*  m1 = transmat_->updateMatrix1 ( globdat );
    AbstractMatrix*  m2 = transmat_->updateMatrix2 ( globdat );

    axpy ( vbuf_, udot, alpha_, u );

    m2->matmul ( v0, vbuf_ );
    m1->matmul ( vbuf_, u );

    axpy ( v0, 1.0,   vbuf_ );
    axpy ( v0, beta_, fint_ );
  }
  else
  {
    axpy ( v0, vhist_, coeffs_.hb, vdot_ );
  }

  uold_[0] = u;

  getHistVector_ ( uhist_, udot,  uold_ );
  getHistVector_ ( vhist_, vdot_, vold_ );

  // Update the velocity vector.

  globvecs_.udot = (1.0 / coeffs_.hb) * (globvecs_.u - uhist_);
}


//-----------------------------------------------------------------------
//   commit_
//-----------------------------------------------------------------------


void Park3Model::commit_ ( const Properties&  globdat )
{
  if ( ! (updated_ & U_INT_VEC_) )
  {
    Properties  params;

    fint_ = 0.0;

    params.set ( ActionParams::INT_VECTOR, fint_ );

    child_->takeAction ( Actions::GET_INT_VECTOR, params, globdat );

    updated_ |= U_INT_VEC_;
  }

  if ( ! (updated_ & U_EXT_VEC_) )
  {
    Properties  params;

    fext_ = 0.0;

    params.set ( ActionParams::EXT_VECTOR, fext_ );

    child_->takeAction ( Actions::GET_EXT_VECTOR, params, globdat );

    updated_ |= U_EXT_VEC_;
  }

  vdot_ = fext_ - fint_;

  uold_[2].swap ( uold_[1] );
  uold_[1].swap ( uold_[0] );

  vold_[2].swap ( vold_[1] );
  vold_[1].swap ( vold_[0] );

  Globdat::commitTime ( globdat );

  istep0_  = istep_;
  status_ &= ~ADVANCING_;
}


//-----------------------------------------------------------------------
//   cancel_
//-----------------------------------------------------------------------


void Park3Model::cancel_ ( const Properties&  globdat )
{
  Globdat::restoreTime ( globdat );

  istep_   = istep0_;
  status_ &= ~ADVANCING_;

  updateCoeffs_ ();
}


//-----------------------------------------------------------------------
//   getIdxVector_
//-----------------------------------------------------------------------


void Park3Model::getIdxVector_

  ( const Vector&      fint,
    const Properties&  globdat )

{
  AbstractMatrix*  m1 = transmat_->updateMatrix1 ( globdat );
  AbstractMatrix*  m2 = transmat_->updateMatrix2 ( globdat );

  const Coeffs_&   c  = coeffs_;
  const Vector&    u  = globvecs_.u;

  vbuf_ = (1.0 + alpha_ * c.hb) * u;

  m2->matmul ( fint, vbuf_ );
  m1->matmul ( vbuf_, u );

  axpy ( fint, c.hb, vbuf_ );
  axpy ( fint, c.hb2 + beta_ * c.hb, fint_ );
}


//-----------------------------------------------------------------------
//   getExtVector_
//-----------------------------------------------------------------------


void Park3Model::getExtVector_

  ( const Vector&      fext,
    const Properties&  globdat )

{
  AbstractMatrix*  m2 = transmat_->updateMatrix2 ( globdat );

  const Coeffs_&   c  = coeffs_;

  m2->matmul ( fext, uhist_ );

  axpy ( fext, c.hb,  vhist_ );
  axpy ( fext, c.hb2, fext_  );
}


//-----------------------------------------------------------------------
//   updateCoeffs_
//-----------------------------------------------------------------------


void Park3Model::updateCoeffs_ ()
{
  if ( istep_ >= 1 && istep_ <= 3 )
  {
    Coeffs_&  c = coeffs_;

    if      ( istep_ == 1_idx )
    {
      c.a[0] = -1.0;
      c.a[1] =  0.0;
      c.a[2] =  0.0;

      c.b[0] =  0.6;
      c.b[1] =  0.4;
      c.b[2] =  0.0;
    }
    else if ( istep_ == 2_idx )
    {
      c.a[0] = -1.2;
      c.a[1] =  0.2;
      c.a[2] =  0.0;

      c.b[0] =  0.6;
      c.b[1] =  0.2;
      c.b[2] =  0.0;
    }
    else if ( istep_ == 3_idx )
    {
      c.a[0] = -1.5;
      c.a[1] =  0.6;
      c.a[2] = -0.1;

      c.b[0] =  0.6;
      c.b[1] =  0.0;
      c.b[2] =  0.0;
    }

    c.hb  = dtime_ * c.b[0];
    c.hb2 = c.hb   * c.hb;

    transmat_->setCoeffs ( c.hb2 + beta_ * c.hb,
                           c.hb,
                           1.0 + alpha_ * c.hb );
  }
}


//-----------------------------------------------------------------------
//   getHistVector_
//-----------------------------------------------------------------------


void Park3Model::getHistVector_

  ( const Vector&  h,
    const Vector&  adot,
    const Vector   aold[3] ) const

{
  const Coeffs_&  c = coeffs_;

  h = (dtime_ * c.b[1]) * adot;

  axpy ( h, -c.a[0], aold[0] );
  axpy ( h, -c.a[1], aold[1] );
  axpy ( h, -c.a[2], aold[2] );
}


//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------


void Park3Model::dofsChanged_ ()
{
  if ( status_ & ADVANCING_ )
  {
    dofs_->resetEvents ();

    throw IllegalOperationException (
      getContext (),
      "DOF space changed while executing a time step"
    );
  }

  Vector  nilVec;

  globvecs_.u   .ref ( nilVec );
  globvecs_.udot.ref ( nilVec );

  status_ = 0;
}


//-----------------------------------------------------------------------
//   consChanged_
//-----------------------------------------------------------------------


void Park3Model::consChanged_ ()
{
  if ( status_ & ADVANCING_ )
  {
    cons_->resetEvents ();

    throw IllegalOperationException (
      getContext (),
      "constraints changed while executing a time step"
    );
  }

  status_ = 0;
}


//-----------------------------------------------------------------------
//   mat2Changed_
//-----------------------------------------------------------------------


void Park3Model::mat2Changed_ ()
{
  if ( status_ & ADVANCING_ )
  {
    AbstractMatrix*  m2 = transmat_->getMatrix ( 2 );

    throw IllegalOperationException (
      getContext (),
      m2->getContext() + " changed while executing a time step"
    );
  }

  status_ = 0;
}


JIVE_END_PACKAGE( implict )
