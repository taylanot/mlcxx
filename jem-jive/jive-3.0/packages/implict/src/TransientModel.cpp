
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
#include <jem/base/array/operators.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/implict/Names.h>
#include <jive/implict/TimeStepper.h>
#include <jive/implict/TransientMatrix.h>
#include <jive/implict/TransientModel.h>


JEM_DEFINE_CLASS( jive::implict::TransientModel );


JIVE_BEGIN_PACKAGE( implict )


using jem::newInstance;
using jem::System;
using jem::io::endl;
using jive::util::joinNames;
using jive::model::STATE1;
using jive::model::STATE2;
using jive::model::StateVector;


//=======================================================================
//   class TransientModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  TransientModel::TYPE_NAME   = "Transient";

const int    TransientModel::STARTED_    = 1 << 0;
const int    TransientModel::ADVANCING_  = 1 << 1;
const int    TransientModel::COMMITTING_ = 1 << 2;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TransientModel::TransientModel ()
{
  init_ ();
}


TransientModel::TransientModel

  ( const String&      name,
    const Ref<Model>&  child,
    Ref<TimeStepper>   stepper,
    Options            options ) :

    Super    ( name ),
    child_   ( child ),
    stepper_ ( stepper )

{
  JEM_PRECHECK ( child && stepper );

  init_ ();

  options_ = options;
}


TransientModel::TransientModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Super ( name )

{
  using jive::model::newModel;

  child_ = newModel (
    joinNames ( name, PropNames::MODEL ),
    conf,
    props,
    globdat
  );

  stepper_ = newTimeStepper (
    joinNames ( name, PropNames::TIME_STEPPER ),
    conf,
    props,
    globdat
  );

  init_ ();
  init  ( globdat );
}


TransientModel::~TransientModel ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void TransientModel::init ( const Properties& globdat )
{
  using jem::min;
  using jem::max;
  using jem::util::connect;
  using jive::util::Globdat;

  int  order = stepper_->getMaxOrder ();

  order = min ( 2, order );
  order = max ( 1, order );

  transmat_ = newInstance<TransientMatrix> (
    joinNames ( myName_, PropNames::MATRIX ),
    order,
    child_,
    globdat
  );

  dofs_   = stepper_->getDofSpace ();
  vspace_ = VectorSpace::get ( dofs_, globdat );
  status_ = 0;

  Globdat::initTime ( globdat );

  connect ( dofs_->newSizeEvent,  this, & Self::dofsChanged_ );
  connect ( dofs_->newOrderEvent, this, & Self::dofsChanged_ );
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void TransientModel::readFrom ( ObjectInput& in )
{
  init_  ();

  decode ( in,
           myName_,
           context_,
           child_,
           stepper_ );

  decode ( in,
           dtMin_,
           dtMax_,
           options_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void TransientModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           context_,
           child_,
           stepper_ );

  encode ( out,
           dtMin_,
           dtMax_,
           options_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* TransientModel::findModel ( const String& name ) const
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


void TransientModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::min;
  using jem::max;
  using jem::minOf;
  using jem::maxOf;

  int  events = stepper_->configure ( props );

  if ( (events & TimeStepper::NEW_COEFFS) && transmat_ )
  {
    newCoeffs_ ();
  }

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    double      dt0     = stepper_->getDeltaTime ();
    double      dt      = dt0;

    if ( myProps.find( dt, PropNames::DELTA_TIME,
                       minOf( dt ), maxOf( dt ) ) )
    {
      stepper_->setDeltaTime ( dt );

      dtMin_ = min ( dt, dtMin_ * (dt / dt0) );
      dtMax_ = max ( dt, dtMax_ * (dt / dt0) );

      if ( transmat_ )
      {
        newCoeffs_ ();
      }

      if ( dt > dt0 )
      {
        // Force a restart to avoid bad estimates of the state vector.

        status_ = 0;
      }
    }

    myProps.find ( dtMin_, PropNames::MIN_DTIME,
                   minOf( dtMin_ ), dt );

    myProps.find ( dtMax_, PropNames::MAX_DTIME,
                   dt, maxOf( dtMax_ ) );
  }

  if ( ! (options_ & MANAGED) )
  {
    child_->configure ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void TransientModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );
  double      dtime  = stepper_->getDeltaTime  ();

  myConf.set ( PropNames::DELTA_TIME, dtime  );
  myConf.set ( PropNames::MIN_DTIME,  dtMin_ );
  myConf.set ( PropNames::MAX_DTIME,  dtMax_ );

  stepper_->getConfig ( conf );

  if ( ! (options_ & MANAGED) )
  {
    child_->getConfig ( conf, globdat );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool TransientModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;

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
    restart_ ( globdat );
  }

  if ( action == Actions::UPDATE )
  {
    transmat_->invalidate  ();

    if ( status_ & ADVANCING_ )
    {
      stepper_ ->updateState ( globdat );
    }

    goto run_child;
  }

  if ( action == Actions::UPD_MATRIX0 )
  {
    Vector  fint;

    params.get ( fint, ActionParams::INT_VECTOR );

    transmat_->update ( fint, globdat );
    getIntVector_     ( fint, globdat );

    return true;
  }

  if ( action == Actions::GET_INT_VECTOR )
  {
    Vector  fint;

    child_->takeAction ( action, params, globdat );

    params.get    ( fint, ActionParams::INT_VECTOR );
    getIntVector_ ( fint, globdat );

    return true;
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
      advance_ ( globdat );
    }

    goto run_child;
  }

  if ( action == Actions::CHECK_COMMIT )
  {
    if ( status_ & ADVANCING_ )
    {
      status_ &= ~ADVANCING_;
      status_ |=  COMMITTING_;
    }

    goto run_child;
  }

  if ( action == Actions::COMMIT )
  {
    if ( status_ & (ADVANCING_ | COMMITTING_) )
    {
      commit_ ( globdat );
    }

    goto run_child;
  }

  if ( action == Actions::CANCEL )
  {
    if ( status_ & (ADVANCING_ | COMMITTING_) )
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


String TransientModel::getContext () const
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


void TransientModel::setContext ( const String& context )
{
  context_ = context;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> TransientModel::makeNew

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


void TransientModel::declare ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void TransientModel::init_ ()
{
  double  dt = 1.0;

  if ( stepper_ )
  {
    dt = stepper_->getDeltaTime ();
  }

  dtMin_   = dt;
  dtMax_   = dt;
  norm0_   = 0.0;
  status_  = 0;
  options_ = 0;
}


//-----------------------------------------------------------------------
//   restart_
//-----------------------------------------------------------------------


void TransientModel::restart_ ( const Properties& globdat )
{
  print ( System::info( myName_ ),
          endl, myName_, " : restarting\n\n" );

  stepper_ ->restart    ( globdat );
  transmat_->invalidate ();

  newCoeffs_ ();

  norm0_  = 0.0;
  status_ = STARTED_;

  dofs_->resetEvents ();
}


//-----------------------------------------------------------------------
//   advance_
//-----------------------------------------------------------------------


void TransientModel::advance_ ( const Properties& globdat )
{
  int  events = stepper_->advance ( globdat );

  if ( events & TimeStepper::NEW_COEFFS )
  {
    newCoeffs_ ();
  }

  if ( stepper_->getTimeStep() > 1 )
  {
    stepper_->predictState ( globdat );
  }

  status_ |= ADVANCING_;
}


//-----------------------------------------------------------------------
//   commit_
//-----------------------------------------------------------------------


void TransientModel::commit_ ( const Properties&  globdat )
{
  using jem::isTiny;

  const int  order = stepper_->getMaxOrder ();

  Vector     v;


  stepper_->commit ( globdat );

  if ( order < 2 )
  {
    StateVector::get ( v, STATE1, dofs_, globdat );
  }
  else
  {
    StateVector::get ( v, STATE2, dofs_, globdat );
  }

  const double  vnorm = vspace_->norm2 ( v );

  if ( stepper_->getTimeStep() < (order + 2) )
  {
    norm0_ = vnorm;
  }
  else if ( ! isTiny( vnorm ) )
  {
    // Compute the new time step size.

    const double  dt0 = stepper_->getDeltaTime ();
    double        dt  = dt0 * (norm0_ / vnorm);

    if      ( dt < 0.5 * dt0 )
    {
      dt = 0.5 * dt0;
    }
    else if ( dt > 2.0 * dt0 )
    {
      dt = 2.0 * dt0;
    }

    if      ( dt < dtMin_ )
    {
      dt = dtMin_;
    }
    else if ( dt > dtMax_ )
    {
      dt = dtMax_;
    }

    // Only update the time step size if the new one differs
    // significantly from the old one. This limits the number of
    // updates to the global matrix.

    if ( dt < (dt0 / 1.2) || dt > (dt0 * 1.2) )
    {
      stepper_->setDeltaTime ( dt );

      newCoeffs_ ();

      norm0_ = vnorm;

      print ( System::debug( myName_ ), myName_,
              " : setting time step size to ", dt, endl, endl );
    }
  }

  status_ &= ~ADVANCING_;
}


//-----------------------------------------------------------------------
//   cancel_
//-----------------------------------------------------------------------


void TransientModel::cancel_ ( const Properties& globdat )
{
  int  events = stepper_->cancel ( globdat );

  if ( events & TimeStepper::NEW_COEFFS )
  {
    newCoeffs_ ();
  }

  status_ &= ~ADVANCING_;
}


//-----------------------------------------------------------------------
//   getIntVector_
//-----------------------------------------------------------------------


void TransientModel::getIntVector_

  ( const Vector&      fint,
    const Properties&  globdat )

{
  AbstractMatrix*  m1 = transmat_->updateMatrix1 ( globdat );
  AbstractMatrix*  m2 = transmat_->updateMatrix2 ( globdat );

  if ( vbuf_.size() != fint.size() )
  {
    vbuf_.resize ( fint.size() );
  }

  if ( m1 )
  {
    Vector  v;

    StateVector::get ( v, STATE1, dofs_, globdat );

    m1->matmul ( vbuf_, v );

    fint += vbuf_;
  }

  if ( m2 )
  {
    Vector  a;

    StateVector::get ( a, STATE2, dofs_, globdat );

    m2->matmul ( vbuf_, a );

    fint += vbuf_;
  }
}


//-----------------------------------------------------------------------
//   newCoeffs_
//-----------------------------------------------------------------------


void TransientModel::newCoeffs_ ()
{
  double  a = stepper_->getTimeCoeff ( 0 );
  double  b = stepper_->getTimeCoeff ( 1 );
  double  c = stepper_->getTimeCoeff ( 2 );

  transmat_->setCoeffs ( a, b, c );
}


//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------


void TransientModel::dofsChanged_ ()
{
  if ( status_ & ADVANCING_ )
  {
    dofs_->resetEvents ();

    throw jem::IllegalOperationException (
      getContext (),
      "DOF space changed while executing a time step"
    );
  }

  status_ = 0;
}


JIVE_END_PACKAGE( implict )
