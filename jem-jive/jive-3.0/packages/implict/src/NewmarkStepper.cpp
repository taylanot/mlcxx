
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/error.h>
#include <jive/util/Globdat.h>
#include <jive/util/DofSpace.h>
#include <jive/model/StateVector.h>
#include <jive/implict/Names.h>
#include <jive/implict/TimeStepperFactory.h>
#include <jive/implict/NewmarkStepper.h>


JEM_DEFINE_CLASS( jive::implict::NewmarkStepper );


JIVE_BEGIN_PACKAGE( implict )


using jem::minOf;
using jem::maxOf;
using jem::newInstance;
using jive::util::Globdat;
using jive::model::STATE0;
using jive::model::STATE1;
using jive::model::STATE2;
using jive::model::StateVector;


//=======================================================================
//   class NewmarkStepper
//=======================================================================

/*
  The class NewmarkStepper implements the Newmark time integration
  scheme. Let 'u' denote the state vector, 'v' the first-order time
  derivative of the state vector, 'a' the second-order time derivative
  and 'g' (gamma) a constant between zero and one. Then the following
  formula's are applied:

  Step 1:
  -------

    u(i+1) = u(i) + dt * v(i+1)
    v(i+1) = v(i) + dt * a(i+1)

    So that:

    v(i+1) = (u(i+1) - u(i)) / dt

    a(i+1) = (v(i+1) - v(i)) / dt
           = (u(i+1) - u(i)) / dt^2 - v(i) / dt

  Step i > 1:
  -----------

    u(i+1) = u(i) + (1-g) * dt * v(i) + g * dt * v(i+1)
    v(i+1) = v(i) + (1-g) * dt * a(i) + g * dt * a(i+1)

    So that:

    v(i+1) = (u(i+1) - u(i)) / (g * dt) + v(i) * (g-1) / g

    a(i+1) = (v(i+1) - v(i)) / (g * dt) + a(i) * (g-1) / g

           = v(i+1) / (g * dt) - v(i) / (g * dt) +
             a(i) * (g-1) / g

           = (u(i+1) - u(i)) / (g * dt)^2 +
             v(i) * (g-1) / (g^2 * dt) - v(i) / (g * dt) +
             a(i) * (g-1) / g

           = (u(i+1) - u(i)) / (g * dt)^2 -
             v(i) / (g^2 * dt) + a(i) * (g-1) / g
 */

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  NewmarkStepper::TYPE_NAME = "Newmark";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


NewmarkStepper::NewmarkStepper ()
{
  init_ ();
}


NewmarkStepper::NewmarkStepper

  ( const String&         name,
    int                   order,
    const Ref<DofSpace>&  dofs ) :

    Super ( name ),
    dofs_ ( dofs )

{
  JEM_PRECHECK ( order >= 1 && order <= 2 && dofs );

  init_ ();

  order_ = order;
}


NewmarkStepper::NewmarkStepper

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Super ( name )

{
  Properties  myProps = props.findProps ( name );
  Properties  myConf  = conf .makeProps ( name );

  init_ ();

  myProps.get ( order_, PropNames::ORDER, 1, 2 );
  myConf .set ( PropNames::ORDER, order_ );

  dofs_ = DofSpace::get ( globdat, getContext() );
}


NewmarkStepper::~NewmarkStepper ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void NewmarkStepper::readFrom ( ObjectInput& in )
{
  init_  ();

  decode ( in,
           myName_,
           dofs_,
           order_ );

  decode ( in,
           gamma_,
           dtime_,
           context_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void NewmarkStepper::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           dofs_,
           order_ );

  encode ( out,
           gamma_,
           dtime_,
           context_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


int NewmarkStepper::configure ( const Properties& props )
{
  int  events = 0;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    if ( myProps.find( gamma_, PropNames::GAMMA, 0.1, 1.0 ) )
    {
      events |= NEW_COEFFS;
    }
  }

  return events;
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void NewmarkStepper::getConfig ( const Properties& conf ) const
{
  Properties  myConf = conf.makeProps ( myName_ );

  Super::getConfig ( conf );

  myConf.set ( PropNames::GAMMA, gamma_ );
}


//-----------------------------------------------------------------------
//   restart
//-----------------------------------------------------------------------


int NewmarkStepper::restart ( const Properties& globdat )
{
  istep_ = istep0_ = 0;

  return NEW_COEFFS;
}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


int NewmarkStepper::advance ( const Properties& globdat )
{
  using jive::model::StateVector;

  int  events = 0;

  istep_ = istep0_;

  if ( istep_ < maxOf( istep_ ) )
  {
    istep_++;
  }

  if ( istep_ <= 2 )
  {
    events |= NEW_COEFFS;
  }

  Globdat    ::advanceTime ( dtime_, globdat );
  StateVector::updateOld   ( dofs_,  globdat );

  return events;
}


//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------


void NewmarkStepper::commit ( const Properties& globdat )
{
  Globdat::commitTime ( globdat );

  istep0_ = istep_;
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


int NewmarkStepper::cancel ( const Properties& globdat )
{
  int  events = 0;

  Globdat::restoreTime ( globdat );

  istep_ = istep0_;

  if ( istep_ < 2 )
  {
    events |= NEW_COEFFS;
  }

  return events;
}


//-----------------------------------------------------------------------
//   updateState
//-----------------------------------------------------------------------


void NewmarkStepper::updateState ( const Properties& globdat )
{
  Vector  u, u0, v;

  StateVector::get    ( u,  STATE0, dofs_, globdat );
  StateVector::get    ( v,  STATE1, dofs_, globdat );
  StateVector::getOld ( u0, STATE0, dofs_, globdat );

  if ( istep_ <= 1 )
  {
    const double  h = 1.0 / dtime_;

    v = (u - u0) * h;

    if ( order_ > 1 )
    {
      Vector  a, v0;

      StateVector::get    ( a,  STATE2, dofs_, globdat );
      StateVector::getOld ( v0, STATE1, dofs_, globdat );

      a = (v - v0) * h;
    }
  }
  else
  {
    const double  g = (gamma_ - 1.0) / gamma_;
    const double  h = 1.0 / (gamma_ * dtime_);

    Vector  v0;

    StateVector::getOld ( v0, STATE1, dofs_, globdat );

    v = (u - u0) * h + v0 * g;

    if ( order_ > 1 )
    {
      Vector  a, a0;

      StateVector::get    ( a,  STATE2, dofs_, globdat );
      StateVector::getOld ( a0, STATE2, dofs_, globdat );

      a = (v - v0) * h + a0 * g;
    }
  }
}


//-----------------------------------------------------------------------
//   predictState
//-----------------------------------------------------------------------


void NewmarkStepper::predictState ( const Properties& globdat )
{
  Vector  u, u0;
  Vector  v, v0;

  StateVector::get    ( u,  STATE0, dofs_, globdat );
  StateVector::get    ( v,  STATE1, dofs_, globdat );
  StateVector::getOld ( u0, STATE0, dofs_, globdat );
  StateVector::getOld ( v0, STATE1, dofs_, globdat );

  if ( order_ > 1 )
  {
    Vector  a, a0;

    StateVector::get    ( a,  STATE2, dofs_, globdat );
    StateVector::getOld ( a0, STATE2, dofs_, globdat );

    a = a0;
    v = v0 + dtime_ * a;
  }
  else
  {
    v = v0;
  }

  u = u0 + (0.5 * dtime_) * (v0 + v);
}


//-----------------------------------------------------------------------
//   getTimeCoeff
//-----------------------------------------------------------------------


double NewmarkStepper::getTimeCoeff ( int order ) const
{
  double  c;

  switch ( order )
  {
  case 0:

    c = 1.0;

    break;

  case 1:
  case 2:

    if ( istep_ <= 1 )
    {
      c = 1.0 / dtime_;
    }
    else
    {
      c = 1.0 / (dtime_ * gamma_);
    }

    if ( order == 2 )
    {
      c = c * c;
    }

    break;

  default:

    c = 0.0;
  }

  return c;
}


//-----------------------------------------------------------------------
//   getMaxOrder
//-----------------------------------------------------------------------


int NewmarkStepper::getMaxOrder () const
{
  return order_;
}


//-----------------------------------------------------------------------
//   getDeltaTime
//-----------------------------------------------------------------------


double NewmarkStepper::getDeltaTime () const
{
  return dtime_;
}


//-----------------------------------------------------------------------
//   setDeltaTime
//-----------------------------------------------------------------------


void NewmarkStepper::setDeltaTime ( double dtime )
{
  JEM_PRECHECK ( dtime > 0.0 );

  dtime_ = dtime;
}


//-----------------------------------------------------------------------
//   getTimeStep
//-----------------------------------------------------------------------


idx_t NewmarkStepper::getTimeStep () const
{
  return istep_;
}


//-----------------------------------------------------------------------
//   getDofSpace
//-----------------------------------------------------------------------


DofSpace* NewmarkStepper::getDofSpace () const
{
  return dofs_.get ();
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String NewmarkStepper::getContext () const
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


void NewmarkStepper::setContext ( const String& context )
{
  context_ = context;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<TimeStepper> NewmarkStepper::makeNew

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


void NewmarkStepper::declare ()
{
  TSFactory::declare ( TYPE_NAME,  Self::makeNew );
  TSFactory::declare ( CLASS_NAME, Self::makeNew );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void NewmarkStepper::init_ ()
{
  order_  = 0;
  istep_  = 0;
  istep0_ = 0;
  gamma_  = 0.5;
  dtime_  = 1.0;
}


JIVE_END_PACKAGE( implict )
