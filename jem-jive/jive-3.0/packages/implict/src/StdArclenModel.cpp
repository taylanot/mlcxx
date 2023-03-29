
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
#include <jem/base/limits.h>
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/Table.h>
#include <jive/util/TableUtils.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/util/VectorManager.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/model/Actions.h>
#include <jive/model/DummyModel.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/implict/Names.h>
#include <jive/implict/ArclenActions.h>
#include <jive/implict/StdArclenModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::implict::StdArclenModel );


JIVE_BEGIN_PACKAGE( implict )


using jem::newInstance;
using jive::util::VectorManager;
using jive::model::Actions;
using jive::model::ActionParams;
using jive::model::StateVector;


//=======================================================================
//   class StdArclenModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdArclenModel::TYPE_NAME    = "StdArclen";
const char*  StdArclenModel::DELTA_STATE_ = "deltaState0";

const int    StdArclenModel::U_LOAD_      = 1 << 0;
const int    StdArclenModel::U_WEIGHTS_   = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdArclenModel::StdArclenModel ()
{
  init_ ();
}


StdArclenModel::StdArclenModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Super ( name  )

{
  using jive::util::joinNames;
  using jive::model::newModel;

  child_ = newModel ( joinNames( name, PropNames::MODEL ),
                      conf, props, globdat );

  init_ ();
  init_ ( globdat );
}


StdArclenModel::~StdArclenModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdArclenModel::readFrom ( ObjectInput& in )
{
  int  fn;

  decode ( in,
           myName_,
           child_,
           fn,
           maxIter_ );

  decode ( in,
           minIncr_,
           maxIncr_,
           loadIncr_,
           loadScale_,
           wtblName_ );

  arcFunc_ = (ArcFunc) fn;
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdArclenModel::writeTo ( ObjectOutput& out ) const
{
  int  fn = (int) arcFunc_;

  encode ( out,
           myName_,
           child_,
           fn,
           maxIter_ );

  encode ( out,
           minIncr_,
           maxIncr_,
           loadIncr_,
           loadScale_,
           wtblName_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool StdArclenModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  if ( ! dofs_ )
  {
    if ( action == Actions::INIT )
    {
      child_->takeAction ( action, params, globdat );

      init_ ( globdat );

      return true;
    }
    else
    {
      return child_->takeAction ( action, params, globdat );
    }
  }

  if ( action == ArclenActions::GET_ARC_FUNC )
  {
    evalArcFunc_ ( params, globdat );

    return true;
  }

  if ( action == ArclenActions::GET_UNIT_LOAD )
  {
    getUnitLoad_ ( params, globdat );

    return true;
  }

  if ( action == Actions::COMMIT )
  {
    child_->takeAction ( action, params, globdat );

    commit_ ( params, globdat );

    return true;
  }

  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* StdArclenModel::findModel ( const String& name ) const
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


void StdArclenModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    String      fnType;


    if ( myProps.find( fnType, PropNames::ARC_FUNC ) )
    {
      if      ( fnType == "UNP" )
      {
        arcFunc_ = UNP;
      }
      else if ( fnType == "IDC" )
      {
        arcFunc_ = IDC;
      }
      else
      {
        myProps.propertyError (
          PropNames::ARC_FUNC,
          String::format (
            "invalid arc-length function type: %s; "
            "valid types are "
            "`UNP\' (Updated Normal Plane) and "
            "`IDC\' (Indirect Displacement Control)",
            fnType
          )
        );
      }
    }

    myProps.find ( maxIter_,   PropNames::MAX_ITER,
                   1,          1000 );
    myProps.find ( minIncr_,   PropNames::MIN_INCR,
                   0.0,        maxOf( minIncr_ ) );
    myProps.find ( maxIncr_,   PropNames::MAX_INCR,
                   minIncr_,   maxOf( maxIncr_ ) );
    myProps.find ( loadIncr_,  PropNames::LOAD_INCR );
    myProps.find ( loadScale_, PropNames::LOAD_SCALE );

    if ( myProps.find( wtblName_, PropNames::WEIGHT_TABLE ) )
    {
      updated_ &= ~U_WEIGHTS_;
    }
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void StdArclenModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  String      fnType;

  if      ( arcFunc_ == UNP )
  {
    fnType = "UNP";
  }
  else if ( arcFunc_ == IDC )
  {
    fnType = "IDC";
  }
  else
  {
    fnType = "unknown";
  }

  myConf.set ( PropNames::ARC_FUNC,     fnType     );
  myConf.set ( PropNames::MAX_ITER,     maxIter_   );
  myConf.set ( PropNames::MIN_INCR,     minIncr_   );
  myConf.set ( PropNames::MAX_INCR,     maxIncr_   );
  myConf.set ( PropNames::LOAD_INCR,    loadIncr_  );
  myConf.set ( PropNames::LOAD_SCALE,   loadScale_ );
  myConf.set ( PropNames::WEIGHT_TABLE, wtblName_  );

  child_->getConfig ( conf, globdat );
}


//-----------------------------------------------------------------------
//   setArcFunc
//-----------------------------------------------------------------------


void StdArclenModel::setArcFunc ( ArcFunc func )
{
  arcFunc_ = func;
}


//-----------------------------------------------------------------------
//   setMaxIter
//-----------------------------------------------------------------------


void StdArclenModel::setMaxIter ( idx_t count )
{
  JEM_PRECHECK ( count > 0 );

  maxIter_ = count;
}


//-----------------------------------------------------------------------
//   setLoadIncr
//-----------------------------------------------------------------------


void StdArclenModel::setLoadIncr ( double incr )
{
  loadIncr_ = incr;

  if ( istep_ > 0 )
  {
    istep_ = 0;
  }
}


//-----------------------------------------------------------------------
//   setLoadScale
//-----------------------------------------------------------------------


void StdArclenModel::setLoadScale ( double scale )
{
  loadScale_ = scale;

  if ( istep_ > 0 )
  {
    istep_ = 0;
  }
}


//-----------------------------------------------------------------------
//   setIncrRange
//-----------------------------------------------------------------------


void StdArclenModel::setIncrRange

  ( double  minIncr,
    double  maxIncr )

{
  JEM_PRECHECK ( minIncr >= 0.0 && minIncr < maxIncr );

  minIncr_ = minIncr;
  maxIncr_ = maxIncr;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> StdArclenModel::makeNew

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


void StdArclenModel::declare ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void StdArclenModel::init_ ()
{
  istep_     = 0;
  updated_   = 0;
  arcFunc_   = UNP;
  maxIter_   = 4;
  minIncr_   = 1.0e-3;
  maxIncr_   = 1.0e+1;
  loadIncr_  = 1.0;
  loadScale_ = 0.0;
  arcLength_ = 0.0;
}


void StdArclenModel::init_ ( const Properties& globdat )
{
  using jem::util::connect;

  dofs_    = DofSpace   ::get ( globdat, getContext() );
  cons_    = Constraints::get ( dofs_,   globdat );
  istep_   = 0;
  updated_ = 0;

  connect ( dofs_->newSizeEvent,   this, & Self::dofsChanged_ );
  connect ( dofs_->newOrderEvent,  this, & Self::dofsChanged_ );
  connect ( cons_->newStructEvent, this, & Self::consChanged_ );

  dofs_->resetEvents ();
  cons_->resetEvents ();
}


//-----------------------------------------------------------------------
//   initLoad_
//-----------------------------------------------------------------------


void StdArclenModel::initLoad_ ( const Properties& globdat )
{
  Properties  params;

  load_.resize ( dofs_->dofCount() );

  load_ = 0.0;

  params.set ( ActionParams::EXT_VECTOR, load_ );

  child_->takeAction  ( Actions::GET_EXT_VECTOR, params, globdat );
  dofs_ ->resetEvents ();

  updated_ |= U_LOAD_;
}


//-----------------------------------------------------------------------
//   initWeights_
//-----------------------------------------------------------------------


void StdArclenModel::initWeights_ ( const Properties& globdat )
{
  using jive::util::zeroSlaveDofs;
  using jive::util::Table;
  using jive::util::TableUtils;

  weights_.resize ( dofs_->dofCount() );

  if ( wtblName_.size() == 0 )
  {
    weights_ = 1.0;
  }
  else
  {
    Ref<Table>  table  =

      Table::get ( wtblName_, dofs_->getItems(),
                   globdat,   getContext() );

    const idx_t  nnz    = table->size ();

    IdxVector    idofs  ( nnz );
    Vector       values ( nnz );

    TableUtils::toVector ( values, idofs, *table, *dofs_ );

    weights_        = 0.0;
    weights_[idofs] = values;
  }

  zeroSlaveDofs ( weights_, *cons_ );

  dofs_->resetEvents ();
  cons_->resetEvents ();

  updated_ |= U_WEIGHTS_;
}


//-----------------------------------------------------------------------
//   evalArcFunc_
//-----------------------------------------------------------------------


void StdArclenModel::evalArcFunc_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jem::max;
  using jem::isTiny;
  using jem::System;
  using jem::io::endl;

  double  fvalue;
  double  fscale;
  double  lambda;
  double  jac11;
  idx_t   iiter;


  params.get ( iiter, ArclenParams::IITER );

  if ( ! vspace_ )
  {
    vspace_ = VectorSpace::get ( dofs_, globdat );
  }

  if ( ! (updated_ & U_LOAD_) )
  {
    initLoad_    ( globdat );
  }

  if ( ! (updated_ & U_WEIGHTS_) )
  {
    initWeights_ ( globdat );
  }

  vtmp_.resize ( weights_.size() );

  if ( iiter > 0 )
  {
    Vector  jac10;
    Vector  u, u0;

    params      .get    ( jac10, ArclenParams::JACOBIAN10 );
    StateVector::get    ( u,  dofs_, globdat );
    StateVector::getOld ( u0, dofs_, globdat );

    vtmp_ = u - u0;

    if ( arcLength_ > 0.0 )
    {
      fscale = 1.0 / arcLength_;
    }
    else
    {
      fscale = 1.0;
    }

    if      ( arcFunc_ == UNP )
    {
      vtmp_ *= weights_;
      lambda = vspace_->product ( vtmp_, vtmp_ );
      jac10  = (2.0 * fscale) * vtmp_;
    }
    else if ( arcFunc_ == IDC )
    {
      lambda = vspace_->product ( weights_, vtmp_ );
      jac10  = fscale * weights_;

      if ( lambda < 0.0 )
      {
        lambda = -lambda;
        jac10 *= -1.0;
      }
    }
    else
    {
      lambda = 0.0;
      jac10  = 0.0;
    }

    fvalue = fscale * (lambda - arcLength_);
    jac11  = 0.0;
  }
  else
  {
    Vector  q;

    double  delta;
    double  tau;

    params.get ( lambda, ArclenParams::LOAD_SCALE );
    params.get ( q,      ArclenParams::LOAD_RESPONSE );

    delta = loadIncr_;

    if ( istep_ > 0 )
    {
      Vector  du;

      double  a, b;

      VectorManager::getVector ( du, DELTA_STATE_, dofs_, globdat );

      // Determine the initial load increment in such a way that the
      // amount of work done in the previous load step is
      // approximately equal to the amount of work in the current load
      // step.

      vtmp_ = load_ * weights_;
      a     = delta * vspace_->product ( du, vtmp_ );
      b     =         vspace_->product ( q,  vtmp_ );

      if ( isTiny( b ) )
      {
        print ( System::warn(), myName_,
                ": zero load increment; falling back to "
                "previous load increment\n" );

        delta = std::fabs ( delta );
      }
      else
      {
        delta = std::fabs ( a / b );
      }

      if      ( delta < minIncr_ )
      {
        delta = minIncr_;
      }
      else if ( delta > maxIncr_ )
      {
        delta = maxIncr_;
      }

      // Reverse the sign of the load increment when global softening
      // occurs.

      if ( vspace_->product( du, q ) < 0.0 )
      {
        delta = -delta;
      }
    }

    print ( System::debug( myName_ ), myName_,
            ": initial load increment = ", delta, endl );

    tau    = loadScale_ + delta;
    fscale = max ( std::fabs( tau ), minIncr_ );

    if ( fscale > 0.0 )
    {
      fscale = 1.0 / fscale;
    }
    else
    {
      fscale = 1.0;
    }

    fvalue = fscale * (lambda - tau);
    jac11  = fscale;

    if      ( arcFunc_ == UNP )
    {
      vtmp_      = q     * weights_;
      arcLength_ = delta * delta * vspace_->product ( vtmp_, vtmp_ );
    }
    else if ( arcFunc_ == IDC )
    {
      arcLength_ = delta * vspace_->product ( weights_, q );
      arcLength_ = std::fabs ( arcLength_ );
    }
    else
    {
      arcLength_ = 0.0;
    }
  }

  params.set ( ArclenParams::JACOBIAN11, jac11  );
  params.set ( ArclenParams::ARC_FUNC,   fvalue );
}


//-----------------------------------------------------------------------
//   getUnitLoad_
//-----------------------------------------------------------------------


void StdArclenModel::getUnitLoad_

  ( const Properties&  params,
    const Properties&  globdat )

{
  Vector  f;

  if ( ! (updated_ & U_LOAD_) )
  {
    initLoad_ ( globdat );
  }

  params.get ( f, ArclenParams::UNIT_LOAD );

  if ( f.size() != load_.size() )
  {
    util::sizeError ( getContext(),
                      "unit load vector", f.size(), load_.size() );
  }

  f = load_;
}


//-----------------------------------------------------------------------
//   commit_
//-----------------------------------------------------------------------


void StdArclenModel::commit_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jem::maxOf;
  using jem::numeric::axpy;

  Vector  u, u0, du;

  double  lambda;
  idx_t   iiter;


  params.get ( iiter,  ArclenParams::IITER );
  params.get ( lambda, ArclenParams::LOAD_SCALE );

  loadIncr_  = lambda - loadScale_;
  loadScale_ = lambda;

  // Adjust the load increment according to the number of iterations.

  if ( iiter > 0 )
  {
    if ( iiter < maxIter_ )
    {
      loadIncr_ *= 1.2;
    }
    else
    {
      loadIncr_ *= (double) maxIter_ / (double) iiter;
    }
  }

  StateVector  ::get       ( u,                dofs_, globdat );
  StateVector  ::getOld    ( u0,               dofs_, globdat );
  VectorManager::getVector ( du, DELTA_STATE_, dofs_, globdat );

  axpy ( du, u, -1.0, u0 );

  if ( istep_ < maxOf( istep_ ) )
  {
    istep_++;
  }
}


//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------


void StdArclenModel::dofsChanged_ ()
{
  updated_ = 0;
}


//-----------------------------------------------------------------------
//   consChanged_
//-----------------------------------------------------------------------


void StdArclenModel::consChanged_ ()
{
  updated_ &= ~U_WEIGHTS_;
}


JIVE_END_PACKAGE( implict )
