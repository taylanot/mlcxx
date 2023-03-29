
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
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/io/Writer.h>
#include <jem/util/Event.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/FuncUtils.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/Names.h>
#include <jive/implict/ConHandler.h>
#include <jive/implict/SolverInfo.h>
#include <jive/implict/SolverBounds.h>
#include <jive/implict/NonlinRunData.h>
#include <jive/implict/NonlinModule.h>


JEM_DEFINE_CLASS( jive::implict::NonlinModule );


JIVE_BEGIN_PACKAGE( implict )


using jem::min;
using jem::max;
using jem::newInstance;
using jem::Float;
using jem::System;
using jem::Exception;
using jem::io::endl;
using jem::numeric::axpy;
using jive::util::FuncUtils;
using jive::model::Actions;
using jive::model::StateVector;

/*
 * The NonlinModule can solve generic non-linear systems of equations
 * and non-linear complementary problems with box constraints. The
 * latter type of problems arise in minimisation problems in which
 * the unknowns (DOFs) must remain within specific bounds.
 *
 * See also:
 *
 *  - "Flexible complementary solvers for large-scale applications",
 *    by Steven J Benson and Todd S Munson, in Optimization Methods
 *    and Software, Vol 21, No 1, February 2006, pp 155-168.
 *
 *  - "An active-set Newton algorithm for large-scale non-linear
 *    programs with box constraints", by Francisco Facchinei,
 *    Joaquim Judice, and Joao Soares, in SIAM J Optim., Vol 8,
 *    No 1, February 1998, pp 158-186.
 */

//=======================================================================
//   class NonlinModule::RunData_
//=======================================================================


class NonlinModule::RunData_ : public NonlinRunData
{
 public:

  typedef NonlinRunData   Super;
  typedef RunData_        Self;


  explicit inline         RunData_

    ( const String&         context );


 public:

  Matrix                  vbuf;
  IdxMatrix               ibuf;

  bool                    validMatrix;


 protected:

  virtual void            dofsChanged_  () override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NonlinModule::RunData_::RunData_ ( const String& ctx ) :

  Super ( ctx )

{
  validMatrix = false;
}


//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------


void NonlinModule::RunData_::dofsChanged_ ()
{
  if ( frozen )
  {
    Super::dofsChanged_ ();
  }
  else
  {
    validMatrix = false;
  }
}


//=======================================================================
//   class NonlinModule::Work_
//=======================================================================


class NonlinModule::Work_ : public ConHandler
{
 public:

                          Work_

    ( NonlinModule&         module,
      const Properties&     globdat );

  inline                 ~Work_             ();

  void                    updateBounds

    ( const Properties&     globdat );

  void                    applyBounds       ();

  inline bool             hasBounds         () const noexcept;

  void                    updateRscale

    ( const Properties&     globdat );

  void                    updateResidual    ();

  void                    updateMatrix

    ( const Properties&     globdat,
      bool                  getFint = true );

  void                    calcIncrement

    ( double                maxIncr );

  void                    doTotalUpdate

    ( const Properties&     globdat );

  bool                    checkConverged

    ( double                tol,
      const Properties&     globdat );

  inline void             reportProgress    ();


 public:

  RunData_&               rundat;

  Vector                  u;
  Vector                  u0;
  Vector                  du;
  Vector                  fext;
  Vector                  fint;
  Vector                  r;
  Vector                  r2;

  IdxVector               slaveDofs;

  idx_t                   iiter;
  double                  rscale;
  double                  dnorm;
  double                  rnorm;
  double                  rnorm0;
  double                  rnorm1;


 private:

  String                  myName_;
  Function*               updateCond_;
  Vector                  lowerBound_;
  Vector                  upperBound_;
  IdxVector               fixedDofs_;
  bool                    bounded_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NonlinModule::Work_::Work_

  ( NonlinModule&      mod,
    const Properties&  globdat ) :

    rundat  ( *mod.rundat_ ),
    myName_ (  mod.getName() )

{
  idx_t  dofCount;
  idx_t  j;


  iiter  = 0;
  rscale = 0.0;
  dnorm  = 0.0;
  rnorm  = 1.0;
  rnorm0 = 1.0;
  rnorm1 = 1.0;

  updateCond_ = mod.updateCond_.get    ();
  bounded_    = mod.bounds_->hasBounds ();

  rundat.updateConstraints ( globdat );
  rundat.dofs->resetEvents ();

  rundat.frozen = true;
  dofCount      = rundat.dofs->dofCount ();

  StateVector::get ( u, rundat.dofs, globdat );

  j = 0;

  rundat.vbuf.resize ( dofCount, 8 );
  rundat.ibuf.resize ( dofCount, 1 );

  u0         .ref ( rundat.vbuf[j++] );
  du         .ref ( rundat.vbuf[j++] );
  fext       .ref ( rundat.vbuf[j++] );
  fint       .ref ( rundat.vbuf[j++] );
  r          .ref ( rundat.vbuf[j++] );
  r2         .ref ( rundat.vbuf[j++] );
  lowerBound_.ref ( rundat.vbuf[j++] );
  upperBound_.ref ( rundat.vbuf[j++] );

  j = 0;

  fixedDofs_ .ref ( rundat.ibuf[j++] );

  mod.bounds_->getBounds ( lowerBound_,
                           upperBound_,
                           *rundat.dofs );

  saveConstraints ( *rundat.cons );

  if ( ! (mod.options_ & DELTA_CONS) )
  {
    adjustConstraints ( *rundat.cons, u );
  }
}


inline NonlinModule::Work_::~Work_ ()
{
  rundat.frozen = false;
}


//-----------------------------------------------------------------------
//   updateBounds
//-----------------------------------------------------------------------


void NonlinModule::Work_::updateBounds

  ( const Properties&  globdat )

{
  bool  result = rundat.getBounds ( lowerBound_,
                                    upperBound_,
                                    globdat );

  bounded_ = bounded_ || result;
}


//-----------------------------------------------------------------------
//   applyBounds
//-----------------------------------------------------------------------


void NonlinModule::Work_::applyBounds ()
{
  if ( bounded_ )
  {
    const idx_t  dofCount = u.size ();

    if ( u          .isContiguous() &&
         lowerBound_.isContiguous() &&
         upperBound_.isContiguous() )
    {
      const double* JEM_RESTRICT  lb = lowerBound_.addr ();
      const double* JEM_RESTRICT  ub = upperBound_.addr ();
      double*       JEM_RESTRICT  x  = u          .addr ();

      for ( idx_t i = 0; i < dofCount; i++ )
      {
        if      ( x[i] < lb[i] )
        {
          x[i] = lb[i];
        }
        else if ( x[i] > ub[i] )
        {
          x[i] = ub[i];
        }
      }
    }
    else
    {
      for ( idx_t i = 0; i < dofCount; i++ )
      {
        if      ( u[i] < lowerBound_[i] )
        {
          u[i] = lowerBound_[i];
        }
        else if ( u[i] > upperBound_[i] )
        {
          u[i] = upperBound_[i];
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   hasBounds()
//-----------------------------------------------------------------------


inline bool NonlinModule::Work_::hasBounds () const noexcept
{
  return bounded_;
}


//-----------------------------------------------------------------------
//   updateRscale
//-----------------------------------------------------------------------


void NonlinModule::Work_::updateRscale

  ( const Properties&  globdat )

{
  using jive::model::ActionParams;

  Properties  params;
  double      rtmp;

  axpy ( r, fext, -1.0, fint );

  rtmp   = std::sqrt( Float::EPSILON ) *

    max ( rundat.vspace->norm2( fext ),
          rundat.vspace->norm2( fint ) );

  rtmp   = max ( rtmp, rundat.vspace->norm2( r ) );
  rscale = max ( rtmp, rscale );

  params.set ( ActionParams::RESIDUAL,   r );
  params.set ( ActionParams::RES_SCALE,  rscale );
  params.set ( ActionParams::INT_VECTOR, fint );
  params.set ( ActionParams::EXT_VECTOR, fext );

  rundat.model->takeAction ( Actions::GET_RES_SCALE,
                             params, globdat );

  params.get ( rscale, ActionParams::RES_SCALE );
}


//-----------------------------------------------------------------------
//   updateResidual
//-----------------------------------------------------------------------


void NonlinModule::Work_::updateResidual ()
{
  using std::fabs;

  axpy           ( r, fext, -1.0, fint );
  evalMasterDofs ( r, *rundat.cons );

  rnorm1 = rnorm0;
  rnorm0 = rnorm;

  if ( ! bounded_ )
  {
    rnorm = rundat.vspace->norm2 ( r );
  }
  else
  {
    const double  epsilon  = std::sqrt ( Float::EPSILON );
    const idx_t   dofCount = u.size    ();

    for ( idx_t i = 0; i < dofCount; i++ )
    {
      double  lb =  lowerBound_[i];
      double  ub =  upperBound_[i];
      double  x  =  u[i];
      double  f  = -r[i];

      lb += epsilon * fabs ( lb );
      ub -= epsilon * fabs ( ub );

      // Note that the condition ((x <= lb) && (x >= ub)) may be true
      // if the lower bound equals the upper bound.

      if ( x <= lb )
      {
        f = min ( 0.0, f );
      }

      if ( x >= ub )
      {
        f = max ( 0.0, f );
      }

      r2[i] = f;
    }

    rnorm = rundat.vspace->norm2 ( r2 );
  }

  if ( rscale > 0.0 )
  {
    rnorm /= rscale;
  }
}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void NonlinModule::Work_::updateMatrix

  ( const Properties&  globdat,
    bool               getFint )

{
  bool  doUpdate;


  if ( ! rundat.validMatrix || ! updateCond_ )
  {
    doUpdate = true;
  }
  else
  {
    double  args[4] = { (double) iiter, rnorm, rnorm0, rnorm1 };
    double  result  = 0.0;

    try
    {
      result = updateCond_->getValue ( args );
    }
    catch ( Exception& ex )
    {
      ex.setContext ( rundat.context );
      throw;
    }

    doUpdate = (result * result >= 0.25);
  }

  if ( doUpdate )
  {
    rundat.updateMatrix ( fint, globdat );

    rundat.validMatrix = true;
  }
  else if ( getFint )
  {
    rundat.getIntVector ( fint, globdat );
  }
}


//-----------------------------------------------------------------------
//   calcIncrement
//-----------------------------------------------------------------------


void NonlinModule::Work_::calcIncrement ( double maxIncr )
{
  using jem::isTiny;

  idx_t   fixedCount = 0;
  double  dnorm0     = dnorm;


  if ( bounded_ )
  {
    const double  epsilon  = std::sqrt ( Float::EPSILON );
    const idx_t   dofCount = u.size    ();

    for ( idx_t i = 0; i < dofCount; i++ )
    {
      double  lb =  lowerBound_[i];
      double  ub =  upperBound_[i];
      double  x  =  u[i];
      double  f  = -r[i];

      lb += epsilon * fabs ( lb );
      ub -= epsilon * fabs ( ub );

      if ( ((x <= lb) && (f > 0.0)) ||
           ((x >= ub) && (f < 0.0)) )
      {
        if ( ! rundat.cons->isSlaveDof( i ) )
        {
          fixedDofs_[fixedCount++] = i;

          rundat.cons->addConstraint ( i );
        }
      }
    }
  }

  if ( ! bounded_ )
  {
    rundat.solver->solve ( du, r );
  }
  else
  {
    try
    {
      rundat.solver->solve ( du, r );
    }
    catch ( const Exception& )
    {
      // Apply a steepest descent step.

      print ( System::info( myName_ ), rundat.context,
              " : singular matrix; falling back "
              "to steepest descent.\n" );

      du = r;
    }
  }

  for ( idx_t i = 0 ; i < fixedCount; i++ )
  {
    rundat.cons->eraseConstraint ( fixedDofs_[i] );
  }

  dnorm = rundat.vspace->norm2 ( du );

  iiter++;

  if ( bounded_ || (iiter <= 2) || isTiny( dnorm0 ) )
  {
    return;
  }

  if ( dnorm > maxIncr * dnorm0 )
  {
    double  scale = maxIncr * dnorm0 / dnorm;

    print ( System::info( myName_ ), rundat.context,
            " : scaling solution increment with factor ",
            rundat.nformat.print( scale ),
            endl );

    du *= scale;
  }
}


//-----------------------------------------------------------------------
//   doTotalUpdate
//-----------------------------------------------------------------------


void NonlinModule::Work_::doTotalUpdate

  ( const Properties& globdat )

{
  rundat.updateModel  (       globdat );
  rundat.getIntVector ( fint, globdat );
  updateResidual      ();
}


//-----------------------------------------------------------------------
//   checkConverged
//-----------------------------------------------------------------------


bool NonlinModule::Work_::checkConverged

  ( double             tol,
    const Properties&  globdat )

{
  using jive::model::ActionParams;

  Properties  params;
  bool        conv;


  conv = (rnorm <= tol);

  params.set ( ActionParams::RESIDUAL,  r );
  params.set ( ActionParams::RES_SCALE, rscale );
  params.set ( ActionParams::CONVERGED, conv );

  rundat.model->takeAction ( Actions::CHECK_CONVERGED,
                             params, globdat );

  params.find ( conv, ActionParams::CONVERGED );

  return conv;
}


//-----------------------------------------------------------------------
//   reportProgress
//-----------------------------------------------------------------------


inline void NonlinModule::Work_::reportProgress ()
{
  print ( System::info( myName_ ), rundat.context,
          " : iter = "          , iiter,
          ", scaled residual = ", rundat.nformat.print( rnorm ),
          endl );

  System::info( myName_ ).flush ();
}


//=======================================================================
//   class NonlinModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  NonlinModule::TYPE_NAME = "Nonlin";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NonlinModule::NonlinModule ( const String& name ) :

  Super ( name )

{
  maxIter_   = 20;
  options_   = 0;
  tiny_      = jem::Limits<double>::TINY_VALUE;
  precision_ = 1.0e-3;
  maxIncr_   = 10.0;
  bounds_    = jem::newInstance<SolverBounds> ( getContext() );
}


NonlinModule::~NonlinModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status NonlinModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  rundat_ = nullptr;

  Ref<RunData_>  newdat = newInstance<RunData_> ( getContext() );
  String         name   = joinNames ( myName_, PropNames::SOLVER );

  newdat->init       ( globdat );
  newdat->initSolver ( name, precision_, conf, props, globdat );

  // Everything OK, so commit the changes

  rundat_.swap ( newdat );

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void NonlinModule::shutdown ( const Properties& globdat )
{
  rundat_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void NonlinModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    String      expr;
    bool        option;


    myProps.find ( maxIter_,   PropNames::MAX_ITER,
                   0,          maxOf( maxIter_ ) );
    myProps.find ( tiny_,      PropNames::TINY,
                   0.0,        1.0e20 );
    myProps.find ( precision_, PropNames::PRECISION,
                   0.0,        1.0e20 );
    myProps.find ( maxIncr_,   PropNames::MAX_INCR,
                   0.0,        1.0e20 );

    FuncUtils::configFunc  ( updateCond_, "i, r, r0, r1",
                             PropNames::UPDATE_COND,
                             myProps, globdat );

    if ( myProps.find( option, PropNames::LINE_SEARCH ) )
    {
      if ( option )
      {
        options_ |=  LINE_SEARCH;
      }
      else
      {
        options_ &= ~LINE_SEARCH;
      }
    }

    if ( myProps.find( option, PropNames::DELTA_CONS ) )
    {
      if ( option )
      {
        options_ |=  DELTA_CONS;
      }
      else
      {
        options_ &= ~DELTA_CONS;
      }
    }

    bounds_->configure ( myProps );
  }

  if ( rundat_ )
  {
    rundat_->solver->configure ( props );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void NonlinModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );


  myConf.set ( PropNames::MAX_ITER,  maxIter_   );
  myConf.set ( PropNames::TINY,      tiny_      );
  myConf.set ( PropNames::PRECISION, precision_ );
  myConf.set ( PropNames::MAX_INCR,  maxIncr_   );

  myConf.set ( PropNames::LINE_SEARCH,
               ((options_ & LINE_SEARCH) != 0)  );

  myConf.set ( PropNames::DELTA_CONS,
               ((options_ & DELTA_CONS)  != 0)  );

  bounds_->getConfig ( myConf );

  FuncUtils::getConfig ( myConf, updateCond_,
                         PropNames::UPDATE_COND );

  if ( rundat_ )
  {
    rundat_->solver->getConfig ( conf );
  }
}


//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------


void NonlinModule::advance ( const Properties& globdat )
{
  if ( ! rundat_ )
  {
    notAliveError ( JEM_FUNC );
  }

  rundat_->advance ( globdat );
}


//-----------------------------------------------------------------------
//   solve
//-----------------------------------------------------------------------


void NonlinModule::solve

  ( const Properties&  info,
    const Properties&  globdat )

{
  using jive::util::Globdat;
  using jive::util::FuncUtils;

  if ( ! rundat_ )
  {
    notAliveError ( JEM_FUNC );
  }

  RunData_&  d = * rundat_;

  Work_      w ( *this, globdat );

  bool       result;


  print ( System::info( myName_ ),
          "Starting the Newton-Raphson solver `",
          myName_, "\' ...\n" );

  if ( updateCond_ )
  {
    try
    {
      FuncUtils::resolve ( *updateCond_, globdat );
    }
    catch ( Exception& ex )
    {
      ex.setContext ( getContext() );
      throw;
    }
  }

  d.getExtVector    ( w.fext, globdat );
  w.updateBounds    ( globdat );
  w.applyBounds     ();
  w.updateMatrix    ( globdat );
  w.updateRscale    ( globdat );
  w.calcIncrement   ( maxIncr_ );
  w.zeroConstraints ( *d.cons  );

  result = solve_   ( w, globdat );

  w.restoreConstraints ( *d.cons );

  info.set ( SolverInfo::CONVERGED,  result  );
  info.set ( SolverInfo::ITER_COUNT, w.iiter );
  info.set ( SolverInfo::RESIDUAL,   w.rnorm );

  if ( Globdat::hasVariable( myName_, globdat ) )
  {
    Properties  vars = Globdat::getVariables ( myName_, globdat );

    vars.set ( SolverInfo::CONVERGED,  result  );
    vars.set ( SolverInfo::ITER_COUNT, w.iiter );
    vars.set ( SolverInfo::RESIDUAL,   w.rnorm );
  }

  if ( ! result )
  {
    throw Exception (
      getContext (),
      String::format (
        "no convergence achieved in %d iterations; "
        "final residual: %e",
        w.iiter,
        w.rnorm
      )
    );
  }

  print ( System::info( myName_ ),
          "The Newton-Raphson solver converged in ",
          w.iiter, " iterations\n\n" );
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void NonlinModule::cancel ( const Properties& globdat )
{
  if ( ! rundat_ )
  {
    notAliveError ( JEM_FUNC );
  }

  rundat_->cancel ( globdat );
}


//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------


bool NonlinModule::commit ( const Properties& globdat )
{
  if ( ! rundat_ )
  {
    notAliveError ( JEM_FUNC );
  }

  return rundat_->commit ( globdat );
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void NonlinModule::setPrecision ( double eps )
{
  JEM_PRECHECK ( eps > 0.0 );

  precision_ = eps;
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double NonlinModule::getPrecision () const
{
  return precision_;
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void NonlinModule::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void NonlinModule::setOptions ( Options options )
{
  options_ = options;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


NonlinModule::Options NonlinModule::getOptions () const
{
  return options_;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> NonlinModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void NonlinModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------


bool NonlinModule::solve_

  ( Work_&             w,
    const Properties&  globdat )

{
  RunData_&  d       = w.rundat;
  bool       lsearch = false;


  axpy ( w.u, 1.0, w.du );

  w.applyBounds  ();
  d.updateModel  ( globdat );
  w.updateMatrix ( globdat );
  w.updateRscale ( globdat );

  print ( System::info( myName_ ), d.context,
          " : residual scale factor = ",
          d.nformat.print( w.rscale ), endl );

  if ( w.rscale <= tiny_ )
  {
    return true;
  }

  w.updateResidual ();

  do
  {
    w.reportProgress ();

    if ( w.checkConverged( precision_, globdat ) )
    {
      return true;
    }

    if ( w.iiter > maxIter_ )
    {
      return false;
    }

    if ( w.rnorm > 1.0e4 )
    {
      if ( options_ & LINE_SEARCH )
      {
        // Undo the last solution increment and execute the line
        // search procedure.

        axpy ( w.u, -1.0, w.du );

        w.applyBounds ();
        d.updateModel ( globdat );
        lineSearch_   ( w, globdat );
      }
      else
      {
        return false;
      }
    }

    w.calcIncrement ( maxIncr_ );

    if ( lsearch )
    {
      double  rnorm0 = w.rnorm;

      lineSearch_ ( w, globdat );

      if ( w.hasBounds() && (w.rnorm > (0.9 * rnorm0)) )
      {
        // Try again with a steepest descent search direction.

        lineSearch2_ ( w, globdat );
      }

      // If no progress has been made, then do a regular Newton step
      // in the next iteration to avoid getting stuck.

      if ( w.rnorm > (0.99 * rnorm0) )
      {
        lsearch = false;
      }
    }
    else
    {
      axpy ( w.u, 1.0, w.du );

      w.applyBounds    ();
      d.updateModel    ( globdat );
      w.updateMatrix   ( globdat );
      w.updateResidual ();

      if ( (options_ & LINE_SEARCH) && (w.iiter >= 2) )
      {
        lsearch = true;
      }
    }
  }
  while ( true );

  return false;
}


//-----------------------------------------------------------------------
//   lineSearch_
//-----------------------------------------------------------------------


void NonlinModule::lineSearch_

  ( Work_&             w,
    const Properties&  globdat )

{
  const int  NMAX     = 10;

  RunData_&  d        = w.rundat;
  Vector     u0       = w.u0;

  double     minScale = 0.1;
  double     maxScale = 1.0;

  double     rmin, rmin0;
  double     rmax;
  double     r[NMAX + 1];

  double     s, s0, ds;
  double     a, b;

  idx_t      i, j, n;


  rmax = max ( 0.5 * w.rnorm, precision_ );
  n    = max ( 4_idx, 2 * w.iiter );
  n    = min ( n, (idx_t) NMAX );
  s    = maxScale;
  u0   = w.u;

  // Do a normal Newton step and check whether enough progress is
  // made. If not, then proceed with the line search algorithm.

  axpy ( w.u, s, w.du );

  w.applyBounds    ();
  d.updateModel    ( globdat );
  w.updateMatrix   ( globdat );
  w.updateResidual ();

  if ( w.rnorm < rmax )
  {
    return;
  }

  r[n] = w.rnorm;

  print ( System::info( myName_ ), w.rundat.context,
          " : starting line search ...\n" );

  s    = minScale;

  axpy ( w.u, u0, s, w.du );

  w.applyBounds   ();
  w.doTotalUpdate ( globdat );

  if ( w.rnorm < rmax )
  {
    goto success;
  }

  r[0] = w.rnorm;
  rmin = 10.0 * rmax;
  s0   = s;

  do
  {
    rmin0 = rmin;

    if ( r[0] < r[n] )
    {
      rmin = r[0];
      j    = 0;
      s0   = minScale;
    }
    else
    {
      rmin = r[n];
      j    = n;
      s0   = maxScale;
    }

    ds = (maxScale - minScale) / (double) n;

    for ( i = 1; i < n; i++ )
    {
      s    = minScale + (double) i * ds;

      axpy ( w.u, u0, s, w.du );

      w.applyBounds   ();
      w.doTotalUpdate ( globdat );

      r[i] = w.rnorm;

      if ( w.rnorm < rmax )
      {
        goto success;
      }

      if ( w.rnorm < rmin )
      {
        rmin = w.rnorm;
        j    = i;
        s0   = s;
      }
    }

    if      ( j == 0 )
    {
      maxScale = minScale + ds;
      r[n]     = r[1];
    }
    else if ( j == n )
    {
      minScale = maxScale - ds;
      r[0]     = r[n - 1];
    }
    else
    {
      s        = minScale + (double) j * ds;
      maxScale = s + ds;
      minScale = s - ds;
      r[0]     = r[j - 1];
      r[n]     = r[j + 1];

      // Try a quadratic approximation.

      a =  0.5 * (r[n] + r[0] - 2.0 * r[j]) / (ds * ds);
      b =  0.5 * (r[n] - r[0]) / ds - 2.0 * s * a;
      s = -0.5 * b / a;

      axpy ( w.u, u0, s, w.du );

      w.applyBounds   ();
      w.doTotalUpdate ( globdat );

      if ( w.rnorm < rmax )
      {
        goto success;
      }

      if ( w.rnorm < rmin )
      {
        rmin = w.rnorm;
        s0   = s;
      }

      j = n / 2;
    }
  }
  while ( (ds > 0.01) && (rmin < (0.9 * rmin0)) );

  // Apply the best scale factor that has been found.

  s = s0;

  axpy ( w.u, u0, s, w.du );

  w.applyBounds    ();
  d.updateModel    ( globdat );
  w.updateMatrix   ( globdat );
  w.updateResidual ();

  goto finally;

 success:

  if ( w.rnorm > precision_ )
  {
    w.updateMatrix ( globdat, false );
  }

 finally:

  print ( System::info( myName_ ), w.rundat.context,
          " : scale factor = ", d.nformat.print( s ),
          endl );
}


//-----------------------------------------------------------------------
//   lineSearch2_
//-----------------------------------------------------------------------


void NonlinModule::lineSearch2_

  ( Work_&             w,
    const Properties&  globdat )

{
  RunData_&  d      =  w.rundat;
  Vector     u0     =  w.u0;

  double     minExp = -20.0;
  double     maxExp =  10.0;

  double     unorm, rnorm;
  double     rmin,  rmax;
  double     s, s0, dt;

  idx_t      n;


  print ( System::info( myName_ ), w.rundat.context,
          " : starting steepest descent line search ...\n" );

  // Roll back to the previous solution.

  w.u = u0;

  w.applyBounds   ();
  w.doTotalUpdate ( globdat );

  // Use the steepest descent direction as the search direction.
  // Scale the search vector so that its magnitude is comparable
  // with the current solution.

  w.du  = w.r;
  unorm = d.vspace->norm2 ( u0 );
  rnorm = d.vspace->norm2 ( w.r );

  if ( (unorm > 0.0) && (rnorm > 0.0) )
  {
    w.du *= unorm / rnorm;
  }

  rmin = w.rnorm;
  rmax = max ( 0.5 * w.rnorm, precision_ );
  s0   = 0.0;
  n    = max (  7_idx, 2 * w.iiter );
  n    = min ( 31_idx, n );
  dt   = (maxExp - minExp) / (double) (n - 1);

  for ( idx_t i = 0; i < n; i++ )
  {
    s = std::exp ( minExp + (double) i * dt );

    axpy ( w.u, u0, s, w.du );

    w.applyBounds   ();
    w.doTotalUpdate ( globdat );

    if ( w.rnorm < rmax )
    {
      goto success;
    }

    if ( w.rnorm < rmin )
    {
      rmin = w.rnorm;
      s0   = s;
    }
  }

  // Apply the best update that has been found.

  s = s0;

  axpy ( w.u, u0, s, w.du );

  w.applyBounds    ();
  d.updateModel    ( globdat );
  w.updateMatrix   ( globdat );
  w.updateResidual ();

  goto finally;

 success:

  if ( w.rnorm > precision_ )
  {
    w.updateMatrix ( globdat, false );
  }

 finally:

  print ( System::info( myName_ ), w.rundat.context,
          " : scale factor = ", d.nformat.print( s ),
          endl );
}


JIVE_END_PACKAGE( implict )
