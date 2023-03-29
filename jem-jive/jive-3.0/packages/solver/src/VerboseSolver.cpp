
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/DirectSolver.h>
#include <jive/solver/IterativeSolver.h>
#include <jive/solver/VerboseSolver.h>


JEM_DEFINE_CLASS( jive::solver::VerboseSolver          );
JEM_DEFINE_CLASS( jive::solver::DirectVerboseSolver    );
JEM_DEFINE_CLASS( jive::solver::IterativeVerboseSolver );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::io::endl;
using jem::io::flush;


//=======================================================================
//   class VerboseSolver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  VerboseSolver::TYPE_NAME = "Verbose";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


VerboseSolver::VerboseSolver

  ( const String&        name,
    const Ref<Solver>&   solver,
    const Ref<Printer>&  out ) :

    Super   (   name ),
    solver_ ( solver ),
    output_ (    out )

{
  using jem::System;


  JEM_PRECHECK ( solver );

  if ( myName_.size() == 0 )
  {
    myName_ = solver->getName ();
  }

  solverName_ = solver_->getName ();

  if ( ! output_ )
  {
    output_ = newInstance<Printer> ( & System::info( myName_ ) );

    output_->nformat.setScientific     ();
    output_->nformat.setFloatWidth     ( 10 );
    output_->nformat.setFractionDigits (  3 );
  }

  printInterval_ = 2.0;
  noiseLevel_    = 1;
  lastTime_      = 0.0;
}


VerboseSolver::~VerboseSolver ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void VerboseSolver::start ()
{}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void VerboseSolver::finish ()
{}


//-----------------------------------------------------------------------
//   improve
//-----------------------------------------------------------------------


void VerboseSolver::improve

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  start_           ();
  solver_->improve ( lhs, rhs );
  stop_            ();
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void VerboseSolver::getInfo ( const Properties& info ) const
{
  solver_->getInfo ( info );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void VerboseSolver::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( printInterval_, PropNames::PRINT_INTERVAL );
    myProps.find ( noiseLevel_,    PropNames::NOISE_LEVEL    );
  }

  solver_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void VerboseSolver::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set  ( PropNames::PRINT_INTERVAL, printInterval_ );
  myProps.set  ( PropNames::NOISE_LEVEL,    noiseLevel_    );

  solver_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   setMode & getMode
//-----------------------------------------------------------------------


void VerboseSolver::setMode ( int mode )
{
  solver_->setMode ( mode );
}


int VerboseSolver::getMode () const
{
  return solver_->getMode ();
}


//-----------------------------------------------------------------------
//   setPrecision & getPrecision
//-----------------------------------------------------------------------


void VerboseSolver::setPrecision ( double eps )
{
  solver_->setPrecision ( eps );
}


double VerboseSolver::getPrecision () const
{
  return solver_->getPrecision ();
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* VerboseSolver::getMatrix () const
{
  return solver_->getMatrix ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* VerboseSolver::getConstraints () const
{
  return solver_->getConstraints ();
}


//-----------------------------------------------------------------------
//   setPrintInterval & getPrintInterval
//-----------------------------------------------------------------------


void VerboseSolver::setPrintInterval ( double dt )
{
  printInterval_ = dt;
}


double VerboseSolver::getPrintInterval () const
{
  return printInterval_;
}


//-----------------------------------------------------------------------
//   setNoiseLevel & getNoiseLevel
//-----------------------------------------------------------------------


void VerboseSolver::setNoiseLevel ( int n )
{
  noiseLevel_ = n;
}


int VerboseSolver::getNoiseLevel () const
{
  return noiseLevel_;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> VerboseSolver::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  return newVerboseSolver (
    name,
    SolverFactory::newInstance (
      joinNames ( name, PropNames::SOLVER ),
      conf,   props,
      params, globdat
    )
  );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void VerboseSolver::declare ()
{
  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   begin_
//-----------------------------------------------------------------------


void VerboseSolver::begin_ ()
{
  if ( noiseLevel_ > 0 )
  {
    Printer&  out = * output_;

    print ( out, "Starting solver `", solverName_, "\' ...\n\n" );
  }
}


//-----------------------------------------------------------------------
//   end_
//-----------------------------------------------------------------------


void VerboseSolver::end_ ()
{
  using jem::io::indent;
  using jem::io::outdent;

  if ( noiseLevel_ > 0 )
  {
    Printer&  out = * output_;

    if ( noiseLevel_ > 1 )
    {
      Properties  info;

      solver_->getInfo ( info );

      print ( out, indent, endl, info, endl, outdent );
    }

    print ( out, "\nSolution obtained in ", timer_, endl, endl );
  }
}


//-----------------------------------------------------------------------
//   start_
//-----------------------------------------------------------------------


void VerboseSolver::start_ ()
{
  timer_.reset ();

  lastTime_ = 0.0;

  begin_ ();

  output_->flush ();
  timer_  .start ();
}


//-----------------------------------------------------------------------
//   stop_
//-----------------------------------------------------------------------


void VerboseSolver::stop_ ()
{
  timer_.stop ();

  end_ ();

  output_->flush ();
}


//=======================================================================
//   class DirectVerboseSolver
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DirectVerboseSolver::DirectVerboseSolver

  ( const String&        name,
    Ref<DirectSolver>    solver,
    const Ref<Printer>&  out ) :

    Super   ( name, solver, out ),

    solver_ ( solver )

{
  using jem::util::connect;

  phase_          = INACTIVE_;
  zeroPivotCount_ = 0;

  connect ( solver_->factorEvent, this, & Self::factorHandler_ );
  connect ( solver_->solveEvent,  this, & Self::solveHandler_  );
  connect ( solver_->zeroPivotEvent,
            this, & Self::zeroPivotHandler_ );
}


DirectVerboseSolver::~DirectVerboseSolver ()
{}


//-----------------------------------------------------------------------
//   begin_
//-----------------------------------------------------------------------


void DirectVerboseSolver::begin_ ()
{
  phase_          = INACTIVE_;
  zeroPivotCount_ = 0;

  if ( noiseLevel_ > 0 )
  {
    const double  threshold = solver_->getZeroThreshold ();
    Printer&      out       = * output_;

    out.nformat.setIntegerWidth ( 3 );

    print ( out, "Starting solver `", solverName_, "\' ...\n\n" );

    print ( out, solverName_, " : zero threshold = ",
            threshold, "\n\n" );
  }
}


//-----------------------------------------------------------------------
//   end_
//-----------------------------------------------------------------------


void DirectVerboseSolver::end_ ()
{
  Super::end_ ();

  phase_ = INACTIVE_;
}


//-----------------------------------------------------------------------
//   factorHandler_
//-----------------------------------------------------------------------


void DirectVerboseSolver::factorHandler_ ( idx_t done )
{
  if ( noiseLevel_ > 0 )
  {
    Printer&  out = * output_;
    double    t   = timer_.toDouble ();

    if ( phase_ == INACTIVE_ )
    {
      print ( out, solverName_, " : factoring matrix ...\n", flush );

      phase_ = FACTORING_;
    }
    else if ( phase_ == FACTORING_ &&
              t      >= lastTime_ + printInterval_ )
    {
      print ( out, solverName_, " : ", done, "% done\n", flush );

      lastTime_ = t;
    }
  }
}


//-----------------------------------------------------------------------
//   zeroPivotHandler_
//-----------------------------------------------------------------------


void DirectVerboseSolver::zeroPivotHandler_ ( double pivot )
{
  const idx_t  MAX_COUNT = 10;

  if ( phase_ == FACTORING_ && noiseLevel_ > 1 )
  {
    Printer&  out = * output_;

    if ( zeroPivotCount_ < MAX_COUNT )
    {
      print ( out, solverName_, " : zero pivot (value = ",
              pivot, ")\n", flush );
    }
    else if ( zeroPivotCount_ == MAX_COUNT )
    {
      print ( out, solverName_, " : more zero pivots\n", flush );
    }

    zeroPivotCount_++;
  }
}


//-----------------------------------------------------------------------
//   solveHandler_
//-----------------------------------------------------------------------


void DirectVerboseSolver::solveHandler_ ( double err )
{
  if ( noiseLevel_ > 0 )
  {
    Printer&  out = * output_;

    if ( phase_ == FACTORING_ )
    {
      print ( out, solverName_, " : factorization complete\n" );

      phase_ = SOLVING_;
    }

    print ( out, solverName_, " : residual = ", err, endl, flush );
  }
}


//=======================================================================
//   class IterativeVerboseSolver
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


IterativeVerboseSolver::IterativeVerboseSolver

  ( const String&         name,
    Ref<IterativeSolver>  solver,
    const Ref<Printer>&   out ) :

    Super     ( name, solver, out ),

    solver_   ( solver ),
    lastIter_ (      0 )

{
  using jem::util::connect;

  connect ( solver_->nextIterEvent,
            this, & Self::nextIterHandler_ );

  connect ( solver_->restartEvent,
            this, & Self::restartHandler_ );
}


IterativeVerboseSolver::~IterativeVerboseSolver ()
{}


//-----------------------------------------------------------------------
//   begin_
//-----------------------------------------------------------------------


void IterativeVerboseSolver::begin_ ()
{
  const double  precision = solver_->getPrecision    ();
  const idx_t   maxIter   = solver_->getMaxIterCount ();
  Printer&      out       = * output_;

  int           width;


  if      ( maxIter < 10 )
  {
    width = 1;
  }
  else if ( maxIter < 100 )
  {
    width = 2;
  }
  else if ( maxIter < 1000 )
  {
    width = 3;
  }
  else
  {
    width = 4;
  }

  out.nformat.setIntegerWidth ( width );

  lastIter_ = 0;

  if ( noiseLevel_ > 0 )
  {
    print ( out, "Starting solver `", solverName_, "\' ...\n\n" );

    print ( out, solverName_, " : maxIter   = ", maxIter, endl );
    print ( out, solverName_, " : precision = ", precision, "\n\n" );
  }
}


//-----------------------------------------------------------------------
//   end_
//-----------------------------------------------------------------------


void IterativeVerboseSolver::end_ ()
{
  if ( noiseLevel_ > 0 )
  {
    Printer& out = * output_;

    print ( out, "\nSolution obtained in ", lastIter_,
            " iterations (", timer_, ").\n\n" );
  }
}


//-----------------------------------------------------------------------
//   nextIterHandler_
//-----------------------------------------------------------------------


void IterativeVerboseSolver::nextIterHandler_

  ( idx_t   iiter,
    double  err )

{
  if ( noiseLevel_ > 0 )
  {
    Printer&  out = * output_;
    double    t   = timer_.toDouble ();

    if ( iiter == lastIter_ || t >= lastTime_ + printInterval_ )
    {
      print ( out, solverName_, " : iter = ", iiter );

      if ( iiter == lastIter_ )
      {
        print ( out, ", actual residual    = " );
      }
      else
      {
        print ( out, ", estimated residual = " );
      }

      print ( out, err, endl, flush );

      lastTime_ = t;
    }
  }

  lastIter_ = iiter;
}


//-----------------------------------------------------------------------
//   restartHandler_
//-----------------------------------------------------------------------


void IterativeVerboseSolver::restartHandler_

  ( idx_t   iiter,
    double  err )

{
  if ( noiseLevel_ > 0 )
  {
    Printer& out = * output_;

    print ( out, solverName_,
            " : restarting after ", iiter, " iterations.\n" );

    out.flush ();
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newVerboseSolver
//-----------------------------------------------------------------------


Ref<VerboseSolver>          newVerboseSolver

  ( const Ref<Solver>&        solver,
    const Ref<PrintWriter>&   out )

{
  return newVerboseSolver ( solver->getName(), solver, out );
}


Ref<VerboseSolver>          newVerboseSolver

  ( const String&             name,
    const Ref<Solver>&        solver,
    const Ref<PrintWriter>&   out )

{
  using jem::dynamicCast;

  Ref<IterativeSolver>  isolver;
  Ref<DirectSolver>     dsolver;


  dsolver = dynamicCast<DirectSolver> ( solver );

  if ( dsolver )
  {
    return newInstance<DirectVerboseSolver> ( name, dsolver, out );
  }

  isolver = dynamicCast<IterativeSolver> ( solver );

  if ( isolver )
  {
    return newInstance<IterativeVerboseSolver> ( name, isolver, out );
  }

  return newInstance<VerboseSolver> ( name, solver, out );
}


JIVE_END_PACKAGE( solver )
