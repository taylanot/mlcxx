
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


#include <jem/base/System.h>
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/SparseLU.h>
#include <jive/solver/SkylineSolver.h>
#include <jive/solver/SkylineLU.h>


JEM_DEFINE_CLASS( jive::solver::SkylineLU );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::System;
using jem::io::endl;


//=======================================================================
//   class SkylineLU
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SkylineLU::TYPE_NAME = "SkylineLU";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SkylineLU::SkylineLU

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<Constraints>     cons ) :

    Super ( name )

{
  JEM_PRECHECK ( matrix );

  skyline_ = newInstance<SkylineSolver> ( name, matrix, cons );
  solver_  = skyline_;
}


SkylineLU::~SkylineLU ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SkylineLU::start ()
{
  // Nothing is done here to make switching between solvers
  // a less tricky business.
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SkylineLU::finish ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SkylineLU::clear ()
{
  solver_->clear ();
}


//-----------------------------------------------------------------------
//   improve
//-----------------------------------------------------------------------


void SkylineLU::improve

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  using jem::Exception;

  SolverScope  scope ( *this );

  connect_ ();

 again:

  if ( sparseLU_ )
  {
    sparseLU_->improve ( lhs, rhs );

    return;
  }

  try
  {
    skyline_->improve ( lhs, rhs );
  }
  catch ( const Exception& ex )
  {
    print       ( System::warn(), ex, endl );
    toSparseLU_ ();
    goto again;
  }

  // Make sure that the desired precision has been obtained. This may
  // not be the case if the solver is operating in lenient or
  // preconditioning mode.

  if ( skyline_->getLastError() > skyline_->getPrecision() )
  {
    toSparseLU_ ();
    goto again;
  }
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void SkylineLU::getInfo ( const Properties& info ) const
{
  solver_->getInfo ( info );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SkylineLU::configure ( const Properties& props )
{
  solver_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SkylineLU::getConfig ( const Properties& conf ) const
{
  solver_->getConfig ( conf );
}


//-----------------------------------------------------------------------
//   setMode
//-----------------------------------------------------------------------


void SkylineLU::setMode ( int mode )
{
  solver_->setMode ( mode );
}


//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


int SkylineLU::getMode () const
{
  return solver_->getMode ();
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void SkylineLU::setPrecision ( double eps )
{
  solver_->setPrecision ( eps );
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double SkylineLU::getPrecision () const
{
  return solver_->getPrecision ();
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* SkylineLU::getMatrix () const
{
  return solver_->getMatrix ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* SkylineLU::getConstraints () const
{
  return solver_->getConstraints ();
}


//-----------------------------------------------------------------------
//   getNullSpace
//-----------------------------------------------------------------------


void SkylineLU::getNullSpace ( Matrix& cmat )
{
  return solver_->getNullSpace ( cmat );
}


//-----------------------------------------------------------------------
//   setZeroThreshold
//-----------------------------------------------------------------------


void SkylineLU::setZeroThreshold ( double eps )
{
  solver_->setZeroThreshold ( eps );
}


//-----------------------------------------------------------------------
//   getZeroThreshold
//-----------------------------------------------------------------------


double SkylineLU::getZeroThreshold () const
{
  return solver_->getZeroThreshold ();
}


//-----------------------------------------------------------------------
//   setMaxZeroPivots
//-----------------------------------------------------------------------


void SkylineLU::setMaxZeroPivots ( idx_t n )
{
  solver_->setMaxZeroPivots ( n );
}


//-----------------------------------------------------------------------
//   getMaxZeroPivots
//-----------------------------------------------------------------------


idx_t SkylineLU::getMaxZeroPivots () const
{
  return solver_->getMaxZeroPivots ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> SkylineLU::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<AbstractMatrix>  mat;
  Ref<Constraints>     cons;


  params.find ( mat,  SolverParams::MATRIX      );
  params.find ( cons, SolverParams::CONSTRAINTS );

  if ( mat && mat->hasExtension<SparseMatrixExt>() )
  {
    return newInstance<Self> ( name, mat, cons );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void SkylineLU::declare ()
{
  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void SkylineLU::connect_ ()
{
  using jem::util::connect;

  DirectSolver&  s = *solver_;

  if (   solveEvent.isConnected() && !
       s.solveEvent.isConnected() )
  {
    connect ( s.solveEvent,     this, & Self::solveEvent_ );
  }

  if (   zeroPivotEvent.isConnected() && !
       s.zeroPivotEvent.isConnected() )
  {
    connect ( s.zeroPivotEvent, this, & Self::pivotEvent_ );
  }

  if (   factorEvent.isConnected() && !
       s.factorEvent.isConnected() )
  {
    connect ( s.factorEvent,    this, & Self::factorEvent_ );
  }
}


//-----------------------------------------------------------------------
//   toSparseLU_
//-----------------------------------------------------------------------


void SkylineLU::toSparseLU_ ()
{
  print ( System::warn(), myName_,
          " : switching to the `SparseLU\' solver\n" );

  // Free memory before creating the new solver.

  skyline_->clear ();

  Ref<SparseLU>  slu =

    newInstance<SparseLU> ( myName_,
                            skyline_->getMatrix      (),
                            skyline_->getConstraints () );

  // Initialize the SparseLU solver.

  SkylineSolver::Options  skyOptions = skyline_->getOptions ();
  SparseLU     ::Options  sluOptions = 0;

  if ( skyOptions & SkylineSolver::PRINT_PIVOTS )
  {
    sluOptions |= SparseLU::PRINT_PIVOTS;
  }

  slu->setOptions ( sluOptions );

  if ( skyOptions & SkylineSolver::REORDER )
  {
    slu->setReorderMethod ( SparseLU::REORDER_MATRIX );
  }
  else
  {
    slu->setReorderMethod ( SparseLU::REORDER_NONE );
  }

  slu->setMode          ( skyline_->getMode          () );
  slu->setPrecision     ( skyline_->getPrecision     () );
  slu->setZeroThreshold ( skyline_->getZeroThreshold () );
  slu->setMaxZeroPivots ( skyline_->getMaxZeroPivots () );

  // No errors, so commit the new solver.

  solver_   = slu;
  sparseLU_ = slu;
  skyline_  = nullptr;

  // Connect to the events of the new solver.

  connect_ ();
}


//-----------------------------------------------------------------------
//   solveEvent_
//-----------------------------------------------------------------------


void SkylineLU::solveEvent_ ( double err )
{
  solveEvent.emit ( err, *this );
}


//-----------------------------------------------------------------------
//   pivotEvent_
//-----------------------------------------------------------------------


void SkylineLU::pivotEvent_ ( double piv )
{
  zeroPivotEvent.emit ( piv, *this );
}


//-----------------------------------------------------------------------
//   factorEvent_
//-----------------------------------------------------------------------


void SkylineLU::factorEvent_ ( idx_t progress )
{
  factorEvent.emit ( progress, *this );
}


JIVE_END_PACKAGE( solver )
