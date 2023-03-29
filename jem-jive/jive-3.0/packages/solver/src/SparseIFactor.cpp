
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


#include <cfloat>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/mp/utilities.h>
#include <jem/mp/Context.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/numeric/sparse/utilities.h>
#include <jem/numeric/sparse/Reorder.h>
#include <jive/util/error.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/utilities.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/solver/typedefs.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverException.h>
#include <jive/solver/SparseIFactor.h>


JEM_DEFINE_CLASS( jive::solver::SparseIFactor );


JIVE_BEGIN_PACKAGE( solver )


using jem::dynamicCast;
using jive::mp::EXCHANGE;
using jive::mp::SCATTER;
using jive::algebra::MPMatrixObj;
using jive::algebra::SparseMatrixExt;


//=======================================================================
//   class SparseIFactor
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  SparseIFactor::ZERO_THRESHOLD  = 1.0e-6;

const int     SparseIFactor::NEW_VALUES_     = 1 << 0;
const int     SparseIFactor::NEW_STRUCT_     = 1 << 1;
const int     SparseIFactor::NEW_CONFIG_     = 1 << 2;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SparseIFactor::SparseIFactor

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<Constraints>     cons ) :

    Super   ( name   ),
    matrix_ ( matrix ),
    cons_   ( cons   )

{
  JEM_PRECHECK ( matrix );

  MPMatrixObj*  mpMat = dynamicCast<MPMatrixObj*> ( matrix );

  if ( mpMat )
  {
    matrix_    = mpMat->getInner        ();
    exchanger_ = mpMat->getExchanger    ();
    xmode_     = mpMat->getExchangeMode ();
  }
  else
  {
    xmode_     = EXCHANGE;
  }

  if ( ! matrix_->hasExtension<SparseMatrixExt>() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      matrix->getContext() +
      " does not implement the sparse matrix extension"
    );
  }

  options_       =  REORDER;
  droptol_       =  1.0e-6;
  dshift_        =  0.0;
  zeroThreshold_ =  ZERO_THRESHOLD;
  symmetric_     =  false;
  events_        = ~0x0;
  started_       =  0;

  connect_ ();
}


SparseIFactor::~SparseIFactor ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SparseIFactor::resetEvents ()
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


SparseIFactor::Shape SparseIFactor::shape () const
{
  return matrix_->shape ();
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SparseIFactor::start ()
{
  if ( ! started_ )
  {
    syncEvents_ ();
  }

  if ( events_ & (~NEW_VALUES_) )
  {
    update_ ();
  }

  started_++;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SparseIFactor::finish ()
{
  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SparseIFactor::update ()
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


void SparseIFactor::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  JEM_PRECHECK ( ! (events_ & NEW_STRUCT_) );

  if ( ! exchanger_ )
  {
    solve_ ( lhs, rhs );
  }
  else
  {
    int  xmode = xmode_;

    if ( symmetric_ )
    {
      xmode = SCATTER;
    }

    exchanger_->startOne ( xmode );
    this      ->solve_   ( lhs, rhs );

    if ( exchanger_->hasOverlap() )
    {
      lhs[borderDofs_] = 0.0;
    }
    else
    {
      lhs[borderDofs_] = borderDiag_ * rhs[borderDofs_];
    }

    exchanger_->endOne ( lhs );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SparseIFactor::configure ( const Properties& props )
{
  using jem::maxOf;

  Super::configure ( props );

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    bool        newConf = false;
    bool        found;
    bool        option;

    if ( myProps.find( option, PropNames::REORDER ) )
    {
      if ( option )
      {
        options_ |= REORDER;
      }
      else
      {
        options_ &= ~REORDER;
      }

      newConf = true;
    }

    found   = myProps.find ( droptol_,
                             PropNames::DROP_TOL,
                             0.0, maxOf ( droptol_ ) );

    newConf = (newConf || found);

    found   = myProps.find ( dshift_,
                             PropNames::DIAG_SHIFT,
                             0.0, maxOf ( dshift_ ) );

    newConf = (newConf || found);

    found   = myProps.find ( zeroThreshold_,
                             PropNames::ZERO_THRESHOLD,
                             0.0, maxOf ( zeroThreshold_ ) );

    newConf = (newConf || found);

    if ( newConf )
    {
      setEvents_          ( NEW_CONFIG_ );
      newValuesEvent.emit ( *this );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SparseIFactor::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Super::getConfig ( props );

  myProps.set ( PropNames::REORDER, (bool) (options_ & REORDER) );
  myProps.set ( PropNames::MAX_FILL,        maxFill_       );
  myProps.set ( PropNames::DROP_TOL,        droptol_       );
  myProps.set ( PropNames::DIAG_SHIFT,      dshift_        );
  myProps.set ( PropNames::ZERO_THRESHOLD,  zeroThreshold_ );
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool SparseIFactor::hasTrait ( const String& trait ) const
{
  using jive::algebra::MatrixTraits;

  if      ( trait == MatrixTraits::SYMMETRIC )
  {
    return matrix_->hasTrait ( trait );
  }
  else if ( trait == MatrixTraits::DISTRIBUTED )
  {
    if ( ! exchanger_ )
    {
      return false;
    }
    else
    {
      return exchanger_->isDistributed ();
    }
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* SparseIFactor::getConstraints () const
{
  return cons_.get ();
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void SparseIFactor::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void SparseIFactor::setOptions ( Options options )
{
  if ( options != options_ )
  {
    options_ = options;
    events_ |= NEW_STRUCT_ | NEW_CONFIG_;

    newStructEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   setDropTol
//-----------------------------------------------------------------------


void SparseIFactor::setDropTol ( double tol )
{
  JEM_PRECHECK2 ( tol >= 0.0, "invalid drop tolerance" );

  setParam_ ( droptol_, tol );
}


//-----------------------------------------------------------------------
//   setDiagShift
//-----------------------------------------------------------------------


void SparseIFactor::setDiagShift ( double dshift )
{
  JEM_PRECHECK2 ( dshift >= 0.0, "invalid diagonal shift factor" );

  setParam_ ( dshift_, dshift );
}


//-----------------------------------------------------------------------
//   setZeroThreshold
//-----------------------------------------------------------------------


void SparseIFactor::setZeroThreshold ( double eps )
{
  JEM_PRECHECK2 ( eps >= 0.0, "invalid zero pivot threshold" );

  setParam_ ( zeroThreshold_, eps );
}


//-----------------------------------------------------------------------
//   setExchangeMode
//-----------------------------------------------------------------------


void SparseIFactor::setExchangeMode ( int xmode )
{
  setParam_ ( xmode_, xmode );
}


//-----------------------------------------------------------------------
//   decodeParams
//-----------------------------------------------------------------------


bool SparseIFactor::decodeParams

  ( Ref<AbstractMatrix>&  matrix,
    Ref<Constraints>&     cons,
    const Properties&     params )

{
  if ( ! params.find( matrix, SolverParams::MATRIX ) )
  {
    return false;
  }

  MPMatrixObj*         mpMat = dynamicCast<MPMatrixObj*> ( matrix );
  Ref<AbstractMatrix>  inner = matrix;

  if ( mpMat )
  {
    inner = mpMat->getInner ();
  }

  if ( inner->isDistributed() )
  {
    matrix = nullptr;

    return false;
  }

  if ( ! inner->hasExtension<SparseMatrixExt>() )
  {
    matrix = nullptr;

    return false;
  }

  params.find ( cons, SolverParams::CONSTRAINTS );

  return true;
}


//-----------------------------------------------------------------------
//   setParam_
//-----------------------------------------------------------------------


void SparseIFactor::setParam_

  ( int&  param,
    int   value )

{
  if ( param != value )
  {
    param = value;

    setEvents_          ( NEW_CONFIG_ );
    newValuesEvent.emit ( *this );
  }
}


void SparseIFactor::setParam_

  ( idx_t&  param,
    idx_t   value )

{
  if ( param != value )
  {
    param = value;

    setEvents_          ( NEW_CONFIG_ );
    newValuesEvent.emit ( *this );
  }
}


void SparseIFactor::setParam_

  ( double&  param,
    double   value )

{
  if ( ! jem::isTiny( param - value ) )
  {
    param = value;

    setEvents_          ( NEW_CONFIG_ );
    newValuesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void SparseIFactor::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "sparse incomplete factor changed while solving "
      "a linear system of equations"
    );
  }

  events_ |= events;
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void SparseIFactor::connect_ ()
{
  using jem::util::connect;

  connect ( matrix_->newValuesEvent, this, &Self::valuesChanged_ );
  connect ( matrix_->newStructEvent, this, &Self::structChanged_ );

  if ( cons_ )
  {
    connect ( cons_->newStructEvent, this, &Self::structChanged_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void SparseIFactor::update_ ()
{
  using jem::find;
  using jem::iarray;
  using jem::isTiny;
  using jem::System;
  using jem::numeric::diag;
  using jem::numeric::Reorder;
  using jive::algebra::markBorderRows;

  SparseMatrixExt*  sx  = matrix_->getExtension<SparseMatrixExt> ();
  SparseMatrix      sm  = sx     ->toSparseMatrix ();

  const Shape  sh       = sm.shape ();
  const idx_t  dofCount = sh[0];

  BoolVector   mask     ( dofCount );


  if ( sh[0] != sh[1] )
  {
    util::nonSquareMatrixError ( getContext (), sh );
  }

  symmetric_ = matrix_->isSymmetric ();

  if ( ! exchanger_ )
  {
    mask = true;

    borderDofs_.resize ( 0 );
    borderDiag_.resize ( 0 );
  }
  else
  {
    mask = false;

    markBorderRows ( mask, *exchanger_, sm.getStructure() );

    find ( borderDofs_, mask );

    mask = ! mask;

    if ( exchanger_->hasOverlap() )
    {
      borderDiag_.resize ( 0 );
    }
    else
    {
      const idx_t  bdCount   = borderDofs_.size ();

      Vector       totalDiag = diag ( sm );

      exchanger_->exchange ( totalDiag );

      borderDiag_.resize ( bdCount );

      borderDiag_ = totalDiag[borderDofs_];

      for ( idx_t i = 0; i < bdCount; i++ )
      {
        if ( isTiny( borderDiag_[i] ) )
        {
          borderDiag_[i] = 1.0;
        }
      }

      borderDiag_ = 1.0 / borderDiag_;
    }
  }

  if ( cons_ )
  {
    mask[cons_->getSlaveDofs()] = false;
  }

  if ( (events_ & NEW_STRUCT_) || dofPerm_.size() != dofCount )
  {
    dofPerm_.resize ( dofCount );

    if ( options_ & REORDER )
    {
      print ( System::debug( myName_ ), myName_,
              " : re-ordering matrix ...\n" );

      Reorder::superReorder ( dofPerm_,
                              sm.getStructure (),
                              & Reorder::rcm );
    }
    else
    {
      dofPerm_ = iarray ( dofCount );
    }
  }

  factor_ ( events_, sm, mask, dofPerm_ );

  events_ = 0;

  Self::resetEvents   ();
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   syncEvents_
//-----------------------------------------------------------------------


void SparseIFactor::syncEvents_ ()
{
  if ( exchanger_ )
  {
    MPContext*  mpx = exchanger_->getMPContext ();

    events_ = allreduce ( *mpx, events_, jem::mp::BOR );
  }

  Self::resetEvents ();
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void SparseIFactor::valuesChanged_ ()
{
  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void SparseIFactor::structChanged_ ()
{
  setEvents_          ( NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( solver )
