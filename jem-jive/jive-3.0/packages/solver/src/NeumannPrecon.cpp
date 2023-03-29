
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/select.h>
#include <jem/base/array/operators.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/mp/utilities.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MPVectorSpace.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/PreconFactory.h>
#include <jive/solver/SchurMatrix.h>
#include <jive/solver/SchurDofSpace.h>
#include <jive/solver/SparseIFactor.h>
#include <jive/solver/NeumannPrecon.h>


JEM_DEFINE_CLASS( jive::solver::NeumannPrecon );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;


//=======================================================================
//   class NeumannPrecon
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  NeumannPrecon::TYPE_NAME   = "Neumann";

const int    NeumannPrecon::NEW_VALUES_ = 1 << 0;
const int    NeumannPrecon::NEW_STRUCT_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


NeumannPrecon::NeumannPrecon

  ( const String&         name,
    Ref<Preconditioner>   inner,
    Ref<VectorExchanger>  vex,
    Ref<SchurMatrix>      smat ) :

    Super      ( name  ),
    inner_     ( inner ),
    exchanger_ ( vex   ),
    schurMat_  ( smat  )

{
  JEM_PRECHECK ( inner );

  if ( schurMat_ )
  {
    exchanger_ = schurMat_->getExchanger ();
  }

  events_  = ~0x0;
  started_ =  0;

  connect_ ();
}


NeumannPrecon::~NeumannPrecon ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void NeumannPrecon::resetEvents ()
{
  inner_->resetEvents ();

  if ( exchanger_ )
  {
    exchanger_->getRecvBorders().resetEvents ();
    exchanger_->getSendBorders().resetEvents ();
  }

  if ( schurMat_ )
  {
    schurMat_->resetEvents ();
  }
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape NeumannPrecon::shape () const
{
  if ( ! schurMat_ )
  {
    return inner_->shape ();
  }
  else
  {
    schurMat_->update ();

    return schurMat_->shape ();
  }
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void NeumannPrecon::start ()
{
  inner_->start ();

  if ( schurMat_ )
  {
    schurMat_->start ();
  }

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


void NeumannPrecon::finish ()
{
  inner_->finish ();

  if ( schurMat_ )
  {
    schurMat_->finish ();
  }

  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void NeumannPrecon::update ()
{
  inner_->update ();

  if ( schurMat_ )
  {
    schurMat_->update ();
  }

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


void NeumannPrecon::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  using jive::mp::SCATTER;
  using jive::util::sizeError;

  JEM_PRECHECK ( ! (events_ & NEW_STRUCT_) );
  JEM_PRECHECK ( lhs.size() == rhs.size() );

  if ( ! exchanger_ )
  {
    inner_->matmul ( lhs, rhs );

    return;
  }

  const idx_t  totalSize = totalRhs_.size ();

  if ( ! schurMat_ )
  {
    if ( rhs.size() != totalSize )
    {
      sizeError ( JEM_FUNC, "rhs vector", rhs.size(), totalSize );
    }

    exchanger_->startOne ( SCATTER );

    totalRhs_  = rhs;
    totalRhs_ *= weights_;

    inner_->matmul ( lhs, totalRhs_ );

    lhs       *= weights_;

    exchanger_->endOne ( lhs );
  }
  else
  {
    SchurDofSpace*  schurDofs  = schurMat_->getSchurDofSpace ();
    IdxVector       borderDofs = schurDofs->getBorderDofs    ();

    const idx_t     schurSize  = borderDofs.size ();

    if ( rhs.size() != schurSize )
    {
      sizeError ( JEM_FUNC, "rhs vector", rhs.size(), schurSize );
    }

    exchanger_->startOne ( SCATTER );

    totalRhs_             = 0.0;
    totalRhs_[borderDofs] = weights_ * rhs;

    inner_->matmul ( totalLhs_, totalRhs_ );

    lhs = weights_ * totalLhs_[borderDofs];

    exchanger_->endOne ( lhs );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void NeumannPrecon::configure ( const Properties& props )
{
  inner_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void NeumannPrecon::getConfig ( const Properties& conf ) const
{
  inner_->getConfig ( conf );
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool NeumannPrecon::hasTrait ( const String& trait ) const
{
  using jive::algebra::MatrixTraits;

  bool  result = false;

  if      ( trait == MatrixTraits::SYMMETRIC )
  {
    result = inner_->hasTrait ( trait );
  }
  else if ( trait == MatrixTraits::DISTRIBUTED )
  {
    if ( ! exchanger_ )
    {
      result = false;
    }
    else
    {
      result = exchanger_->isDistributed ();
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* NeumannPrecon::getConstraints () const
{
  return inner_->getConstraints ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Precon> NeumannPrecon::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::dynamicCast;
  using jive::util::joinNames;
  using jive::algebra::MPMatrixObj;
  using jive::algebra::MPVectorSpace;

  const String  innerName =

    joinNames ( name, PropNames::PRECONDITIONER );

  Ref<AbstractMatrix>   matrix;
  Ref<Preconditioner>   inner;
  Ref<VectorExchanger>  exchanger;
  Ref<VectorSpace>      vspace;
  Ref<Constraints>      cons;


  params.find ( matrix, SolverParams::MATRIX );

  if ( ! matrix )
  {
    return nullptr;
  }

  // Set defaults for the inner preconditioner.

  Properties  innerProps = props.makeProps ( innerName );
  String      type;

  if ( innerProps.find( type, PropNames::TYPE ) )
  {
    if ( type == "LU"       ||
         type == "SparseLU" ||
         type == "Skyline"  ||
         type == "SkylineLU" )
    {
      if ( ! innerProps.contains( PropNames::MAX_ZERO_PIVOTS ) )
      {
        innerProps.set ( PropNames::MAX_ZERO_PIVOTS, -1 );
      }

      if ( ! innerProps.contains( PropNames::ZERO_THRESHOLD ) )
      {
        innerProps.set ( PropNames::ZERO_THRESHOLD,
                         SparseIFactor::ZERO_THRESHOLD );
      }
    }
  }
  else
  {
    // Use ILU0 as the default inner preconditioner.

    innerProps.set ( PropNames::TYPE, "ILU0" );
  }

  // Handle the trivial case.

  if ( ! matrix->isDistributed() )
  {
    return newPrecon ( innerName, conf, props, params, globdat );
  }

  params.find ( vspace, SolverParams::VECTOR_SPACE );
  params.find ( cons,   SolverParams::CONSTRAINTS  );

  // Unwrap the matrix and vector space.

  Ref<SchurMatrix>  schurMat =

    dynamicCast<SchurMatrix> ( matrix );

  if ( schurMat )
  {
    matrix    = schurMat->getTotalMatrix   ();
    exchanger = schurMat->getExchanger     ();
    cons      = schurMat->getTotalCons     ();
  }
  else
  {
    MPMatrixObj*  mpMat = dynamicCast<MPMatrixObj*> ( matrix );

    if ( mpMat )
    {
      matrix    = mpMat->getInner     ();
      exchanger = mpMat->getExchanger ();
    }
  }

  // Make sure that the inner matrix is not distributed.

  if ( matrix->isDistributed() )
  {
    return nullptr;
  }

  Ref<MPVectorSpace>  mpVSpace =

    dynamicCast<MPVectorSpace> ( vspace );

  if ( mpVSpace )
  {
    vspace = mpVSpace->getInner ();
  }

  // Make sure that the inner vector space is not distributed.

  if ( vspace->isDistributed() )
  {
    return nullptr;
  }

  // Set up the parameters for the inner preconditioner.

  Properties  innerParams ( joinNames( innerName, "params" ) );

  innerParams.set ( SolverParams::MATRIX, matrix );

  if ( cons )
  {
    innerParams.set ( SolverParams::CONSTRAINTS, cons );
  }

  if ( vspace )
  {
    innerParams.set ( SolverParams::VECTOR_SPACE, vspace );
  }

  Ref<Object>  extra;

  if ( params.find( extra, SolverParams::EXTRA ) )
  {
    innerParams.set ( SolverParams::EXTRA, extra );
  }

  // Create the inner preconditioner.

  inner = newPrecon ( innerName, conf, props,
                      innerParams, globdat );

  if ( ! inner )
  {
    return nullptr;
  }

  return newInstance<Self> ( name, inner, exchanger, schurMat );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void NeumannPrecon::declare ()
{
  PreconFactory::declare ( TYPE_NAME,  & Self::makeNew );
  PreconFactory::declare ( CLASS_NAME, & Self::makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void NeumannPrecon::connect_ ()
{
  using jem::util::connect;

  connect ( inner_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( inner_->newStructEvent, this, & Self::structChanged_ );

  if ( exchanger_ )
  {
    connect ( exchanger_->getRecvBorders().changeEvent(),
              this, & Self::structChanged_ );

    if ( exchanger_->hasOverlap() )
    {
      connect ( exchanger_->getSendBorders().changeEvent(),
                this, & Self::structChanged_ );
    }
  }

  if ( schurMat_ )
  {
    connect ( schurMat_->newStructEvent,
              this, & Self::structChanged_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void NeumannPrecon::update_ ()
{
  using jive::util::zeroSlaveDofs;

  // This is to avoid infinite recursions:

  if ( ! events_ )
  {
    return;
  }

  events_ = 0;

  if      ( schurMat_ )
  {
    Constraints*  schurCons = schurMat_->getSchurCons ();
    const idx_t   schurSize = schurMat_->schurSize    ();
    const idx_t   totalSize = schurMat_->totalSize    ();


    weights_ .resize ( schurSize );
    totalRhs_.resize ( totalSize );
    totalLhs_.resize ( totalSize );

    weights_  = 1.0;
    totalRhs_ = 0.0;
    totalLhs_ = 0.0;

    exchanger_->scatter ( weights_ );

    weights_  = 1.0 / weights_;

    if ( schurCons )
    {
      zeroSlaveDofs ( weights_, *schurCons );
    }
  }
  else if ( exchanger_ )
  {
    Constraints*  totalCons = inner_->getConstraints ();
    const idx_t   totalSize = inner_->size           (0);

    weights_ .resize ( totalSize );
    totalRhs_.resize ( totalSize );

    weights_  = 1.0;
    totalRhs_ = 0.0;

    exchanger_->scatter ( weights_ );

    weights_ = 1.0 / weights_;

    if ( totalCons )
    {
      zeroSlaveDofs ( weights_, *totalCons );
    }
  }

  events_ = 0;

  Self::resetEvents   ();

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void NeumannPrecon::valuesChanged_ ()
{
  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void NeumannPrecon::structChanged_ ()
{
  Vector  empty;

  weights_ .ref ( empty );
  totalRhs_.ref ( empty );
  totalLhs_.ref ( empty );

  setEvents_          ( NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   syncEvents_
//-----------------------------------------------------------------------


void NeumannPrecon::syncEvents_ ()
{
  if ( exchanger_ )
  {
    MPContext*  mpx = exchanger_->getMPContext ();

    events_ = allreduce ( *mpx, events_, jem::mp::BOR );
  }

  Self::resetEvents ();
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void NeumannPrecon::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "Neumann preconditioner changed while "
      "solving a linear system of equations"
    );
  }

  events_ |= events;
}


JIVE_END_PACKAGE( solver )
