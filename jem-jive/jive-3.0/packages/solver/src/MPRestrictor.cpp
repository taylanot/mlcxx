
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/util/Event.h>
#include <jem/mp/utilities.h>
#include <jive/util/error.h>
#include <jive/util/DofSpace.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/solver/typedefs.h>
#include <jive/solver/MPRestrictor.h>


JEM_DEFINE_CLASS( jive::solver::MPRestrictor );


JIVE_BEGIN_PACKAGE( solver )


using jive::mp::SCATTER;
using jive::util::sizeError;


//=======================================================================
//   class MPRestrictor
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  MPRestrictor::NEW_VALUES_ = 1 << 0;
const int  MPRestrictor::NEW_STRUCT_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MPRestrictor::MPRestrictor

  ( Ref<Restrictor>       rt,
    Ref<VectorExchanger>  vex ) :

  inner_     ( rt ),
  exchanger_ ( vex )

{
  JEM_PRECHECK ( rt );

  myName_  =  rt->getName ();
  events_  = ~0x0;
  started_ =  0;

  connect_ ();
}


MPRestrictor::~MPRestrictor ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void MPRestrictor::start ()
{
  inner_->start ();

  if ( ! started_ )
  {
    syncEvents_ ();
  }

  if ( events_ )
  {
    update_ ();
  }

  started_++;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void MPRestrictor::finish ()
{
  if ( started_ )
  {
    started_--;
  }

  inner_->finish ();
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MPRestrictor::update ()
{
  inner_->update ();

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
//   updated
//-----------------------------------------------------------------------


bool MPRestrictor::updated () const
{
  return (events_ == 0);
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void MPRestrictor::resetEvents ()
{
  inner_->resetEvents ();

  if ( exchanger_ )
  {
    exchanger_->getDofSpace   ()->resetEvents ();
    exchanger_->getRecvBorders() .resetEvents ();
    exchanger_->getSendBorders() .resetEvents ();
  }
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape MPRestrictor::shape () const
{
  return inner_->shape ();
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void MPRestrictor::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  JEM_PRECHECK ( updated() );

  if ( ! exchanger_ )
  {
    inner_->matmul ( lhs, rhs );
  }
  else
  {
    const idx_t  dofCount = vbuffer_.size ();

    if ( rhs.size() != dofCount )
    {
      sizeError ( JEM_FUNC, "rhs vector", rhs.size(), dofCount );
    }

    vbuffer_ = rhs * weights_;

    inner_->matmul ( lhs, vbuffer_ );
  }
}


//-----------------------------------------------------------------------
//   transmul
//-----------------------------------------------------------------------


void MPRestrictor::transmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  JEM_PRECHECK ( updated() );

  if ( ! exchanger_ )
  {
    inner_->transmul ( lhs, rhs );
  }
  else
  {
    exchanger_->startOne ( SCATTER );
    inner_    ->transmul ( lhs, rhs );

    lhs *= weights_;

    exchanger_->endOne   ( lhs );
  }
}


//-----------------------------------------------------------------------
//   multiTransmul
//-----------------------------------------------------------------------


void MPRestrictor::multiTransmul

  ( Matrix&           lhsVecs,
    IdxVector&        lhsTags,
    const Matrix&     rhsVecs,
    const IdxVector&  rhsTags ) const

{
  using jem::reshape;

  JEM_PRECHECK ( updated() );
  JEM_PRECHECK ( rhsVecs.size(1) == rhsTags.size() );

  const idx_t  rhsCount = rhsTags .size ();
  const idx_t  dofCount = vbuffer_.size ();

  if ( ! exchanger_ )
  {
    lhsVecs.resize ( dofCount, rhsCount );
    lhsTags.resize ( rhsCount );

    lhsTags = rhsTags;

    for ( idx_t j = 0; j < rhsCount; j++ )
    {
      inner_->transmul ( lhsVecs[j], rhsVecs[j] );
    }
  }
  else
  {
    Shape   tmpShape ( dofCount, rhsCount );
    Matrix  tmpVecs;

    exchanger_->startSome ( SCATTER, rhsTags );

    if ( rhsCount == 1 )
    {
      tmpVecs.ref    ( reshape( vbuffer_, tmpShape ) );
    }
    else
    {
      tmpVecs.resize ( tmpShape );
    }

    for ( idx_t j = 0; j < rhsCount; j++ )
    {
      inner_->transmul ( tmpVecs[j], rhsVecs[j] );

      tmpVecs[j] *= weights_;
    }

    exchanger_->endSome ( lhsVecs, lhsTags, tmpVecs, rhsTags );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MPRestrictor::configure ( const Properties& props )
{
  inner_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MPRestrictor::getConfig ( const Properties& conf ) const
{
  inner_->getConfig ( conf );
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool MPRestrictor::hasTrait ( const String& trait ) const
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
//   getExtByID
//-----------------------------------------------------------------------


void* MPRestrictor::getExtByID ( ExtensionID extID ) const
{
  if ( extID == MPMatrixExt::ID )
  {
    MPMatrixExt*  ext = const_cast<Self*> ( this );

    return ext;
  }
  else
  {
    return inner_->getExtByID ( extID );
  }
}


//-----------------------------------------------------------------------
//   getExchanger
//-----------------------------------------------------------------------


VectorExchanger* MPRestrictor::getExchanger () const
{
  return exchanger_.get ();
}


//-----------------------------------------------------------------------
//   getExchangeMode
//-----------------------------------------------------------------------


int MPRestrictor::getExchangeMode () const
{
  return SCATTER;
}


//-----------------------------------------------------------------------
//   getWeights
//-----------------------------------------------------------------------


Vector MPRestrictor::getWeights () const
{
  return weights_;
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void MPRestrictor::connect_ ()
{
  using jem::util::connect;

  connect ( inner_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( inner_->newStructEvent, this, & Self::structChanged_ );

  if ( exchanger_ )
  {
    DofSpace*  dofs = exchanger_->getDofSpace ();

    connect ( dofs->newSizeEvent,  this, & Self::structChanged_ );
    connect ( dofs->newOrderEvent, this, & Self::structChanged_ );

    connect ( exchanger_->getRecvBorders().changeEvent(),
              this, & Self::valuesChanged_ );

    if ( exchanger_->hasOverlap() )
    {
      connect ( exchanger_->getSendBorders().changeEvent(),
                this, & Self::valuesChanged_ );
    }
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void MPRestrictor::update_ ()
{
  int  events = events_;

  // This is to avoid infinite recursions:

  if ( ! events )
  {
    return;
  }

  events_ = 0;

  if ( (events & NEW_STRUCT_) && exchanger_ )
  {
    const DofSpace*  dofs     = exchanger_->getDofSpace ();
    const idx_t      dofCount = dofs      ->dofCount    ();

    if ( dofCount != inner_->size(1) )
    {
      throw jem::IllegalArgumentException (
        getContext (),
        "dof space incompatible with restriction operator "
        "(size mismatch)"
      );
    }

    vbuffer_.resize ( dofCount );
    weights_.resize ( dofCount );

    vbuffer_ = 0.0;
    weights_ = 1.0;

    exchanger_->scatter ( weights_ );

    weights_ = 1.0 / weights_;
  }

  Self::resetEvents ();

  // This is to make sure that all processes emit the same events.
  // Some other classes (CoarseDofSpace and CoarseMatrix)
  // depend on this property; they do not need to execute a global
  // reduction operation to get the same events on all processes.

  if ( events & NEW_STRUCT_ )
  {
    newStructEvent.emit ( *this );
  }

  if ( events & NEW_VALUES_ )
  {
    newValuesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void MPRestrictor::valuesChanged_ ()
{
  setEvents_    ( NEW_VALUES_ );
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void MPRestrictor::structChanged_ ()
{
  Vector  empty;

  vbuffer_.ref ( empty );
  weights_.ref ( empty );

  setEvents_    ( NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   syncEvents_
//-----------------------------------------------------------------------


void MPRestrictor::syncEvents_ ()
{
  if ( exchanger_ )
  {
    // Make sure that all processes have a consistent view of the
    // state of the restriction operators.

    MPContext*  mpx = exchanger_->getMPContext ();

    events_ = allreduce ( *mpx, events_, jem::mp::BOR );
  }

  Self::resetEvents ();
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void MPRestrictor::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "coarse grid restrictor changed while "
      "solving a linear system of equations"
    );
  }

  events_ |= events;
}


JIVE_END_PACKAGE( solver )
