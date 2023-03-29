
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
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Event.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/mp/utilities.h>
#include <jive/log.h>
#include <jive/util/error.h>
#include <jive/util/DofSpace.h>
#include <jive/mp/ItemMask.h>
#include <jive/algebra/MPVectorSpace.h>


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::MPVectorSpace );


JIVE_BEGIN_PACKAGE( algebra )


using jem::newInstance;
using jem::mp::SUM;
using jem::mp::allsum;
using jem::mp::RecvBuffer;
using jem::mp::SendBuffer;
using jive::util::sizeError;


//=======================================================================
//   class MPVectorSpace
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MPVectorSpace::MPVectorSpace ()
{}


MPVectorSpace::MPVectorSpace

  ( const Ref<Super>&      inner,
    const Ref<MPContext>&  mpx,
    const Ref<ItemMask>&   mask ) :

    vspace_ ( inner ),
    mpx_    ( mpx   ),
    mask_   ( mask  )

{
  using jem::dynamicCast;
  using jem::System;
  using jem::io::Writer;

  JEM_PRECHECK2 ( inner && mpx,
                  "invalid argument (NULL pointer)" );

  Writer&  debug = System::debug      ( CLASS_NAME );
  Self*    self  = dynamicCast<Self*> ( vspace_ );

  if ( self )
  {
    vspace_ = self->vspace_;
  }

  if ( vspace_->isDistributed() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "attempt to wrap a distributed vector space "
      "in an MPVectorSpace"
    );
  }

  if ( mask )
  {
    DofSpace*  dofs = vspace_->getDofSpace ();

    if ( mask->getItems() != dofs->getItems() )
    {
      throw jem::IllegalArgumentException (
        CLASS_NAME,
        "incompatible item mask and dof space (different item sets)"
      );
    }
  }

  print ( debug, "Wrapping vector space (",
          vspace_->toString(), ") in an MP vector space\n" );

  print ( debug, "Synchronizing all processes ...\n" );

  debug.flush ();

  // Make sure that all processes get to this point. This is to catch
  // programming mistakes as early as possible.

  mpx->barrier ();

  print ( debug, "OK, all processes synchronized\n\n" );

  connect_ ();
}


MPVectorSpace::~MPVectorSpace ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void MPVectorSpace::readFrom ( ObjectInput& in )
{
  decode   ( in, vspace_, mpx_, mask_ );
  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void MPVectorSpace::writeTo ( ObjectOutput& out ) const
{
  encode ( out, vspace_, mpx_, mask_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> MPVectorSpace::clone () const
{
  return newInstance<Self> ( vspace_, mpx_, mask_ );
}


//-----------------------------------------------------------------------
//   globalSize
//-----------------------------------------------------------------------


idx_t MPVectorSpace::globalSize () const
{
  idx_t  sz = vspace_->size ();

  return allsum ( *mpx_, sz );
}


//-----------------------------------------------------------------------
//   isDistributed
//-----------------------------------------------------------------------


bool MPVectorSpace::isDistributed () const
{
  return (mpx_->size() > 1);
}


//-----------------------------------------------------------------------
//   product
//-----------------------------------------------------------------------


double MPVectorSpace::product

  ( const Vector&  x,
    const Vector&  y ) const

{
  double  p;

  if ( ! mask_ )
  {
    p = vspace_->product ( x, y );
  }
  else
  {
    JEM_PRECHECK2 ( x.size() == y.size(), "Array size mismatch" );

    update_ ();

    if ( x.size() != weights_.size() )
    {
      sizeError ( JEM_FUNC, "vector", x.size(), weights_.size() );
    }

    vbuf_ = weights_ * x;
    p     = vspace_->product ( vbuf_, y );
  }

  return allsum ( *mpx_, p );
}


//-----------------------------------------------------------------------
//   products
//-----------------------------------------------------------------------


void MPVectorSpace::products

  ( const Vector&  a,
    const Matrix&  x,
    const Matrix&  y ) const

{
  const idx_t  n = a.size ();

  if ( n != rbuf_.size() )
  {
    Self*  self = const_cast<Self*> ( this );

    self->rbuf_.resize ( n );
    self->sbuf_.resize ( n );
  }

  if ( ! mask_ )
  {
    vspace_->products ( sbuf_, x, y );
  }
  else
  {
    JEM_PRECHECK2 ( a.size()  == x.size(1) &&
                    x.size(0) == y.size(0) &&
                    x.size(1) == y.size(1),
                    "Array shape mismatch" );

    update_ ();

    if ( x.size(0) != weights_.size() )
    {
      sizeError ( JEM_FUNC, "vector", x.size(0), weights_.size() );
    }

    for ( idx_t i = 0; i < n; i++ )
    {
      vbuf_    = weights_ * x[i];
      sbuf_[i] = vspace_->product ( vbuf_, y[i] );
    }
  }

  mpx_->allreduce ( RecvBuffer ( rbuf_.addr(), n ),
                    SendBuffer ( sbuf_.addr(), n ), SUM );

  a = rbuf_;
}


//-----------------------------------------------------------------------
//   project
//-----------------------------------------------------------------------


void MPVectorSpace::project

  ( const Vector&  a,
    const Vector&  x,
    const Matrix&  y ) const

{
  const idx_t  n = a.size ();

  if ( n != rbuf_.size() )
  {
    Self*  self = const_cast<Self*> ( this );

    self->rbuf_.resize ( a.size() );
    self->sbuf_.resize ( a.size() );
  }

  if ( ! mask_ )
  {
    vspace_->project ( sbuf_, x, y );
  }
  else
  {
    JEM_PRECHECK2 ( a.size() >= y.size(1),
                    "coefficient array is too small" );
    JEM_PRECHECK2 ( x.size() == y.size(0),
                    "Array shape mismatch" );

    update_ ();

    if ( x.size() != weights_.size() )
    {
      sizeError ( JEM_FUNC, "vector", x.size(), weights_.size() );
    }

    vbuf_ = weights_ * x;

    vspace_->project   ( sbuf_, vbuf_, y );
  }

  mpx_->allreduce ( RecvBuffer ( rbuf_.addr(), n ),
                    SendBuffer ( sbuf_.addr(), n ), SUM );

  a = rbuf_;
}


//-----------------------------------------------------------------------
//   getDofSpace
//-----------------------------------------------------------------------


DofSpace* MPVectorSpace::getDofSpace () const
{
  return vspace_->getDofSpace ();
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void MPVectorSpace::connect_ ()
{
  using jem::util::connect;

  if ( ! mask_ )
  {
    updated_ = true;
  }
  else
  {
    DofSpace*  dofs = vspace_->getDofSpace ();

    updated_ = false;

    connect ( mask_->newValuesEvent, this, & Self::invalidate_ );
    connect ( dofs ->newSizeEvent,   this, & Self::invalidate_ );
    connect ( dofs ->newOrderEvent,  this, & Self::invalidate_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


inline void MPVectorSpace::update_ () const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update2_ ();
  }
}


//-----------------------------------------------------------------------
//   update2_
//-----------------------------------------------------------------------


void MPVectorSpace::update2_ ()
{
  using jem::iarray;

  JEM_PRECHECK2 ( mask_, "NULL ItemMask" );

  DofSpace*    dofs     = vspace_->getDofSpace ();

  const idx_t  dofCount = dofs->dofCount  ();

  BoolVector   mvalues  = mask_->toVector ();

  IdxVector    iitems  ( dofCount );
  IdxVector    itypes  ( dofCount );
  IdxVector    idofs   ( dofCount );

  idx_t        iitem;
  idx_t        idof;


  print ( JIVE_LOG_DEBUG, CLASS_NAME,
          " : initializing weight vector ...\n" );

  weights_.resize ( dofCount );
  vbuf_   .resize ( dofCount );

  idofs = iarray  ( dofCount );

  dofs->decodeDofIndices ( iitems, itypes, idofs );

  for ( idof = 0; idof < dofCount; idof++ )
  {
    iitem = iitems[idof];

    if ( mvalues[iitem] )
    {
      weights_[idof] = 1.0;
    }
    else
    {
      weights_[idof] = 0.0;
    }
  }

  updated_ = true;

  mask_->resetEvents ();
  dofs ->resetEvents ();
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void MPVectorSpace::invalidate_ ()
{
  updated_ = false;
}


JIVE_END_PACKAGE( algebra )
