
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
#include <jem/base/RuntimeException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/algebra/ConstrainedMatrix.h>


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::ConstrainedMatrix );


JIVE_BEGIN_PACKAGE( algebra )


using jive::util::sizeError;


//=======================================================================
//   class ConstrainedMatrix
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ConstrainedMatrix::ConstrainedMatrix ()
{
  updated_ = false;
}


ConstrainedMatrix::ConstrainedMatrix

  ( Ref<AbstractMatrix>  inner,
    Ref<Constraints>     cons ) :

    inner_ ( inner ),
    cons_  (  cons )

{
  JEM_PRECHECK2 ( inner && cons,
                  "invalid argument (NULL pointer)" );

  connect_ ();
}


ConstrainedMatrix::~ConstrainedMatrix ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void ConstrainedMatrix::resetEvents ()
{
  inner_->resetEvents ();
  cons_ ->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ConstrainedMatrix::readFrom ( ObjectInput& in )
{
  decode   ( in, inner_, cons_ );
  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ConstrainedMatrix::writeTo ( ObjectOutput& out ) const
{
  encode ( out, inner_, cons_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> ConstrainedMatrix::clone () const
{
  return jem::newInstance<Self> ( inner_, cons_ );
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape ConstrainedMatrix::shape () const
{
  return inner_->shape ();
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void ConstrainedMatrix::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  using jive::util::evalSlaveDofs;
  using jive::util::evalMasterDofs;

  if ( ! updated_ )
  {
    const_cast<Self*>( this )->update ();
  }

  const idx_t  dofCount = tmpRhs_.size ();

  if ( lhs.size() != dofCount )
  {
    sizeError ( getContext(), "lhs vector", lhs.size(), dofCount );
  }

  if ( rhs.size() != dofCount )
  {
    sizeError ( getContext(), "rhs vector", rhs.size(), dofCount );
  }

  tmpRhs_ = rhs;

  evalSlaveDofs  ( tmpRhs_, *cons_, 0.0 );
  inner_->matmul ( lhs, tmpRhs_ );
  evalMasterDofs ( lhs, *cons_  );
}


//-----------------------------------------------------------------------
//   multiMatmul
//-----------------------------------------------------------------------


void ConstrainedMatrix::multiMatmul

  ( Matrix&           lhsVecs,
    IdxVector&        lhsTags,
    const Matrix&     rhsVecs,
    const IdxVector&  rhsTags ) const

{
  JEM_PRECHECK2 ( rhsVecs.size(1) == rhsTags.size(),
                  "Array shape mismatch" );

  using jem::reshape;
  using jive::util::evalSlaveDofs;
  using jive::util::evalMasterDofs;

  if ( ! updated_ )
  {
    const_cast<Self*>( this )->update ();
  }

  MultiMatmulExt*    mmx = inner_->getExtension<MultiMatmulExt> ();

  const idx_t   rhsCount = rhsTags.size ();
  const idx_t   dofCount = tmpRhs_.size ();

  Matrix        tmpVecs;


  if ( rhsCount > 0 && rhsVecs.size(0) != dofCount )
  {
    sizeError ( getContext(), "rhs vector",
                rhsVecs.size(0), dofCount );
  }

  if ( rhsCount == 1 )
  {
    tmpVecs.ref (
      reshape ( tmpRhs_, jem::shape( dofCount, 1_idx ) )
    );
  }
  else
  {
    tmpVecs.resize ( dofCount, rhsCount );
  }

  tmpVecs = rhsVecs;

  for ( idx_t j = 0; j < rhsCount; j++ )
  {
    evalSlaveDofs ( tmpVecs[j], *cons_, 0.0 );
  }

  mmx->multiMatmul ( lhsVecs, lhsTags, tmpVecs, rhsTags );

  for ( idx_t j = 0; j < lhsTags.size(); j++ )
  {
    evalMasterDofs ( lhsVecs[j], *cons_  );
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* ConstrainedMatrix::getExtByID ( ExtensionID extID ) const
{
  if ( extID == MultiMatmulExt::ID &&
       inner_->hasExtension<MultiMatmulExt>() )
  {
    MultiMatmulExt*  ext = const_cast<Self*> ( this );

    return ext;
  }
  else
  {
    return inner_->getExtByID ( extID );
  }
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool ConstrainedMatrix::hasTrait ( const String& trait ) const
{
  using jive::algebra::MatrixTraits;

  if ( trait == MatrixTraits::SYMMETRIC ||
       trait == MatrixTraits::DISTRIBUTED )
  {
    return inner_->hasTrait ( trait );
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void ConstrainedMatrix::update ()
{
  if ( ! updated_ )
  {
    const idx_t  dofCount = cons_ ->dofCount ();
    const Shape  matShape = inner_->shape    ();

    if ( matShape[0] != matShape[1] )
    {
      util::nonSquareMatrixError ( getContext(), matShape );
    }

    if ( dofCount != matShape[0] )
    {
      throw jem::RuntimeException (
        getContext (),
        String::format (
          "matrix size (%d) differs from number of dofs (%d) "
          "(incompatible matrix and constraints)",
          matShape[0],
          dofCount
        )
      );
    }

    tmpRhs_.resize ( dofCount );

    Self::resetEvents ();
  }

  updated_ = true;
}


//-----------------------------------------------------------------------
//   getRhs
//-----------------------------------------------------------------------


void ConstrainedMatrix::getRhs

  ( const Vector&  g,
    const Vector&  f ) const

{
  using jive::util::setSlaveDofs;
  using jive::util::evalMasterDofs;

  JEM_PRECHECK2 ( updated(),
                  "ConstrainedMatrix has not been updated" );
  JEM_PRECHECK2 ( g.size() == f.size(),
                  "Array size mismatch" );

  const idx_t  dofCount = tmpRhs_.size ();

  if ( f.size() != dofCount )
  {
    sizeError ( getContext(), "rhs vector", f.size(), dofCount );
  }

  tmpRhs_ = 0.0;

  setSlaveDofs   ( tmpRhs_, *cons_, -1.0 );
  inner_->matmul ( g, tmpRhs_ );

  g += f;

  evalMasterDofs ( g, *cons_ );
}


//-----------------------------------------------------------------------
//   getLhs
//-----------------------------------------------------------------------


void ConstrainedMatrix::getLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  using jive::util::evalSlaveDofs;

  JEM_PRECHECK2 ( updated(),
                  "ConstrainedMatrix has not been updated" );
  JEM_PRECHECK2 ( x.size() == y.size(),
                  "Array size mismatch" );

  const idx_t  dofCount = tmpRhs_.size ();

  if ( x.size() != dofCount )
  {
    sizeError ( getContext(), "lhs vector", x.size(), dofCount );
  }

  x = y;

  evalSlaveDofs ( x, *cons_ );
}


//-----------------------------------------------------------------------
//   initLhs
//-----------------------------------------------------------------------


void ConstrainedMatrix::initLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  using jive::util::zeroSlaveDofs;

  JEM_PRECHECK2 ( updated(),
                  "ConstrainedMatrix has not been updated" );
  JEM_PRECHECK2 ( x.size() == y.size(),
                  "Array size mismatch" );

  const idx_t  dofCount = tmpRhs_.size ();

  if ( x.size() != dofCount )
  {
    sizeError ( getContext(), "lhs vector", x.size(), dofCount );
  }

  x = y;

  zeroSlaveDofs ( x, *cons_ );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void ConstrainedMatrix::connect_ ()
{
  using jem::util::connect;

  connect ( inner_->newValuesEvent,  this, & Self::valuesChanged_ );
  connect ( inner_->newStructEvent,  this, & Self::structChanged_ );
  connect ( cons_ ->newStructEvent,  this, & Self::structChanged_ );

  myName_  = inner_->getName ();
  updated_ = false;
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void ConstrainedMatrix::structChanged_ ()
{
  updated_ = false;

  newStructEvent.emit ( * this );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void ConstrainedMatrix::valuesChanged_ ()
{
  newValuesEvent.emit ( * this );
}


JIVE_END_PACKAGE( algebra )
