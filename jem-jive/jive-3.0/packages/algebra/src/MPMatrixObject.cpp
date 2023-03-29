
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/typedefs.h>
#include <jive/algebra/MPMatrixObject.h>


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::MPMatrixObject );


JIVE_BEGIN_PACKAGE( algebra )


using jem::checkedCast;
using jem::newInstance;
using jive::mp::EXCHANGE;
using jive::mp::SCATTER;


//=======================================================================
//   class MPMatrixObject
//=======================================================================
//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MPMatrixObject::MPMatrixObject ()
{}


MPMatrixObject::MPMatrixObject

  ( Ref<AbstractMatrix>   inner,
    Ref<VectorExchanger>  vex ) :

    inner_     ( inner ),
    exchanger_ ( vex   )

{
  using jem::System;
  using jem::io::Writer;

  JEM_PRECHECK2 ( inner && vex,
                  "invalid argument (NULL pointer)" );

  Writer&  debug = System::debug ( inner->getName() );
  Self*    self  = jem::dynamicCast<Self*> ( inner );

  if ( self )
  {
    inner_ = self->getInner ();
  }

  myName_ = inner_->getName ();

  if ( inner_->isDistributed() )
  {
    throw jem::IllegalArgumentException (
      getContext (),
      "attempt to wrap a distributed matrix in an MPMatrix"
    );
  }

  print ( debug, "Wrapping matrix `", myName_,
          "\' in an MP matrix\n" );

  print ( debug, myName_, " : synchronizing all processes ...\n" );

  debug.flush ();

  // Make sure that all processes get to this point. This is to catch
  // programming mistakes as early as possible.

  vex->barrier ();

  print ( debug, myName_, " : OK, all processes synchronized\n\n" );

  xmode_ = EXCHANGE;

  connect_ ();
}


MPMatrixObject::MPMatrixObject ( const Self& rhs )
{
  inner_     = checkedCast<AbstractMatrix> ( rhs.inner_->clone() );
  exchanger_ = rhs.exchanger_;
  xmode_     = rhs.xmode_;

  connect_ ();
}


MPMatrixObject::~MPMatrixObject ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void MPMatrixObject::resetEvents ()
{
  inner_->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void MPMatrixObject::readFrom ( ObjectInput& in )
{
  decode   ( in, inner_, exchanger_, xmode_ );
  connect_ ();
}


void MPMatrixObject::writeTo ( ObjectOutput& out ) const
{
  encode ( out, inner_, exchanger_, xmode_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> MPMatrixObject::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape MPMatrixObject::shape () const
{
  return inner_->shape ();
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void MPMatrixObject::matmul

  ( const Vector& lhs,
    const Vector& rhs ) const

{
  exchanger_->startOne ( xmode_ );
  inner_    ->matmul   ( lhs, rhs );
  exchanger_->endOne   ( lhs );
}


//-----------------------------------------------------------------------
//   multiMatmul
//-----------------------------------------------------------------------


void MPMatrixObject::multiMatmul

  ( Matrix&           lhsVecs,
    IdxVector&        lhsTags,
    const Matrix&     rhsVecs,
    const IdxVector&  rhsTags ) const

{
  JEM_PRECHECK2 ( rhsVecs.size(1) == rhsTags.size(),
                  "Array shape mismatch" );

  MultiMatmulExt*  mmx = inner_->getExtension<MultiMatmulExt> ();

  Matrix           tmpVecs;


  exchanger_->startSome ( xmode_, rhsTags );

  if ( mmx )
  {
    mmx->multiMatmul ( tmpVecs, lhsTags, rhsVecs, rhsTags );
  }
  else
  {
    tmpVecs.resize ( rhsVecs.shape() );

    for ( idx_t j = 0; j < rhsTags.size(); j++ )
    {
      inner_->matmul ( tmpVecs[j], rhsVecs[j] );
    }
  }

  exchanger_->endSome ( lhsVecs, lhsTags, tmpVecs, rhsTags );
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool MPMatrixObject::hasTrait ( const String& trait ) const
{
  if ( trait == MatrixTraits::DISTRIBUTED )
  {
    return exchanger_->isDistributed ();
  }
  else
  {
    return inner_->hasTrait ( trait );
  }
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void MPMatrixObject::printTo ( PrintWriter& out ) const
{
  inner_->printTo ( out );
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* MPMatrixObject::getExtByID ( ExtensionID extID ) const
{
  if ( extID == MPMatrixExt::ID )
  {
    MPMatrixExt*     ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == MultiMatmulExt::ID )
  {
    MultiMatmulExt*  ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == DiagMatrixExt::ID &&
       inner_->hasExtension<DiagMatrixExt>() )
  {
    DiagMatrixExt*   ext = const_cast<Self*> ( this );

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


VectorExchanger* MPMatrixObject::getExchanger () const
{
  return exchanger_.get ();
}


//-----------------------------------------------------------------------
//   getExchangeMode
//-----------------------------------------------------------------------


int MPMatrixObject::getExchangeMode () const
{
  return xmode_;
}


//-----------------------------------------------------------------------
//   setExchangeMode
//-----------------------------------------------------------------------


void MPMatrixObject::setExchangeMode ( int mode )
{
  if ( mode != xmode_ )
  {
    xmode_ = mode;

    newValuesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   getDiagonal
//-----------------------------------------------------------------------


void MPMatrixObject::getDiagonal ( const Vector& diag ) const
{
  DiagMatrixExt*  ext = inner_->getExtension<DiagMatrixExt> ();

  exchanger_->startOne    ( xmode_ );
  ext       ->getDiagonal ( diag   );
  exchanger_->endOne      ( diag   );
}


//-----------------------------------------------------------------------
//   getRowScales
//-----------------------------------------------------------------------


void MPMatrixObject::getRowScales ( const Vector& rscales ) const
{
  DiagMatrixExt*  ext = inner_->getExtension<DiagMatrixExt> ();

  exchanger_->startOne     ( xmode_  );
  ext       ->getRowScales ( rscales );
  exchanger_->endOne       ( rscales );

  if ( ! exchanger_->hasOverlap() || xmode_ == SCATTER )
  {
    Vector  w ( rscales.size() );

    w = 1.0;

    exchanger_->exchange ( w );

    rscales /= w;
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void MPMatrixObject::connect_ ()
{
  using jem::util::connect;

  connect ( inner_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( inner_->newStructEvent, this, & Self::structChanged_ );

  myName_ = inner_->getName ();
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void MPMatrixObject::valuesChanged_ ()
{
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void MPMatrixObject::structChanged_ ()
{
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( algebra )
