
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
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/base/array/select.h>
#include <jem/base/array/operators.h>
#include <jem/util/Event.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/util/error.h>
#include <jive/util/Constraints.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/solver/typedefs.h>
#include <jive/solver/Preconditioner.h>
#include <jive/solver/SchurDofSpace.h>
#include <jive/solver/SchurMatrix.h>


JEM_DEFINE_CLASS( jive::solver::SchurMatrix );


JIVE_BEGIN_PACKAGE( solver )


using jem::numeric::axpy;
using jive::util::sizeError;
using jive::util::zeroSlaveDofs;
using jive::util::evalSlaveDofs;
using jive::util::evalMasterDofs;


//=======================================================================
//   class SchurMatrix
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  SchurMatrix::NEW_CONS_   = 1 << 0;
const int  SchurMatrix::NEW_STRUCT_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SchurMatrix::SchurMatrix

  ( const String&        name,
    Ref<MPContext>       mpx,
    Ref<AbstractMatrix>  inner,
    Ref<Preconditioner>  precon,
    Ref<SchurDofSpace>   dofs,
    Ref<Constraints>     cons ) :

    Super      ( name   ),
    inner_     ( inner  ),
    precon_    ( precon ),
    schurDofs_ ( dofs   ),
    totalCons_ ( cons   )

{
  using jem::dynamicCast;
  using jem::newInstance;
  using jem::IllegalArgumentException;
  using jive::mp::EXCHANGE;
  using jive::algebra::MPMatrixObj;

  JEM_PRECHECK ( mpx    &&
                 inner  &&
                 precon &&
                 dofs   );

  MPMatrixObj*  mpMat = dynamicCast<MPMatrixObj*> ( inner );

  if ( mpMat )
  {
    inner_ = mpMat->getInner        ();
    xmode_ = mpMat->getExchangeMode ();
  }
  else
  {
    xmode_ = EXCHANGE;
  }

  if ( inner_->isDistributed() )
  {
    throw IllegalArgumentException (
      getContext (),
      "distributed inner matrix"
    );
  }

  innerCons_ = precon->getConstraints ();

  if ( innerCons_ && totalCons_ )
  {
    if ( innerCons_->getDofSpace() != totalCons_->getDofSpace() )
    {
      throw IllegalArgumentException (
        getContext (),
        "inner and total constraints are associated with "
        "different DOF spaces"
      );
    }
  }

  if ( totalCons_ )
  {
    if ( totalCons_->getDofSpace() != dofs->getTotalDofSpace() )
    {
      throw IllegalArgumentException (
        getContext (),
        "constraints are associated with the wrong DOF space"
      );
    }

    schurCons_ = newInstance<Constraints> ( dofs );
  }

  exchanger_ = newInstance<VectorExchanger>

    ( mpx,
      dofs,
      dofs->getRecvBorders(),
      dofs->getSendBorders() );

  events_  = NEW_STRUCT_;
  started_ = 0;

  if ( totalCons_ )
  {
    events_ |= NEW_CONS_;
  }

  connect_ ();
}


SchurMatrix::~SchurMatrix ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SchurMatrix::start ()
{
  precon_   ->start ();
  schurDofs_->start ();

  if ( ! started_ )
  {
    resetEvents ();
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


void SchurMatrix::finish ()
{
  precon_   ->finish ();
  schurDofs_->finish ();

  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SchurMatrix::update ()
{
  precon_   ->update ();
  schurDofs_->update ();

  if ( ! started_ )
  {
    resetEvents ();
  }

  if ( events_ )
  {
    update_ ();
  }
}


//-----------------------------------------------------------------------
//   updated
//-----------------------------------------------------------------------


bool SchurMatrix::updated () const
{
  return (events_ == 0);
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SchurMatrix::resetEvents ()
{
  inner_    ->resetEvents ();
  precon_   ->resetEvents ();
  schurDofs_->resetEvents ();

  if ( totalCons_ )
  {
    totalCons_->resetEvents ();
  }
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape SchurMatrix::shape () const
{
  const idx_t  size = this->schurSize ();

  return Shape ( size, size );
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void SchurMatrix::matmul

  ( const Vector&  schurLhs,
    const Vector&  schurRhs ) const

{
  if ( events_ )
  {
    const_cast<Self*>( this )->update_ ();
  }

  exchanger_->startOne ( xmode_ );
  localMatmul_         ( schurLhs, schurRhs );
  exchanger_->endOne   ( schurLhs );

  if ( schurCons_ )
  {
    evalMasterDofs ( schurLhs, *schurCons_ );
  }
}


//-----------------------------------------------------------------------
//   multiMatmul
//-----------------------------------------------------------------------


void SchurMatrix::multiMatmul

  ( Matrix&           lhsVecs,
    IdxVector&        lhsTags,
    const Matrix&     rhsVecs,
    const IdxVector&  rhsTags ) const

{
  JEM_PRECHECK ( rhsVecs.size(1) == rhsTags.size() );

  if ( events_ )
  {
    const_cast<Self*>( this )->update_ ();
  }

  Matrix  tmpVecs ( rhsVecs.shape() );

  exchanger_->startSome ( xmode_, rhsTags );

  for ( idx_t j = 0; j < rhsTags.size(); j++ )
  {
    localMatmul_ ( tmpVecs[j], rhsVecs[j] );
  }

  exchanger_->endSome ( lhsVecs, lhsTags, tmpVecs, rhsTags );

  if ( schurCons_ )
  {
    for ( idx_t j = 0 ; j < lhsTags.size(); j++ )
    {
      evalMasterDofs ( lhsVecs[j], *schurCons_ );
    }
  }
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool SchurMatrix::hasTrait ( const String& trait ) const
{
  using jive::algebra::MatrixTraits;

  if      ( trait == MatrixTraits::SYMMETRIC )
  {
    return inner_->hasTrait ( trait );
  }
  else if ( trait == MatrixTraits::DISTRIBUTED )
  {
    return exchanger_->isDistributed ();
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* SchurMatrix::getExtByID ( ExtensionID extID ) const
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

  return 0;
}


//-----------------------------------------------------------------------
//   getExchanger
//-----------------------------------------------------------------------


VectorExchanger* SchurMatrix::getExchanger () const
{
  return exchanger_.get ();
}


//-----------------------------------------------------------------------
//   getExchangeMode
//-----------------------------------------------------------------------


int SchurMatrix::getExchangeMode () const
{
  return xmode_;
}


//-----------------------------------------------------------------------
//   setExchangeMode
//-----------------------------------------------------------------------


void SchurMatrix::setExchangeMode ( int mode )
{
  if ( mode != xmode_ )
  {
    xmode_ = mode;

    newValuesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   schurSize
//-----------------------------------------------------------------------


idx_t SchurMatrix::schurSize () const
{
  if ( events_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return schurDofs_->dofCount ();
}


//-----------------------------------------------------------------------
//   totalSize
//-----------------------------------------------------------------------


idx_t SchurMatrix::totalSize () const
{
  if ( events_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return vbuf_.size (0);
}


//-----------------------------------------------------------------------
//   getSchurRhs
//-----------------------------------------------------------------------


void SchurMatrix::getSchurRhs

  ( const Vector&  schurRhs,
    const Vector&  totalRhs ) const

{
  JEM_PRECHECK ( updated() );

  IdxVector    innerDofs  = schurDofs_->getInnerDofs  ();
  IdxVector    borderDofs = schurDofs_->getBorderDofs ();

  const idx_t  schurSize  = borderDofs.size ();
  const idx_t  totalSize  = vbuf_     .size (0);

  Vector       a = vbuf_[0];
  Vector       b = vbuf_[1];


  if ( schurRhs.size() != schurSize )
  {
    sizeError ( getContext(), "Schur rhs vector",
                schurRhs.size(), schurSize );
  }

  if ( totalRhs.size() != totalSize )
  {
    sizeError ( getContext(), "full rhs vector",
                totalRhs.size(), totalSize );
  }

  exchanger_->startOne ( xmode_ );

  a             = -totalRhs;
  a[borderDofs] = 0.0;

  precon_->matmul ( b, a );

  if ( totalCons_ )
  {
    evalSlaveDofs ( b, *totalCons_, 0.0 );

    if ( totalCons_->masterDofCount() > 0 )
    {
      b[borderDofs] = 0.0;
    }
  }

  inner_->matmul ( a, b );

  schurRhs = a[borderDofs];

  exchanger_->endOne ( schurRhs );

  if ( schurCons_ )
  {
    evalMasterDofs ( schurRhs, *schurCons_ );
  }

  schurRhs += totalRhs[borderDofs];
}


//-----------------------------------------------------------------------
//   getTotalLhs
//-----------------------------------------------------------------------


void SchurMatrix::getTotalLhs

  ( const Vector&  totalLhs,
    const Vector&  schurLhs,
    const Vector&  totalRhs ) const

{
  JEM_PRECHECK ( updated() && totalLhs.size() == totalRhs.size() );

  IdxVector    innerDofs  = schurDofs_->getInnerDofs  ();
  IdxVector    borderDofs = schurDofs_->getBorderDofs ();

  const idx_t  schurSize  = borderDofs.size ();
  const idx_t  totalSize  = vbuf_     .size (0);

  Vector       a = vbuf_[0];
  Vector       b = vbuf_[1];


  if ( totalLhs.size() != totalSize )
  {
    sizeError ( getContext(), "full rhs vector",
                totalLhs.size(), totalSize );
  }

  if ( schurLhs.size() != schurSize )
  {
    sizeError ( getContext(), "Schur lhs vector",
                schurLhs.size(), schurSize );
  }

  if ( schurCons_ )
  {
    evalSlaveDofs ( schurLhs, *schurCons_, 0.0 );
  }

  a             =  0.0;
  a[borderDofs] = -schurLhs;

  if ( schurCons_ )
  {
    zeroSlaveDofs ( schurLhs, *schurCons_ );
  }

  inner_->matmul ( b, a );

  b[borderDofs] = 0.0;

  if ( totalCons_ )
  {
    evalMasterDofs ( b, *totalCons_ );

    if ( totalCons_->masterDofCount() > 0 )
    {
      b[borderDofs] = 0.0;
    }
  }

  b[innerDofs] += totalRhs[innerDofs];

  precon_->matmul ( totalLhs, b );

  totalLhs[borderDofs] = schurLhs;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SchurMatrix::configure ( const Properties& props )
{
  precon_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SchurMatrix::getConfig ( const Properties& conf ) const
{
  precon_->getConfig ( conf );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void SchurMatrix::connect_ ()
{
  using jem::util::connect;

  connect ( inner_ ->newValuesEvent,  this, & Self::valuesChanged_ );
  connect ( inner_ ->newStructEvent,  this, & Self::structChanged_ );
  connect ( precon_->newValuesEvent,  this, & Self::valuesChanged_ );
  connect ( precon_->newStructEvent,  this, & Self::structChanged_ );
  connect ( schurDofs_->newSizeEvent, this, & Self::structChanged_ );

  if ( totalCons_ )
  {
    connect ( totalCons_->newStructEvent,
              this, & Self::consChanged_ );

    connect ( totalCons_->newRvaluesEvent,
              this, & Self::consChanged_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void SchurMatrix::update_ ()
{
  int  events = events_;

  // This is to make sure that this function is not called recursively
  // because of some event that is emitted by the inner matrix or
  // another object.

  events_ = 0;

  if ( ! events )
  {
    return;
  }

  if ( events & NEW_STRUCT_ )
  {
    DofSpace*  totalDofs = schurDofs_->getTotalDofSpace ();

    vbuf_.resize ( totalDofs->dofCount(), 2 );

    if ( totalCons_ )
    {
      events |= NEW_CONS_;
    }
  }

  if ( (events & NEW_CONS_) && totalCons_ )
  {
    updateCons_ ();
  }

  events_ = 0;

  Self::resetEvents ();
}


//-----------------------------------------------------------------------
//   updateCons_
//-----------------------------------------------------------------------


void SchurMatrix::updateCons_ ()
{
  if ( innerCons_ )
  {
    IdxVector  borderDofs = schurDofs_->getBorderDofs ();

    if ( innerCons_ != totalCons_ )
    {
      innerCons_->copy ( *totalCons_ );
    }

    for ( idx_t i = 0; i < borderDofs.size(); i++ )
    {
      idx_t  idof = borderDofs[i];

      if ( ! innerCons_->isSlaveDof( idof ) )
      {
        innerCons_->addConstraint ( idof );
      }
    }
  }

  if ( schurCons_ )
  {
    const idx_t  slaveCount = totalCons_->slaveDofCount ();

    IdxVector    dofMap     = schurDofs_->getDofMap     ();

    BoolVector   rvalMask   ( slaveCount );
    IdxVector    slaveDofs  ( slaveCount );

    IdxVector    conOffsets;
    IdxVector    dofIndices;
    Vector       conCoeffs;

    idx_t        k, n;


    schurCons_->clear ();

    totalCons_->getSlaveDofs  ( slaveDofs );
    totalCons_->getRvalueMask ( rvalMask, slaveDofs );

    k = n = 0;

    for ( idx_t i = 0; i < slaveCount; i++ )
    {
      idx_t  idof = slaveDofs[i];

      if ( dofMap[idof] >= 0 )
      {
        idx_t  m = totalCons_->masterDofCount ( idof );

        k += 1;
        n += m;
      }
    }

    conOffsets.resize ( k + 1 );
    dofIndices.resize ( k + n );
    conCoeffs .resize ( k + n );

    k = n = 0;

    for ( idx_t i = 0; i < slaveCount; i++ )
    {
      idx_t  idof = slaveDofs[i];

      if ( dofMap[idof] >= 0 )
      {
        idx_t  m = totalCons_->masterDofCount ( idof );
        idx_t  p = n + m + 1;

        rvalMask  [k] = rvalMask[i];
        conOffsets[k] = n;
        dofIndices[n] = dofMap[idof];

        totalCons_->getConstraint ( conCoeffs [n],
                                    dofIndices[slice(n + 1,p)],
                                    conCoeffs [slice(n + 1,p)],
                                    idof );

        k++; n++;

        for ( idx_t j = n; j < p; j++ )
        {
          idx_t  jdof = dofMap[dofIndices[j]];

          if ( jdof >= 0 )
          {
            dofIndices[n] = jdof;
            conCoeffs [n] = conCoeffs[j];
            n++;
          }
        }
      }
    }

    conOffsets[k] = n;

    rvalMask  .reshape ( k );
    dofIndices.reshape ( n );
    conCoeffs .reshape ( n );

    schurCons_->setConstraints (
      SparseMatrix ( jem::shape ( conOffsets.size() - 1,
                                  schurDofs_->dofCount() ),
                     conOffsets,
                     dofIndices,
                     conCoeffs ),
      rvalMask
    );
  }
}


//-----------------------------------------------------------------------
//   localMatmul_
//-----------------------------------------------------------------------


void SchurMatrix::localMatmul_

  ( const Vector&  schurLhs,
    const Vector&  schurRhs ) const

{
  JEM_ASSERT ( updated() );

  IdxVector    innerDofs  = schurDofs_->getInnerDofs  ();
  IdxVector    borderDofs = schurDofs_->getBorderDofs ();

  const idx_t  schurSize  = borderDofs.size ();

  Vector       a = vbuf_[0];
  Vector       b = vbuf_[1];


  if ( schurLhs.size() != schurSize )
  {
    sizeError ( getContext(), "Schur lhs vector",
                schurLhs.size(), schurSize );
  }

  if ( schurRhs.size() != schurSize )
  {
    sizeError ( getContext(), "Schur rhs vector",
                schurRhs.size(), schurSize );
  }

  schurLhs = schurRhs;

  if ( schurCons_ )
  {
    evalSlaveDofs ( schurLhs, *schurCons_, 0.0 );
  }

  a             = 0.0;
  a[borderDofs] = schurLhs;

  inner_->matmul ( b, a );

  schurLhs = b[borderDofs];

  b[borderDofs] = 0.0;

  if ( totalCons_ )
  {
    evalMasterDofs ( b, *totalCons_ );

    if ( totalCons_->masterDofCount() > 0 )
    {
      b[borderDofs] = 0.0;
    }
  }

  precon_->matmul ( a, b );

  if ( totalCons_ )
  {
    evalSlaveDofs ( a, *totalCons_, 0.0 );

    if ( totalCons_->masterDofCount() > 0 )
    {
      a[borderDofs] = 0.0;
    }
  }

  inner_->matmul ( b, a );

  schurLhs -= b[borderDofs];
}


//-----------------------------------------------------------------------
//   consChanged_
//-----------------------------------------------------------------------


void SchurMatrix::consChanged_ ()
{
  setEvents_          ( NEW_CONS_ );
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void SchurMatrix::valuesChanged_ ()
{
  setEvents_          ( 0 );
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void SchurMatrix::structChanged_ ()
{
  setEvents_          ( NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void SchurMatrix::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "Schur matrix structure changed while solving a "
      "linear system of equations"
    );
  }

  events_ |= events;
}


JIVE_END_PACKAGE( solver )
