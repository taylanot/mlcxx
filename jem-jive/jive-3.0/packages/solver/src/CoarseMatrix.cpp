
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
#include <jem/base/array/operators.h>
#include <jem/mp/utilities.h>
#include <jem/util/Event.h>
#include <jive/algebra/MultiMatmulExtension.h>
#include <jive/solver/MPRestrictor.h>
#include <jive/solver/CoarseDofSpace.h>
#include <jive/solver/CoarseMatrix.h>


JEM_DEFINE_CLASS( jive::solver::CoarseMatrix );


JIVE_BEGIN_PACKAGE( solver )


using jem::dynamicCast;


//=======================================================================
//   class CoarseMatrix
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


CoarseMatrix::CoarseMatrix

  ( const String&        name,
    Ref<AbstractMatrix>  inner,
    Ref<CoarseDofSpace>  dofs ) :

    Super       ( name ),
    inner_      ( inner ),
    coarseDofs_ ( dofs )

{
  JEM_PRECHECK ( inner && dofs );
  JEM_PRECHECK ( inner->isDistributed() == dofs->isDistributed() );

  rtor_ = dofs->getRestrictor ();

  connect_ ();
}


CoarseMatrix::~CoarseMatrix ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void CoarseMatrix::start ()
{
  rtor_      ->start ();
  coarseDofs_->start ();
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void CoarseMatrix::finish ()
{
  rtor_      ->finish ();
  coarseDofs_->finish ();
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void CoarseMatrix::update ()
{
  rtor_      ->update ();
  coarseDofs_->update ();
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void CoarseMatrix::resetEvents ()
{
  rtor_ ->resetEvents ();
  inner_->resetEvents ();
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape CoarseMatrix::shape () const
{
  const idx_t  size = rtor_->size (0);

  return Shape ( size, size );
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void CoarseMatrix::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  const idx_t  n = inner_->size (0);

  if ( n != totalRhs_.size() )
  {
    Self*  self = const_cast<Self*> ( this );

    self->totalRhs_.resize ( n );
    self->totalLhs_.resize ( n );
  }

  rtor_ ->transmul ( totalRhs_, rhs );
  inner_->matmul   ( totalLhs_, totalRhs_ );
  rtor_ ->matmul   ( lhs,       totalLhs_ );
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool CoarseMatrix::hasTrait ( const String& trait ) const
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
//   assemble
//-----------------------------------------------------------------------


idx_t CoarseMatrix::assemble ( Matrix& cmat ) const
{
  if ( inner_->hasExtension<MultiMatmulExt>() )
  {
    if ( dynamicCast<MPRestrictor*>( rtor_ ) )
    {
      return assembleMPX_ ( cmat );
    }
    else
    {
      return assembleMMX_ ( cmat );
    }
  }
  else
  {
    return assemble_ ( cmat );
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void CoarseMatrix::connect_ ()
{
  using jem::util::connect;

  connect ( rtor_ ->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( rtor_ ->newStructEvent, this, & Self::structChanged_ );
  connect ( inner_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( inner_->newStructEvent, this, & Self::valuesChanged_ );
  connect ( coarseDofs_
            ->newSizeEvent,         this, & Self::structChanged_ );
}


//-----------------------------------------------------------------------
//   assemble_
//-----------------------------------------------------------------------


idx_t CoarseMatrix::assemble_ ( Matrix& cmat ) const
{
  const idx_t  localSize  = coarseDofs_->dofCount       ();
  const idx_t  globalSize = coarseDofs_->globalDofCount ();
  const idx_t  firstDof   = coarseDofs_->getMyFirstDof  ();

  Vector       localVec   ( localSize );


  cmat.resize ( localSize, globalSize );

  cmat     = 0.0;
  localVec = 0.0;

  for ( idx_t jcol = 0; jcol < globalSize; jcol++ )
  {
    idx_t  j = jcol - firstDof;

    if ( j >= 0 && j < localSize )
    {
      localVec[j] = 1.0;
    }

    matmul ( cmat[jcol], localVec );

    if ( j >= 0 && j < localSize )
    {
      localVec[j] = 0.0;
    }
  }

  return globalSize;
}


//-----------------------------------------------------------------------
//   assembleMMX_
//-----------------------------------------------------------------------


idx_t CoarseMatrix::assembleMMX_ ( Matrix& cmat ) const
{
  using jem::min;
  using jem::Slice;

  MultiMatmulExt*     mmx = inner_->getExtension<MultiMatmulExt> ();

  const idx_t  BATCH_SIZE = 4;

  const idx_t  totalSize  = inner_     ->size           (0);
  const idx_t  localSize  = coarseDofs_->dofCount       ();
  const idx_t  globalSize = coarseDofs_->globalDofCount ();
  const idx_t  firstDof   = coarseDofs_->getMyFirstDof  ();

  Matrix       totalRhs   ( totalSize, BATCH_SIZE );
  Matrix       totalLhs   ( totalSize, BATCH_SIZE );
  Vector       localVec   ( localSize );

  IdxVector    rhsTags    ( BATCH_SIZE );
  IdxVector    lhsTags    ( BATCH_SIZE );


  cmat.resize ( localSize, globalSize );

  localVec = 0.0;

  for ( idx_t iouter = 0; iouter < globalSize; iouter += BATCH_SIZE )
  {
    idx_t  n = min ( BATCH_SIZE, globalSize - iouter );

    for ( idx_t j = 0; j < n; j++ )
    {
      idx_t  jcol = iouter + j;
      idx_t  idof = jcol   - firstDof;

      if ( idof >= 0 && idof < localSize )
      {
        localVec[idof] = 1.0;
      }

      rhsTags[j] = jcol;

      rtor_->transmul ( totalRhs[j], localVec );

      if ( idof >= 0 && idof < localSize )
      {
        localVec[idof] = 0.0;
      }
    }

    Slice  s = slice ( 0, n );

    mmx->multiMatmul ( totalLhs,    lhsTags,
                       totalRhs[s], rhsTags[s] );

    JEM_ASSERT ( lhsTags.size() == n );

    for ( idx_t j = 0; j < n; j++ )
    {
      idx_t  jcol = iouter + j;

      rtor_->matmul ( cmat[jcol], totalLhs[j] );
    }
  }

  return globalSize;
}


//-----------------------------------------------------------------------
//   assembleMPX_
//-----------------------------------------------------------------------


idx_t CoarseMatrix::assembleMPX_ ( Matrix& cmat ) const
{
  using jem::min;
  using jem::max;
  using jem::Slice;
  using jem::mp::allmax;

  MultiMatmulExt*     mmx = inner_->getExtension<MultiMatmulExt> ();
  MPRestrictor*       mpr = dynamicCast<MPRestrictor*>    ( rtor_ );
  MPContext*          mpx = coarseDofs_->getMPContext            ();

  const idx_t  BATCH_SIZE = 4;

  const idx_t  localSize  = coarseDofs_->dofCount       ();
  const idx_t  globalSize = coarseDofs_->globalDofCount ();
  const idx_t  firstDof   = coarseDofs_->getMyFirstDof  ();

  IdxVector    localTags  ( BATCH_SIZE );
  Matrix       localVecs  ( localSize, BATCH_SIZE );
  Vector       localVec   = localVecs[0];

  IdxVector    totalTags1, totalTags2;
  Matrix       totalVecs1, totalVecs2;

  Slice        s;

  idx_t        innerCount;
  idx_t        outerCount;


  innerCount= outerCount = localSize;

  if ( mpx )
  {
    outerCount = allmax ( *mpx, localSize );
  }

  cmat.resize ( localSize, globalSize );

  cmat = 0.0;

  for ( idx_t iouter = 0; iouter < outerCount; iouter += BATCH_SIZE )
  {
    idx_t  n = min ( BATCH_SIZE, localSize - iouter );

    n = max   ( 0_idx, n );
    s = slice ( 0, n );

    if ( n > 0 )
    {
      localVecs[s] = 0.0;

      for ( idx_t j = 0; j < n; j++ )
      {
        idx_t  i = iouter + j;

        localVecs(i,j) = 1.0;
        localTags[j]   = i + firstDof;
      }
    }

    mpr->multiTransmul ( totalVecs1,   totalTags1,
                         localVecs[s], localTags[s] );

    if ( iouter == 0 && mpx )
    {
      innerCount = allmax ( *mpx, totalTags1.size() );
    }

    for ( idx_t iinner = 0; iinner < innerCount; iinner += BATCH_SIZE )
    {
      n = min ( BATCH_SIZE, totalTags1.size() - iinner );

      if ( n <= 0 )
      {
        s = slice ( 0, 0 );
      }
      else
      {
        s = slice ( iinner, iinner + n );
      }

      mmx->multiMatmul ( totalVecs2,    totalTags2,
                         totalVecs1[s], totalTags1[s] );

      for ( idx_t j = 0; j < totalTags2.size(); j++ )
      {
        idx_t  jcol = totalTags2[j];

        mpr->matmul ( localVec, totalVecs2[j] );

        cmat[jcol] += localVec;
      }
    }
  }

  return (outerCount * (innerCount / BATCH_SIZE));
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void CoarseMatrix::valuesChanged_ ()
{
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void CoarseMatrix::structChanged_ ()
{
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( solver )
