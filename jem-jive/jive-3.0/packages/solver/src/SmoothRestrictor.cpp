
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


#include <jem/base/limits.h>
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/util/Event.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/mp/utilities.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MultiMatmulExtension.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/solver/LocalRestrictor.h>
#include <jive/solver/SmoothRestrictor.h>


JEM_DEFINE_CLASS( jive::solver::SmoothRestrictor );


JIVE_BEGIN_PACKAGE( solver )


using jem::IllegalArgumentException;


//=======================================================================
//   class SmoothRestrictor
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SmoothRestrictor::SmoothRestrictor

  ( Ref<Restrictor>       inner,
    Ref<AbstractMatrix>   matrix,
    Ref<Constraints>      cons,
    Ref<VectorExchanger>  vex ) :

    inner_     ( inner ),
    matrix_    ( matrix ),
    cons_      ( cons ),
    exchanger_ ( vex )

{
  JEM_PRECHECK ( inner && matrix );

  myName_ = inner->getName ();

  if ( ! inner->getExtension<LocalRestrictor>() )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      inner->getContext() + " is not a local restrictor"
    );
  }

  if ( ! matrix->hasExtension<SparseMatrixExt>() )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      matrix->getContext() +
      " does not implement the sparse matrix extension"
    );
  }

  if ( matrix->isDistributed() )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      getContext() + " does not support distributed matrices"
    );
  }

  connect_ ();
}


SmoothRestrictor::~SmoothRestrictor ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SmoothRestrictor::start ()
{
  inner_->start ();
  Super ::start ();
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SmoothRestrictor::finish ()
{
  inner_->finish ();
  Super ::finish ();
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SmoothRestrictor::update ()
{
  inner_->update ();
  Super ::update ();
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SmoothRestrictor::resetEvents ()
{
  matrix_->resetEvents ();

  if ( cons_ )
  {
    cons_->resetEvents ();
  }

  if ( exchanger_ )
  {
    exchanger_->getRecvBorders().resetEvents ();
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SmoothRestrictor::configure ( const Properties& props )
{
  inner_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SmoothRestrictor::getConfig ( const Properties& conf ) const
{
  inner_->getConfig ( conf );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void SmoothRestrictor::connect_ ()
{
  using jem::util::connect;
  using jive::mp::BorderSet;

  inner_->resetEvents ();

  connect ( inner_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( inner_->newStructEvent, this, & Self::structChanged_ );

  if ( exchanger_ )
  {
    BorderSet  borders = exchanger_->getRecvBorders ();

    borders.resetEvents ();

    connect ( borders.changeEvent(), this, & Self::valuesChanged_ );
  }

  if ( cons_ )
  {
    cons_->resetEvents ();

    connect ( cons_->newStructEvent, this, & Self::valuesChanged_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void SmoothRestrictor::update_ ( int events )
{
  using jem::min;
  using jem::System;
  using jive::util::evalSlaveDofs;
  using jive::util::evalMasterDofs;

  LocalRestrictor*  rt  = inner_ ->getExtension<LocalRestrictor> ();
  MultiMatmulExt*   mmx = matrix_->getExtension<MultiMatmulExt>  ();

  const idx_t  BATCH_SIZE = 4;

  const idx_t  basisSize  = inner_->size  (0);
  const idx_t  dofCount   = inner_->size  (1);

  Vector       diag       = getDiagonal_   ();
  IdxVector    border     = getBorderDofs_ ();

  Matrix       rhsVecs    ( dofCount, BATCH_SIZE );
  Matrix       lhsVecs    ( dofCount, BATCH_SIZE );

  IdxVector    rhsTags    ( BATCH_SIZE );
  IdxVector    lhsTags    ( BATCH_SIZE );


  print ( System::debug( myName_ ), myName_,
          " : smoothing restriction operator ...\n" );

  if ( dofCount != matrix_->size(0) )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "matrix size and restriction operator size do not match"
    );
  }

  basis_.resize ( dofCount, basisSize );

  for ( idx_t iouter = 0; iouter < basisSize; iouter += BATCH_SIZE )
  {
    idx_t  n = min ( BATCH_SIZE, basisSize - iouter );

    for ( idx_t i = 0; i < n; i++ )
    {
      Vector  v = rhsVecs[i];
      idx_t   j = iouter + i;

      rt->getBasis ( v, j );

      v[border]  = 0.0;
      basis_ [j] = v;
      rhsTags[i] = j;

      if ( cons_ )
      {
        evalSlaveDofs ( v, *cons_, 0.0 );
      }
    }

    if ( mmx )
    {
      mmx->multiMatmul ( lhsVecs, lhsTags,
                         rhsVecs[slice(BEGIN,n)],
                         rhsTags[slice(BEGIN,n)] );
    }
    else
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        matrix_->matmul ( lhsVecs[i], rhsVecs[i] );
      }
    }

    for ( idx_t i = 0; i < n; i++ )
    {
      Vector  v = lhsVecs[i];
      idx_t   j = iouter + i;

      if ( cons_ )
      {
        evalMasterDofs ( v, *cons_ );
      }

      basis_[j] -= diag * v;
    }
  }

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getDiagonal_
//-----------------------------------------------------------------------


Vector SmoothRestrictor::getDiagonal_ () const
{
  using jem::isTiny;
  using jem::IllegalInputException;

  SparseMatrix  sm    =

    matrix_->getExtension<SparseMatrixExt>()->toSparseMatrix ();

  const idx_t   rowCount = sm.size (0);
  const idx_t   colCount = sm.size (1);

  IdxVector     offsets  = sm.getRowOffsets    ();
  IdxVector     indices  = sm.getColumnIndices ();
  Vector        values   = sm.getValues        ();

  Vector        diag     ( rowCount );
  Vector        cnorm    ( colCount );


  if ( rowCount != colCount )
  {
    throw IllegalInputException (
      getContext (),
      matrix_->getContext() + " is not square"
    );
  }

  cnorm = 0.0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t   n = offsets[irow + 1];
    double  d = 0.0;

    for ( idx_t j = offsets[irow]; j < n; j++ )
    {
      idx_t   jcol = indices[j];
      double  t    = values [j];

      cnorm[jcol] += t * t;

      if ( jcol == irow )
      {
        d = t;
      }
    }

    diag[irow] = d;
  }

  for ( idx_t jcol = 0; jcol < colCount; jcol++ )
  {
    if ( isTiny( cnorm[jcol] ) )
    {
      diag[jcol] = 1.0;
    }
    else
    {
      diag[jcol] /= cnorm[jcol];
    }
  }

  return diag;
}


//-----------------------------------------------------------------------
//   getBorderDofs_
//-----------------------------------------------------------------------


IdxVector SmoothRestrictor::getBorderDofs_ () const
{
  using jem::find;
  using jive::mp::BorderSet;
  using jive::mp::markBorderDofs;

  IdxVector  borderDofs;

  if ( exchanger_ )
  {
    BorderSet    borders  = exchanger_->getRecvBorders ();
    DofSpace*    dofs     = exchanger_->getDofSpace    ();
    const idx_t  dofCount = dofs      ->dofCount       ();

    BoolVector   mask ( dofCount );

    mask = false;

    markBorderDofs ( mask, borders, *dofs );
    find ( borderDofs, mask );
  }

  return borderDofs;
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void SmoothRestrictor::valuesChanged_ ()
{
  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void SmoothRestrictor::structChanged_ ()
{
  setEvents_          ( NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( solver )
