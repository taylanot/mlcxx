
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
#include <jem/base/IllegalOperationException.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/ConDistiller.h>
#include <jive/algebra/ConMatrixBuilder.h>


JEM_DEFINE_CLASS( jive::algebra::ConMatrixBuilder );


JIVE_BEGIN_PACKAGE( algebra )


using jive::util::indexError;


//=======================================================================
//   class ConMatrixBuilder
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ConMatrixBuilder::ConMatrixBuilder

  ( Ref<MatrixBuilder>  mbe,
    Ref<ConDistiller>   cdist ) :

    mbe_   ( mbe   ),
    cdist_ ( cdist )

{
  using jem::util::connect;

  JEM_PRECHECK2 ( mbe && cdist,
                  "invalid argument (NULL pointer)" );

  DofSpace*  dofs  = cdist->getDofSpace    ();
  DofSpace*  rdofs = cdist->getReducedDofs ();

  myName_ = mbe->getName ();

  connect ( dofs ->newSizeEvent, this, & Self::invalidate_ );
  connect ( rdofs->newSizeEvent, this, & Self::invalidate_ );

  doUpdate_ ();
}


ConMatrixBuilder::~ConMatrixBuilder ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ConMatrixBuilder::clear ()
{
  mbe_->clear     ();
  mbe_->shapeHint ( beSize_, beSize_ );
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void ConMatrixBuilder::scale ( double factor )
{
  mbe_->scale ( factor );
}


//-----------------------------------------------------------------------
//   setToZero
//-----------------------------------------------------------------------


void ConMatrixBuilder::setToZero ()
{
  mbe_->setToZero ();
  mbe_->shapeHint ( beSize_, beSize_ );
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void ConMatrixBuilder::reserve ( idx_t nz )
{
  if ( mySize_ > 0 )
  {
    double  scale = (double) beSize_ / (double) mySize_;

    mbe_->reserve ( (idx_t) (scale * (double) nz) );
  }
}


//-----------------------------------------------------------------------
//   shapeHint
//-----------------------------------------------------------------------


void ConMatrixBuilder::shapeHint

  ( idx_t  rowCount,
    idx_t  colCount )

{
  mbe_->shapeHint ( beSize_, beSize_ );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void ConMatrixBuilder::trimToSize ()
{
  mbe_->trimToSize ();
}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void ConMatrixBuilder::updateMatrix ()
{
  mbe_->updateMatrix ();
}


//-----------------------------------------------------------------------
//   setMultiplier
//-----------------------------------------------------------------------


void ConMatrixBuilder::setMultiplier ( double mult )
{
  mbe_->setMultiplier ( mult );
}


//-----------------------------------------------------------------------
//   getMultiplier
//-----------------------------------------------------------------------


double ConMatrixBuilder::getMultiplier () const
{
  return mbe_->getMultiplier ();
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void ConMatrixBuilder::setData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  values )

{
  Scratch_&  s = scratch_;

  if ( icount * jcount == 1_idx )
  {
    constrain_  ( irows[0], jcols[0], values[0] );
  }
  else
  {
    constrain_  ( irows, icount, jcols, jcount, values );
  }

  mbe_->setData ( s.irows.addr(), s.irows.size(),
                  s.jcols.addr(), s.jcols.size(),
                  s.block.addr() );
}


//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------


void ConMatrixBuilder::addData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  values )

{
  Scratch_&  s = scratch_;

  if ( icount * jcount == 1_idx )
  {
    constrain_  ( irows[0], jcols[0], values[0] );
  }
  else
  {
    constrain_  ( irows, icount, jcols, jcount, values );
  }

  mbe_->addData ( s.irows.addr(), s.irows.size(),
                  s.jcols.addr(), s.jcols.size(),
                  s.block.addr() );
}


//-----------------------------------------------------------------------
//   eraseData
//-----------------------------------------------------------------------


idx_t ConMatrixBuilder::eraseData

  ( const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount )

{
  Scratch_&  s = scratch_;


  if ( icount * jcount == 1_idx )
  {
    constrain_ ( irows[0], jcols[0], 0.0 );
  }
  else
  {
    Matrix  dummy ( icount, jcount );

    dummy = 0.0;

    constrain_ ( irows, icount, jcols, jcount, dummy.addr() );
  }

  return mbe_->eraseData ( s.irows.addr(), s.irows.size(),
                           s.jcols.addr(), s.jcols.size() );
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void ConMatrixBuilder::getData

  ( double*       buf,
    const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount ) const

{
  readError_ ();
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* ConMatrixBuilder::getMatrix () const
{
  return mbe_->getMatrix ();
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


inline void ConMatrixBuilder::update_ ()
{
  cdist_->update ();

  if ( ! updated_ )
  {
    doUpdate_ ();
  }
}


//-----------------------------------------------------------------------
//   doUpdate_
//-----------------------------------------------------------------------


void ConMatrixBuilder::doUpdate_ ()
{
  Scratch_&  s     = scratch_;

  DofSpace*  dofs  = cdist_->getDofSpace    ();
  DofSpace*  rdofs = cdist_->getReducedDofs ();


  mySize_ = dofs ->dofCount ();
  beSize_ = rdofs->dofCount ();

  if ( s.ibuf.size() != beSize_ )
  {
    s.ibuf.resize ( beSize_ );
    s.accu.resize ( beSize_ );

    s.accu = 0.0;
  }

  updated_ = true;

  dofs ->resetEvents ();
  rdofs->resetEvents ();
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void ConMatrixBuilder::invalidate_ ()
{
  updated_ = false;
}


//-----------------------------------------------------------------------
//   constrain_ (a single value)
//-----------------------------------------------------------------------


void ConMatrixBuilder::constrain_

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  update_ ();

  Scratch_&      s = scratch_;

  const idx_t*   offsets = cdist_->getOffsetPtr ();
  const idx_t*   indices = cdist_->getIndexPtr  ();
  const double*  coeffs  = cdist_->getCoeffPtr  ();

  double         tmp;

  idx_t          i, j, ii, jj;
  idx_t          p, q;


  if ( irow < 0 || irow >= mySize_ )
  {
    indexError ( getContext(), "row",    irow, mySize_ );
  }

  if ( jcol < 0 || jcol >= mySize_ )
  {
    indexError ( getContext(), "column", jcol, mySize_ );
  }

  ii = offsets[irow];
  jj = offsets[jcol];
  p  = offsets[irow + 1] - ii;
  q  = offsets[jcol + 1] - jj;

  if ( s.irows.size() != p )
  {
    s.irows.resize ( p );
  }

  if ( s.jcols.size() != q )
  {
    s.jcols.resize ( q );
  }

  if ( s.block.size(0) != p || s.block.size(1) != q )
  {
    s.block.resize ( p, q );
  }

  for ( i = 0; i < p; i++ )
  {
    s.irows[i] = indices[ii + i];
  }

  for ( j = 0; j < q; j++ )
  {
    s.jcols[j] =         indices[jj + j];
    tmp        = value * coeffs [jj + j];

    for ( i = 0; i < p; i++ )
    {
      s.block(i,j) = tmp * coeffs[ii + i];
    }
  }
}


//-----------------------------------------------------------------------
//   constrain_ (a matrix)
//-----------------------------------------------------------------------


void ConMatrixBuilder::constrain_

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  values )

{
  update_ ();

  Scratch_&      s = scratch_;

  const idx_t*   offsets = cdist_->getOffsetPtr ();
  const idx_t*   indices = cdist_->getIndexPtr  ();
  const double*  coeffs  = cdist_->getCoeffPtr  ();

  idx_t*         ibuf    = s.ibuf.addr ();
  double*        accu    = s.accu.addr ();

  double         tmp;

  idx_t          irow, jcol;
  idx_t          i, j, k;
  idx_t          p, q, r;


  // Find the constrained row indices

  p = 0;

  for ( i = 0; i < icount; i++ )
  {
    irow = irows[i];

    if ( irow < 0 || irow >= mySize_ )
    {
      indexError ( getContext(), "row", irow, mySize_ );
    }

    r = offsets[irow + 1];

    for ( k = offsets[irow]; k < r; k++ )
    {
      irow = indices[k];

      if ( accu[irow] < 1.0 )
      {
        accu[irow] = 2.0;
        ibuf[p++]  = irow;
      }
    }
  }

  if ( s.irows.size() != p )
  {
    s.irows.resize ( p );
  }

  for ( i = 0; i < p; i++ )
  {
    irow       = ibuf[i];
    s.irows[i] = irow;
    accu[irow] = 0.0;
  }

  // Find the constrained columns indices

  q = 0;

  for ( j = 0; j < jcount; j++ )
  {
    jcol = jcols[j];

    if ( jcol < 0 || jcol >= mySize_ )
    {
      indexError ( getContext(), "column", jcol, mySize_ );
    }

    r = offsets[jcol + 1];

    for ( k = offsets[jcol]; k < r; k++ )
    {
      jcol = indices[k];

      if ( accu[jcol] < 1.0 )
      {
        accu[jcol] = 2.0;
        ibuf[q++]  = jcol;
      }
    }
  }

  if ( s.jcols.size() != q )
  {
    s.jcols.resize ( q );
  }

  for ( j = 0; j < q; j++ )
  {
    jcol       = ibuf[j];
    s.jcols[j] = jcol;
    accu[jcol] = 0.0;
  }

  if ( s.mbuf.size(0) != icount || s.mbuf.size(1) != q )
  {
    s.mbuf.resize ( icount, q );
  }

  if ( s.block.size(0) != p || s.block.size(1) != q )
  {
    s.block.resize ( p, q );
  }

  if ( p * q == 0 )
  {
    return;
  }

  // Compute s.mbuf = block * T, with T the constraint matrix

  for ( i = 0; i < icount; i++, values++ )
  {
    for ( j = 0; j < jcount; j++ )
    {
      tmp  = values [j * icount];
      jcol = jcols  [j];
      r    = offsets[jcol + 1];

      for ( k = offsets[jcol]; k < r; k++ )
      {
        jcol        =       indices[k];
        accu[jcol] += tmp * coeffs [k];
      }
    }

    for ( j = 0; j < q; j++ )
    {
      jcol        = s.jcols[j];
      s.mbuf(i,j) = accu[jcol];
      accu[jcol]  = 0.0;
    }
  }

  // Compute s.block = T.transpose() * s.mbuf, which is equivalent
  // with computing s.block.transpose() = s.mbuf.transpose() * T.

  for ( j = 0; j < q; j++ )
  {
    for ( i = 0; i < icount; i++ )
    {
      tmp  = s.mbuf(i,j);
      irow = irows[i];
      r    = offsets[irow + 1];

      for ( k = offsets[irow]; k < r; k++ )
      {
        irow        =       indices[k];
        accu[irow] += tmp * coeffs [k];
      }
    }

    for ( i = 0; i < p; i++ )
    {
      irow         = s.irows[i];
      s.block(i,j) = accu[irow];
      accu[irow]   = 0.0;
    }
  }
}


//-----------------------------------------------------------------------
//   readError_
//-----------------------------------------------------------------------


void ConMatrixBuilder::readError_ () const
{
  throw jem::IllegalOperationException (
    getContext (),
    "reading of matrix values is not supported"
  );
}


JIVE_END_PACKAGE( algebra )
