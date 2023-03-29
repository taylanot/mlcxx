
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


#include <jem/pragmas.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/util/error.h>
#include <jive/solver/SparseRestrictor.h>


JEM_DEFINE_CLASS( jive::solver::SparseRestrictor );


JIVE_BEGIN_PACKAGE( solver )


using jive::util::sizeError;


//=======================================================================
//   class SparseRestrictor
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  SparseRestrictor::NEW_VALUES_ = 1 << 0;
const int  SparseRestrictor::NEW_STRUCT_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SparseRestrictor::SparseRestrictor ( const String& name ) :

  Super ( name )

{
  shape_   =  0;
  events_  = ~0x0;
  started_ =  0;
}


SparseRestrictor::~SparseRestrictor ()
{}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape SparseRestrictor::shape () const
{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update ();
  }

  return shape_;
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SparseRestrictor::start ()
{
  if ( ! started_ )
  {
    resetEvents ();
  }

  if ( events_ & NEW_STRUCT_ )
  {
    update ();
  }

  started_++;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SparseRestrictor::finish ()
{
  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SparseRestrictor::update ()
{
  using jem::isTiny;
  using jem::numeric::norm2;

  int  events = events_;

  if ( ! events )
  {
    return;
  }

  // This is to avoid infinite recursions.

  events_ = 0;

  update_ ( events );

  // Normalize the restriction operator and remove zero vectors.

  idx_t  nnz  = 0;
  idx_t  irow = 0;

  for ( idx_t i = 0; i < shape_[0]; i++ )
  {
    idx_t   rbegin = rowOffsets_[i];
    idx_t   rend   = rowOffsets_[i + 1];
    double  rnorm  = norm2 ( rtValues_[slice(rbegin,rend)] );

    rowOffsets_[irow] = nnz;

    if ( ! isTiny( rnorm ) )
    {
      rnorm = 1.0 / rnorm;

      for ( idx_t j = rbegin; j < rend; j++ )
      {
        rtValues_  [nnz] = rtValues_  [j] * rnorm;
        colIndices_[nnz] = colIndices_[j];

        nnz++;
      }

      irow++;
    }
  }

  rowOffsets_[irow] = nnz;

  if ( irow < shape_[0] )
  {
    shape_[0] = irow;

    rowOffsets_.reshape ( irow + 1 );
  }

  if ( nnz < rtValues_.size() )
  {
    rtValues_  .reshape ( nnz );
    colIndices_.reshape ( nnz );
  }

  resetEvents ();
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void SparseRestrictor::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update ();
  }

  const idx_t*  JEM_RESTRICT  idx = colIndices_.addr ();
  const double* JEM_RESTRICT  rtv = rtValues_  .addr ();

  const idx_t   rowCount = shape_[0];
  const idx_t   colCount = shape_[1];


  if ( rhs.size() != colCount )
  {
    sizeError ( JEM_FUNC, "right-hand vector",
                rhs.size(), colCount );
  }

  if ( lhs.size() != rowCount )
  {
    sizeError ( JEM_FUNC, "left-hand vector",
                lhs.size(), rowCount );
  }

  if ( rhs.isContiguous() )
  {
    const double* JEM_RESTRICT  rpt = rhs.addr ();

    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      const idx_t  n = rowOffsets_[irow + 1];
      double       t = 0.0;

      for ( idx_t i = rowOffsets_[irow]; i < n; i++ )
      {
        t += rtv[i] * rpt[idx[i]];
      }

      lhs[irow] = t;
    }
  }
  else
  {
    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      const idx_t  n = rowOffsets_[irow + 1];
      double       t = 0.0;

      for ( idx_t i = rowOffsets_[irow]; i < n; i++ )
      {
        t += rtv[i] * rhs[idx[i]];
      }

      lhs[irow] = t;
    }
  }
}


//-----------------------------------------------------------------------
//   transmul
//-----------------------------------------------------------------------


void SparseRestrictor::transmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update ();
  }

  const idx_t*  JEM_RESTRICT  idx = colIndices_.addr ();
  const double* JEM_RESTRICT  rtv = rtValues_  .addr ();

  const idx_t   rowCount = shape_[0];
  const idx_t   colCount = shape_[1];


  if ( rhs.size() != rowCount )
  {
    sizeError ( JEM_FUNC, "right-hand vector",
                rhs.size(), rowCount );
  }

  if ( lhs.size() != colCount )
  {
    sizeError ( JEM_FUNC, "left-hand vector",
                lhs.size(), colCount );
  }

  lhs = 0.0;

  if ( lhs.isContiguous() )
  {
    double* JEM_RESTRICT  lpt = lhs.addr ();

    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      const idx_t   n = rowOffsets_[irow + 1];
      const double  t = rhs[irow];

JEM_IVDEP

      for ( idx_t i = rowOffsets_[irow]; i < n; i++ )
      {
        lpt[idx[i]] += t * rtv[i];
      }
    }
  }
  else
  {
    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      const idx_t   n = rowOffsets_[irow + 1];
      const double  t = rhs[irow];

JEM_IVDEP

      for ( idx_t i = rowOffsets_[irow]; i < n; i++ )
      {
        lhs[idx[i]] += t * rtv[i];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getBasis
//-----------------------------------------------------------------------


void SparseRestrictor::getBasis

  ( const Vector&  v,
    idx_t          j ) const

{
  using jive::util::indexError;

  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update ();
  }

  if ( j < 0 || j >= shape_[0] )
  {
    indexError ( JEM_FUNC, "basis vector", j,        shape_[0] );
  }

  if ( v.size() != shape_[1] )
  {
    sizeError  ( JEM_FUNC, "basis vector", v.size(), shape_[1] );
  }

  const idx_t*  JEM_RESTRICT  idx = colIndices_.addr ();
  const double* JEM_RESTRICT  rtv = rtValues_  .addr ();

  const idx_t   n = rowOffsets_[j + 1];


  v = 0.0;

  if ( v.isContiguous() )
  {
    double* JEM_RESTRICT  vpt = v.addr ();

JEM_IVDEP

    for ( idx_t i = rowOffsets_[j]; i < n; i++ )
    {
      vpt[idx[i]] = rtv[i];
    }
  }
  else
  {
JEM_IVDEP

    for ( idx_t i = rowOffsets_[j]; i < n; i++ )
    {
      v[idx[i]] = rtv[i];
    }
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* SparseRestrictor::getExtByID ( ExtensionID extID ) const
{
  if ( extID == LocalRestrictor::ID )
  {
    LocalRestrictor*  ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == SparseMatrixExt::ID )
  {
    SparseMatrixExt*  ext = const_cast<Self*> ( this );

    return ext;
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   getStructure
//-----------------------------------------------------------------------


SparseStruct SparseRestrictor::getStructure () const
{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update ();
  }

  return SparseStruct ( shape_, rowOffsets_, colIndices_ );
}


//-----------------------------------------------------------------------
//   toSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix SparseRestrictor::toSparseMatrix () const
{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update ();
  }

  return SparseMatrix ( shape_,      rowOffsets_,
                        colIndices_, rtValues_ );
}


//-----------------------------------------------------------------------
//   cloneSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix SparseRestrictor::cloneSparseMatrix () const
{
  return Self::toSparseMatrix().clone ();
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void SparseRestrictor::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "coarse grid restrictor changed while solving a "
      "linear system of equations"
    );
  }

  events_ |= events;
}


JIVE_END_PACKAGE( solver )
