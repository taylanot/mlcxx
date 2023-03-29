
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
#include <jem/base/array/intrinsics.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/io/PrintWriter.h>
#include <jive/util/error.h>
#include <jive/algebra/DiagMatrixObject.h>


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::DiagMatrixObject );


JIVE_BEGIN_PACKAGE( algebra )


using jem::iarray;
using jive::util::indexError;


//=======================================================================
//   class DiagMatrixObject
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DiagMatrixObject::DiagMatrixObject ( const String& name ) :

  Super ( name )

{}


DiagMatrixObject::DiagMatrixObject

  ( const String&  name,
    const Vector&  diag ) :

    Super   ( name ),
    values_ ( diag )

{}


DiagMatrixObject::~DiagMatrixObject ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void DiagMatrixObject::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, values_ );
}


void DiagMatrixObject::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, values_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> DiagMatrixObject::clone () const
{
  return jem::newInstance<Self> ( myName_, values_.clone() );
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape DiagMatrixObject::shape () const
{
  return Shape ( values_.size(), values_.size() );
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void DiagMatrixObject::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  lhs = values_ * rhs;
}


//-----------------------------------------------------------------------
//   multiMatmul
//-----------------------------------------------------------------------


void DiagMatrixObject::multiMatmul

  ( Matrix&           lhsVecs,
    IdxVector&        lhsTags,
    const Matrix&     rhsVecs,
    const IdxVector&  rhsTags ) const

{
  JEM_PRECHECK2 ( rhsVecs.size(1) == rhsTags.size(),
                  "Array shape mismatch" );

  const idx_t  rhsCount = rhsTags.size ();

  lhsVecs.resize ( rhsVecs.shape() );
  lhsTags.resize ( rhsCount );

  lhsTags = rhsTags;

  for ( idx_t j = 0; j < rhsCount; j++ )
  {
    lhsVecs[j] = values_ * rhsVecs[j];
  }
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool DiagMatrixObject::hasTrait ( const String& trait ) const
{
  if ( trait == MatrixTraits::SYMMETRIC )
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* DiagMatrixObject::getExtByID ( ExtensionID extID ) const
{
  if ( extID == DiagMatrixExt::ID )
  {
    DiagMatrixExt*    ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == MultiMatmulExt::ID )
  {
    MultiMatmulExt*   ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == DirectMatrixExt::ID )
  {
    DirectMatrixExt*  ext = const_cast<Self*> ( this );

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
//   getValue
//-----------------------------------------------------------------------


double DiagMatrixObject::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  const idx_t  msize = values_.size ();

  if ( irow < 0 || irow >= msize )
  {
    indexError ( getContext(), "row", irow, msize );
  }

  if ( jcol < 0 || jcol >= msize )
  {
    indexError ( getContext(), "column", jcol, msize );
  }

  if ( irow == jcol )
  {
    return values_[jcol];
  }
  else
  {
    return 0.0;
  }
}


//-----------------------------------------------------------------------
//   getBlock
//-----------------------------------------------------------------------


void DiagMatrixObject::getBlock

  ( const Matrix&     block,
    const IdxVector&  irows,
    const IdxVector&  jcols ) const

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array shape mismatch" );

  const idx_t  icount = irows.size ();
  const idx_t  jcount = jcols.size ();

  const idx_t  msize  = values_.size ();


  for ( idx_t i = 0; i < icount; i++ )
  {
    const idx_t  irow = irows[i];

    if ( irow < 0 || irow >= msize )
    {
      indexError ( getContext(), "row", irow, msize );
    }
  }

  for ( idx_t j = 0; j < jcount; j++ )
  {
    const idx_t  jcol = jcols[j];

    if ( jcol < 0 || jcol >= msize )
    {
      indexError ( getContext(), "column", jcol, msize );
    }

    for ( idx_t i = 0; i < icount; i++ )
    {
      if ( irows[i] == jcol )
      {
        block(i,j) = values_[jcol];
      }
      else
      {
        block(i,j) = 0.0;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getDiagonal
//-----------------------------------------------------------------------


void DiagMatrixObject::getDiagonal ( const Vector& diag ) const
{
  diag = values_;
}


//-----------------------------------------------------------------------
//   getRowScales
//-----------------------------------------------------------------------


void DiagMatrixObject::getRowScales ( const Vector& rscales ) const
{
  rscales = abs ( values_ );
}


//-----------------------------------------------------------------------
//   getStructure
//-----------------------------------------------------------------------


SparseStruct DiagMatrixObject::getStructure () const
{
  const idx_t  msize = values_.size ();

  IdxVector  offsets ( iarray( msize + 1 ) );
  IdxVector  indices ( iarray( msize )     );

  return SparseStruct ( Shape( msize, msize ), offsets, indices );
}


//-----------------------------------------------------------------------
//   toSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix DiagMatrixObject::toSparseMatrix () const
{
  const idx_t  msize = values_.size ();

  IdxVector  offsets ( iarray( msize + 1 ) );
  IdxVector  indices ( iarray( msize )     );

  return SparseMatrix ( Shape ( msize, msize ),
                        offsets, indices, values_ );
}


//-----------------------------------------------------------------------
//   cloneSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix DiagMatrixObject::cloneSparseMatrix () const
{
  const idx_t  msize = values_.size ();

  IdxVector  offsets ( iarray( msize + 1 ) );
  IdxVector  indices ( iarray( msize )     );

  return SparseMatrix ( Shape ( msize, msize ),
                        offsets, indices, values_.clone() );
}


//-----------------------------------------------------------------------
//   setValues
//-----------------------------------------------------------------------


void DiagMatrixObject::setValues ( const Vector& diag )
{
  if ( & diag != & values_ )
  {
    idx_t  oldSize = values_.size ();
    idx_t  newSize = diag   .size ();

    values_.ref ( diag );

    if ( newSize == oldSize )
    {
      newValuesEvent.emit ( *this );
    }
    else
    {
      newStructEvent.emit ( *this );
    }
  }
}


//-----------------------------------------------------------------------
//   handleNewValues
//-----------------------------------------------------------------------


void DiagMatrixObject::handleNewValues ()
{
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void DiagMatrixObject::printTo ( PrintWriter& out ) const
{
  using jem::io::space;

  const idx_t  msize = values_.size ();

  print ( out, '[' );

  for ( idx_t i = 0; i < msize; i++ )
  {
    print ( out, space, values_[i] );
  }

  print ( out, space, ']' );
}


JIVE_END_PACKAGE( algebra )
