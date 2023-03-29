
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/error.h>
#include <jive/algebra/NullMatrixObject.h>


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::NullMatrixObject );


JIVE_BEGIN_PACKAGE( algebra )


using jive::util::indexError;


//=======================================================================
//   class NullMatrixObject
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NullMatrixObject::NullMatrixObject

  ( const String&  name,
    idx_t          size ) :

    Super ( name ),
    size_ ( size )

{
  JEM_PRECHECK2 ( size >= 0, "invalid matrix size" );
}


NullMatrixObject::~NullMatrixObject ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void NullMatrixObject::readFrom ( ObjectInput& in )
{
  uidx_t  sz;

  decode ( in, myName_, sz );

  size_ = (idx_t) sz;
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void NullMatrixObject::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, size_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> NullMatrixObject::clone () const
{
  return jem::newInstance<NullMatrixObject> ( myName_, size_ );
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape NullMatrixObject::shape () const
{
  return Shape ( size_, size_ );
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void NullMatrixObject::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  lhs = 0.0;
}


//-----------------------------------------------------------------------
//   multiMatmul
//-----------------------------------------------------------------------


void NullMatrixObject::multiMatmul

  ( Matrix&           lhsVecs,
    IdxVector&        lhsTags,
    const Matrix&     rhsVecs,
    const IdxVector&  rhsTags ) const

{
  JEM_PRECHECK2 ( rhsVecs.size(1) == rhsTags.size(),
                  "Array shape mismatch" );

  lhsVecs.resize ( rhsVecs.shape() );
  lhsTags.resize ( rhsTags.size()  );

  lhsVecs = 0.0;
  lhsTags = rhsTags;
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* NullMatrixObject::getExtByID ( ExtensionID  extID ) const
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


double NullMatrixObject::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  if ( irow < 0 || irow >= size_ )
  {
    indexError ( getContext(), "row", irow, size_ );
  }

  if ( jcol < 0 || jcol >= size_ )
  {
    indexError ( getContext(), "column", jcol, size_ );
  }

  return 0.0;
}


//-----------------------------------------------------------------------
//   getBlock
//-----------------------------------------------------------------------


void NullMatrixObject::getBlock

  ( const Matrix&     block,
    const IdxVector&  irows,
    const IdxVector&  jcols ) const

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array shape mismatch" );

  const idx_t  icount = irows.size ();
  const idx_t  jcount = jcols.size ();


  for ( idx_t i = 0; i < icount; i++ )
  {
    const idx_t  irow = irows[i];

    if ( irow < 0 || irow >= size_ )
    {
      indexError ( getContext(), "row", irow, size_ );
    }
  }

  for ( idx_t j = 0; j < jcount; j++ )
  {
    const idx_t  jcol = jcols[j];

    if ( jcol < 0 || jcol >= size_ )
    {
      indexError ( getContext(), "column", jcol, size_ );
    }
  }

  block = 0.0;
}


//-----------------------------------------------------------------------
//   getDiagonal
//-----------------------------------------------------------------------


void NullMatrixObject::getDiagonal ( const Vector& diag ) const
{
  diag = 0.0;
}


//-----------------------------------------------------------------------
//   getRowScales
//-----------------------------------------------------------------------


void NullMatrixObject::getRowScales ( const Vector& rscales ) const
{
  rscales = 0.0;
}


//-----------------------------------------------------------------------
//   getStructure
//-----------------------------------------------------------------------


SparseStruct NullMatrixObject::getStructure () const
{
  IdxVector  offsets ( size_ + 1 );
  IdxVector  indices;

  offsets = 0;

  return SparseStruct ( Shape( size_, size_ ), offsets, indices );
}


//-----------------------------------------------------------------------
//   toSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix NullMatrixObject::toSparseMatrix () const
{
  IdxVector  offsets ( size_ + 1 );
  IdxVector  indices;
  Vector     values;

  offsets = 0;

  return SparseMatrix ( Shape ( size_, size_ ),
                        offsets, indices, values );
}


//-----------------------------------------------------------------------
//   cloneSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix NullMatrixObject::cloneSparseMatrix () const
{
  return Self::toSparseMatrix ();
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void NullMatrixObject::resize ( idx_t n )
{
  JEM_PRECHECK2 ( n >= 0, "invalid matrix size" );

  idx_t  sz = size_;

  size_ = n;

  if ( sz != n )
  {
    newStructEvent.emit ( *this );
  }
}


JIVE_END_PACKAGE( algebra )
