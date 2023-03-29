
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


#include <cmath>
#include <jem/base/ClassTemplate.h>
#include <jem/base/tuple/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/io/PrintWriter.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/util/error.h>
#include <jive/algebra/utilities.h>
#include <jive/algebra/DenseMatrixObject.h>


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::DenseMatrixObject );


JIVE_BEGIN_PACKAGE( algebra )


using jive::util::sizeError;
using jive::util::indexError;


//=======================================================================
//   class DenseMatrixObject
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  DenseMatrixObject::SYMMETRIC = 1 << 0;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DenseMatrixObject::DenseMatrixObject

  ( const String&  name,
    int            traits ) :

    Super   ( name   ),
    traits_ ( traits )

{}


DenseMatrixObject::DenseMatrixObject

  ( const String&  name,
    const Matrix&  mat,
    int            traits ) :

    Super   ( name   ),
    matrix_ ( mat    ),
    traits_ ( traits )

{}


DenseMatrixObject::~DenseMatrixObject ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void DenseMatrixObject::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, matrix_, traits_ );
}


void DenseMatrixObject::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, matrix_, traits_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> DenseMatrixObject::clone () const
{
  return jem::newInstance<Self> ( myName_, matrix_.clone(),
                                  traits_ );
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape DenseMatrixObject::shape () const
{
  return matrix_.shape ();
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void DenseMatrixObject::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  jem::numeric::matmul ( lhs, matrix_, rhs );
}


//-----------------------------------------------------------------------
//   multiMatmul
//-----------------------------------------------------------------------


void DenseMatrixObject::multiMatmul

  ( Matrix&           lhsVecs,
    IdxVector&        lhsTags,
    const Matrix&     rhsVecs,
    const IdxVector&  rhsTags ) const

{
  JEM_PRECHECK2 ( rhsVecs.size(1) == rhsTags.size(),
                  "Array shape mismatch" );

  const idx_t  rhsCount = rhsTags.size ();

  lhsVecs.resize ( matrix_.size(0), rhsCount );
  lhsTags.resize ( rhsCount );

  if ( rhsCount > 0 )
  {
    lhsTags = rhsTags;

    jem::numeric::matmul ( lhsVecs, matrix_, rhsVecs );
  }
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool DenseMatrixObject::hasTrait ( const String& trait ) const
{
  if ( trait == MatrixTraits::SYMMETRIC )
  {
    return (traits_ & SYMMETRIC);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* DenseMatrixObject::getExtByID ( ExtensionID extID ) const
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


double DenseMatrixObject::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  if ( irow < 0 || irow >= matrix_.size(0) )
  {
    indexError ( getContext(), "row", irow, matrix_.size(0) );
  }

  if ( jcol < 0 || jcol >= matrix_.size(1) )
  {
    indexError ( getContext(), "column", jcol, matrix_.size(1) );
  }

  return matrix_(irow,jcol);
}


//-----------------------------------------------------------------------
//   getBlock
//-----------------------------------------------------------------------


void DenseMatrixObject::getBlock

  ( const Matrix&     block,
    const IdxVector&  irows,
    const IdxVector&  jcols ) const

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array shape mismatch" );

  const idx_t  icount   = irows.size ();
  const idx_t  jcount   = jcols.size ();

  const idx_t  rowCount = matrix_.size(0);
  const idx_t  colCount = matrix_.size(1);


  for ( idx_t i = 0; i < icount; i++ )
  {
    const idx_t  irow = irows[i];

    if ( irow < 0 || irow >= rowCount )
    {
      indexError ( getContext(), "row", irow, rowCount );
    }
  }

  for ( idx_t j = 0; j < jcount; j++ )
  {
    const idx_t  jcol = jcols[j];

    if ( jcol < 0 || jcol >= colCount )
    {
      indexError ( getContext(), "column", jcol, colCount );
    }

    for ( idx_t i = 0; i < icount; i++ )
    {
      block(i,j) = matrix_(irows[i],jcol);
    }
  }
}


//-----------------------------------------------------------------------
//   getDiagonal
//-----------------------------------------------------------------------


void DenseMatrixObject::getDiagonal ( const Vector& diag ) const
{
  const idx_t  diagSize = jem::min ( matrix_.shape() );


  if ( diag.size() != diagSize )
  {
    sizeError ( getContext(), "diagonal vector",
                diag.size(), diagSize );
  }

  for ( idx_t i = 0; i < diagSize; i++ )
  {
    diag[i] = matrix_(i,i);
  }
}


//-----------------------------------------------------------------------
//   getRowScales
//-----------------------------------------------------------------------


void DenseMatrixObject::getRowScales ( const Vector& rscales ) const
{
  const idx_t  rowCount = matrix_.size(0);
  const idx_t  colCount = matrix_.size(1);


  if ( rscales.size() != rowCount )
  {
    sizeError ( getContext(), "row scale vector",
		rscales.size(), rowCount );
  }

  for ( idx_t i = 0; i < rowCount; i++ )
  {
    rscales[i] = 0.0;

    for ( idx_t j = 0; j < colCount; j++ )
    {
      double  s = std::fabs ( matrix_(i,j) );

      if ( s > rscales[i] )
      {
        rscales[i] = s;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getStructure
//-----------------------------------------------------------------------


SparseStruct DenseMatrixObject::getStructure () const
{
  return denseToSparse ( matrix_.size(0), matrix_.size(1) );
}


//-----------------------------------------------------------------------
//   toSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix DenseMatrixObject::toSparseMatrix () const
{
  return denseToSparse ( matrix_ );
}


//-----------------------------------------------------------------------
//   cloneSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix DenseMatrixObject::cloneSparseMatrix () const
{
  return denseToSparse ( matrix_.clone() );
}


//-----------------------------------------------------------------------
//   setMatrix
//-----------------------------------------------------------------------


void DenseMatrixObject::setMatrix

  ( const Matrix&  mat,
    int            traits )

{
  traits_ = traits;

  if ( & mat != & matrix_ )
  {
    Shape  oldShape = matrix_.shape ();
    Shape  newShape = mat    .shape ();

    matrix_.ref ( mat );

    if ( jem::equal( oldShape, newShape ) )
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


void DenseMatrixObject::handleNewValues ()
{
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void DenseMatrixObject::printTo ( PrintWriter& out ) const
{
  using jem::io::endl;
  using jem::io::space;

  const idx_t  rowCount = matrix_.size(0);
  const idx_t  colCount = matrix_.size(1);


  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    if ( irow > 0 )
    {
      print ( out, endl );
    }

    print ( out, '[' );

    for ( idx_t jcol = 0; jcol < colCount; jcol++ )
    {
      print ( out, space, matrix_(irow,jcol) );
    }

    if ( colCount > 0 )
    {
      print ( out, space );
    }

    print ( out, ']' );
  }
}


JIVE_END_PACKAGE( algebra )
