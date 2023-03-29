
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
#include <jem/base/array/utilities.h>
#include <jive/algebra/MatrixBuilder.h>


JEM_DEFINE_CLASS( jive::algebra::MatrixBuilder );


JIVE_BEGIN_PACKAGE( algebra )


using jem::makeContiguous;


//=======================================================================
//   class MatrixBuilder
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MatrixBuilder::MatrixBuilder ( const String& name ) :

  Super ( name )

{}


MatrixBuilder::~MatrixBuilder ()
{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String MatrixBuilder::getContext () const
{
  return NamedObject::makeContext ( "matrix builder", myName_ );
}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void MatrixBuilder::configure ( const Properties& props )
{}


void MatrixBuilder::getConfig ( const Properties& conf ) const
{}


//-----------------------------------------------------------------------
//   setToZero
//-----------------------------------------------------------------------


void MatrixBuilder::setToZero ()
{
  scale ( 0.0 );
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void MatrixBuilder::reserve ( idx_t )
{}


//-----------------------------------------------------------------------
//   shapeHint
//-----------------------------------------------------------------------


void MatrixBuilder::shapeHint

  ( idx_t  rowCount,
    idx_t  colCount )

{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void MatrixBuilder::trimToSize ()
{}


//-----------------------------------------------------------------------
//   setBlock
//-----------------------------------------------------------------------


void MatrixBuilder::setBlock

  ( const IdxVector&  irows,
    const IdxVector&  jcols,
    const Matrix&     block )

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array shape mismatch" );

  const idx_t  m = irows.size ();
  const idx_t  n = jcols.size ();


  if ( block.stride(0) + irows.stride() +
       jcols.stride() == 3_idx &&  block.stride(1) == m )
  {
    setData ( irows.addr(), m, jcols.addr(), n, block.addr() );
  }
  else
  {
    Matrix     bb = makeContiguous ( block );
    IdxVector  ii = makeContiguous ( irows );
    IdxVector  jj = makeContiguous ( jcols );

    setData ( ii.addr(), m, jj.addr(), n, bb.addr() );
  }
}


//-----------------------------------------------------------------------
//   addBlock
//-----------------------------------------------------------------------


void MatrixBuilder::addBlock

  ( const IdxVector&  irows,
    const IdxVector&  jcols,
    const Matrix&     block )

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array shape mismatch" );

  const idx_t  m = irows.size ();
  const idx_t  n = jcols.size ();


  if ( block.stride(0) + irows.stride() +
       jcols.stride() == 3_idx && block.stride(1) == m )
  {
    addData ( irows.addr(), m, jcols.addr(), n, block.addr() );
  }
  else
  {
    Matrix     bb = makeContiguous ( block );
    IdxVector  ii = makeContiguous ( irows );
    IdxVector  jj = makeContiguous ( jcols );

    addData ( ii.addr(), m, jj.addr(), n, bb.addr() );
  }
}


//-----------------------------------------------------------------------
//   eraseBlock
//-----------------------------------------------------------------------


idx_t MatrixBuilder::eraseBlock

  ( const IdxVector&  irows,
    const IdxVector&  jcols )

{
  const idx_t  m = irows.size ();
  const idx_t  n = jcols.size ();

  idx_t        k;


  if ( irows.stride() + jcols.stride() == 2_idx )
  {
    k = eraseData ( irows.addr(), m, jcols.addr(), n );
  }
  else
  {
    IdxVector  ii = makeContiguous ( irows );
    IdxVector  jj = makeContiguous ( jcols );

    k = eraseData ( ii.addr(), m, jj.addr(), n );
  }

  return k;
}


//-----------------------------------------------------------------------
//   getBlock
//-----------------------------------------------------------------------


void MatrixBuilder::getBlock

  ( const Matrix&     block,
    const IdxVector&  irows,
    const IdxVector&  jcols ) const

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array shape mismatch" );

  const idx_t  m = irows.size ();
  const idx_t  n = jcols.size ();


  if ( block.stride(0) + irows.stride() +
       jcols.stride() == 3_idx && block.stride(1) == m )
  {
    getData ( block.addr(), irows.addr(), m, jcols.addr(), n );
  }
  else
  {
    Matrix     bb = makeContiguous ( block );
    IdxVector  ii = makeContiguous ( irows );
    IdxVector  jj = makeContiguous ( jcols );

    getData ( bb.addr(), ii.addr(), m, jj.addr(), n );

    if ( block.addr() != bb.addr() )
    {
      block = bb;
    }
  }
}


//-----------------------------------------------------------------------
//   setRowValues
//-----------------------------------------------------------------------


void MatrixBuilder::setRowValues

  ( idx_t             irow,
    const IdxVector&  jcols,
    const Vector&     rvals )

{
  JEM_PRECHECK2 ( rvals.size() >= jcols.size(),
                  "row-value array is too small" );

  const idx_t  n = jcols.size ();


  if ( rvals.stride() + jcols.stride() == 2_idx )
  {
    setData ( &irow, 1, jcols.addr(), n, rvals.addr() );
  }
  else
  {
    Vector     rv = makeContiguous ( rvals );
    IdxVector  jj = makeContiguous ( jcols );

    setData ( &irow, 1, jj.addr(), n, rv.addr() );
  }
}


//-----------------------------------------------------------------------
//   addRowValues
//-----------------------------------------------------------------------


void MatrixBuilder::addRowValues

  ( idx_t             irow,
    const IdxVector&  jcols,
    const Vector&     rvals )

{
  JEM_PRECHECK2 ( rvals.size() >= jcols.size(),
                  "row-value array is too small" );

  const idx_t  n = jcols.size ();


  if ( rvals.stride() + jcols.stride() == 2_idx )
  {
    addData ( &irow, 1, jcols.addr(), n, rvals.addr() );
  }
  else
  {
    Vector     rv = makeContiguous ( rvals );
    IdxVector  jj = makeContiguous ( jcols );

    addData ( &irow, 1, jj.addr(), n, rv.addr() );
  }
}


//-----------------------------------------------------------------------
//   eraseRowValues
//-----------------------------------------------------------------------


idx_t MatrixBuilder::eraseRowValues

  ( idx_t               irow,
    const IdxVector&  jcols )

{
  const idx_t  n = jcols.size ();

  idx_t        k;


  if ( jcols.isContiguous() )
  {
    k = eraseData ( &irow, 1, jcols.addr(), n );
  }
  else
  {
    IdxVector  jj = jcols.clone ();

    k = eraseData ( &irow, 1, jj.addr(), n );
  }

  return k;
}


//-----------------------------------------------------------------------
//   getRowValues
//-----------------------------------------------------------------------


void MatrixBuilder::getRowValues

  ( const Vector&     rvals,
    idx_t             irow,
    const IdxVector&  jcols ) const

{
  JEM_PRECHECK2 ( rvals.size() >= jcols.size(),
                  "output array is too small" );

  const idx_t  n = jcols.size ();


  if ( rvals.stride() + jcols.stride() == 2_idx )
  {
    getData ( rvals.addr(), &irow, 1, jcols.addr(), n );
  }
  else
  {
    Vector     rv = makeContiguous ( rvals );
    IdxVector  jj = makeContiguous ( jcols );

    getData ( rv.addr(), &irow, 1, jj.addr(), n );

    if ( rvals.addr() != rv.addr() )
    {
      rvals = rv;
    }
  }
}


//-----------------------------------------------------------------------
//   setColValues
//-----------------------------------------------------------------------


void MatrixBuilder::setColValues

  ( const IdxVector&  irows,
    idx_t             jcol,
    const Vector&     cvals )

{
  JEM_PRECHECK2 ( cvals.size() >= irows.size(),
                  "column-value array is too small" );

  const idx_t  m = irows.size ();


  if ( cvals.stride() + irows.stride() == 2_idx )
  {
    setData ( irows.addr(), m, &jcol, 1, cvals.addr() );
  }
  else
  {
    Vector     cv = makeContiguous ( cvals );
    IdxVector  ii = makeContiguous ( irows );

    setData ( ii.addr(), m, &jcol, 1, cv.addr() );
  }
}


//-----------------------------------------------------------------------
//   addColValues
//-----------------------------------------------------------------------


void MatrixBuilder::addColValues

  ( const IdxVector&  irows,
    idx_t             jcol,
    const Vector&     cvals )

{
  JEM_PRECHECK2 ( cvals.size() >= irows.size(),
                  "column-value array is too small" );

  const idx_t  m = irows.size ();


  if ( cvals.stride() + irows.stride() == 2_idx )
  {
    addData ( irows.addr(), m, &jcol, 1, cvals.addr() );
  }
  else
  {
    Vector     cv = makeContiguous ( cvals );
    IdxVector  ii = makeContiguous ( irows );

    addData ( ii.addr(), m, &jcol, 1, cv.addr() );
  }
}


//-----------------------------------------------------------------------
//   eraseColValues
//-----------------------------------------------------------------------


idx_t MatrixBuilder::eraseColValues

  ( const IdxVector&  irows,
    idx_t             jcol )

{
  const idx_t  m = irows.size ();

  idx_t        k;


  if ( irows.isContiguous() )
  {
    k = eraseData ( irows.addr(), m, &jcol, 1 );
  }
  else
  {
    IdxVector  ii = irows.clone ();

    k = eraseData ( ii.addr(), m, &jcol, 1 );
  }

  return k;
}


//-----------------------------------------------------------------------
//   getColValues
//-----------------------------------------------------------------------


void MatrixBuilder::getColValues

  ( const Vector&     cvals,
    const IdxVector&  irows,
    idx_t             jcol ) const

{
  JEM_PRECHECK2 ( cvals.size() >= irows.size(),
                  "output array is too small" );

  const idx_t  n = irows.size ();


  if ( cvals.stride() + irows.stride() == 2_idx )
  {
    getData ( cvals.addr(), irows.addr(), n, &jcol, 1 );
  }
  else
  {
    Vector     cv = makeContiguous ( cvals );
    IdxVector  ii = makeContiguous ( irows );

    getData ( cv.addr(), ii.addr(), n, &jcol, 1 );

    if ( cvals.addr() != cv.addr() )
    {
      cvals = cv;
    }
  }
}


JIVE_END_PACKAGE( algebra )
