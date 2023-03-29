
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
#include <jive/util/SparseTable.h>
#include <jive/util/XTable.h>


JEM_DEFINE_CLASS( jive::util::XTable );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;
using jem::makeContiguous;


//=======================================================================
//   class XTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


XTable::XTable ( const String& name ) : Super ( name )
{}


XTable::~XTable ()
{}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void XTable::reserve ( idx_t )
{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void XTable::trimToSize ()
{}


//-----------------------------------------------------------------------
//   addColumns
//-----------------------------------------------------------------------


IdxVector XTable::addColumns

  ( const StringVector&  names )

{
  const idx_t  n = names.size ();

  IdxVector    icols ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    icols[i] = addColumn ( names[i] );
  }

  return icols;
}


//-----------------------------------------------------------------------
//   scaleRows
//-----------------------------------------------------------------------


void XTable::scaleRows ( const Vector& scale )
{
  JEM_PRECHECK2 ( scale.size() == rowCount(),
                  "row scale vector has wrong size" );

  const idx_t  m = rowCount    ();
  const idx_t  n = columnCount ();

  double       s, t;

  for ( idx_t i = 0; i < m; i++ )
  {
    s = scale[i];

    for ( idx_t j = 0; j < n; j++ )
    {
      if ( findValue( t, i, j ) )
      {
        t *= s;

        setValue ( i, j, t );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   setBlock
//-----------------------------------------------------------------------


void XTable::setBlock

  ( const IdxVector&  irows,
    const IdxVector&  jcols,
    const Matrix&     block )

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array size mismatch" );

  const idx_t  m = irows.size ();
  const idx_t  n = jcols.size ();

  if ( block.stride(0) + irows.stride() +
       jcols.stride() == 3_idx && block.stride(1) == m )
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


void XTable::addBlock

  ( const IdxVector&  irows,
    const IdxVector&  jcols,
    const Matrix&     block )

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array size mismatch" );

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


idx_t XTable::eraseBlock

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
//   setRowValues
//-----------------------------------------------------------------------


void XTable::setRowValues

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


void XTable::addRowValues

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


idx_t XTable::eraseRowValues

  ( idx_t             irow,
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
//   setColValues
//-----------------------------------------------------------------------


void XTable::setColValues

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


void XTable::addColValues

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


idx_t XTable::eraseColValues

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
//   find
//-----------------------------------------------------------------------


Ref<XTable> XTable::find

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  return jem::dynamicCast<XTable>

    ( Super::find( name, items, globdat ) );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<XTable> XTable::get

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<XTable>  table = find ( name, items, globdat );

  if ( ! table )
  {
    table = newInstance<SparseTable> ( name, items );

    table->store ( name, globdat );
  }

  return table;
}


JIVE_END_PACKAGE( util )
