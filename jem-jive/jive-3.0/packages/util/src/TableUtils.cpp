
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
#include <jem/base/Class.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jive/util/Table.h>
#include <jive/util/TableException.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/DofSpace.h>
#include <jive/util/TableUtils.h>


JIVE_BEGIN_PACKAGE( util )


using jem::shape;
using jem::iarray;
using jem::reshape;


//=======================================================================
//   class TableUtils::Utils_
//=======================================================================


class TableUtils::Utils_
{
 public:

  static IdxVector        getDofTypeMap

    ( const Table&          table,
      const DofSpace&       dofs );

  static void             noSuchDofError

    ( const String&         context,
      const String&         dofPath );

};


//-----------------------------------------------------------------------
//   getDofTypeMap
//-----------------------------------------------------------------------


IdxVector TableUtils::Utils_::getDofTypeMap

  ( const Table&     table,
    const DofSpace&  dofs )

{
  const idx_t  colCount = table.columnCount ();

  IdxVector    itypes ( colCount );
  String       name;


  for ( idx_t jcol = 0; jcol < colCount; jcol++ )
  {
    name         = table.getColumnName ( jcol );
    itypes[jcol] = dofs.findType       ( name );

    if ( itypes[jcol] < 0 )
    {
      throw TableException (
        table.getContext (),
        String::format (
          "undefined degree of freedom type: `%s\'", name
        )
      );
    }
  }

  return itypes;
}


//-----------------------------------------------------------------------
//   noSuchDofError
//-----------------------------------------------------------------------


void TableUtils::Utils_::noSuchDofError

  ( const String&  context,
    const String&  dofPath )

{
  throw TableException (
    context,
    String::format ( "undefined degree of freedom: `%s\'",
                      dofPath )
  );
}


//=======================================================================
//   class TableUtils
//=======================================================================

//-----------------------------------------------------------------------
//   toVector (dense)
//-----------------------------------------------------------------------


idx_t TableUtils::toVector

  ( const Vector&    vec,
    const Table&     table,
    const DofSpace&  dofs )

{
  JEM_PRECHECK2 ( vec.size() == dofs.dofCount(),
                  "Array size mismatch" );

  Vector     tmp   ( table.size() );
  IdxVector  idofs ( tmp  .size() );

  toVector ( tmp, idofs, table, dofs );

  vec[idofs] = tmp;

  return idofs.size ();
}


//-----------------------------------------------------------------------
//   toVector (sparse)
//-----------------------------------------------------------------------


void TableUtils::toVector

  ( const Vector&     values,
    const IdxVector&  idofs,
    const Table&      table,
    const DofSpace&   dofs )

{
  JEM_PRECHECK2 ( values.size() == idofs.size() &&
                  values.size() == table.size(),
                  "Array size mismatch" );
  JEM_PRECHECK2 ( table.rowCount() == dofs.itemCount(),
                  "Table and DofSpace do not match" );

  SparseMatrix  mat        = table.toMatrix ();

  const idx_t  rowCount    = mat.size(0);

  IdxVector    rowOffsets  = mat.getRowOffsets    ();
  IdxVector    colIndices  = mat.getColumnIndices ();
  Vector       tableValues = mat.getValues        ();

  IdxVector    itypes;
  idx_t        last;


  itypes.ref ( Utils_::getDofTypeMap( table, dofs ) );

  last = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  n = rowOffsets[irow + 1];

    for ( idx_t i = rowOffsets[irow]; i < n; i++ )
    {
      idx_t  jcol = colIndices[i];
      idx_t  idof = dofs.findDofIndex ( irow, itypes[jcol] );

      if ( idof < 0 )
      {
        Utils_::noSuchDofError (
          table.getContext (),
          dofs.getDofPath ( irow, itypes[jcol] )
        );
      }

      values[last] = tableValues[i];
      idofs[last]  = idof;
      last++;
    }
  }
}


//-----------------------------------------------------------------------
//   printVector (dense)
//-----------------------------------------------------------------------


void TableUtils::printVector

  ( PrintWriter&     out,
    const Vector&    vec,
    const DofSpace&  dofs )

{
  JEM_PRECHECK2 ( vec.size() == dofs.dofCount(),
                  "Array size mismatch" );

  StringVector  colNames = dofs.getTypeNames ();

  const idx_t   dofCount = vec .size      ();
  const idx_t   rowCount = dofs.itemCount ();
  const idx_t   colCount = dofs.typeCount ();

  Vector        vbuf   ( dofCount );
  IdxMatrix     ibuf   ( rowCount, 5 );

  IdxVector     rowIDs ( ibuf[0] );
  IdxVector     irows  ( ibuf[1] );
  IdxVector     jrows  ( ibuf[2] );
  IdxVector     idofs  ( ibuf[3] );
  IdxVector     iperm  ( ibuf[4] );

  IdxVector     iirows;
  IdxVector     jjrows;
  IdxVector     iidofs;
  IdxVector     iiperm;

  idx_t         jfirst;
  idx_t         jlast;
  idx_t         vlast;
  idx_t         irow;
  idx_t         m, n;


  for ( irow = 0; irow < rowCount; irow++ )
  {
    irows[irow] = irow;
  }

  dofs.getItems()->getItemMap()->getItemIDs ( rowIDs, irows );

  jfirst = 0;

  while ( jfirst < colCount )
  {
    m = dofs.getDofsForType ( idofs, irows, jfirst );

    if ( m == 0 )
    {
      jfirst++;
      continue;
    }

    iidofs.ref ( idofs[slice(BEGIN,m)] );
    iirows.ref ( irows[slice(BEGIN,m)] );
    jjrows.ref ( jrows[slice(BEGIN,m)] );
    iiperm.ref ( iperm[slice(BEGIN,m)] );

    vbuf[slice(BEGIN,m)] = vec[iidofs];

    vlast = m;
    jlast = jfirst + 1;

    while ( jlast < colCount )
    {
      n = dofs.getDofsForType ( idofs, jrows, jlast );

      if ( n != m )
      {
        break;
      }

      if ( jem::equal( iirows, jjrows ) )
      {
        vbuf[slice(vlast,vlast + m)] = vec[iidofs];

        vlast += m;
        jlast++;
      }
      else
      {
        break;
      }
    }

    iidofs = rowIDs[iirows];
    iiperm = iarray ( iiperm.size()  );
    n      = jlast - jfirst;

    sort ( iiperm, iidofs );

    Table::printSection (
      out,
      reshape ( vbuf[slice(BEGIN,vlast)], shape( m, n ) ),
      colNames[slice(jfirst,jlast)],
      iidofs,
      iiperm
    );

    jfirst = jlast;
  }
}


//-----------------------------------------------------------------------
//   printVector (sparse)
//-----------------------------------------------------------------------


void TableUtils::printVector

  ( PrintWriter&      out,
    const Vector&     vec,
    const IdxVector&  idofs,
    const DofSpace&   dofs )

{
  using jem::numeric::SparseUtils;


  JEM_PRECHECK2 ( idofs.size() == vec.size(),
                  "Array size mismatch" );

  const ItemMap&  imap      = *dofs.getItems()->getItemMap ();

  const idx_t     colCount  = dofs.typeCount ();
  const idx_t     idofCount = idofs.size     ();

  StringVector    colNames   ( dofs.getTypeNames() );

  IdxMatrix       ibuf       ( idofCount, 4 );

  IdxVector       irows      ( ibuf[0] );
  IdxVector       jcols      ( ibuf[1] );

  IdxVector       rowPerm    ( ibuf[2] );
  IdxVector       rowIndices ( ibuf[3] );
  IdxVector       colOffsets ( colCount + 1 );
  Vector          values     ( idofCount );

  IdxVector       rowIDs;
  IdxVector       iperm;

  idx_t           jfirst;
  idx_t           jlast;
  idx_t           jcol;
  idx_t           i, j, k, m, n;


  dofs.decodeDofIndices ( irows, jcols, idofs );

  colOffsets = 0;

  for ( i = 0; i < idofCount; i++ )
  {
    colOffsets[jcols[i]]++;
  }

  SparseUtils::sumOffsets ( colOffsets );

  for ( i = 0; i < idofCount; i++ )
  {
    jcol          = jcols[i];
    j             = colOffsets[jcol];
    rowIndices[j] = irows[i];
    values[j]     = vec[i];

    colOffsets[jcol]++;
  }

  SparseUtils::shiftOffsets ( colOffsets );

  jfirst = 0;

  while ( jfirst < colCount )
  {
    i = colOffsets[jfirst];
    j = colOffsets[jfirst + 1];
    m = j - i;

    irows .ref ( rowIndices[slice(i,j)]  );
    rowIDs.ref ( jcols[slice(BEGIN,m)]   );
    iperm .ref ( rowPerm[slice(BEGIN,m)] );

    imap.getItemIDs ( rowIDs, irows );

    iperm = iarray ( m );

    sort ( iperm, rowIDs );

    jlast = jfirst + 1;

    while ( jlast < colCount )
    {
      i = colOffsets[jlast];
      j = colOffsets[jlast + 1];

      if ( jem::equal( irows, rowIndices[slice(i,j)] ) )
      {
        jlast++;
      }
      else
      {
        break;
      }
    }

    k = colOffsets[jfirst];
    n = jlast - jfirst;

    Table::printSection (
      out,
      reshape (
        values[slice(k,k + m * n )],
        shape ( m, n )
      ),
      colNames[slice(jfirst,jlast)],
      rowIDs,
      iperm
    );

    jfirst = jlast;
  }
}


JIVE_END_PACKAGE( util )
