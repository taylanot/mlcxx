
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
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/utilities.h>
#include <jem/util/HashDictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/util/Properties.h>
#include <jem/xml/print.h>
#include <jem/xml/utilities.h>
#include <jive/util/error.h>
#include <jive/util/typedefs.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/Globdat.h>
#include <jive/util/TableException.h>
#include <jive/util/Table.h>


JEM_DEFINE_CLASS( jive::util::Table );


JIVE_BEGIN_PACKAGE( util )


using jem::shape;
using jem::staticCast;
using jem::dynamicCast;
using jem::newInstance;
using jem::makeContiguous;


//=======================================================================
//   class Table
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Table::Table ( const String& name ) :

  Super ( name )

{}


Table::~Table ()
{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Table::getContext () const
{
  String  itemName = getRowItems()->getItemName ();

  return NamedObject::makeContext ( itemName + " table", myName_ );
}


//-----------------------------------------------------------------------
//   rowCount
//-----------------------------------------------------------------------


idx_t Table::rowCount () const
{
  return getRowItems()->size ();
}


//-----------------------------------------------------------------------
//   findColumn
//-----------------------------------------------------------------------


idx_t Table::findColumn ( const String& name ) const
{
  const idx_t  n = columnCount ();

  for ( idx_t jcol = 0; jcol < n; jcol++ )
  {
    if ( name == getColumnName(jcol) )
    {
      return jcol;
    }
  }

  return -1;
}


//-----------------------------------------------------------------------
//   getColumnNames
//-----------------------------------------------------------------------


StringVector Table::getColumnNames () const
{
  const idx_t   n = columnCount ();

  StringVector  names ( n );

  for ( idx_t jcol = 0; jcol < n; jcol++ )
  {
    names[jcol] = getColumnName ( jcol );
  }

  return names;
}


//-----------------------------------------------------------------------
//   findBlock
//-----------------------------------------------------------------------


idx_t Table::findBlock

  ( const Matrix&     block,
    const IdxVector&  irows,
    const IdxVector&  jcols ) const

{
  JEM_PRECHECK2 ( block.size(0) == irows.size() &&
                  block.size(1) == jcols.size(),
                  "Array size mismatch" );

  const idx_t  m = irows.size ();
  const idx_t  n = jcols.size ();

  idx_t        k;


  if ( block.stride(0) + irows.stride() +
       jcols.stride() == 3_idx && block.stride(1) == m )
  {
    k = getData ( block.addr(), irows.addr(), m, jcols.addr(), n );
  }
  else
  {
    Matrix     bb = makeContiguous ( block );
    IdxVector  ii = makeContiguous ( irows );
    IdxVector  jj = makeContiguous ( jcols );

    k = getData ( bb.addr(), ii.addr(), m, jj.addr(), n );

    if ( block.addr() != bb.addr() )
    {
      block = bb;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getBlock
//-----------------------------------------------------------------------


void Table::getBlock

  ( const Matrix&     block,
    const IdxVector&  irows,
    const IdxVector&  jcols ) const

{
  const idx_t  k = findBlock ( block, irows, jcols );

  if ( k != irows.size() * jcols.size() )
  {
    missingValueError_ ( irows, jcols );
  }
}


//-----------------------------------------------------------------------
//   findRowValues
//-----------------------------------------------------------------------


idx_t Table::findRowValues

  ( const Vector&     rvals,
    idx_t             irow,
    const IdxVector&  jcols ) const

{
  JEM_PRECHECK2 ( rvals.size() >= jcols.size(),
                  "output array is too small" );

  const idx_t  n = jcols.size ();

  idx_t        k;


  if ( rvals.stride() + jcols.stride() == 2_idx )
  {
    k = getData ( rvals.addr(), &irow, 1, jcols.addr(), n );
  }
  else
  {
    Vector     rv = makeContiguous ( rvals );
    IdxVector  jj = makeContiguous ( jcols );

    k = getData ( rv.addr(), &irow, 1, jj.addr(), n );

    if ( rvals.addr() != rv.addr() )
    {
      rvals = rv;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getRowValues
//-----------------------------------------------------------------------


void Table::getRowValues

  ( const Vector&     rvals,
    idx_t             irow,
    const IdxVector&  jcols ) const

{
  const idx_t  k = findRowValues ( rvals, irow, jcols );

  if ( k != jcols.size() )
  {
    missingValueError_ ( irow, jcols );
  }
}


//-----------------------------------------------------------------------
//   findColValues
//-----------------------------------------------------------------------


idx_t Table::findColValues

  ( const Vector&     cvals,
    const IdxVector&  irows,
    idx_t               jcol ) const

{
  JEM_PRECHECK2 ( cvals.size() >= irows.size(),
                  "output array is too small" );

  const idx_t  n = irows.size ();

  idx_t        k;


  if ( cvals.stride() + irows.stride() == 2_idx )
  {
    k = getData ( cvals.addr(), irows.addr(), n, &jcol, 1 );
  }
  else
  {
    Vector     cv = makeContiguous ( cvals );
    IdxVector  ii = makeContiguous ( irows );

    k = getData ( cv.addr(), ii.addr(), n, &jcol, 1 );

    if ( cvals.addr() != cv.addr() )
    {
      cvals = cv;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getColValues
//-----------------------------------------------------------------------


void Table::getColValues

  ( const Vector&     cvals,
    const IdxVector&  irows,
    idx_t             jcol ) const

{
  const idx_t  k = findColValues ( cvals, irows, jcol );

  if ( k != irows.size() )
  {
    missingValueError_ ( irows, jcol );
  }
}


//-----------------------------------------------------------------------
//   findAllValues
//-----------------------------------------------------------------------


idx_t Table::findAllValues ( const Matrix& buf ) const
{
  JEM_PRECHECK2 ( buf.size(0) == rowCount() &&
                  buf.size(1) == columnCount(),
                  "Array shape mismatch" );

  idx_t  k;

  if ( buf.stride(0) == 1_idx )
  {
    k = getAllData ( buf.addr(), buf.stride(1) );
  }
  else
  {
    Matrix  tmp ( buf.shape() );

    k = getAllData ( tmp.addr(), tmp.stride(1) );

    buf = tmp;
  }

  return k;
}


//-----------------------------------------------------------------------
//   getAllValues
//-----------------------------------------------------------------------


void Table::getAllValues ( const Matrix& buf ) const
{
  JEM_PRECHECK2 ( buf.size(0) == rowCount() &&
                  buf.size(1) == columnCount(),
                  "Array shape mismatch" );

  idx_t  k;

  if ( buf.stride(0) == 1_idx )
  {
    k = getAllData ( buf.addr(), buf.stride(1) );
  }
  else
  {
    Matrix  tmp ( buf.shape() );

    k = getAllData ( tmp.addr(), tmp.stride(1) );

    buf = tmp;
  }

  if ( k != buf.size(0) * buf.size(1) )
  {
    const String myName = getName ();

    throw TableException (
      getContext (),
      String::format (
        "table `%s\' is not completely filled", myName
      )
    );
  }
}


//-----------------------------------------------------------------------
//   getTagName
//-----------------------------------------------------------------------


String Table::getTagName () const
{
  return getTagName ( getRowItems()->getItemName() );
}


String Table::getTagName

  ( const String& rowItemName )

{
  return ( jem::xml::makeTag ( rowItemName ) + "Table" );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void Table::printTo ( PrintWriter& out ) const
{
  const idx_t  m = rowCount    ();
  const idx_t  n = columnCount ();

  IdxVector    ibuf   ( 2 * m );

  IdxVector    irows  ( ibuf[slice(BEGIN,m)] );
  IdxVector    rowIDs ( ibuf[slice(m,END)]   );


  for ( idx_t i = 0; i < m; i++ )
  {
    irows[i] = i;
  }

  getRowItems()->getItemMap()->getItemIDs ( rowIDs, irows );

  if ( m * n == size() )
  {
    Matrix  table ( m, n );

    sort ( irows, rowIDs );

    getAllValues ( table );
    printSection ( out, table, getColumnNames(), rowIDs, irows );
  }
  else
  {
    printMatrix  ( out, toMatrix(), getColumnNames(), rowIDs );
  }
}


//-----------------------------------------------------------------------
//   printSome
//-----------------------------------------------------------------------


void Table::printSome

  ( PrintWriter&   out,
    const Dict&    tables,
    const String&  tag )

{
  using jem::xml::beginStartTag;
  using jem::xml::endStartTag;
  using jem::xml::endTag;
  using jem::xml::attribute;

  Ref<DictEnum>  e = tables.enumerate();


  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<Object>  obj   = e->getValue ();
    Table*       table = dynamicCast<Table*> ( obj );

    if ( ! table )
    {
      continue;
    }

    String  tableName = e->getKey ();
    String  tagName   = tag;
    String  tableType;

    if ( tagName.size() == 0 )
    {
      tagName = table->getTagName ();
    }

    if ( table->isDense() )
    {
      tableType = "Dense";
    }
    else
    {
      tableType = "Sparse";
    }

    print ( out,
            beginStartTag( tagName ),
            attribute( "name", tableName ),
            attribute( "type", tableType ),
            endStartTag );

    table->printTo ( out );

    print ( out, endTag( tagName ) );
  }
}


//-----------------------------------------------------------------------
//   printAll
//-----------------------------------------------------------------------


void Table::printAll

  ( PrintWriter&       out,
    const Properties&  globdat )

{
  Ref<Dict>      list = ItemSet::getAll ( globdat );
  Ref<DictEnum>  e    = list->enumerate ();


  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<ItemSet>  items = dynamicCast<ItemSet> ( e->getValue() );

    if ( items )
    {
      printSome ( out, * getFor ( items, globdat ) );
    }
  }
}


//-----------------------------------------------------------------------
//   printMatrix
//-----------------------------------------------------------------------


void Table::printMatrix

  ( PrintWriter&         out,
    const SparseMatrix&  mat,
    const StringVector&  colNames,
    const IdxVector&     rowIDs )

{
  using jem::iarray;


  JEM_PRECHECK2 ( mat.size(0) == rowIDs  .size() &&
                  mat.size(1) == colNames.size(),
                  "Array size mismatch" );

  SparseMatrix  trmat   = mat.transpose ();

  const idx_t  rowCount   = trmat.size (1);
  const idx_t  colCount   = trmat.size (0);

  IdxVector    colOffsets = trmat.getRowOffsets    ();
  IdxVector    rowIndices = trmat.getColumnIndices ();
  Vector       values     = trmat.getValues        ();

  IdxVector    rwids      ( rowCount );
  IdxVector    iperm      ( rowCount );

  IdxVector    irows;
  idx_t        jfirst;
  idx_t        jlast;
  idx_t        i, j, k, m, n;


  jfirst = 0;

  while ( jfirst < colCount )
  {
    i = colOffsets[jfirst];
    j = colOffsets[jfirst + 1];
    m = j - i;

    irows.ref    ( rowIndices[slice(i,j)] );
    rwids.resize ( m );
    iperm.resize ( m );

    rwids = rowIDs[irows];
    iperm = iarray ( m );
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

    sort ( iperm, rwids );

    printSection (
      out,
      reshape (
        values[slice(k,k + m * n)],
        shape ( m , n )
      ),
      colNames[slice(jfirst,jlast)],
      rwids,
      iperm
    );

    jfirst = jlast;
  }
}


//-----------------------------------------------------------------------
//   printSection
//-----------------------------------------------------------------------


void Table::printSection

  ( PrintWriter&         out,
    const Matrix&        mat,
    const StringVector&  colNames,
    const IdxVector&     rowIDs,
    const IdxVector&     rowPerm )

{
  using jem::io::endl;
  using jem::io::space;
  using jem::xml::beginStartTag;
  using jem::xml::endStartTag;
  using jem::xml::endTag;
  using jem::xml::attribute;

  JEM_PRECHECK2 ( mat.size(0) == rowIDs  .size() &&
                  mat.size(0) == rowPerm .size() &&
                  mat.size(1) == colNames.size(),
                  "Array size mismatch" );

  const idx_t  rowCount = mat.size(0);
  const idx_t  colCount = mat.size(1);


  print ( out, beginStartTag( "Section" ), " columns = \"" );

  for ( idx_t j = 0; j < colCount; j++ )
  {
    if ( j > 0 )
    {
      print ( out, " | " );
    }

    print ( out, colNames[j] );
  }

  print ( out, '\"', endStartTag );

  if ( rowCount * colCount > 0 )
  {
    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      idx_t  i = rowPerm[irow];

      if ( irow > 0 )
      {
        print ( out, endl );
      }

      print ( out, rowIDs[i] );

      for ( idx_t j = 0; j < colCount; j++ )
      {
        print ( out, space, mat(i,j) );
      }
    }
  }

  print ( out, endTag( "Section" ) );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void Table::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  getFor ( getRowItems(), globdat ) -> insert

    ( name, Ref<Object>( const_cast<Self*>( this ) ) );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<Table> Table::find

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Ref<Dict>  tables = findFor ( items, globdat );

  if ( ! tables )
  {
    return nullptr;
  }

  return dynamicCast<Table> ( tables->get( name ) );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Table> Table::get

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const Properties&    globdat,
    const String&        context )

{
  Ref<Table>  table = find ( name, items, globdat );

  if ( ! table )
  {
    String  itemName;

    if ( items )
    {
      itemName = items->getItemName ();
    }
    else
    {
      itemName = "item";
    }

    throw jem::IllegalInputException (
      context,
      String::format ( "undefined %s table: `%s\'",
                       itemName, name )
    );
  }

  return table;
}


//-----------------------------------------------------------------------
//   findFor
//-----------------------------------------------------------------------


Ref<Dictionary> Table::findFor

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Ref<Dict>  tables;

  Globdat::findFor ( tables, Globdat::TABLES, items, globdat );

  return tables;
}


//-----------------------------------------------------------------------
//   getFor
//-----------------------------------------------------------------------


Ref<Dictionary> Table::getFor

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  using jem::util::HashDictionary;

  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Properties  itemdat ( Globdat::getDataFor( items, globdat ) );
  Ref<Dict>   tables;

  if ( ! itemdat.find( tables, Globdat::TABLES ) )
  {
    tables = newInstance<HashDictionary> ();

    itemdat.set ( Globdat::TABLES, tables );
  }

  return tables;
}


//-----------------------------------------------------------------------
//   noSuchColumnError_
//-----------------------------------------------------------------------


void Table::noSuchColumnError_

  ( const String&  colName ) const

{
  throw TableException (
    getContext (),
    String::format (
      "table `%s\' has no column named `%s\'",
      myName_,
      colName
    )
  );
}


//-----------------------------------------------------------------------
//   noSuchValueError_
//-----------------------------------------------------------------------


void Table::noSuchValueError_

  ( idx_t  irow,
    idx_t  jcol ) const

{
  const ItemSet&  items   = * getRowItems   ();
  const idx_t     itemID  = items.getItemID ( irow );
  const String    colName = getColumnName   ( jcol );

  throw TableException (
    getContext (),
    String::format (
      "undefined entry (%d,%s) in table `%s\'",
      itemID,
      colName,
      myName_
    )
  );
}


//-----------------------------------------------------------------------
//   missingValueError_
//-----------------------------------------------------------------------


void Table::missingValueError_

  ( const IdxVector&  irows,
    const IdxVector&  jcols ) const

{
  const idx_t  m = irows.size ();
  const idx_t  n = jcols.size ();

  double       dummy;

  for ( idx_t i = 0; i < m; i++ )
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      if ( ! findValue( dummy, irows[i], jcols[j] ) )
      {
        noSuchValueError_ ( irows[i], jcols[j] );
      }
    }
  }
}


void Table::missingValueError_

  ( idx_t             irow,
    const IdxVector&  jcols ) const

{
  const idx_t  n = jcols.size ();

  double       dummy;

  for ( idx_t j = 0; j < n; j++ )
  {
    if ( ! findValue( dummy, irow, jcols[j] ) )
    {
      noSuchValueError_ ( irow, jcols[j] );
    }
  }
}


void Table::missingValueError_

  ( const IdxVector&  irows,
    idx_t             jcol ) const

{
  const idx_t  n = irows.size ();

  double       dummy;

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( ! findValue( dummy, irows[i], jcol ) )
    {
      noSuchValueError_ ( irows[i], jcol );
    }
  }
}


JIVE_END_PACKAGE( util )
