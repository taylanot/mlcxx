
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
#include <jem/util/HashDictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/util/Properties.h>
#include <jem/xml/print.h>
#include <jem/xml/utilities.h>
#include <jive/util/typedefs.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/util/Globdat.h>
#include <jive/util/Database.h>


JEM_DEFINE_CLASS( jive::util::Database );


JIVE_BEGIN_PACKAGE( util )


using jem::staticCast;
using jem::dynamicCast;
using jem::newInstance;
using jem::Array;
using jem::IllegalInputException;
using jem::util::HashDict;


//=======================================================================
//   utility functions
//=======================================================================

//-----------------------------------------------------------------------
//   printDBColumn
//-----------------------------------------------------------------------


template <class DBCol>

  static inline void  printDBColumn

  ( PrintWriter&        out,
    const DBCol&        col )

{
  using jem::io::space;

  typedef typename DBCol::ValueType  T;

  const ItemMap&  imap = * col.getRowItems()->getItemMap ();

  IdxVector  irows ( col.getNonEmptyRows() );
  Array<T>   buf   ( col.maxRowSize()      );

  idx_t      n     = irows.size ();


  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  irow = irows[i];

    if ( i > 0 )
    {
      out.printLine ();
    }

    print ( out, imap.getItemID( irow ) );

    idx_t  k = col.getData ( buf.addr(), buf.size(), irow );

    for ( idx_t j = 0; j < k; j++ )
    {
      print ( out, space, buf[j] );
    }

    print ( out, ';' );
  }
}


//-----------------------------------------------------------------------
//   clearDBColumns
//-----------------------------------------------------------------------


template <class DBCol>

  static inline void    clearDBColumns

  ( const Array<DBCol*>&  cols )

{
  const idx_t  n = cols.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    cols[i]->clear ();
  }
}


//-----------------------------------------------------------------------
//   trimDBColumns
//-----------------------------------------------------------------------


template <class DBCol>

  static inline void    trimDBColumns

  ( const Array<DBCol*>&  cols )

{
  const idx_t  n = cols.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    cols[i]->trimToSize ();
  }
}


//-----------------------------------------------------------------------
//   printDBColumns
//-----------------------------------------------------------------------


template <class DBCol>

  static inline void    printDBColumns

  ( PrintWriter&          out,
    const Array<DBCol*>&  cols,
    const String&         type )

{
  using jem::xml::beginStartTag;
  using jem::xml::endStartTag;
  using jem::xml::endTag;
  using jem::xml::attribute;

  const idx_t  n = cols.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    print ( out,
            beginStartTag( "Column" ),
            attribute( "type", type ),
            attribute( "name", cols[i]->getName() ),
            endStartTag );

    cols[i]->printTo ( out );

    print ( out, endTag( "Column" ) );
  }
}


//-----------------------------------------------------------------------
//   findDBColumn
//-----------------------------------------------------------------------


template <class DBCol>

  static inline DBCol*  findDBColumn

  ( const String&         name,
    const Array<DBCol*>&  cols )

{
  const idx_t  n = cols.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( cols[i]->getName() == name )
    {
      return cols[i];
    }
  }

  return (DBCol*) 0;
}


//=======================================================================
//   class DBColumn
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DBColumn::DBColumn ()
{}


DBColumn::~DBColumn ()
{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String DBColumn::getContext () const
{
  String  myName   = getName ();
  String  dbName   = getDatabase()->getName     ();
  String  itemName = getRowItems()->getItemName ();

  if ( dbName.size() > 0 )
  {
    return String::format (
      "%s database column `%s.%s\'", itemName, dbName, myName
    );
  }
  else
  {
    return String::format (
      "%s database column `%s\'", itemName, myName
    );
  }
}


String DBColumn::getContext ( idx_t irow ) const
{
  String           myName   = getName           ();
  const Database&  dbase    = * getDatabase     ();
  String           dbName   = dbase.getName     ();
  const ItemSet&   items    = * getRowItems     ();
  String           itemName = items.getItemName ();
  idx_t            itemID   = items.getItemID   ( irow );


  if ( dbName.size() > 0 )
  {
    return String::format (
      "%s database row `%s.%s[%d]\'",
      itemName,
      dbName,
      myName,
      itemID
    );
  }
  else
  {
    return String::format (
      "%s database row `%s[%d]\'",
      itemName,
      myName,
      itemID
    );
  }
}


//-----------------------------------------------------------------------
//   rowCount
//-----------------------------------------------------------------------


idx_t DBColumn::rowCount () const
{
  return getRowItems()->size ();
}


//-----------------------------------------------------------------------
//   maxRowSize
//-----------------------------------------------------------------------


idx_t DBColumn::maxRowSize () const
{
  IdxVector  irows ( getNonEmptyRows() );

  idx_t      m = 0;
  idx_t      n = irows.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    m = jem::max ( m, rowSize( irows[i] ) );
  }

  return m;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void DBColumn::clear ()
{
  IdxVector  irows ( getNonEmptyRows() );

  idx_t      n = irows.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    clearRow ( irows[i] );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void DBColumn::trimToSize ()
{}


//=======================================================================
//   class IntDBColumn
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


IntDBColumn::~IntDBColumn ()
{}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void IntDBColumn::printTo ( PrintWriter& out ) const
{
  printDBColumn ( out, * this );
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


idx_t IntDBColumn::getValue ( idx_t irow ) const
{
  idx_t  value;

  if ( ! getData( &value, 1, irow ) )
  {
    missingDataError_ ( irow, 0, 1 );
  }

  return value;
}


//-----------------------------------------------------------------------
//   findValues
//-----------------------------------------------------------------------


idx_t IntDBColumn::findValues

  ( const IdxVector&  buf,
    idx_t             count,
    idx_t             irow ) const

{
  JEM_PRECHECK2 ( count <= buf.size(),
                  "output array is too small" );

  idx_t  n;

  if ( buf.isContiguous() )
  {
    n = getData ( buf.addr(), count, irow );
  }
  else
  {
    IdxVector  tmp ( count );

    n = getData ( tmp.addr(), count, irow );

    buf[slice(BEGIN,n)] = tmp[slice(BEGIN,n)];
  }

  return n;
}


//-----------------------------------------------------------------------
//   getValues
//-----------------------------------------------------------------------


void IntDBColumn::getValues

  ( const IdxVector&  buf,
    idx_t             count,
    idx_t             irow ) const

{
  JEM_PRECHECK2 ( count <= buf.size(),
                  "output array is too small" );

  idx_t  n;

  if ( buf.isContiguous() )
  {
    n = getData ( buf.addr(), count, irow );
  }
  else
  {
    IdxVector  tmp ( count );

    n = getData ( tmp.addr(), count, irow );

    buf[slice(BEGIN,n)] = tmp[slice(BEGIN,n)];
  }

  if ( n != count )
  {
    missingDataError_ ( irow, n, count );
  }
}


void IntDBColumn::getValues

  ( const IdxMatrix&  buf,
    idx_t             irow ) const

{
  const idx_t  count = buf.size(0) * buf.size(1);

  idx_t  n;

  if ( buf.isContiguous() )
  {
    n   = getData ( buf.addr(), count, irow );
  }
  else
  {
    IdxMatrix  tmp ( buf.shape() );

    n   = getData ( tmp.addr(), count, irow );
    buf = tmp;
  }

  if ( n != count )
  {
    missingDataError_ ( irow, n, count );
  }
}


void IntDBColumn::getValues

  ( const IdxCubix&  buf,
    idx_t            irow ) const

{
  const idx_t  count = buf.size(0) * buf.size(1) * buf.size(2);

  idx_t        n;


  if ( buf.isContiguous() )
  {
    n   = getData ( buf.addr(), count, irow );
  }
  else
  {
    IdxCubix  tmp ( buf.shape() );

    n   = getData ( tmp.addr(), count, irow );
    buf = tmp;
  }

  if ( n != count )
  {
    missingDataError_ ( irow, n, count );
  }
}


//-----------------------------------------------------------------------
//   setValues
//-----------------------------------------------------------------------


void IntDBColumn::setValues

  ( idx_t             irow,
    const IdxVector&  buf,
    idx_t             count )

{
  JEM_PRECHECK2 ( count <= buf.size(),
                  "output array is too small" );

  if ( buf.isContiguous() )
  {
    setData ( irow, buf.addr(), count );
  }
  else
  {
    IdxVector  tmp ( count );

    tmp = buf[slice(BEGIN,count)];

    setData ( irow, tmp.addr(), count );
  }
}


void IntDBColumn::setValues

  ( idx_t             irow,
    const IdxMatrix&  buf )

{
  const idx_t  count = buf.size(0) * buf.size(1);

  if ( buf.isContiguous() )
  {
    setData ( irow, buf.addr(), count );
  }
  else
  {
    IdxMatrix  tmp ( buf.shape() );

    tmp = buf;

    setData ( irow, tmp.addr(), count );
  }
}


void IntDBColumn::setValues

  ( idx_t            irow,
    const IdxCubix&  buf )

{
  const idx_t  count = buf.size(0) * buf.size(1) * buf.size(2);

  if ( buf.isContiguous() )
  {
    setData ( irow, buf.addr(), count );
  }
  else
  {
    IdxCubix  tmp ( buf.shape() );

    tmp = buf;

    setData ( irow, tmp.addr(), count );
  }
}


//-----------------------------------------------------------------------
//   missingDataError_
//-----------------------------------------------------------------------


void IntDBColumn::missingDataError_

  ( idx_t  irow,
    idx_t  actualCount,
    idx_t  formalCount ) const

{
  if ( formalCount == 1 )
  {
    throw IllegalInputException (
      getContext ( irow ),
      String::format (
        "missing data: expected one integer (got none)"
      )
    );
  }
  else
  {
    throw IllegalInputException (
      getContext ( irow ),
      String::format (
        "missing data: expected %d integers (got only %d)",
        formalCount,
        actualCount
      )
    );
  }
}


//=======================================================================
//   class FloatDBColumn
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


FloatDBColumn::~FloatDBColumn ()
{}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void FloatDBColumn::printTo ( PrintWriter& out ) const
{
  printDBColumn ( out, * this );
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


double FloatDBColumn::getValue ( idx_t irow ) const
{
  double  value;

  if ( ! getData( &value, 1, irow ) )
  {
    missingDataError_ ( irow, 0, 1 );
  }

  return value;
}


//-----------------------------------------------------------------------
//   findValues
//-----------------------------------------------------------------------


idx_t FloatDBColumn::findValues

  ( const Vector&  buf,
    idx_t          count,
    idx_t          irow ) const

{
  JEM_PRECHECK2 ( count <= buf.size(),
                  "output array is too small" );

  idx_t  n;

  if ( buf.isContiguous() )
  {
    n = getData ( buf.addr(), count, irow );
  }
  else
  {
    Vector  tmp ( count );

    n = getData ( tmp.addr(), count, irow );

    buf[slice(BEGIN,n)] = tmp[slice(BEGIN,n)];
  }

  return n;
}


//-----------------------------------------------------------------------
//   getValues
//-----------------------------------------------------------------------


void FloatDBColumn::getValues

  ( const Vector&  buf,
    idx_t          count,
    idx_t          irow ) const

{
  JEM_PRECHECK2 ( count <= buf.size(),
                  "output array is too small" );

  idx_t  n;

  if ( buf.isContiguous() )
  {
    n = getData ( buf.addr(), count, irow );
  }
  else
  {
    Vector  tmp ( count );

    n = getData ( tmp.addr(), count, irow );

    buf[slice(BEGIN,n)] = tmp[slice(BEGIN,n)];
  }

  if ( n != count )
  {
    missingDataError_ ( irow, n, count );
  }
}


void FloatDBColumn::getValues

  ( const Matrix&  buf,
    idx_t          irow ) const

{
  const idx_t  count = buf.size(0) * buf.size(1);

  idx_t        n;


  if ( buf.isContiguous() )
  {
    n   = getData ( buf.addr(), count, irow );
  }
  else
  {
    Matrix  tmp ( buf.shape() );

    n   = getData ( tmp.addr(), count, irow );
    buf = tmp;
  }

  if ( n != count )
  {
    missingDataError_ ( irow, n, count );
  }
}


void FloatDBColumn::getValues

  ( const Cubix&  buf,
    idx_t         irow ) const

{
  const idx_t  count = buf.size(0) * buf.size(1) * buf.size(2);

  idx_t         n;


  if ( buf.isContiguous() )
  {
    n   = getData ( buf.addr(), count, irow );
  }
  else
  {
    Cubix  tmp ( buf.shape() );

    n   = getData ( tmp.addr(), count, irow );
    buf = tmp;
  }

  if ( n != count )
  {
    missingDataError_ ( irow, n, count );
  }
}


//-----------------------------------------------------------------------
//   setValues
//-----------------------------------------------------------------------


void FloatDBColumn::setValues

  ( idx_t          irow,
    const Vector&  buf,
    idx_t          count )

{
  JEM_PRECHECK2 ( count <= buf.size(),
                  "output array is too small" );

  if ( buf.isContiguous() )
  {
    setData ( irow, buf.addr(), count );
  }
  else
  {
    Vector  tmp ( count );

    tmp = buf[slice(BEGIN,count)];

    setData ( irow, tmp.addr(), count );
  }
}


void FloatDBColumn::setValues

  ( idx_t          irow,
    const Matrix&  buf )

{
  const idx_t  count = buf.size(0) * buf.size(1);

  if ( buf.isContiguous() )
  {
    setData ( irow, buf.addr(), count );
  }
  else
  {
    Matrix  tmp ( buf.shape() );

    tmp = buf;

    setData ( irow, tmp.addr(), count );
  }
}


void FloatDBColumn::setValues

  ( idx_t         irow,
    const Cubix&  buf )

{
  const idx_t  count = buf.size(0) * buf.size(1) * buf.size(2);

  if ( buf.isContiguous() )
  {
    setData ( irow, buf.addr(), count );
  }
  else
  {
    Cubix  tmp ( buf.shape() );

    tmp = buf;

    setData ( irow, tmp.addr(), count );
  }
}


//-----------------------------------------------------------------------
//   missingDataError_
//-----------------------------------------------------------------------


void FloatDBColumn::missingDataError_

  ( idx_t  irow,
    idx_t  actualCount,
    idx_t  formalCount ) const

{
  if ( formalCount == 1 )
  {
    throw IllegalInputException (
      getContext ( irow ),
      String::format (
        "missing data: expected one double (got none)"
      )
    );
  }
  else
  {
    throw IllegalInputException (
      getContext ( irow ),
      String::format (
        "missing data: expected %d doubles (got only %d)",
        actualCount,
        formalCount
      )
    );
  }
}


//=======================================================================
//   class Database
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Database::MAIN_DB = "main";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Database::Database ( const String& name ) :

  Super ( name )

{}


Database::~Database ()
{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Database::getContext () const
{
  String  itemName = getRowItems()->getItemName ();

  return NamedObject::makeContext ( itemName + " database",
                                    myName_ );
}


//-----------------------------------------------------------------------
//   getTagName
//-----------------------------------------------------------------------


String Database::getTagName () const
{
  return getTagName ( getRowItems()->getItemName() );
}


String Database::getTagName

  ( const String& rowItemName )

{
  return ( jem::xml::makeTag ( rowItemName ) + "Database" );
}


//-----------------------------------------------------------------------
//   clearData
//-----------------------------------------------------------------------


void Database::clearData ()
{
  clearDBColumns ( getIntColumns()   );
  clearDBColumns ( getFloatColumns() );
  resetEvents    ();
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void Database::trimToSize ()
{
  trimDBColumns ( getIntColumns()   );
  trimDBColumns ( getFloatColumns() );
}


//-----------------------------------------------------------------------
//   findIntColumn
//-----------------------------------------------------------------------


Database::IntColumn* Database::findIntColumn

  ( const String& name ) const

{
  return findDBColumn ( name, getIntColumns() );
}


//-----------------------------------------------------------------------
//   getIntColumn
//-----------------------------------------------------------------------


Database::IntColumn* Database::getIntColumn

  ( const String&  name ) const

{
  IntColumn*  col = findIntColumn ( name );

  if ( ! col )
  {
    throw IllegalInputException (
      getContext (),
      String::format (
        "undefined integer column: `%s\'", name
      )
    );
  }

  return col;
}


//-----------------------------------------------------------------------
//   findFloatColumn
//-----------------------------------------------------------------------


Database::FloatColumn* Database::findFloatColumn

  ( const String& name ) const

{
  return findDBColumn ( name, getFloatColumns() );
}


//-----------------------------------------------------------------------
//   getFloatColumn
//-----------------------------------------------------------------------


Database::FloatColumn* Database::getFloatColumn

  ( const String&  name ) const

{
  FloatColumn*  col = findFloatColumn ( name );

  if ( ! col )
  {
    throw IllegalInputException (
      getContext (),
      String::format (
        "undefined floating point column: `%s\'", name
      )
    );
  }

  return col;
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void Database::printTo ( PrintWriter& out ) const
{
  printDBColumns ( out, getIntColumns(),   "int"   );
  printDBColumns ( out, getFloatColumns(), "float" );
}


//-----------------------------------------------------------------------
//   printSome
//-----------------------------------------------------------------------


void Database::printSome

  ( PrintWriter&   out,
    const Dict&    dbases,
    const String&  tag )

{
  using jem::xml::beginStartTag;
  using jem::xml::endStartTag;
  using jem::xml::endTag;
  using jem::xml::attribute;

  Ref<DictEnum>  e = dbases.enumerate ();


  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<Object>  obj   = e->getValue ();
    Database*    dbase = dynamicCast<Database*> ( obj );

    if ( ! dbase )
    {
      continue;
    }

    String  dbaseName = e->getKey ();
    String  tagName   = tag;

    if ( tagName.size() == 0 )
    {
      tagName = dbase->getTagName ();
    }

    print ( out,
            beginStartTag( tagName ),
            attribute( "name", dbaseName ),
            endStartTag );

    dbase->printTo ( out );

    print ( out, endTag( tagName ) );
  }
}


//-----------------------------------------------------------------------
//   printAll
//-----------------------------------------------------------------------


void Database::printAll

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
//   store
//-----------------------------------------------------------------------


void Database::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  Ref<ItemSet>  items   = getRowItems ();
  Properties    itemdat = Globdat::getDataFor ( items, globdat );
  Ref<Object>   self    = const_cast<Self*>   ( this );

  Ref<Dict>     dbases;


  if ( ! itemdat.find( dbases, Globdat::DATABASES ) )
  {
    dbases = newInstance<HashDict> ();

    itemdat.set ( Globdat::DATABASES, dbases );
  }

  dbases->insert ( name, self );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<Database> Database::find

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Properties   itemdat = Globdat::getDataFor ( items, globdat );

  Ref<Dict>    dbases;


  if ( ! itemdat.find( dbases, Globdat::DATABASES ) )
  {
    return nullptr;
  }

  return dynamicCast<Database> ( dbases->get( name ) );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Database> Database::get

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const Properties&    globdat,
    const String&        context )

{
  Ref<Database>  dbase = find ( name, items, globdat );

  if ( ! dbase )
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

    throw IllegalInputException (
      context,
      String::format ( "undefined %s database: `%s\'",
                       itemName, name )
    );
  }

  return dbase;
}


//-----------------------------------------------------------------------
//   findFor
//-----------------------------------------------------------------------


Ref<Dictionary> Database::findFor

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Properties  itemdat ( Globdat::getDataFor( items, globdat ) );
  Ref<Dict>   dbases;

  itemdat.find ( dbases, Globdat::DATABASES );

  return dbases;
}


//-----------------------------------------------------------------------
//   getFor
//-----------------------------------------------------------------------


Ref<Dictionary> Database::getFor

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat ( Globdat::getDataFor( items, globdat ) );
  Ref<Dict>   dbases;

  if ( ! itemdat.find( dbases, Globdat::DATABASES ) )
  {
    dbases = newInstance<HashDict> ();

    itemdat.set ( Globdat::DATABASES, dbases );
  }

  return dbases;
}


JIVE_END_PACKAGE( util )
