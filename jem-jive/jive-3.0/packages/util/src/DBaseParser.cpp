
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


#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/TagParser.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/ParseUtils.h>
#include <jive/util/ParserActions.h>
#include <jive/util/Database.h>
#include <jive/util/IncludeParser.h>
#include <jive/util/DBaseParser.h>


JEM_DEFINE_CLASS( jive::util::DBaseParser );


JIVE_BEGIN_PACKAGE( util )


using jem::dynamicCast;
using jem::newInstance;
using jem::Error;
using jem::xml::ParseLog;
using jem::xml::Tokenizer;
using jem::xml::TagParser;


//=======================================================================
//   class DBColumnParser
//=======================================================================


class DBColumnParser : public TagParser
{
 public:

  typedef DBColumnParser    Self;
  typedef TagParser         Super;

  typedef Dictionary        Dict;
  typedef ParseUtils        Utils;
  typedef ParserScope       Scope;


  enum                      ColumnType
  {
                              INT_TYPE,
                              FLOAT_TYPE
  };


                            DBColumnParser  ();

  virtual void              reset           () override;

  virtual void              parseContents

    ( State&                  state )          override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params )         override;

  inline void               setDatabase

    ( const Ref<Database>&    dbase );

  inline void               setRowGroups

    ( const Ref<Dictionary>&  groups );


 protected:

  virtual                  ~DBColumnParser  ();

  virtual void              handleAttribute

    ( State&                  state,
      const String&           name,
      const String&           value,
      idx_t                   index )          override;


 private:

  void                      getColumn_

    ( State&                  state );

  void                      parseColumn_

    ( State&                  state );

  void                      scanColumn_

    ( State&                  state );

  void                      parseRow_

    ( State&                  state,
      idx_t                   rowID );

  void                      parseRow_

    ( State&                  state,
      idx_t                   first,
      idx_t                   last,
      idx_t                   stride );

  void                      parseRow_

    ( State&                  state,
      const String&           groupName );

  inline void               skipRow_

    ( State&                  state );

  idx_t                     readIntValues_

    ( State&                  state );

  idx_t                     readFloatValues_

    ( State&                  state );

  Ref<ItemGroup>            getRowGroup_

    ( State&                  state,
      const String&           groupName )      const;

  inline void               logProgress_

    ( State&                  state,
      idx_t                   rowCount )       const;

  void                      noSuchColumnError_

    ( State&                  state,
      const String&           name,
      const String&           type )           const;

  void                      noSuchRowError_

    ( State&                  state,
      idx_t                   rowID )          const;

  void                      noSuchGroupError_

    ( State&                  state,
      const String&           name )           const;


 private:

  static const char*        ATTRIBS_;

  static const idx_t        NAME_ATTRIB_;
  static const idx_t        TYPE_ATTRIB_;

  Ref<Scope>                scope_;
  Ref<Database>             dbase_;
  Ref<ItemSet>              rowItems_;
  Ref<ItemMap>              rowMap_;
  Ref<Dictionary>           rowGroups_;

  bool                      noNewCols_;
  bool                      skipUnknown_;

  String                    colName_;
  ColumnType                colType_;

  union
  {
    IntDBColumn*            icolumn_;
    FloatDBColumn*          fcolumn_;
  };

  IdxVector                 ivalues_;
  Vector                    fvalues_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DBColumnParser::ATTRIBS_     = "name | type";
const idx_t  DBColumnParser::NAME_ATTRIB_ = 0;
const idx_t  DBColumnParser::TYPE_ATTRIB_ = 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DBColumnParser::DBColumnParser () :

  Super        ( ATTRIBS_ ),
  noNewCols_   (    false ),
  skipUnknown_ (    false ),
  colType_     ( INT_TYPE ),
  icolumn_     (  nullptr )

{}


DBColumnParser::~DBColumnParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void DBColumnParser::reset ()
{
  Super::reset ();

  colName_ = "";
  colType_ = INT_TYPE;
  icolumn_ = nullptr;
}


//-----------------------------------------------------------------------
//   parseContents
//-----------------------------------------------------------------------


void DBColumnParser::parseContents ( State&  state )
{
  if ( dbase_ )
  {
    try
    {
      getColumn_   ( state );
      parseColumn_ ( state );
    }
    catch ( const ItemIDException& ex )
    {
      Utils::itemIDError ( state, *rowItems_, ex );
    }
  }
  else
  {
    try
    {
      scanColumn_ ( state );
    }
    catch ( const ItemIDException& ex )
    {
      inputError ( state, ex.what() );
    }
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void DBColumnParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  Super::takeAction ( action, params );

  if ( action == ParserActions::NEW_SCOPE )
  {
    params.get ( scope_, ParserParams::SCOPE );

    if ( scope_->isGlobal() )
    {
      scope_ = nullptr;
    }
  }
  else if ( action == ParserActions::SET_OPTIONS )
  {
    params.find ( noNewCols_,   DBaseParser::NO_NEW_COLUMNS );
    params.find ( skipUnknown_, ParserOptions::SKIP_UNKNOWN );
  }
  else if ( action == ParserActions::RESET_OPTIONS )
  {
    noNewCols_   = false;
    skipUnknown_ = false;
  }
}


//-----------------------------------------------------------------------
//   setDatabase
//-----------------------------------------------------------------------


inline void DBColumnParser::setDatabase

  ( const Ref<Database>&  dbase )

{
  dbase_ = dbase;

  if ( dbase_ )
  {
    rowItems_ = dbase_   ->getRowItems ();
    rowMap_   = rowItems_->getItemMap  ();
  }
  else
  {
    rowItems_ = nullptr;
    rowMap_   = nullptr;
  }
}


//-----------------------------------------------------------------------
//   setRowGroups
//-----------------------------------------------------------------------


inline void DBColumnParser::setRowGroups ( const Ref<Dict>&  groups )
{
  rowGroups_ = groups;
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void DBColumnParser::handleAttribute

  ( State&         state,
    const String&  name,
    const String&  value,
    idx_t          index )

{
  if      ( index == NAME_ATTRIB_ )
  {
    colName_ = value.stripWhite ();
  }
  else if ( index == TYPE_ATTRIB_ )
  {
    String  type = value.toUpper ();

    type = type.stripWhite ();

    if      ( type == "INT"   || type == "I" )
    {
      colType_ = INT_TYPE;
    }
    else if ( type == "FLOAT" || type == "F" )
    {
      colType_ = FLOAT_TYPE;
    }
    else
    {
      inputError (
        state,
        String::format (
          "invalid column type: `%s\' "
          "(should be `int\' or `float\')",
          value
        )
      );
    }
  }
  else
  {
    throw Error (
      JEM_FUNC,
      "invalid attribute index: " + String ( index )
    );
  }
}


//-----------------------------------------------------------------------
//   getColumn_
//-----------------------------------------------------------------------


void DBColumnParser::getColumn_ ( State& state )
{
  if ( colType_ == INT_TYPE )
  {
    if ( noNewCols_ )
    {
      icolumn_ = dbase_->findIntColumn ( colName_ );
    }
    else
    {
      icolumn_ = dbase_->addIntColumn  ( colName_ );
    }

    if ( ! icolumn_ )
    {
      noSuchColumnError_ ( state, colName_, "integer" );
    }
  }
  else
  {
    if ( noNewCols_ )
    {
      fcolumn_ = dbase_->findFloatColumn ( colName_ );
    }
    else
    {
      fcolumn_ = dbase_->addFloatColumn  ( colName_ );
    }

    if ( ! fcolumn_ )
    {
      noSuchColumnError_ ( state, colName_, "floating point" );
    }
  }
}


//-----------------------------------------------------------------------
//   parseColumn_
//-----------------------------------------------------------------------


void DBColumnParser::parseColumn_ ( State& state )
{
  const Scope*  scope =   scope_.get ();
  Tokenizer&    input = * state.input;

  idx_t         first, last, stride;
  idx_t         rowCount;
  int           token;


  rowCount = 0;

  while ( true )
  {
    token = input.nextToken ();

    rowCount++;

    if ( token == Tokenizer::INTEGER_TOKEN )
    {
      parseRow_ ( state, Utils::getItemID( input, scope ) );
    }
    else if ( token == '[' )
    {
      Utils::parseRange ( state, first, last, stride );
      parseRow_         ( state, first, last, stride );
    }
    else if ( token == Tokenizer::SYMBOL_TOKEN )
    {
      parseRow_ ( state, Utils::getSymbol( input, scope ) );
    }
    else if ( token == Tokenizer::STRING_TOKEN )
    {
      parseRow_ ( state, Utils::getString( input, scope ) );
    }
    else
    {
      break;
    }

    logProgress_ ( state, rowCount );
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   scanColumn_
//-----------------------------------------------------------------------


void DBColumnParser::scanColumn_ ( State& state )
{
  Tokenizer&  input = * state.input;

  idx_t       first, last, stride;
  idx_t       rowCount;
  int         token;


  rowCount = 0;

  while ( true )
  {
    token = input.nextToken ();

    rowCount++;

    if ( token == Tokenizer::INTEGER_TOKEN )
    {
      skipRow_ ( state );
    }
    else if ( token == '[' )
    {
      Utils::parseRange ( state, first, last, stride );
      skipRow_          ( state );
    }
    else if ( token == Tokenizer::SYMBOL_TOKEN )
    {
      skipRow_ ( state );
    }
    else if ( token == Tokenizer::STRING_TOKEN )
    {
      skipRow_ ( state );
    }
    else
    {
      break;
    }

    logProgress_ ( state, rowCount );
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   parseRow_ (given a rowID)
//-----------------------------------------------------------------------


void DBColumnParser::parseRow_

  ( State&  state,
    idx_t   rowID )

{
  idx_t  irow;
  idx_t  n;


  irow = rowMap_->findItem ( rowID );

  if ( irow < 0 )
  {
    if ( skipUnknown_ )
    {
      skipRow_ ( state );
      return;
    }
    else
    {
      noSuchRowError_ ( state, rowID );
    }
  }

  if ( colType_ == INT_TYPE )
  {
    n = readIntValues_  ( state );
    icolumn_->setValues ( irow, ivalues_, n );
  }
  else
  {
    n = readFloatValues_ ( state );
    fcolumn_->setValues  ( irow, fvalues_, n );
  }
}


//-----------------------------------------------------------------------
//   parseRow_ (given a range of row IDs)
//-----------------------------------------------------------------------


void DBColumnParser::parseRow_

  ( State&  state,
    idx_t   first,
    idx_t   last,
    idx_t   stride )

{
  IdxVector  ibuf;
  IdxVector  rowIDs;
  IdxVector  irows;

  idx_t      rowCount;
  idx_t      irow;
  idx_t      i, j, n;


  rowCount = last - first;

  if ( rowCount > 0 )
  {
    rowCount = 1 + (rowCount - 1) / stride;
  }

  if ( rowCount > rowItems_->size() )
  {
    inputError ( state, "range is too large" );
  }

  ibuf.resize ( 2 * rowCount );

  rowIDs.ref ( ibuf[slice(BEGIN,rowCount)] );
  irows .ref ( ibuf[slice(rowCount,END)]   );

  if ( ! scope_ )
  {
    for ( i = 0; i < rowCount; i++, first += stride )
    {
      rowIDs[i] = first;
    }
  }
  else
  {
    for ( i = 0; i < rowCount; i++, first += stride )
    {
      rowIDs[i] = scope_->expandID ( first );
    }
  }

  j = rowMap_->findItems ( irows, rowIDs );

  if ( j < rowCount )
  {
    j = 0;

    for ( i = 0; i < rowCount; i++ )
    {
      irow = irows[i];

      if ( irow >= 0 )
      {
        irows[j++] = irow;
      }
      else if ( ! skipUnknown_ )
      {
        noSuchRowError_ ( state, rowIDs[i] );
      }
    }

    rowCount = j;
    irows.ref ( irows[slice(BEGIN,rowCount)] );
  }

  if ( colType_ == INT_TYPE )
  {
    n = readIntValues_ ( state );

    for ( i = 0; i < rowCount; i++ )
    {
      icolumn_->setValues ( irows[i], ivalues_, n );
    }
  }
  else
  {
    n = readFloatValues_ ( state );

    for ( i = 0; i < rowCount; i++ )
    {
      fcolumn_->setValues ( irows[i], fvalues_, n );
    }
  }
}


//-----------------------------------------------------------------------
//   parseRow_ (given a row group)
//-----------------------------------------------------------------------


void DBColumnParser::parseRow_

  ( State&         state,
    const String&  groupName )

{
  Ref<ItemGroup>  rowGroup;
  IdxVector       irows;
  idx_t           rowCount;
  idx_t           i, n;


  rowGroup = getRowGroup_ ( state, groupName );

  if ( ! rowGroup )
  {
    if ( skipUnknown_ )
    {
      skipRow_ ( state );
      return;
    }
    else
    {
      noSuchGroupError_ ( state, groupName );
    }
  }

  irows.ref ( rowGroup->getIndices() );

  rowCount = irows.size ();

  if ( colType_ == INT_TYPE )
  {
    n = readIntValues_ ( state );

    for ( i = 0; i < rowCount; i++ )
    {
      icolumn_->setValues ( irows[i], ivalues_, n );
    }
  }
  else
  {
    n = readFloatValues_ ( state );

    for ( i = 0; i < rowCount; i++ )
    {
      fcolumn_->setValues ( irows[i], fvalues_, n );
    }
  }
}


//-----------------------------------------------------------------------
//   skipRow_
//-----------------------------------------------------------------------


inline void DBColumnParser::skipRow_ ( State& state )
{
  if ( colType_ == INT_TYPE )
  {
    readIntValues_   ( state );
  }
  else
  {
    readFloatValues_ ( state );
  }
}


//-----------------------------------------------------------------------
//   readIntValues_
//-----------------------------------------------------------------------


idx_t DBColumnParser::readIntValues_ ( State& state )

{
  Tokenizer&  input = * state.input;

  int         token;
  idx_t       i, n;


  i = 0;
  n = ivalues_.size ();

  while ( true )
  {
    token = input.nextToken ();

    if ( token != Tokenizer::INTEGER_TOKEN )
    {
      break;
    }

    if ( i == n )
    {
      IdxVector  tmp ( 2 * n + 16 );

      tmp[slice(BEGIN,n)] = ivalues_;
      n                   = tmp.size ();

      ivalues_.swap ( tmp );
    }

    ivalues_[i++] = (idx_t) input.getInteger ();
  }

  if ( token != ';' )
  {
    parseError ( state );
  }

  return i;
}


//-----------------------------------------------------------------------
//   readFloatValues_
//-----------------------------------------------------------------------


idx_t DBColumnParser::readFloatValues_ ( State& state )
{
  Tokenizer&  input = * state.input;

  int         token;
  idx_t       i, n;


  i = 0;
  n = fvalues_.size ();

  while ( true )
  {
    token = input.nextToken ();

    if ( token != Tokenizer::INTEGER_TOKEN &&
         token != Tokenizer::FLOAT_TOKEN )
    {
      break;
    }

    if ( i == n )
    {
      Vector  tmp ( 2 * n + 16 );

      tmp[slice(BEGIN,n)] = fvalues_;
      n                   = tmp.size ();

      fvalues_.swap ( tmp );
    }

    fvalues_[i++] = input.getFloat ();
  }

  if ( token != ';' )
  {
    parseError ( state );
  }

  return i;
}


//-----------------------------------------------------------------------
//   getRowGroup_
//-----------------------------------------------------------------------


Ref<ItemGroup> DBColumnParser::getRowGroup_

  ( State&         state,
    const String&  groupName ) const

{
  Ref<ItemGroup>  group;

  if ( rowGroups_ )
  {
    group = dynamicCast<ItemGroup> ( rowGroups_->get( groupName ) );
  }

  if ( group && group->getItems() != rowItems_.get() )
  {
    group = nullptr;
  }

  return group;
}


//-----------------------------------------------------------------------
//   logProgress_
//-----------------------------------------------------------------------


void DBColumnParser::logProgress_

  ( State&   state,
    idx_t    rowCount ) const

{
  if ( rowCount % 1000_idx == 0 )
  {
    state.log->logEvent (
      state,
      ParseLog::PROGRESS,
      String::format ( "number of rows read : %d", rowCount )
    );
  }
}


//-----------------------------------------------------------------------
//   noSuchColumnError_
//-----------------------------------------------------------------------


void DBColumnParser::noSuchColumnError_

  ( State&         state,
    const String&  name,
    const String&  type ) const

{
  inputError (
    state,
    String::format (
      "undefined %s column: `%s\'", type, name
    )
  );
}


//-----------------------------------------------------------------------
//   noSuchRowError_
//-----------------------------------------------------------------------


void DBColumnParser::noSuchRowError_

  ( State&  state,
    idx_t     rowID ) const

{
  String  rowName = rowItems_->getItemName ();

  inputError (
    state,
    String::format ( "undefined %s: %d", rowName, rowID )
  );
}


//-----------------------------------------------------------------------
//   noSuchGroupError_
//-----------------------------------------------------------------------


void DBColumnParser::noSuchGroupError_

  ( State&         state,
    const String&  groupName ) const

{
  String  rowName = rowItems_->getItemName ();

  inputError (
    state,
    String::format (
      "undefined %s group: `%s\'", rowName, groupName
    )
  );
}


//=======================================================================
//   class DBaseParser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DBaseParser::NO_NEW_COLUMNS = "noNewDBColumns";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DBaseParser::DBaseParser

  ( const Ref<MPContext>&   mpx,
    const Ref<Database>&    dbase,
    const Ref<Dictionary>&  rowGroups ) :

    mpx_ ( mpx )

{
  parser_ = newInstance<DBColumnParser> ();

  parser_->setDatabase  ( dbase     );
  parser_->setRowGroups ( rowGroups );
}


DBaseParser::~DBaseParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void DBaseParser::reset ()
{
  parser_->reset ();
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void DBaseParser::parse ( State& state )
{
  Tokenizer&  input = * state.input;


  if ( input.nextToken() == Tokenizer::START_TAG_TOKEN )
  {
    String  tagName = input.getTagName ();

    if ( tagName.equalsIgnoreCase( "Column" ) )
    {
      state.pushParser ( this );
      state.pushParser ( parser_ );

      return;
    }
    else if ( tagName.equalsIgnoreCase( "Include" ) )
    {
      state.pushParser ( this );
      state.pushParser ( newInstance<IncludeParser>( mpx_ ) );

      return;
    }
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void DBaseParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  parser_->takeAction ( action, params );
}


//-----------------------------------------------------------------------
//   setDatabase
//-----------------------------------------------------------------------


void DBaseParser::setDatabase ( const Ref<Database>& dbase )
{
  parser_->setDatabase ( dbase );
}


//-----------------------------------------------------------------------
//   setRowGroups
//-----------------------------------------------------------------------


void DBaseParser::setRowGroups ( const Ref<Dictionary>& dict )
{
  parser_->setRowGroups ( dict );
}


JIVE_END_PACKAGE( util )
