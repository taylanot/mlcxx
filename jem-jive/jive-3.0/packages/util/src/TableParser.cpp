
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
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/util/StringUtils.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/TagParser.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/ParseUtils.h>
#include <jive/util/ParserActions.h>
#include <jive/util/XTable.h>
#include <jive/util/IncludeParser.h>
#include <jive/util/TableParser.h>


JEM_DEFINE_CLASS( jive::util::TableParser );


JIVE_BEGIN_PACKAGE( util )


using jem::dynamicCast;
using jem::newInstance;
using jem::util::StringUtils;
using jem::xml::ParseLog;
using jem::xml::Tokenizer;
using jem::xml::TagParser;


//=======================================================================
//   class TBSectionParser
//=======================================================================


class TBSectionParser : public TagParser
{
 public:

  typedef TBSectionParser   Self;
  typedef TagParser         Super;
  typedef ParseUtils        Utils;
  typedef ParserScope       Scope;


                            TBSectionParser   ();

  virtual void              reset             () override;

  virtual void              parseContents

    ( State&                  state  )           override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params )           override;

  inline void               setTable

    ( const Ref<XTable>&      table  );

  inline void               setRowGroups

    ( const Ref<Dictionary>&  groups );


 protected:

  virtual                  ~TBSectionParser   ();

  virtual void              handleAttribute

    ( State&                  state,
      const String&           name,
      const String&           value,
      idx_t                   index )            override;


 private:

  void                      parseSection_

    ( State&                  state  );

  void                      scanSection_

    ( State&                  state  );

  void                      parseRow_

    ( State&                  state,
      idx_t                   rowID  )           const;

  void                      parseRow_

    ( State&                  state,
      idx_t                   first,
      idx_t                   last,
      idx_t                   stride )           const;

  void                      parseRow_

    ( State&                  state,
      const String&           groupName )        const;

  void                      skipRow_

    ( State&                  state )            const;

  inline double             readValue_

    ( State&                  state )            const;

  Ref<ItemGroup>            getRowGroup_

    ( State&                  state,
      const String&           groupName )        const;

  inline void               logProgress_

    ( State&                  state,
      idx_t                   rowCount )         const;

  void                      noSuchRowError_

    ( State&                  state,
      idx_t                   rowID )            const;

  void                      noSuchGroupError_

    ( State&                  state,
      const String&           groupName )        const;


 private:

  Ref<Scope>                scope_;
  Ref<XTable>               table_;
  Ref<ItemSet>              rowItems_;
  Ref<ItemMap>              rowMap_;
  Ref<Dictionary>           rowGroups_;

  bool                      noNewCols_;
  bool                      skipUnknown_;

  IdxVector                 jcols_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


TBSectionParser::TBSectionParser () :

  Super        ( "columns" ),
  noNewCols_   (     false ),
  skipUnknown_ (     false )

{}


TBSectionParser::~TBSectionParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void TBSectionParser::reset ()
{
  Super::reset  ();
  jcols_.resize ( 0 );
}


//-----------------------------------------------------------------------
//   parseContents
//-----------------------------------------------------------------------


void TBSectionParser::parseContents ( State& state )
{
  if ( table_ )
  {
    try
    {
      parseSection_ ( state );
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
      scanSection_ ( state );
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


void TBSectionParser::takeAction

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
    params.find ( noNewCols_,   TableParser::NO_NEW_COLUMNS );
    params.find ( skipUnknown_, ParserOptions::SKIP_UNKNOWN );
  }
  else if ( action == ParserActions::RESET_OPTIONS )
  {
    noNewCols_   = false;
    skipUnknown_ = false;
  }
}


//-----------------------------------------------------------------------
//   setTable
//-----------------------------------------------------------------------


inline void TBSectionParser::setTable ( const Ref<XTable>& table )
{
  table_ = table;

  if ( table_ )
  {
    rowItems_ = table    ->getRowItems ();
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


inline void TBSectionParser::setRowGroups

  ( const Ref<Dictionary>& groups )

{
  rowGroups_ = groups;
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void TBSectionParser::handleAttribute

  ( State&         state,
    const String&  name,
    const String&  value,
    idx_t          index )

{
  StringVector  columns  =

    StringUtils::split ( value.stripWhite(), '|' );

  const idx_t   colCount = columns.size ();


  for ( idx_t j = 0; j < colCount; j++ )
  {
    columns[j] = columns[j].stripWhite ();

    if ( columns[j].size() == 0 )
    {
      inputError ( state, "empty column name" );
    }
  }

  jcols_.resize ( colCount );

  if ( ! table_ )
  {
    jcols_ = -1;
    return;
  }

  for ( idx_t j = 0; j < colCount; j++ )
  {
    idx_t  jcol = table_->findColumn ( columns[j] );

    if ( jcol < 0 )
    {
      if ( noNewCols_ )
      {
        inputError (
          state,
          String::format (
            "invalid column name: `%s\'", columns[j]
          )
        );
      }
      else
      {
        jcol = table_->addColumn ( columns[j] );
      }
    }

    jcols_[j] = jcol;
  }
}


//-----------------------------------------------------------------------
//   parseSection_
//-----------------------------------------------------------------------


void TBSectionParser::parseSection_ ( State& state )
{
  const Scope*  scope =   scope_.get ();
  Tokenizer&    input = * state.input;

  idx_t         first, last, stride;
  idx_t         rowCount;
  int           token;


  table_->reserve ( rowItems_->size() / 4 + 32 );

  rowCount = 0;
  token    = input.nextToken ();

  while ( true )
  {
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

    token = input.nextToken ();

    // A line-terminating ';' is optional

    if ( token == ';' )
    {
      token = input.nextToken ();
    }
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   scanSection_
//-----------------------------------------------------------------------


void TBSectionParser::scanSection_ ( State&  state )
{
  Tokenizer&  input = * state.input;

  idx_t       first, last, stride;
  idx_t       rowCount;
  int         token;


  rowCount = 0;
  token    = input.nextToken ();

  while ( true )
  {
    rowCount++;

    if      ( token == Tokenizer::INTEGER_TOKEN )
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

    token = input.nextToken ();

    // A line-terminating ';' is optional

    if ( token == ';' )
    {
      token = input.nextToken ();
    }
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   parseRow_ (given a rowID)
//-----------------------------------------------------------------------


void TBSectionParser::parseRow_

  ( State&  state,
    idx_t   rowID ) const

{
  const idx_t  colCount = jcols_.size ();

  idx_t        irow     = rowMap_->findItem ( rowID );

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

  for ( idx_t j = 0; j < colCount; j++ )
  {
    table_->setValue ( irow, jcols_[j], readValue_( state ) );
  }
}


//-----------------------------------------------------------------------
//   parseRow_ (given a range of row IDs)
//-----------------------------------------------------------------------


void TBSectionParser::parseRow_

  ( State&  state,
    idx_t   first,
    idx_t   last,
    idx_t   stride ) const

{
  const idx_t  colCount = jcols_.size ();

  Matrix       values;
  IdxVector    ibuf;
  IdxVector    rowIDs;
  IdxVector    irows;
  double       value;
  idx_t        irow;
  idx_t        rowCount;
  idx_t        i, j;


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

  values.resize ( rowCount, colCount );

  for ( j = 0; j < colCount; j++ )
  {
    value = readValue_ ( state );

    for ( i = 0; i < rowCount; i++ )
    {
      values(i,j) = value;
    }
  }

  table_->setBlock ( irows, jcols_, values );
}


//-----------------------------------------------------------------------
//   parseRow_ (given a row group)
//-----------------------------------------------------------------------


void TBSectionParser::parseRow_

  ( State&         state,
    const String&  groupName ) const

{
  const idx_t     colCount = jcols_.size ();

  Ref<ItemGroup>  rowGroup;
  Matrix          values;
  double          value;
  idx_t           rowCount;


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

  rowCount = rowGroup->size ();

  values.resize ( rowCount, colCount );

  for ( idx_t j = 0; j < colCount; j++ )
  {
    value = readValue_ ( state );

    for ( idx_t i = 0; i < rowCount; i++ )
    {
      values(i,j) = value;
    }
  }

  table_->setBlock ( rowGroup->getIndices(), jcols_, values );
}


//-----------------------------------------------------------------------
//   skipRow_
//-----------------------------------------------------------------------


void TBSectionParser::skipRow_ ( State& state ) const
{
  const idx_t  colCount = jcols_.size ();

  for ( idx_t j = 0; j < colCount; j++ )
  {
    readValue_ ( state );
  }
}


//-----------------------------------------------------------------------
//   readValue_
//-----------------------------------------------------------------------


inline double TBSectionParser::readValue_ ( State&  state ) const
{
  int  token = state.input->nextToken ();

  if      ( token == Tokenizer::INTEGER_TOKEN )
  {
    return (double) state.input->getInteger ();
  }
  else if ( token == Tokenizer::FLOAT_TOKEN )
  {
    return state.input->getFloat ();
  }
  else
  {
    parseError ( state, "expected a floating point number" );
    return 0.0;
  }
}


//-----------------------------------------------------------------------
//   getRowGroup_
//-----------------------------------------------------------------------


Ref<ItemGroup> TBSectionParser::getRowGroup_

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


void TBSectionParser::logProgress_

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
//   noSuchRowError_
//-----------------------------------------------------------------------


void TBSectionParser::noSuchRowError_

  ( State&  state,
    idx_t   rowID ) const

{
  String rowName = rowItems_->getItemName ();

  inputError (
    state,
    String::format ( "undefined %s: %d", rowName, rowID )
  );
}


//-----------------------------------------------------------------------
//   noSuchGroupError_
//-----------------------------------------------------------------------


void TBSectionParser::noSuchGroupError_

  ( State&         state,
    const String&  groupName ) const

{
  String rowName = rowItems_->getItemName ();

  inputError (
    state,
    String::format (
      "undefined %s group: `%s\'", rowName, groupName
    )
  );
}


//=======================================================================
//   class TableParser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  TableParser::NO_NEW_COLUMNS = "noNewTableColumns";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


TableParser::TableParser

  ( const Ref<MPContext>&   mpx,
    const Ref<XTable>&      table,
    const Ref<Dictionary>&  rowGroups ) :

    mpx_ ( mpx )

{
  parser_ = newInstance<TBSectionParser> ();

  parser_->setTable     ( table     );
  parser_->setRowGroups ( rowGroups );
}


TableParser::~TableParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void TableParser::reset ()
{
  parser_->reset ();
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void TableParser::parse ( State& state )
{
  Tokenizer&  input = * state.input;

  if ( input.nextToken() == Tokenizer::START_TAG_TOKEN )
  {
    String  tagName = input.getTagName ();

    if      ( tagName.equalsIgnoreCase( "Section" ) )
    {
      state.pushParser ( this    );
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


void TableParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  parser_->takeAction ( action, params );
}


//-----------------------------------------------------------------------
//   setTable
//-----------------------------------------------------------------------


void TableParser::setTable ( const Ref<XTable>& table )
{
  parser_->setTable ( table );
}


//-----------------------------------------------------------------------
//   setRowGroups
//-----------------------------------------------------------------------


void TableParser::setRowGroups ( const Ref<Dictionary>& dict )
{
  parser_->setRowGroups ( dict );
}


JIVE_END_PACKAGE( util )
