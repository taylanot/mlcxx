
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
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ParserScope.h>
#include <jive/util/ParserActions.h>
#include <jive/util/DenseTable.h>
#include <jive/util/SparseTable.h>
#include <jive/util/TableParser.h>
#include <jive/util/TablesParser.h>


JEM_DEFINE_CLASS( jive::util::TablesParser );


using jem::dynamicCast;
using jem::newInstance;
using jem::Error;
using jem::util::Pattern;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class TablesParser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  TablesParser::ATTRIBS_     = "name | type";
const idx_t  TablesParser::NAME_ATTRIB_ = 0;
const idx_t  TablesParser::TYPE_ATTRIB_ = 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


TablesParser::TablesParser

  ( const Ref<MPContext>&   mpx,
    const Ref<ItemSet>&     items,
    const Ref<Dictionary>&  tables,
    const Ref<Dictionary>&  rowGroups,
    const String&           filter ) :

    Super       (      ATTRIBS_ ),
    rowItems_   (         items ),
    tables_     (        tables ),
    tableType_  ( SPARSE_TABLE_ ),
    nameFilter_ (        filter )

{
  tableParser_ = newInstance<TableParser> ( mpx, nullptr, rowGroups );

  Pattern::check ( filter );
}


TablesParser::~TablesParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void TablesParser::reset ()
{
  Super::reset ();

  table_     = nullptr;
  tableName_ = "";
  tableType_ = SPARSE_TABLE_;
}


//-----------------------------------------------------------------------
//   parseContents
//-----------------------------------------------------------------------


void TablesParser::parseContents ( State& state )
{
  if ( Pattern::matches( nameFilter_, tableName_ ) )
  {
    newTable_ ( state );

    tableParser_->setTable ( table_ );
    state.pushParser       ( tableParser_ );
  }
  else
  {
    state.input->skipToEndTag  ( getTagName() );
    state.input->pushBackToken ();
  }
}


//-----------------------------------------------------------------------
//   handleEmptyContents
//-----------------------------------------------------------------------


void TablesParser::handleEmptyContents ( State& state )
{
  if ( Pattern::matches( nameFilter_, tableName_ ) )
  {
    newTable_ ( state );
  }
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void TablesParser::finish ( State& state )
{
  if ( tables_ && table_ )
  {
    tables_->insert ( tableName_, table_ );
  }

  table_     = nullptr;
  tableName_ = "";
  tableType_ = SPARSE_TABLE_;
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void TablesParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  Super       ::takeAction ( action, params );
  tableParser_->takeAction ( action, params );

  if ( action == ParserActions::NEW_SCOPE )
  {
    params.get ( scope_, ParserParams::SCOPE );

    if ( scope_->isGlobal() )
    {
      scope_ = nullptr;
    }
  }
}


//-----------------------------------------------------------------------
//   setTables
//-----------------------------------------------------------------------


void TablesParser::setTables

  ( const Ref<ItemSet>&     items,
    const Ref<Dictionary>&  tables )

{
  rowItems_ = items;
  tables_   = tables;
}


//-----------------------------------------------------------------------
//   setRowGroups
//-----------------------------------------------------------------------


void TablesParser::setRowGroups ( const Ref<Dictionary>& rowGroups )
{
  tableParser_->setRowGroups ( rowGroups );
}


//-----------------------------------------------------------------------
//   setNameFilter
//-----------------------------------------------------------------------


void TablesParser::setNameFilter ( const String& pattern )
{
  Pattern::check ( pattern );

  nameFilter_ = pattern;
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void TablesParser::handleAttribute

  ( State&         state,
    const String&  name,
    const String&  value,
    idx_t          index )

{
  if      ( index == NAME_ATTRIB_ )
  {
    tableName_ = value.stripWhite ();

    if ( tableName_.size() == 0 )
    {
      inputError ( state, "empty name" );
    }
    if ( scope_ )
    {
      tableName_ = scope_->expandName ( tableName_ );
    }
  }
  else if ( index == TYPE_ATTRIB_ )
  {
    String  type = value.toUpper ();

    type = type.stripWhite ();

    if      ( type == "DENSE" )
    {
      tableType_ = DENSE_TABLE_;
    }
    else if ( type == "SPARSE" )
    {
      tableType_ = SPARSE_TABLE_;
    }
    else
    {
      inputError (
        state,
        String::format (
          "invalid table type: `%s\'; "
          "should be `Dense\' or `Sparse\'", value
        )
      );
    }
  }
  else
  {
    throw Error (
      JEM_FUNC,
      "invalid attribute index: " + String( index )
    );
  }
}


//-----------------------------------------------------------------------
//   newTable_
//-----------------------------------------------------------------------


void TablesParser::newTable_ ( State& state )
{
  table_ = nullptr;

  if ( tables_ )
  {
    Ref<Table>  t =

      dynamicCast<Table> ( tables_->get( tableName_ ) );


    table_ = dynamicCast<XTable> ( t );

    if ( t && table_ == nullptr )
    {
      String  itemName = t->getRowItems()->getItemName ();

      inputError (
        state,
        String::format ( "%s table `%s\' can not be modified",
                         itemName,
                         tableName_ )
      );
    }
  }

  if ( table_ == nullptr && rowItems_ != nullptr )
  {
    String  name = joinNames ( rowItems_->getName(), tableName_ );

    if ( tableType_ == SPARSE_TABLE_ )
    {
      table_ = newInstance<SparseTable> ( name, rowItems_ );
    }
    else
    {
      table_ = newInstance<DenseTable>  ( name, rowItems_ );
    }
  }
}


JIVE_END_PACKAGE( util )
