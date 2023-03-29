
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
#include <jive/util/StdDatabase.h>
#include <jive/util/DBaseParser.h>
#include <jive/util/DBasesParser.h>


JEM_DEFINE_CLASS( jive::util::DBasesParser );


using jem::dynamicCast;
using jem::newInstance;
using jem::Error;
using jem::util::Pattern;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class DBasesParser
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DBasesParser::ATTRIBS_     = "| name";
const idx_t  DBasesParser::NAME_ATTRIB_ = 0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DBasesParser::DBasesParser

  ( const Ref<MPContext>&   mpx,
    const Ref<ItemSet>&     items,
    const Ref<Dictionary>&  dbases,
    const Ref<Dictionary>&  rowGroups,
    const String&           filter ) :

    Super       ( ATTRIBS_ ),
    rowItems_   (    items ),
    dbases_     (   dbases ),
    nameFilter_ (   filter )

{
  dbaseParser_ = newInstance<DBaseParser> ( mpx, nullptr, rowGroups );
  dbaseName_   = Database::MAIN_DB;

  Pattern::check ( filter );
}


DBasesParser::~DBasesParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void DBasesParser::reset ()
{
  Super::reset ();

  dbase_     = nullptr;
  dbaseName_ = Database::MAIN_DB;
}


//-----------------------------------------------------------------------
//   parseContents
//-----------------------------------------------------------------------


void DBasesParser::parseContents ( State& state )
{
  if ( Pattern::matches( nameFilter_, dbaseName_ ) )
  {
    newDatabase_ ();

    dbaseParser_->setDatabase ( dbase_ );
    state.pushParser          ( dbaseParser_ );
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


void DBasesParser::handleEmptyContents ( State& state )
{
  if ( Pattern::matches( nameFilter_, dbaseName_ ) )
  {
    newDatabase_ ();
  }
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void DBasesParser::finish ( State& state )
{
  if ( dbases_ && dbase_ )
  {
    dbases_->insert ( dbaseName_, dbase_ );
  }

  dbase_     = nullptr;
  dbaseName_ = Database::MAIN_DB;
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void DBasesParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  Super       ::takeAction ( action, params );
  dbaseParser_->takeAction ( action, params );

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
//   setDatabases
//-----------------------------------------------------------------------


void DBasesParser::setDatabases

  ( const Ref<ItemSet>&     items,
    const Ref<Dictionary>&  dbases )

{
  rowItems_ = items;
  dbases_   = dbases;
}


//-----------------------------------------------------------------------
//   setRowGroups
//-----------------------------------------------------------------------


void DBasesParser::setRowGroups ( const Ref<Dictionary>& groups )
{
  dbaseParser_->setRowGroups ( groups );
}


//-----------------------------------------------------------------------
//   setNameFilter
//-----------------------------------------------------------------------


void DBasesParser::setNameFilter ( const String& pattern )
{
  Pattern::check ( pattern );

  nameFilter_ = pattern;
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void DBasesParser::handleAttribute

  ( State&         state,
    const String&  name,
    const String&  value,
    idx_t          index )

{
  if ( index == NAME_ATTRIB_ )
  {
    dbaseName_ = value.stripWhite ();

    if ( dbaseName_.size() == 0 )
    {
      inputError ( state, "empty name" );
    }
    if ( scope_ )
    {
      dbaseName_ = scope_->expandName ( dbaseName_ );
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
//   newDatabase_
//-----------------------------------------------------------------------


void DBasesParser::newDatabase_ ()
{
  dbase_ = nullptr;

  if ( dbases_ )
  {
    dbase_ = dynamicCast<Database> ( dbases_->get( dbaseName_ ) );
  }

  if ( dbase_ == nullptr && rowItems_ != nullptr )
  {
    String name = joinNames ( rowItems_->getName(), dbaseName_ );

    dbase_ = newInstance<StdDatabase> ( name, rowItems_ );
  }
}


JIVE_END_PACKAGE( util )
