
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
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ParserScope.h>
#include <jive/util/ParserActions.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/util/ItemGroupParser.h>
#include <jive/util/ItemGroupsParser.h>


JEM_DEFINE_CLASS( jive::util::ItemGroupsParser );


using jem::dynamicCast;
using jem::newInstance;
using jem::util::Pattern;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class ItemGroupsParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ItemGroupsParser::ItemGroupsParser

  ( const Ref<ItemSet>&     items,
    const Ref<Dictionary>&  groups,
    const String&           filter ) :

    Super       ( "name" ),
    items_      (  items ),
    groups_     ( groups ),
    nameFilter_ ( filter )

{
  groupParser_ = newInstance<ItemGroupParser> ( nullptr, groups );

  Pattern::check ( filter );
}


ItemGroupsParser::~ItemGroupsParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void ItemGroupsParser::reset ()
{
  Super       ::reset ();
  groupParser_->reset ();

  groupName_ = "";
}


//-----------------------------------------------------------------------
//   parseContents
//-----------------------------------------------------------------------


void ItemGroupsParser::parseContents ( State& state )
{
  if ( Pattern::matches( nameFilter_, groupName_ ) )
  {
    newGroup_ ( state );

    groupParser_->setGroup ( group_ );
    state.pushParser ( groupParser_ );
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


void ItemGroupsParser::handleEmptyContents ( State& state )
{
  if ( Pattern::matches( nameFilter_, groupName_ ) )
  {
    newGroup_ ( state );
  }
}

//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void ItemGroupsParser::finish ( State& state )
{
  if ( groups_ && group_ )
  {
    groups_->insert ( groupName_, group_ );
  }

  group_     = nullptr;
  groupName_ = "";
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void ItemGroupsParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  Super       ::takeAction ( action, params );
  groupParser_->takeAction ( action, params );

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
//   setGroups
//-----------------------------------------------------------------------


void ItemGroupsParser::setGroups

  ( const Ref<ItemSet>&     items,
    const Ref<Dictionary>&  groups )

{
  items_  = items;
  groups_ = groups;

  groupParser_->setGroups ( groups );
}


//-----------------------------------------------------------------------
//   setNameFilter
//-----------------------------------------------------------------------


void ItemGroupsParser::setNameFilter ( const String& pattern )
{
  Pattern::check ( pattern );

  nameFilter_ = pattern;
}


//-----------------------------------------------------------------------
//   handleAttribute
//-----------------------------------------------------------------------


void ItemGroupsParser::handleAttribute

  ( State&         state,
    const String&  name,
    const String&  value,
    idx_t          index )

{
  groupName_ = value.stripWhite ();

  if ( groupName_.size() == 0 )
  {
    inputError ( state, "empty name" );
  }

  if ( scope_ )
  {
    groupName_ = scope_->expandName ( groupName_ );
  }
}


//-----------------------------------------------------------------------
//   newGroup_
//-----------------------------------------------------------------------


void ItemGroupsParser::newGroup_ ( State& state )
{
  group_ = nullptr;

  if ( groups_ )
  {
    Ref<ItemGroup>  g =

      dynamicCast<ItemGroup> ( groups_->get( groupName_ ) );

    group_ = dynamicCast<XItemGroup> ( g );

    if ( g != nullptr && group_ == nullptr )
    {
      String  itemName = g->getItems()->getItemName ();

      inputError (
        state,
        String::format ( "%s group `%s\' can not be modified",
                         itemName,
                         groupName_ )
      );
    }
  }

  if ( group_ == nullptr && items_ != nullptr )
  {
    group_ = newInstance<ArrayItemGroup> ( items_ );
  }
}


JIVE_END_PACKAGE( util )
