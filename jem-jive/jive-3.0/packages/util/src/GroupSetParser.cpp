
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
#include <jem/util/ArrayBuffer.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/XGroupSet.h>
#include <jive/util/ParseUtils.h>
#include <jive/util/ParserActions.h>
#include <jive/util/GroupSetParser.h>


JEM_DEFINE_CLASS( jive::util::GroupSetParser );


JIVE_BEGIN_PACKAGE( util )


using jem::util::ArrayBuffer;
using jem::xml::ParseLog;


//=======================================================================
//   class GroupSetParser::Utils_
//=======================================================================


class GroupSetParser::Utils_ : public ParseUtils
{
 public:

  typedef
    ArrayBuffer<idx_t>      IdxBuffer;


  static void               parse

    ( State&                  state,
      XGroupSet&              groups,
      const Scope*            scope,
      bool                    skipUnknown );

  static void               parseFixed

    ( State&                  state,
      XGroupSet&              groups,
      const Scope*            scope,
      idx_t                   groupSize );

  static void               scan

    ( State&                  state,
      const Scope*            scope );

  static void               parseGroup

    ( State&                  state,
      IdxBuffer&              iitems,
      const ItemSet&          items,
      const ItemMap&          imap,
      const Scope*            scope,
      bool                    skipUnknown );

  static void               parseFixedGroup

    ( State&                  state,
      const IdxVector&        iitems,
      const ItemSet&          items,
      const ItemMap&          imap,
      const Scope*            scope );

  static inline void        logProgress

    ( State&                  state,
      idx_t&                  counter,
      const String&           pointName,
      idx_t                   pointID );

  static inline void        checkForSemiColon

    ( State&                  state,
      int                     token );

  static void               noSuchItemError

    ( State&                  state,
      idx_t                   itemID,
      const ItemSet&          items );

};


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void GroupSetParser::Utils_::parse

  ( State&        state,
    XGroupSet&    groups,
    const Scope*  scope,
    bool          skipUnknown )

{
  const String    groupName =   groups.getItemName     ();
  const ItemSet&  items     = * groups.getGroupedItems ();
  const ItemMap&  itemMap   = * items.getItemMap       ();
  Tokenizer&      input     = * state.input;

  IdxBuffer       iitems;
  idx_t           groupID;
  idx_t           counter;


  counter = 0;

  while ( input.nextToken() == Tokenizer::INTEGER_TOKEN )
  {
    counter++;

    groupID = getItemID ( input, scope );

    try
    {
      parseGroup ( state,   iitems, items,
                   itemMap, scope,  skipUnknown );
    }
    catch ( const ItemIDException& ex )
    {
      itemIDError ( state, items, ex );
    }

    groups.addGroup ( groupID, iitems.toArray() );
    logProgress     ( state, counter, groupName, groupID );
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   parseFixed
//-----------------------------------------------------------------------


void GroupSetParser::Utils_::parseFixed

  ( State&        state,
    XGroupSet&    groups,
    const Scope*  scope,
    idx_t         groupSize )

{
  const String    groupName =   groups.getItemName     ();
  const ItemSet&  items     = * groups.getGroupedItems ();
  const ItemMap&  itemMap   = * items.getItemMap       ();
  Tokenizer&      input     = * state.input;

  IdxVector       iitems ( groupSize );
  idx_t           groupID;
  idx_t           counter;


  counter = 0;

  while ( input.nextToken() == Tokenizer::INTEGER_TOKEN )
  {
    counter++;

    groupID = getItemID ( input, scope );

    try
    {
      parseFixedGroup ( state, iitems, items, itemMap, scope );
    }
    catch ( const ItemIDException& ex )
    {
      itemIDError ( state, items, ex );
    }

    groups.addGroup ( groupID, iitems );
    logProgress     ( state, counter,  groupName, groupID );
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   scan
//-----------------------------------------------------------------------


void GroupSetParser::Utils_::scan

  ( State&        state,
    const Scope*  scope )

{
  const String  groupName =   "group";
  Tokenizer&    input     = * state.input;

  int           token;
  idx_t         first, last, stride;
  idx_t         groupID;
  idx_t         counter;


  counter = 0;

  while ( input.nextToken() == Tokenizer::INTEGER_TOKEN )
  {
    counter++;

    groupID = getItemID ( input, scope );

    while ( true )
    {
      token = input.nextToken ();

      if ( token == '[' )
      {
        parseRange ( state, first, last, stride );
      }
      else if ( token != Tokenizer::INTEGER_TOKEN )
      {
        break;
      }
    }

    checkForSemiColon ( state, token );
    logProgress       ( state, counter, groupName, groupID );
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   parseGroup
//-----------------------------------------------------------------------


void GroupSetParser::Utils_::parseGroup

  ( State&          state,
    IdxBuffer&      iitems,
    const ItemSet&  items,
    const ItemMap&  imap,
    const Scope*    scope,
    bool            skipUnknown )

{
  Tokenizer&  input = *state.input;

  int         token;
  idx_t       first, last, stride;
  idx_t       itemID;
  idx_t       iitem;


  iitems.clear ();

  while ( true )
  {
    token = input.nextToken ();

    if ( token == Tokenizer::INTEGER_TOKEN )
    {
      itemID = getItemID ( input, scope );
      iitem  = imap.findItem ( itemID );

      if ( iitem >= 0 )
      {
        iitems.pushBack ( iitem );
      }
      else if ( ! skipUnknown )
      {
        noSuchItemError ( state, itemID, items );
      }
    }
    else if ( token == '[' )
    {
      parseRange ( state, first, last, stride );

      for ( ; first < last; first += stride )
      {
        if ( scope )
        {
          iitem = imap.findItem ( scope->expandID( first ) );
        }
        else
        {
          iitem = imap.findItem ( first );
        }

        if ( iitem >= 0 )
        {
          iitems.pushBack ( iitem );
        }
        else if ( ! skipUnknown )
        {
          noSuchItemError ( state, first, items );
        }
      }
    }
    else
    {
      break;
    }
  }

  checkForSemiColon ( state, token );
}


//-----------------------------------------------------------------------
//   parseFixedGroup
//-----------------------------------------------------------------------


void GroupSetParser::Utils_::parseFixedGroup

  ( State&            state,
    const IdxVector&  iitems,
    const ItemSet&    items,
    const ItemMap&    imap,
    const Scope*      scope )

{
  Tokenizer&    input = *state.input;
  const idx_t   n     =  iitems.size ();

  int           token;
  idx_t         first, last, stride;
  idx_t         itemID;
  idx_t         iitem;
  idx_t         i;


  i = 0;

  while ( i < n )
  {
    token = input.nextToken ();

    if ( token == Tokenizer::INTEGER_TOKEN )
    {
      itemID = getItemID ( input, scope );
      iitem  = imap.findItem ( itemID );

      if ( iitem < 0 )
      {
        noSuchItemError ( state, itemID, items );
      }

      iitems[i++] = iitem;
    }
    else if ( token == '[' )
    {
      parseRange ( state, first, last, stride );

      for ( ; first < last && i < n; first += stride )
      {
        if ( scope )
        {
          iitem = imap.findItem ( scope->expandID( first ) );
        }
        else
        {
          iitem = imap.findItem ( first );
        }

        if ( iitem < 0 )
        {
          noSuchItemError ( state, first, items );
        }

        iitems[i++] = iitem;
      }

      if ( first < last )
      {
        parseError ( state, "range is too large" );
      }
    }
    else
    {
      parseError ( state );
    }
  }

  checkForSemiColon ( state, input.nextToken() );
}


//-----------------------------------------------------------------------
//   logProgress
//-----------------------------------------------------------------------


inline void GroupSetParser::Utils_::logProgress

  ( State&         state,
    idx_t&         counter,
    const String&  groupName,
    idx_t          groupID )

{
  if ( counter > 1000_idx )
  {
    counter = 0;

    state.log->logEvent (
      state,
      ParseLog::PROGRESS,
      String::format ( "at %s %d", groupName, groupID )
    );
  }
}


//-----------------------------------------------------------------------
//   checkForSemiColon
//-----------------------------------------------------------------------


inline void GroupSetParser::Utils_::checkForSemiColon

  ( State&  state,
    int     token )

{
  if ( token != ';' )
  {
    parseError ( state, "expected a `;\'" );
  }
}


//-----------------------------------------------------------------------
//   noSuchItemError
//-----------------------------------------------------------------------


void GroupSetParser::Utils_::noSuchItemError

  ( State&          state,
    idx_t           itemID,
    const ItemSet&  items )

{
  String itemName = items.getItemName ();

  inputError (
    state,
    String::format (
      "%s %d has not been defined", itemName, itemID
    )
  );
}


//=======================================================================
//   class GroupSetParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GroupSetParser::GroupSetParser

  ( const Ref<XGroupSet>&  groups,
    idx_t                  groupSize ) :

    groups_      (    groups ),
    groupSize_   ( groupSize ),
    skipUnknown_ (     false )

{}


GroupSetParser::~GroupSetParser ()
{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void GroupSetParser::parse ( State& state )
{
  ParserScope*  scope = scope_.get ();


  if ( ! groups_ )
  {
    try
    {
      Utils_::scan ( state, scope );
    }
    catch ( const ItemIDException& ex )
    {
      inputError ( state, ex.what() );
    }
  }
  else
  {
    try
    {
      if ( groupSize_ >= 0 )
      {
        Utils_::parseFixed ( state, * groups_,
                             scope,   groupSize_ );
      }
      else
      {
        Utils_::parse      ( state, * groups_,
                             scope,   skipUnknown_ );
      }
    }
    catch ( const ItemIDException& ex )
    {
      ParseUtils::itemIDError ( state, *groups_, ex );
    }
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void GroupSetParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
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
    params.find ( skipUnknown_, ParserOptions::SKIP_UNKNOWN );
  }
  else if ( action == ParserActions::RESET_OPTIONS )
  {
    skipUnknown_ = false;
  }
}


//-----------------------------------------------------------------------
//   setGroups
//-----------------------------------------------------------------------


void GroupSetParser::setGroups ( const Ref<XGroupSet>& groups )
{
  groups_ = groups;
}


//-----------------------------------------------------------------------
//   setGroupSize
//-----------------------------------------------------------------------


void GroupSetParser::setGroupSize ( idx_t groupSize )
{
  groupSize_ = groupSize;
}


JIVE_END_PACKAGE( util )
