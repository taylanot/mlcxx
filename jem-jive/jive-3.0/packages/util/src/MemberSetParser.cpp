
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
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/ParseUtils.h>
#include <jive/util/ParserActions.h>
#include <jive/util/XMemberSet.h>
#include <jive/util/MemberSetParser.h>


JEM_DEFINE_CLASS( jive::util::MemberSetParser );


using jem::xml::ParseLog;
using jem::xml::Tokenizer;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class MemberSetParser::Utils_
//=======================================================================


class MemberSetParser::Utils_
{
 public:

  static inline void  logProgress

    ( State&            state,
      idx_t&            counter,
      const String&     memberName,
      idx_t             memberID );

};


//-----------------------------------------------------------------------
//   logProgress
//-----------------------------------------------------------------------


inline void MemberSetParser::Utils_::logProgress

  ( State&         state,
    idx_t&         counter,
    const String&  memberName,
    idx_t          memberID )

{
  if ( counter > 1000_idx )
  {
    counter = 0;

    state.log->logEvent (
      state,
      ParseLog::PROGRESS,
      String::format ( "at %s %d", memberName, memberID )
    );
  }
}


//=======================================================================
//   class MemberSetParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MemberSetParser::MemberSetParser

  ( const Ref<XMemberSet>&  members,
    idx_t                   maxLocalIndex ) :

    members_       (       members ),
    maxLocalIndex_ ( maxLocalIndex ),
    skipUnknown_   (         false )

{}


MemberSetParser::~MemberSetParser ()
{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void MemberSetParser::parse ( State& state )
{
  if ( ! members_ )
  {
    try
    {
      scan_ ( state );
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
      parse_ ( state );
    }
    catch ( const ItemIDException& ex )
    {
      ParseUtils::itemIDError ( state, *members_, ex );
    }
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void MemberSetParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if      ( action == ParserActions::NEW_SCOPE )
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
//   setMembers
//-----------------------------------------------------------------------


void MemberSetParser::setMembers ( const Ref<XMemberSet>& members )
{
  members_ = members;
}


//-----------------------------------------------------------------------
//   setMaxLocalIndex
//-----------------------------------------------------------------------


void MemberSetParser::setMaxLocalIndex ( idx_t maxIndex )
{
  maxLocalIndex_ = maxIndex;
}


//-----------------------------------------------------------------------
//   parse_
//-----------------------------------------------------------------------


void MemberSetParser::parse_ ( State& state )
{
  const ItemSet&      items    = * members_->getCompoundItems ();
  const ItemMap&      itemMap  = * items.getItemMap           ();
  const String        itemName =   items.getItemName          ();
  const String        mbrName  =   members_->getItemName      ();
  const ParserScope*  scope    =   scope_.get                 ();

  Tokenizer&          input    = * state.input;

  idx_t               mbrID;
  idx_t               itemID;
  idx_t               iitem;
  idx_t               ilocal;
  idx_t               counter;
  int                 token;


  members_->reserve ( items.size() / 16 + 32 );

  itemID  = 0;
  counter = 0;
  token   = input.nextToken ();

  while ( token == Tokenizer::INTEGER_TOKEN )
  {
    mbrID = ParseUtils::getItemID ( input, scope );

    if ( input.nextToken() != Tokenizer::INTEGER_TOKEN )
    {
      inputError (
        state,
        String::format ( "illegal %s ID", mbrName )
      );
    }

    try
    {
      itemID = ParseUtils::getItemID ( input, scope );
    }
    catch ( const ItemIDException& ex )
    {
      ParseUtils::itemIDError ( state, items, ex );
    }

    iitem = itemMap.findItem ( itemID );

    if ( iitem < 0 && ! skipUnknown_ )
    {
      inputError (
        state,
        String::format (
          "undefined %s: %d", itemName, itemID
        )
      );
    }

    if ( input.nextToken() != Tokenizer::INTEGER_TOKEN )
    {
      parseError ( state, mbrName + " index expected" );
    }

    ilocal = (idx_t) input.getInteger ();

    if ( ilocal < 0 ||
         (maxLocalIndex_ >= 0 && ilocal > maxLocalIndex_) )
    {
      inputError (
        state,
        String::format (
          "invalid %s index: %d", itemName, ilocal
        )
      );
    }

    if ( iitem >= 0 )
    {
      members_->addMember ( mbrID, iitem, ilocal );
    }

    Utils_::logProgress ( state, counter, mbrName, mbrID );

    token = input.nextToken ();

    if ( token == ';' )
    {
      token = input.nextToken ();
    }
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   scan_
//-----------------------------------------------------------------------


void MemberSetParser::scan_ ( State& state )
{
  const String        mbrName  =   "member";
  const ParserScope*  scope    =   scope_.get ();

  Tokenizer&          input    = * state.input;

  idx_t               mbrID;
  idx_t               ilocal;
  idx_t               counter;
  int                 token;
  idx_t               i;


  counter = 0;
  token   = input.nextToken ();

  while ( token == Tokenizer::INTEGER_TOKEN )
  {
    mbrID = ParseUtils::getItemID ( input, scope );

    for ( i = 0; i < 2; i++ )
    {
      if ( input.nextToken() != Tokenizer::INTEGER_TOKEN )
      {
        parseError ( state, "expected an integer" );
      }
    }

    ilocal = (idx_t) input.getInteger ();

    if ( ilocal < 0 ||
         (maxLocalIndex_ >= 0 && ilocal > maxLocalIndex_) )
    {
      inputError (
        state,
        String::format (
          "invalid member index: %d", ilocal
        )
      );
    }

    Utils_::logProgress ( state, counter, mbrName, mbrID );

    token = input.nextToken ();

    if ( token == ';' )
    {
      token = input.nextToken ();
    }
  }

  input.pushBackToken ();
}


JIVE_END_PACKAGE( util )
