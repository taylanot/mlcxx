
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <jem/util/Properties.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/ParserActions.h>
#include <jem/xml/TagParser.h>


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class TagParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


TagParser::TagParser ( const String& attribs ) :

  AttributeParser ( attribs ),
  isLenient_      (   false ),
  state_          (  START_ )

{}


TagParser::~TagParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void TagParser::reset ()
{
  state_ = START_;
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void TagParser::parse ( State& state )
{
  Tokenizer&  input = * state.input;

  int         token;

  if ( state_ == START_ )
  {
    if ( input.getToken() != Tokenizer::START_TAG_TOKEN )
    {
      parseError ( state, "current token is not a start tag" );
    }

    tagName_ = input.getTagName ();

    state.log->logEvent ( state,
                          ParseLog::TAG_START,
                          tagName_ );

    token = parseAttributes ( state, isLenient_ );

    if ( token == Tokenizer::TAG_END_TOKEN )
    {
      state_ = FINISH_;
      state.pushParser ( this );
      parseContents    ( state );
    }
    else
    {
      handleEmptyContents ( state );
      finish              ( state );
    }
  }
  else
  {
    state_ = START_;

    parseEndTag ( state, tagName_ );
    finish      ( state );

    state.log->logEvent ( state,
                          ParseLog::TAG_END,
                          tagName_ );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void TagParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if      ( action == ParserActions::SET_OPTIONS )
  {
    params.find ( isLenient_, ParserOptions::PARSE_LENIENT );
  }
  else if ( action == ParserActions::RESET_OPTIONS )
  {
    isLenient_ = false;
  }
}


//-----------------------------------------------------------------------
//   parseContents
//-----------------------------------------------------------------------


void TagParser::parseContents ( State& )
{}


//-----------------------------------------------------------------------
//   handleEmptyContents
//-----------------------------------------------------------------------


void TagParser::handleEmptyContents ( State& )
{}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void TagParser::finish ( State& )
{}


//-----------------------------------------------------------------------
//   getTagName
//-----------------------------------------------------------------------


const String& TagParser::getTagName () const noexcept
{
  return tagName_;
}


JEM_END_PACKAGE( xml )
