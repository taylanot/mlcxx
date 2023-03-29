
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


#include <jem/base/PrecheckException.h>
#include <jem/util/Properties.h>
#include <jem/util/HashDictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserActions.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/CompoundParser.h>


using jem::util::Dictionary;
using jem::util::HashDictionary;


JEM_BEGIN_PACKAGE( xml )


typedef jem::util::DictionaryEnumerator  DictEnum;


//=======================================================================
//   class CompoundParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


CompoundParser::CompoundParser ()
{
  parserTable_ = newInstance<HashDictionary>( );
  isLenient_   = false;
}


CompoundParser::~CompoundParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void CompoundParser::reset ()
{
  Ref<DictEnum> e = parserTable_->enumerate ();

  for ( ; ! e->atEnd(); e->toNext() )
  {
    staticCast<Parser>( e->getValue() )->reset ();
  }

  if ( defaultParser_ )
  {
    defaultParser_->reset ();
  }
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void CompoundParser::parse ( State& state )
{
  Tokenizer&   input = *state.input;

  Ref<Object>  parser;
  String       tagName;
  int          token;


  token = input.nextToken ();

  if ( token == Tokenizer::EOF_TOKEN )
  {
    return;
  }

  if ( token == Tokenizer::START_TAG_TOKEN )
  {
    tagName = input.getTagName ();

    if ( parserTable_->find( parser, tagName.toUpper() ) )
    {
      state.pushParser ( this );
      state.pushParser ( staticCast<Parser>( parser ) );
      return;
    }
    else if ( defaultParser_ )
    {
      state.pushParser ( this );
      state.pushParser ( defaultParser_ );
      return;
    }
    else if ( isLenient_ )
    {
      state.log->logEvent ( state,
                            ParseLog::TAG_SKIP,
                            tagName );

      token = input.skipAttributes ();

      if ( token == Tokenizer::TAG_END_TOKEN )
      {
        input.skipToEndTag ( tagName );
      }
    }
  }

  input.pushBackToken ();
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void CompoundParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  Ref<DictEnum>  e = parserTable_->enumerate ();


  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<Parser>  p = staticCast<Parser> ( e->getValue() );

    p->takeAction ( action, params );
  }

  if ( defaultParser_ )
  {
    defaultParser_->takeAction ( action, params );
  }

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
//   addParser
//-----------------------------------------------------------------------


void CompoundParser::addParser

  ( const String&       tagName,
    const Ref<Parser>&  parser )

{
  if ( ! parser )
  {
    parserTable_->erase ( tagName.toUpper() );
  }
  else
  {
    parserTable_->insert ( tagName.toUpper(), parser );
  }
}


//-----------------------------------------------------------------------
//   setDefaultParser
//-----------------------------------------------------------------------


void CompoundParser::setDefaultParser ( const Ref<Parser>& parser )
{
  defaultParser_ = parser;
}


JEM_END_PACKAGE( xml )
