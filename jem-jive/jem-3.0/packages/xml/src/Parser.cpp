
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


#include <jem/base/assert.h>
#include <jem/base/Integer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/FileReader.h>
#include <jem/io/IOException.h>
#include <jem/util/StringUtils.h>
#include <jem/xml/Parser.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jem/xml/ParserException.h>
#include <jem/xml/Parser.h>


JEM_DEFINE_CLASS( jem::xml::Parser );


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class PopInputParser
//=======================================================================


class PopInputParser : public Parser
{
 public:

  typedef PopInputParser    Self;
  typedef Parser            Super;


  explicit                  PopInputParser

    ( const Ref<Tokenizer>&   savedInput );

  virtual void              parse

    ( State&                  state )      override;


 private:

  Ref<Tokenizer>            savedInput_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


PopInputParser::PopInputParser

  ( const Ref<Tokenizer>&  savedInput ) :

    savedInput_ ( savedInput )

{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void PopInputParser::parse

  ( State&  state )

{
  if ( state.input->nextToken() != Tokenizer::EOF_TOKEN )
  {
    parseError ( state, "end of file expected" );
  }

  String  source = state.sourceStack.pop ();

  state.input    = savedInput_;

  state.log->logEvent ( state,
                        ParseLog::SOURCE_POPPED,
                        source );
}


//=======================================================================
//   class Parser
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Parser::~Parser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void Parser::reset ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void Parser::takeAction

  ( const String&      action,
    const Properties&  params )

{}


//-----------------------------------------------------------------------
//   parseInput
//-----------------------------------------------------------------------


void Parser::parseInput

  ( const Ref<Tokenizer>&  input,
    const String&          source,
    const Ref<ParseLog>&   log )

{
  JEM_PRECHECK ( input );

  Ref<Parser>  p;
  State        state;
  lint         line;
  int          token;


  state.input = input;
  state.log   = log;

  if ( ! state.log )
  {
    state.log = newInstance<ParseLog> ();
  }

  state.log->reset ();
  this     ->reset ();

  state.sourceStack.push ( source );
  state.parserStack.push (   this );

  try
  {
    while ( state.parserStack.size() > 0 )
    {
      p = state.parserStack.pop ();
      p->parse ( state );
    }
  }
  catch ( const IllegalInputException& e )
  {
    if ( ! Integer::parse( line, e.where() ) )
    {
      line = state.input->getLineNumber ();
    }

    throw ParserException (
      state.sourceStack.top (),
      line,
      e.what ()
    );
  }

  catch ( const io::IOException& e )
  {
    throw ParserException (
      state.sourceStack.top (),
      state.input->getLineNumber (),
      e.what ()
    );
  }

  if ( ! state.input )
  {
    return;
  }

  token = state.input->nextToken ();

  if ( token == Tokenizer::START_TAG_TOKEN )
  {
    illegalTagError ( state, state.input->getTagName() );
  }
  else if ( token != Tokenizer::EOF_TOKEN )
  {
    parseError ( state, "end of file expected" );
  }
}


//-----------------------------------------------------------------------
//   parseFile
//-----------------------------------------------------------------------


void Parser::parseFile

  ( const String&         fname,
    const Ref<ParseLog>&  log )

{
  parseInput (
    newInstance<Tokenizer> (
      newInstance<io::FileReader> ( fname )
    ),
    fname,
    log
  );
}


//-----------------------------------------------------------------------
//   pushInput
//-----------------------------------------------------------------------


void Parser::pushInput

  ( State&                 state,
    const Ref<Tokenizer>&  input,
    const String&          source,
    const Ref<Parser>&     parser )

{
  JEM_PRECHECK ( input );

  Ref<Tokenizer>  in = state.input;
  Ref<Parser>     p  = parser;

  if ( ! p )
  {
    p = state.parserStack.top ();

    if ( ! p )
    {
      return;
    }
  }

  state.input = input;

  state.sourceStack.push ( source );
  state.parserStack.push ( newInstance<PopInputParser>( in ) );
  state.parserStack.push ( p );

  state.log->logEvent    ( state,
                           ParseLog::SOURCE_PUSHED,
                           source );
}


//-----------------------------------------------------------------------
//   parseEndTag
//-----------------------------------------------------------------------


void Parser::parseEndTag

  ( State&         state,
    const String&  tagName )

{
  Tokenizer& input = * state.input;

  if ( input.nextToken() != Tokenizer::END_TAG_TOKEN ||
       ! tagName.equalsIgnoreCase( input.getTagName() ) )
  {
    parseError (
      state,
      String::format (
        "expected the end tag `</%s>\'", tagName
      )
    );
  }
}


//-----------------------------------------------------------------------
//   parseError
//-----------------------------------------------------------------------


void Parser::parseError ( State& state )
{
  String token = util::StringUtils::shorten (
    state.input->getTokenString ()
  );

  throw ParserException (
    state.sourceStack.top      (),
    state.input->getLineNumber (),
    String::format (
      "parse error before `%s\'", token
    )
  );
}


//-----------------------------------------------------------------------
//   parseError
//-----------------------------------------------------------------------


void Parser::parseError

  ( State&         state,
    const String&  msg )

{
  String token = util::StringUtils::shorten (
    state.input->getTokenString ()
  );

  throw ParserException (
    state.sourceStack.top      (),
    state.input->getLineNumber (),
    String::format (
      "parse error before `%s\' : %s", token, msg
    )
  );
}


//-----------------------------------------------------------------------
//   inputError
//-----------------------------------------------------------------------


void Parser::inputError

  ( State&         state,
    const String&  msg )

{
  throw ParserException (
    state.sourceStack.top      (),
    state.input->getLineNumber (),
    msg
  );
}


//-----------------------------------------------------------------------
//   illegalTagError
//-----------------------------------------------------------------------


void Parser::illegalTagError

  ( State&         state,
    const String&  tagName )

{
  throw ParserException (
    state.sourceStack.top      (),
    state.input->getLineNumber (),
    String::format (
      "illegal tag `%s\'", tagName
    )
  );
}


//-----------------------------------------------------------------------
//   illegalAttributeError
//-----------------------------------------------------------------------


void Parser::illegalAttributeError

  ( State&         state,
    const String&  attribName )

{
  throw ParserException (
    state.sourceStack.top      (),
    state.input->getLineNumber (),
    String::format (
      "illegal attribute name `%s\'", attribName
    )
  );
}


//-----------------------------------------------------------------------
//   missingAttributeError
//-----------------------------------------------------------------------


void Parser::missingAttributeError

  ( State&         state,
    const String&  attribName )

{
  throw ParserException (
    state.sourceStack.top      (),
    state.input->getLineNumber (),
    String::format (
      "missing attribute `%s\'", attribName
    )
  );
}


JEM_END_PACKAGE( xml )
