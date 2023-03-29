
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


#include <cctype>
#include <cstdio>
#include <jem/base/StringBuffer.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/io/NullWriter.h>
#include <jem/util/Flex.h>
#include <jem/util/StringUtils.h>
#include <jem/xml/Tokenizer.h>


JEM_BEGIN_PACKAGE( xml )


using jem::io::NullWriter;
using jem::util::Flex;
using jem::util::StringUtils;
using jem::util::BasicScanner;


//=======================================================================
//   class Tokenizer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Tokenizer::NULL_TOKEN           = -100;
const int  Tokenizer::EOF_TOKEN            =   -1;
const int  Tokenizer::COMMENT_TOKEN        =   -2;
const int  Tokenizer::START_TAG_TOKEN      =   -3;
const int  Tokenizer::TAG_END_TOKEN        =   -4;
const int  Tokenizer::EMPTY_TAG_END_TOKEN  =   -5;
const int  Tokenizer::END_TAG_TOKEN        =   -6;
const int  Tokenizer::SYMBOL_TOKEN         =   -7;
const int  Tokenizer::STRING_TOKEN         =   -8;
const int  Tokenizer::BOOLEAN_TOKEN        =   -9;
const int  Tokenizer::INTEGER_TOKEN        =  -10;
const int  Tokenizer::FLOAT_TOKEN          =  -11;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Tokenizer::Tokenizer ( const String&  in ) :

  scanner_ ( in )

{
  init_ ();
}


Tokenizer::Tokenizer

  ( const Ref<Reader>&  in,
    idx_t               bufsize ) :

    scanner_ ( in, bufsize )

{
  init_ ();
}


Tokenizer::~Tokenizer ()
{}


//-----------------------------------------------------------------------
//   setInput
//-----------------------------------------------------------------------


void Tokenizer::setInput ( const String& str )
{
  scanner_.setInput ( str );

  token_ = pushedToken_ = NULL_TOKEN;
}


//-----------------------------------------------------------------------
//   nextToken
//-----------------------------------------------------------------------


int Tokenizer::nextToken ()
{
  int  c;


  if ( pushedToken_ != NULL_TOKEN )
  {
    token_       = pushedToken_;
    pushedToken_ = NULL_TOKEN;

    return token_;
  }

  if ( options_ & PARSE_COMMENTS )
  {
    scanner_.skipWhite ();
  }
  else
  {
    scanner_.skipXmlWhite ();
  }

  c = scanner_.read ();

  if ( c < 0 )
  {
    scanner_.setTokenString ( "end of file" );

    return token_ = EOF_TOKEN;
  }

  token_ = NULL_TOKEN;

  scanner_.token.clear ();
  scanner_.token += (char) c;

  if ( c == '<' )
  {
    return token_ = readTag_ ();
  }

  if ( (std::isalpha(c) || c == '_') && (options_ & PARSE_SYMBOLS) )
  {
    scanner_.readSymbol ();

    if ( (options_ & PARSE_BOOLEANS) && scanner_.isBoolToken() )
    {
      token_ = BOOLEAN_TOKEN;
    }
    else
    {
      token_ = SYMBOL_TOKEN;
    }

    return token_;
  }

  if ( std::isdigit(c) && (options_ & PARSE_NUMBERS) )
  {
    if ( scanner_.readNumber() == Scanner::INTEGER )
    {
      token_ = INTEGER_TOKEN;
    }
    else
    {
      token_ = FLOAT_TOKEN;
    }

    return token_;
  }

  if ( c == '.' && (options_ & PARSE_NUMBERS) )
  {
    int  next = scanner_.read ();

    if ( std::isdigit(next) )
    {
      scanner_.token += (char) next;
      scanner_.readFloat ();

      token_ = FLOAT_TOKEN;
    }
    else
    {
      scanner_.unread ( next );

      token_ = c;
    }

    return token_;
  }

  if ( (c == '\'' || c == '\"') && (options_ & PARSE_STRINGS) )
  {
    int  delim = c;

    do
    {
      scanner_.readString ( delim );

      if ( options_ & PARSE_COMMENTS )
      {
        scanner_.skipWhite ();
      }
      else
      {
        scanner_.skipXmlWhite ();
      }

      c = scanner_.read ();
    }
    while ( c == '\'' || c == '\"' );

    scanner_.unread ( c );
    scanner_.token += (char) delim;

    return token_ = STRING_TOKEN;
  }

  if ( (c == '-' || c == '+') && (options_ & PARSE_SIGNED_NUMBERS) )
  {
    int  next = scanner_.tryReadNumber ();

    if      ( next == Scanner::INTEGER )
    {
      token_ = INTEGER_TOKEN;
    }
    else if ( next == Scanner::FLOAT )
    {
      token_ = FLOAT_TOKEN;
    }
    else
    {
      token_ = c;
    }

    return token_;
  }

  return token_ = c;
}


//-----------------------------------------------------------------------
//   pushBackToken
//-----------------------------------------------------------------------


void Tokenizer::pushBackToken ()
{
  pushedToken_ = token_;
}


//-----------------------------------------------------------------------
//   getToken
//-----------------------------------------------------------------------


int Tokenizer::getToken () const
{
  return token_;
}


//-----------------------------------------------------------------------
//   getLineNumber
//-----------------------------------------------------------------------


lint Tokenizer::getLineNumber () const noexcept
{
  return scanner_.lineno;
}


//-----------------------------------------------------------------------
//   setLineNumber
//-----------------------------------------------------------------------


void Tokenizer::setLineNumber ( lint lineno )
{
  scanner_.lineno = lineno;
}


//-----------------------------------------------------------------------
//   getTokenString
//-----------------------------------------------------------------------


String Tokenizer::getTokenString () const
{
  return scanner_.token[ALL];
}


//-----------------------------------------------------------------------
//   getComment
//-----------------------------------------------------------------------


String Tokenizer::getComment () const
{
  if ( token_ == COMMENT_TOKEN )
  {
    idx_t  len = scanner_.token.size ();

    return scanner_.token[slice(4,len - 3)];
  }
  else
  {
    throw IllegalOperationException (
      JEM_FUNC,
      "current token is not a comment token"
    );
  }
}


//-----------------------------------------------------------------------
//   getTagName
//-----------------------------------------------------------------------


String Tokenizer::getTagName () const
{
  if ( token_ == START_TAG_TOKEN )
  {
    return scanner_.token[slice(1,END)];
  }
  else if ( token_ == END_TAG_TOKEN )
  {
    idx_t  len = scanner_.token.size ();

    return scanner_.token[slice(2,len - 1)];
  }
  else
  {
    throw IllegalOperationException (
      JEM_FUNC,
      "current token is not a tag"
    );
  }
}


//-----------------------------------------------------------------------
//   getSymbol
//-----------------------------------------------------------------------


String Tokenizer::getSymbol () const
{
  if ( token_ == SYMBOL_TOKEN )
  {
    return scanner_.token[ALL];
  }
  else
  {
    throw IllegalOperationException (
      JEM_FUNC,
      "current token is not a symbol"
    );
  }
}


//-----------------------------------------------------------------------
//   getString
//-----------------------------------------------------------------------


String Tokenizer::getString () const
{
  if ( token_ == STRING_TOKEN )
  {
    idx_t  len = scanner_.token.size ();

    return scanner_.token[slice(1,len - 1)];
  }
  else
  {
    throw IllegalOperationException (
      JEM_FUNC,
      "current token is not a string"
    );
  }
}


//-----------------------------------------------------------------------
//   getBoolean
//-----------------------------------------------------------------------


bool Tokenizer::getBoolean () const
{
  if ( token_ == BOOLEAN_TOKEN )
  {
    if ( scanner_.token.size() == 4 )
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  else
  {
    throw IllegalOperationException (
      JEM_FUNC,
      "current token is not a boolean"
    );
  }
}


//-----------------------------------------------------------------------
//   getInteger
//-----------------------------------------------------------------------


lint Tokenizer::getInteger () const
{
  if ( token_ == INTEGER_TOKEN )
  {
    return scanner_.getInteger ();
  }
  else
  {
    throw IllegalOperationException (
      JEM_FUNC,
      "current token is not an integer"
    );
  }
}


//-----------------------------------------------------------------------
//   getFloat
//-----------------------------------------------------------------------


double Tokenizer::getFloat () const
{
  if      ( token_ == FLOAT_TOKEN )
  {
    return scanner_.getFloat ();
  }
  else if ( token_ == INTEGER_TOKEN )
  {
    return (double) scanner_.getInteger ();
  }
  else
  {
    throw IllegalOperationException (
      JEM_FUNC,
      "current token is not a floating point number"
    );
  }
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void Tokenizer::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void Tokenizer::setOptions ( Options options )
{
  options_ = options;

  if ( options & PARSE_SIGNED_NUMBERS )
  {
    options_ = options_ | PARSE_NUMBERS;
  }
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


Tokenizer::Options Tokenizer::getOptions () const noexcept
{
  return options_;
}


//-----------------------------------------------------------------------
//   readAttribute
//-----------------------------------------------------------------------


bool Tokenizer::readAttribute ( String& name, String& value )
{
  int  c;

  scanner_.skipWhite ();
  c = scanner_.read  ();

  if ( c < 0 )
  {
    scanner_.parseError ( "unexpected end of file" );
  }

  token_ = pushedToken_ = NULL_TOKEN;

  scanner_.token.clear ();
  scanner_.token += (char) c;

  if ( c == '>' )
  {
    token_ = TAG_END_TOKEN;
    return   false;
  }

  if ( c == '/' )
  {
    c = scanner_.read ();

    if ( c != '>' )
    {
      scanner_.parseError ( "expected a `>\'" );
    }

    scanner_.token += (char) c;
    token_ = EMPTY_TAG_END_TOKEN;

    return false;
  }

  if ( std::isalpha(c) || c == '_' )
  {
    scanner_.readName ();
  }
  else
  {
    scanner_.parseError ( "missing attribute name" );
  }

  name = scanner_.token[ALL];

  scanner_.skipWhite ();
  c = scanner_.read  ();

  if ( c != '=' )
  {
    scanner_.unread ( c );

    value  = "true";
    token_ = SYMBOL_TOKEN;

    return true;
  }

  scanner_.skipWhite   ();
  scanner_.token.clear ();
  c = scanner_  .read  ();

  if ( c == '\'' || c == '\"' )
  {
    int  delim = c;

    scanner_.token += (char) delim;

    do
    {
      scanner_.readString ( delim );
      scanner_.skipWhite  ();
      c = scanner_.read   ();
    }
    while ( c == '\'' || c == '\"' );

    scanner_.token += (char) delim;

    token_ = STRING_TOKEN;
    value  = getString ();
  }

  else
  {
    while ( c >= 0 && c != '>' && ! std::isspace(c) )
    {
      scanner_.token += (char) c;
      c = scanner_.read ();
    }

    token_ = SYMBOL_TOKEN;
    value  = getSymbol ();
  }

  scanner_.unread ( c );

  return true;
}


//-----------------------------------------------------------------------
//   skipAttributes
//-----------------------------------------------------------------------


int Tokenizer::skipAttributes ()
{
  Ref<NullWriter>  bitBucket = newInstance<NullWriter> ();

  return readAttributes ( *bitBucket );
}


//-----------------------------------------------------------------------
//   readAttributes
//-----------------------------------------------------------------------


int Tokenizer::readAttributes ( TextOutput& out )
{
  int  c;


  token_ = pushedToken_ = NULL_TOKEN;

  scanner_.token.clear ();

  while ( true )
  {
    c = scanner_.read ();

    if ( c < 0 )
    {
      scanner_.parseError ( "unexpected end of file" );
    }

    scanner_.token += (char) c;

    if ( c == '\n' )
    {
      scanner_.lineno++;
      continue;
    }

    if ( c == '\'' || c == '\"' )
    {
      scanner_.readString ( c );
      scanner_.token += (char) c;
      continue;
    }

    if ( c == '>' )
    {
      token_ = TAG_END_TOKEN;
      break;
    }

    if ( c == '/' )
    {
      c = scanner_.read ();

      if ( c != '>' )
      {
        scanner_.parseError ( "expected a `>\'" );
      }

      scanner_.token += (char) c;
      token_ = EMPTY_TAG_END_TOKEN;

      break;
    }
  }

  out.write ( scanner_.token.addr(), scanner_.token.size() );

  return token_;
}


//-----------------------------------------------------------------------
//   skipToEndTag
//-----------------------------------------------------------------------


void Tokenizer::skipToEndTag ( const String& tagName )
{
  Ref<NullWriter>  bitBucket = newInstance<NullWriter> ();

  readToEndTag ( *bitBucket, tagName );
}


//-----------------------------------------------------------------------
//   readToEndTag
//-----------------------------------------------------------------------


String Tokenizer::readToEndTag ( const String& tagName )
{
  StringBuffer  buf;

  readToEndTag ( buf, tagName );

  return buf.toString ();
}


void Tokenizer::readToEndTag

  ( TextOutput&    out,
    const String&  tagName )

{
  Flex<String>  tags;
  String        name;
  int           c;


  tags.reserve  ( 8 );
  tags.pushBack ( tagName );

  while ( tags.size() > 0 )
  {
    c = scanner_.read ();

    if ( c < 0 )
    {
      break;
    }

    if ( c == '\n' )
    {
      out.write ( c );
      scanner_.lineno++;
      continue;
    }

    if ( std::isspace( c ) )
    {
      out.write ( c );
      continue;
    }

    // Parse strings (if specified) so that '<' characters in strings
    // are properly ignored.

    if ( (c == '\'' || c == '\"') && (options_ & PARSE_STRINGS) )
    {
      token_ = pushedToken_ = NULL_TOKEN;

      scanner_.token.clear ();
      scanner_.token += (char) c;
      scanner_.readString    ( c );
      scanner_.token += (char) c;

      token_ = STRING_TOKEN;

      out.write ( scanner_.token.addr(), scanner_.token.size() );

      continue;
    }

    if ( c == '<' )
    {
      token_ = pushedToken_ = NULL_TOKEN;

      scanner_.token.clear ();
      scanner_.token += (char) c;
      token_ = readTag_ ();

      out.write ( scanner_.token.addr(), scanner_.token.size() );

      if ( token_ == START_TAG_TOKEN )
      {
        name = getTagName ();

        if ( readAttributes( out ) == TAG_END_TOKEN )
        {
          tags.pushBack ( std::move( name ) );
        }
      }
      else if ( token_ == END_TAG_TOKEN )
      {
        if ( ! tags.back().equalsIgnoreCase( getTagName() ) )
        {
          endTagError_ ( tags.back() );
        }
        else
        {
          tags.popBack ();
        }
      }

      continue;
    }

    out.write ( c );
  }

  if ( tags.size() > 0 )
  {
    scanner_.parseError (
      String::format (
        "missing end tag `</%s>\'", tagName
      )
    );
  }
}


//-----------------------------------------------------------------------
//   readText
//-----------------------------------------------------------------------


String Tokenizer::readText ()
{
  StringBuffer  buf;

  readText ( buf );

  return buf.toString ();
}


int Tokenizer::readText ( TextOutput& out )
{
  int  c;

  token_ = pushedToken_ = NULL_TOKEN;

  scanner_.token.clear ();

  while ( true )
  {
    c = scanner_.read ();

    if ( c < 0 )
    {
      nextToken ();
      break;
    }

    if ( c == '<' )
    {
      scanner_.unread ( c );
      nextToken       ();

      if ( token_ != '<' )
      {
        break;
      }
    }

    if ( c == '\n' )
    {
      scanner_.lineno++;
    }

    out.write ( c );
  }

  return token_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void Tokenizer::init_ ()
{
  token_       = NULL_TOKEN;
  pushedToken_ = NULL_TOKEN;

  options_     = (PARSE_SYMBOLS | PARSE_STRINGS | PARSE_BOOLEANS |
                  PARSE_NUMBERS | PARSE_SIGNED_NUMBERS);

  scanner_.setTokenString ( "start of file" );
}


//-----------------------------------------------------------------------
//   readTag_
//-----------------------------------------------------------------------


int Tokenizer::readTag_ ()
{
  int  c = scanner_.read ();

  if ( c == '!' && scanner_.scan( "--", 2 ) )
  {
    scanner_.token += '!';
    scanner_.token += '-';
    scanner_.token += '-';

    scanner_.readComment ();

    return COMMENT_TOKEN;
  }
  else if ( std::isalpha(c) || c == '_' )
  {
    scanner_.token += (char) c;
    scanner_.readName ();

    return START_TAG_TOKEN;
  }
  else if ( c == '/' )
  {
    scanner_.token += '/';
    c = scanner_.read ();

    if ( ! ( std::isalpha(c) || c == '_' ) )
    {
      scanner_.parseError ( "illegal end tag" );
    }

    scanner_.token += (char) c;

    scanner_.readName  ();
    scanner_.skipWhite ();
    c = scanner_.read  ();

    if ( c != '>' )
    {
      scanner_.parseError ( "missing `>\' in end tag" );
    }

    scanner_.token += '>';

    return END_TAG_TOKEN;
  }
  else
  {
    scanner_.unread ( c );
    return '<';
  }
}


//-----------------------------------------------------------------------
//   endTagError_
//-----------------------------------------------------------------------


void Tokenizer::endTagError_ ( const String& tagName ) const
{
  String  token = StringUtils::shorten ( getTokenString () );

  scanner_.parseError (
    String::format (
      "parse error before `%s\': expected the end tag `</%s>\'",
      token,
      tagName
    )
  );
}


JEM_END_PACKAGE( xml )
