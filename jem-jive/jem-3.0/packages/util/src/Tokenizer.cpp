
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
#include <jem/base/IllegalOperationException.h>
#include <jem/util/StringUtils.h>
#include <jem/util/Tokenizer.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class Tokenizer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Tokenizer::NULL_TOKEN           = -100;
const int  Tokenizer::EOF_TOKEN            =   -1;
const int  Tokenizer::EOL_TOKEN            =   -2;
const int  Tokenizer::COMMENT_TOKEN        =   -3;
const int  Tokenizer::SYMBOL_TOKEN         =   -4;
const int  Tokenizer::STRING_TOKEN         =   -5;
const int  Tokenizer::BOOLEAN_TOKEN        =   -6;
const int  Tokenizer::INTEGER_TOKEN        =   -7;
const int  Tokenizer::FLOAT_TOKEN          =   -8;

const int  Tokenizer::PLUS_TOKEN           =  -20;
const int  Tokenizer::MINUS_TOKEN          =  -21;
const int  Tokenizer::MULT_TOKEN           =  -22;
const int  Tokenizer::DIV_TOKEN            =  -23;
const int  Tokenizer::MOD_TOKEN            =  -24;
const int  Tokenizer::BOR_TOKEN            =  -25;
const int  Tokenizer::BAND_TOKEN           =  -26;
const int  Tokenizer::BNOT_TOKEN           =  -27;
const int  Tokenizer::BXOR_TOKEN           =  -28;
const int  Tokenizer::LSHIFT_TOKEN         =  -29;
const int  Tokenizer::RSHIFT_TOKEN         =  -30;
const int  Tokenizer::PLUS_IS_TOKEN        =  -31;
const int  Tokenizer::MINUS_IS_TOKEN       =  -32;
const int  Tokenizer::MULT_IS_TOKEN        =  -33;
const int  Tokenizer::DIV_IS_TOKEN         =  -34;
const int  Tokenizer::MOD_IS_TOKEN         =  -35;
const int  Tokenizer::BOR_IS_TOKEN         =  -36;
const int  Tokenizer::BAND_IS_TOKEN        =  -37;
const int  Tokenizer::BNOT_IS_TOKEN        =  -38;
const int  Tokenizer::BXOR_IS_TOKEN        =  -39;
const int  Tokenizer::LSHIFT_IS_TOKEN      =  -40;
const int  Tokenizer::RSHIFT_IS_TOKEN      =  -41;
const int  Tokenizer::LOR_TOKEN            =  -42;
const int  Tokenizer::LAND_TOKEN           =  -43;
const int  Tokenizer::LNOT_TOKEN           =  -44;
const int  Tokenizer::ASSIGN_TOKEN         =  -45;
const int  Tokenizer::EQUAL_TOKEN          =  -46;
const int  Tokenizer::UNEQUAL_TOKEN        =  -47;
const int  Tokenizer::LESS_THAN_TOKEN      =  -48;
const int  Tokenizer::LESS_EQUAL_TOKEN     =  -49;
const int  Tokenizer::GREATER_THAN_TOKEN   =  -50;
const int  Tokenizer::GREATER_EQUAL_TOKEN  =  -51;
const int  Tokenizer::COMMA_TOKEN          =  -52;
const int  Tokenizer::DOT_TOKEN            =  -53;
const int  Tokenizer::COLON_TOKEN          =  -54;
const int  Tokenizer::DOUBLE_COLON_TOKEN   =  -55;
const int  Tokenizer::SEMI_COLON_TOKEN     =  -56;
const int  Tokenizer::OPEN_PAREN_TOKEN     =  -57;
const int  Tokenizer::CLOSE_PAREN_TOKEN    =  -58;
const int  Tokenizer::OPEN_BRACE_TOKEN     =  -59;
const int  Tokenizer::CLOSE_BRACE_TOKEN    =  -60;
const int  Tokenizer::OPEN_BRACKET_TOKEN   =  -61;
const int  Tokenizer::CLOSE_BRACKET_TOKEN  =  -62;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Tokenizer::Tokenizer ( const String& in ) :

  scanner_ ( in )

{
  init_ ();
}


Tokenizer::Tokenizer ( const Ref<Reader>& in, idx_t bufsize ) :

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
    scanner_.skipCppWhite ();
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

  if ( c == '\n' )
  {
    scanner_.lineno++;

    return token_ = EOL_TOKEN;
  }

  if ( (std::isalpha(c) || c == '_') && (options_ & PARSE_SYMBOLS) )
  {
    scanner_.readSymbol ();

    if ( (options_ & PARSE_BOOLEANS) && scanner_.isBoolToken() )
    {
      return (token_ = BOOLEAN_TOKEN);
    }
    else
    {
      return (token_ = SYMBOL_TOKEN);
    }
  }

  if ( std::isdigit(c) && (options_ & PARSE_NUMBERS) )
  {
    if ( scanner_.readNumber () == CppScanner::INTEGER )
    {
      return (token_ = INTEGER_TOKEN);
    }
    else
    {
      return (token_ = FLOAT_TOKEN);
    }
  }

  if ( c == '.' && (options_ & PARSE_NUMBERS) )
  {
    int  next = scanner_.read ();

    if ( std::isdigit(next) )
    {
      scanner_.token += (char) next;
      scanner_.readFloat ();

      return (token_ = FLOAT_TOKEN);
    }

    scanner_.unread ( next );
  }

  if ( (c == '\'' || c == '\"') && (options_ & PARSE_STRINGS) )
  {
    int  delim = c;

    if ( options_ & CATENATE_STRINGS )
    {
      do
      {
        scanner_.readString ( delim );

        if ( options_ & PARSE_COMMENTS )
        {
          scanner_.skipWhite ();
        }
        else
        {
          scanner_.skipCppWhite ();
        }

        c = scanner_.read ();
      }
      while ( c == '\'' || c == '\"' );

      scanner_.unread ( c );
    }
    else
    {
      scanner_.readString ( delim );
    }

    scanner_.token += (char) delim;

    return (token_ = STRING_TOKEN);
  }

  if ( (c == '-' || c == '+') && (options_ & PARSE_SIGNED_NUMBERS) )
  {
    int  next = scanner_.tryReadNumber ();

    if      ( next == CppScanner::INTEGER )
    {
      return (token_ = INTEGER_TOKEN);
    }
    else if ( next == CppScanner::FLOAT )
    {
      return (token_ = FLOAT_TOKEN);
    }
  }

  if ( c == '/' && (options_ & PARSE_COMMENTS) )
  {
    int  next = scanner_.read ();

    if ( next == '/' )
    {
      scanner_.token += (char) next;
      scanner_.readLine ();

      return (token_ = COMMENT_TOKEN);
    }
    else if ( next == '*' )
    {
      scanner_.token += (char) next;
      scanner_.readComment ();

      return (token_ = COMMENT_TOKEN);
    }

    scanner_.unread ( next );
  }

  if ( options_ & PARSE_OPERATORS )
  {
    return (token_ = parseOperator_( c ));
  }

  return (token_ = c);
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
//   skipLine
//-----------------------------------------------------------------------


void Tokenizer::skipLine ()
{
  scanner_.skipLine ();

  token_       = EOL_TOKEN;
  pushedToken_ = NULL_TOKEN;
}


//-----------------------------------------------------------------------
//   getLineNumber
//-----------------------------------------------------------------------


lint Tokenizer::getLineNumber () const
{
  return scanner_.lineno;
}


//-----------------------------------------------------------------------
//   setLineNumber
//-----------------------------------------------------------------------


void Tokenizer::setLineNumber ( lint line )
{
  scanner_.lineno = line;
}


//-----------------------------------------------------------------------
//   getTokenString
//-----------------------------------------------------------------------


String Tokenizer::getTokenString () const
{
  return scanner_.token[ALL];
}


//-----------------------------------------------------------------------
//   getTokenChars
//-----------------------------------------------------------------------


const char* Tokenizer::getTokenChars ( idx_t& n ) const
{
  n = scanner_.token.size ();

  return scanner_.token.addr ();
}


//-----------------------------------------------------------------------
//   getComment
//-----------------------------------------------------------------------


String Tokenizer::getComment () const
{
  if ( token_ == COMMENT_TOKEN )
  {
    if ( scanner_.token[1] == '/' )
    {
      return scanner_.token[slice(2,END)];
    }
    else
    {
      idx_t  len = scanner_.token.size ();

      return scanner_.token[slice(2,len - 2)];
    }
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
//   getStringType
//-----------------------------------------------------------------------


int Tokenizer::getStringType () const
{
  if ( token_ == STRING_TOKEN )
  {
    return scanner_.token.front ();
  }
  else
  {
    return NULL_TOKEN;
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
  CppScanner::Options  scanOpts = scanner_.getOptions ();

  options_ = options;

  if ( options & PARSE_SIGNED_NUMBERS )
  {
    options_ = options_ | PARSE_NUMBERS;
  }

  if ( options & PARSE_END_OF_LINE )
  {
    scanOpts |=  CppScanner::PARSE_EOL;
  }
  else
  {
    scanOpts &= ~CppScanner::PARSE_EOL;
  }

  scanner_.setOptions ( scanOpts );
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


Tokenizer::Options Tokenizer::getOptions () const noexcept
{
  return options_;
}


//-----------------------------------------------------------------------
//   parseError
//-----------------------------------------------------------------------


void Tokenizer::parseError () const
{
  String  token = StringUtils::shorten ( scanner_.token[ALL] );

  scanner_.parseError (
    String::format (
      "parse error before `%s\'", token
    )
  );
}


void Tokenizer::parseError ( const String& msg ) const
{
  String  token = StringUtils::shorten ( scanner_.token[ALL] );

  scanner_.parseError (
    String::format (
      "parse error before `%s\': %s", token, msg
    )
  );
}


//-----------------------------------------------------------------------
//   rangeError
//-----------------------------------------------------------------------


void Tokenizer::rangeError

  ( lint  value,
    lint  lbound,
    lint  ubound ) const

{
  scanner_.parseError (
    String::format (
      "integer value (%d) out of range; valid range is "
      "[%d,%d]",
      value,
      lbound,
      ubound
    )
  );
}


void Tokenizer::rangeError

  ( double  value,
    double  ubound ) const

{
  scanner_.parseError (
    String::format (
      "floating point value (%g) out of range; valid range is "
      "[%g,%g]",
      value,
      -ubound,
      ubound
    )
  );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void Tokenizer::init_ ()
{
  token_       = NULL_TOKEN;
  pushedToken_ = NULL_TOKEN;

  options_     = ( PARSE_SYMBOLS        |
                   PARSE_STRINGS        |
                   PARSE_BOOLEANS       |
                   PARSE_NUMBERS        |
                   PARSE_SIGNED_NUMBERS |
                   CATENATE_STRINGS );

  scanner_.setTokenString ( "start of file" );
}


//-----------------------------------------------------------------------
//   parseOperator_
//-----------------------------------------------------------------------


int Tokenizer::parseOperator_ ( int c )
{
  int  next;

  switch ( c )
  {
  case '+':

    return parseOperator2_ ( PLUS_TOKEN,  PLUS_IS_TOKEN );

  case '-':

    return parseOperator2_ ( MINUS_TOKEN, MINUS_IS_TOKEN );

  case '*':

    return parseOperator2_ ( MULT_TOKEN,  MULT_IS_TOKEN );

  case '/':

    return parseOperator2_ ( DIV_TOKEN,   DIV_IS_TOKEN );

  case '%':

    return parseOperator2_ ( MOD_TOKEN,   MOD_IS_TOKEN );

  case '|':

    next = scanner_.read ();

    if      ( next == '|' )
    {
      return LOR_TOKEN;
    }
    else if ( next == '=' )
    {
      return BOR_IS_TOKEN;
    }
    else
    {
      scanner_.unread ( next );

      return BOR_TOKEN;
    }

  case '&':

    next = scanner_.read ();

    if      ( next == '&' )
    {
      return LAND_TOKEN;
    }
    else if ( next == '=' )
    {
      return BAND_IS_TOKEN;
    }
    else
    {
      scanner_.unread ( next );

      return BAND_TOKEN;
    }

  case '~':

    return parseOperator2_ ( BNOT_TOKEN,  BNOT_IS_TOKEN );

  case '^':

    return parseOperator2_ ( BXOR_TOKEN,  BXOR_IS_TOKEN );

  case '<':

    next = scanner_.read ();

    if      ( next == '<' )
    {
      return parseOperator2_ ( LSHIFT_TOKEN, LSHIFT_IS_TOKEN );
    }
    else if ( next == '=' )
    {
      return LESS_EQUAL_TOKEN;
    }
    else
    {
      scanner_.unread ( next );

      return LESS_THAN_TOKEN;
    }

  case '>':

    next = scanner_.read ();

    if      ( next == '>' )
    {
      return parseOperator2_ ( RSHIFT_TOKEN, RSHIFT_IS_TOKEN );
    }
    else if ( next == '=' )
    {
      return GREATER_EQUAL_TOKEN;
    }
    else
    {
      scanner_.unread ( next );

      return GREATER_THAN_TOKEN;
    }

  case '!':

    return parseOperator2_ ( LNOT_TOKEN,   UNEQUAL_TOKEN );

  case '=':

    return parseOperator2_ ( ASSIGN_TOKEN, EQUAL_TOKEN );

  case ',':

    return COMMA_TOKEN;

  case '.':

    return DOT_TOKEN;

  case ':':

    next = scanner_.read ();

    if ( next == ':' )
    {
      return DOUBLE_COLON_TOKEN;
    }
    else
    {
      scanner_.unread ( next );

      return COLON_TOKEN;
    }

  case ';':

    return SEMI_COLON_TOKEN;

  case '(':

    return OPEN_PAREN_TOKEN;

  case ')':

    return CLOSE_PAREN_TOKEN;

  case '{':

    return OPEN_BRACE_TOKEN;

  case '}':

    return CLOSE_BRACE_TOKEN;

  case '[':

    return OPEN_BRACKET_TOKEN;

  case ']':

    return CLOSE_BRACKET_TOKEN;
  }

  return c;
}


//-----------------------------------------------------------------------
//   parseOperator2_
//-----------------------------------------------------------------------


int Tokenizer::parseOperator2_

  ( int  tok,
    int  alt )

{
  int  next = scanner_.read ();

  if ( next == '=' )
  {
    return alt;
  }
  else
  {
    scanner_.unread ( next );

    return tok;
  }
}


JEM_END_PACKAGE( util )
