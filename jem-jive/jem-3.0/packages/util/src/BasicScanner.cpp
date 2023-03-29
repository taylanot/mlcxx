
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


#include <cstdlib>
#include <cctype>
#include <cstdio>
#include <jem/base/limits.h>
#include <jem/base/ParseException.h>
#include <jem/io/StringReader.h>
#include <jem/util/BasicScanner.h>


JEM_BEGIN_PACKAGE( util )


using jem::io::StringReader;


//=======================================================================
//   class BasicScanner
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


BasicScanner::BasicScanner ( const String& str )
{
  input = newInstance<StringReader> ( str );

  init_ ();
}


BasicScanner::BasicScanner

  ( const Ref<Reader>&  in,
    idx_t               bufsize ) :

    input  (      in ),
    buffer ( bufsize )

{
  init_ ();
}


//-----------------------------------------------------------------------
//   setInput
//-----------------------------------------------------------------------


void BasicScanner::setInput ( const String& str )
{
  StringReader*  r = dynamic_cast<StringReader*> ( input.get() );

  if ( r )
  {
    r->setString ( str );
  }
  else
  {
    input = newInstance<StringReader> ( str );
  }

  buffer.reset ();
  token .clear ();

  lineno = 1L;
}


//-----------------------------------------------------------------------
//   skipWhite
//-----------------------------------------------------------------------


void BasicScanner::skipWhite ()
{
  int  c = read ();

  while ( c >= 0 && isSpace_( c ) )
  {
    if ( c == '\n' )
    {
      lineno++;
    }

    c = read ();
  }

  unread ( c );
}


//-----------------------------------------------------------------------
//   skipLine
//-----------------------------------------------------------------------


void BasicScanner::skipLine ()
{
  int  c = read ();

  while ( c >= 0 && c != '\n' )
  {
    c = read ();
  }

  if ( c == '\n' )
  {
    lineno++;
  }
}


//-----------------------------------------------------------------------
//   readSymbol
//-----------------------------------------------------------------------


void BasicScanner::readSymbol ()
{
  int  c = read ();

  while ( c >= 0 && (std::isalnum(c) || c == '_' ) )
  {
    token += (char) c;
    c      = read  ();
  }

  unread ( c );
}


//-----------------------------------------------------------------------
//   readString
//-----------------------------------------------------------------------


void BasicScanner::readString ( int delim )
{
  lint  line = lineno;
  int   c    = read ();

  while ( c >= 0 && c != delim )
  {
    if ( c == '\n' )
    {
      if ( options_ & PARSE_EOL )
      {
        unread ( c );
        break;
      }

      lineno++;
      token += (char) c;
      c      = read  ();

      while ( c >= 0 && std::isspace(c) && c != '\n' )
      {
        c = read ();
      }

      continue;
    }

    if ( c == '\\' )
    {
      int  next = read ();

      if ( next == delim )
      {
        c = delim;
      }
      else if ( next == ' ' )
      {
        c = ' ';
      }
      else if ( next == 'b' )
      {
        c = '\b';
      }
      else if ( next == 'f' )
      {
        c = '\f';
      }
      else if ( next == 'n' )
      {
        c = '\n';
      }
      else if ( next == 'r' )
      {
        c = '\r';
      }
      else if ( next == 't' )
      {
        c = '\t';
      }
      else if ( next == 'v' )
      {
        c = '\v';
      }
      else if ( next == '\\' )
      {
        c = '\\';
      }
      else if ( next == '\n' )
      {
        lineno++;
        c = read ();

        while ( c >= 0 && std::isspace(c) && c != '\n' )
        {
          c = read();
        }
      }
      else
      {
        skipString_ ( delim );

        parseError  (
          String::format (
            "illegal escape sequence `\\%c\' in string literal",
            next
          )
        );
      }
    }

    token += (char) c;
    c      = read  ();
  }

  if ( c != delim )
  {
    parseError (
      String::format (
        "unterminated string literal starting at line %d",
        line
      )
    );
  }
}


//-----------------------------------------------------------------------
//   readLine
//-----------------------------------------------------------------------


void BasicScanner::readLine ()
{
  int  c = read ();

  while ( c >= 0 && c != '\n' )
  {
    token += (char) c;
    c      = read  ();
  }

  if ( c == '\n' )
  {
    lineno++;
  }
}


//-----------------------------------------------------------------------
//   readNumber
//-----------------------------------------------------------------------


BasicScanner::NumberType BasicScanner::readNumber ()
{
  int  c = read ();

  if ( token.back() == '0' && ( c == 'x' || c == 'X' ) )
  {
    token += (char) c;
    c      = read  ();

    while ( c >= 0 && std::isxdigit(c) )
    {
      token += (char) c;
      c      = read  ();
    }

    unread      ( c );
    setHexInt_  ();

    return INTEGER;
  }
  else
  {
    while ( c >= 0 && std::isdigit(c) )
    {
      token += (char) c;
      c      = read  ();
    }

    if ( c == '.' )
    {
      token += (char) c;

      readFraction_ ();
      setFloat_     ();

      return FLOAT;
    }
    else if ( c == 'e' || c == 'E' )
    {
      token += (char) c;

      readExponent_ ();
      setFloat_     ();

      return FLOAT;
    }
    else
    {
      unread         ( c );
      setDecimalInt_ ();

      return INTEGER;
    }
  }
}


//-----------------------------------------------------------------------
//   tryReadNumber
//-----------------------------------------------------------------------


int BasicScanner::tryReadNumber ()
{
  int  c = read ();

  if ( std::isdigit(c) )
  {
    token += (char) c;

    return readNumber ();
  }

  if ( c == '.' )
  {
    int  next = read ();

    if ( std::isdigit(next) )
    {
      token += (char) c;
      token += (char) next;

      readFraction_  ();

      return FLOAT;
    }

    unread ( next );
  }

  unread ( c );

  return -1;
}


//-----------------------------------------------------------------------
//   isBoolToken
//-----------------------------------------------------------------------


bool BasicScanner::isBoolToken () const
{
  const idx_t  n = token.size ();
  const char*  s = token.addr ();


  if      ( n    == 4   &&
            s[0] == 't' &&
            s[1] == 'r' &&
            s[2] == 'u' &&
            s[3] == 'e' )
  {
    return true;
  }
  else if ( n    == 5   &&
            s[0] == 'f' &&
            s[1] == 'a' &&
            s[2] == 'l' &&
            s[3] == 's' &&
            s[4] == 'e' )
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   setTokenString
//-----------------------------------------------------------------------


void BasicScanner::setTokenString ( const String& str )
{
  token = str;
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void BasicScanner::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void BasicScanner::setOptions ( Options options )
{
  options_ = options;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


BasicScanner::Options BasicScanner::getOptions () const noexcept
{
  return options_;
}


//-----------------------------------------------------------------------
//   parseError
//-----------------------------------------------------------------------


void BasicScanner::parseError ( const String& what ) const
{
  throw ParseException ( String( lineno ), what );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void BasicScanner::init_ ()
{
  token.reserve ( 128 );

  lineno   = 1_lint;
  xvalue_  = 0.0;
  options_ = 0;
}


//-----------------------------------------------------------------------
//   skipString_
//-----------------------------------------------------------------------


void BasicScanner::skipString_ ( int delim )
{
  int  c = read ();

  while ( c >= 0 && c != delim && c != '\n' )
  {
    c = read ();
  }

  if ( c == '\n' )
  {
    unread ( c );
  }
}


//-----------------------------------------------------------------------
//   readFraction_
//-----------------------------------------------------------------------


void BasicScanner::readFraction_ ()
{
  int  c = read ();

  while ( c >= 0 && std::isdigit(c) )
  {
    token += (char) c;
    c      = read  ();
  }

  if ( c == 'e' || c == 'E' )
  {
    token += (char) c;

    readExponent_  ();
  }
  else
  {
    unread ( c );
  }
}


//-----------------------------------------------------------------------
//   readExponent_
//-----------------------------------------------------------------------


void BasicScanner::readExponent_ ()
{
  int  c = read ();

  if ( c == '+' || c == '-' )
  {
    token += (char) c;
    c      = read  ();
  }

  while ( c >= 0 && std::isdigit(c) )
  {
    token += (char) c;
    c      = read  ();
  }

  unread ( c );
}


//-----------------------------------------------------------------------
//   setDecimalInt_
//-----------------------------------------------------------------------


void BasicScanner::setDecimalInt_()
{
  const lint    MAX_VALUE = (maxOf<lint>() - 9) / 10;

  const char*   s = token.addr ();
  const idx_t   n = token.size ();

  lint          value;
  lint          sign;
  idx_t         i;


  if ( n == 0 )
  {
    parseError ( "badly formed integer: no digits" );
  }

  value = 0;
  sign  = 1;
  i     = 0;

  if ( s[i] == '-' )
  {
    sign = -1;
    i++;
  }
  else if ( s[i] == '+' )
  {
    i++;
  }

  for ( ; i < n; i++ )
  {
    if ( value > MAX_VALUE )
    {
      parseError ( "integer out of range" );
    }

    value = 10 * value + (idx_t) ( s[i] - '0' );
  }

  ivalue_ = sign * value;
}


//-----------------------------------------------------------------------
//   setHexInt_
//-----------------------------------------------------------------------


void BasicScanner::setHexInt_ ()
{
  const lint   MAX_VALUE = (maxOf<lint>() - 15) / 16;

  const char*  s = token.addr ();
  const idx_t  n = token.size ();

  lint         value;
  lint         sign;
  idx_t        i;


  if ( n == 0 )
  {
    parseError ( "badly formed integer: no digits" );
  }

  value = 0;
  sign  = 1;
  i     = 0;

  if ( s[i] == '-' )
  {
    sign = -1;
    i++;
  }
  else if ( s[i] == '+' )
  {
    i++;
  }

  // Next two characters are '0' and 'x'; skip them

  i += 2;

  for ( ; i < n; i++ )
  {
    if ( value > MAX_VALUE )
    {
      parseError ( "integer out of range" );
    }

    if ( std::isdigit(s[i]) )
    {
      value = 16 * value + (idx_t) ( s[i] - '0' );
    }
    else
    {
      value = 16 * value + (idx_t) ( toupper(s[i]) - 'A' ) + 10;
    }
  }

  ivalue_ = sign * value;
}


//-----------------------------------------------------------------------
//   setFloat_
//-----------------------------------------------------------------------


void BasicScanner::setFloat_ ()
{
  char*   end = token.xalloc ( 1 );
  double  value;


  end[0] = '\0';
  value  = strtod ( token.addr(), &end );

  if ( token[0] == '\0' || end[0] != '\0' )
  {
    parseError ( "badly formed floating point number" );
  }

  xvalue_ = value;
}


JEM_END_PACKAGE( util )
