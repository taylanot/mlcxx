
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
#include <cstdlib>
#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ParseException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/io/Reader.h>


JEM_DEFINE_CUSTOM_CLASS( jem::io::Reader );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class Reader
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Reader::BUFSIZE_ = 128;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Reader::~Reader ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Reader::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis ( "a text input stream" );

  return myClass;
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Reader> Reader::dup ()
{
  throw IllegalOperationException (
    JEM_FUNC,
    "text input stream can not be duplicated"
  );

  return nullptr;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void Reader::close ()
{}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t Reader::poll ( const Time& timeout )
{
  return 0;
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t Reader::read ( char* buf, idx_t n )
{
  idx_t  i;

  for ( i = 0; i < n; i++ )
  {
    int  c = read ();

    if ( c < 0 )
    {
      break;
    }

    buf[i] = (char) c;
  }

  return i;
}


//-----------------------------------------------------------------------
//   readUntil
//-----------------------------------------------------------------------


idx_t Reader::readUntil ( char* buf, idx_t n, int delim )
{
  idx_t  i;

  for ( i = 0; i < n; i++ )
  {
    int  c = read ();

    if ( c == delim )
    {
      pushBack ( c );
      break;
    }
    else if ( c < 0 )
    {
      break;
    }
    else
    {
      buf[i] = (char) c;
    }
  }

  return i;
}


//-----------------------------------------------------------------------
//   skipUntil
//-----------------------------------------------------------------------


void Reader::skipUntil ( int delim )
{
  char  buf[BUFSIZE_];

  int   c = read ();

  while ( c >= 0 && c != delim )
  {
    readUntil ( buf, BUFSIZE_, delim );

    c = read  ();
  }

  pushBack ( c );
}


//-----------------------------------------------------------------------
//   skipWhite
//-----------------------------------------------------------------------


void Reader::skipWhite ()
{
  int  c = read ();

  while ( c >= 0 && std::isspace(c) )
  {
    c = read ();
  }

  pushBack ( c );
}


//-----------------------------------------------------------------------
//   skip
//-----------------------------------------------------------------------


idx_t Reader::skip ( idx_t count )
{
  char  buf[BUFSIZE_];

  idx_t  k = 0;

  while ( count > 0 )
  {
    idx_t  n = read ( buf, jem::min( (idx_t) BUFSIZE_, count ) );

    if ( n == 0 )
    {
      break;
    }

    count -= n;
    k     += n;
  }

  return k;
}


//-----------------------------------------------------------------------
//   readLine
//-----------------------------------------------------------------------


String Reader::readLine ()
{
  StringBuffer  buf;

  int  c = read ();

  while ( c >= 0 && c != '\n' )
  {
    buf += (char) c;

    idx_t  n = readUntil ( buf.xalloc( BUFSIZE_ ), BUFSIZE_, '\n' );

    buf.commit ( n );

    c = read ();
  }

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   parseByte
//-----------------------------------------------------------------------


byte Reader::parseByte ()
{
  lint  i = parseLong ();

  if ( i < 0 || i > maxOf<byte>() )
  {
    throw IllegalInputException (
      JEM_FUNC,
      String::format(
        "byte out of range; valid range is [%x,%x]",
        0, maxOf<byte> ()
      )
    );
  }

  return (byte) i;
}


//-----------------------------------------------------------------------
//   parseBool
//-----------------------------------------------------------------------


bool Reader::parseBool ()
{
  bool  b;
  int   c;

  skipWhite ();
  c = read  ();

  if      ( c == '0' )
  {
    b = false;
  }
  else if ( c == '1' )
  {
    b = true;
  }
  else if ( (c == 'f' || c == 'F') && matchIgnoreCase_( "alse" ) )
  {
    b = false;
  }
  else if ( (c == 't' || c == 'T') && matchIgnoreCase_( "rue" )  )
  {
    b = true;
  }
  else
  {
    throw ParseException ( JEM_FUNC, "badly formed boolean" );
  }

  return b;
}


//-----------------------------------------------------------------------
//   parseChar
//-----------------------------------------------------------------------


char Reader::parseChar ()
{
  int  c;

  skipWhite ();
  c = read  ();

  if ( c < 0 )
  {
    throw ParseException ( JEM_FUNC, "unexpected end of input" );
  }

  return (char) c;
}


//-----------------------------------------------------------------------
//   parseShort
//-----------------------------------------------------------------------


short Reader::parseShort ()
{
  lint  i = parseLint ();

  if ( i < (lint) minOf<short>() || i > (lint) maxOf<short>() )
  {
    throw IllegalInputException (
      JEM_FUNC,
      String::format (
        "short integer out of range; valid range is [%d,%d]",
        (int) minOf<short> (),
        (int) maxOf<short> ()
      )
    );
  }

  return (short) i;
}


//-----------------------------------------------------------------------
//   parseInt
//-----------------------------------------------------------------------


int Reader::parseInt ()
{
  lint  i = parseLint ();

  if ( i < (lint) minOf<int>() || i > (lint) maxOf<int>() )
  {
    throw IllegalInputException (
      JEM_FUNC,
      String::format (
        "integer out of range; valid range is [%d,%d]",
        minOf<int> (),
        maxOf<int> ()
      )
    );
  }

  return (int) i;
}


//-----------------------------------------------------------------------
//   parseLong
//-----------------------------------------------------------------------


long Reader::parseLong ()
{
  lint  i = parseLint ();

  if ( i < (lint) minOf<long>() || i > (lint) maxOf<long>() )
  {
    throw IllegalInputException (
      JEM_FUNC,
      String::format (
        "long integer out of range; valid range is [%d,%d]",
        minOf<long> (),
        maxOf<long> ()
      )
    );
  }

  return (long) i;
}

//-----------------------------------------------------------------------
//   parseLint
//-----------------------------------------------------------------------


lint Reader::parseLint ()
{
  char   buf[BUFSIZE_ + 1];
  char*  end;

  lint   i;
  int    j;
  int    c;

  skipWhite ();

  i = 0;
  j = 0;
  c = read ();

  if ( c == '+' || c == '-' )
  {
    buf[j++] = (char) c;
    c        = read ();
  }

  if ( c < 0 || ! std::isdigit(c) )
  {
    throw ParseException (
      JEM_FUNC,
      "badly formed integer: no digits"
    );
  }

  buf[j++] = (char) c;
  c        = read ();

  if ( buf[j - 1] == '0' && ( c == 'x' || c == 'X' ) )
  {
    buf[j++] = (char) c;
    c        = read ();

    while ( j < BUFSIZE_ && c >= 0 && std::isxdigit(c) )
    {
      buf[j++] = (char) c;
      c        = read ();
    }
  }
  else
  {
    while ( j < BUFSIZE_ && c >= 0 && std::isdigit(c) )
    {
      buf[j++] = (char) c;
      c        = read ();
    }
  }

  if ( j == BUFSIZE_ )
  {
    throw ParseException (
      JEM_FUNC,
      "badly formed integer: too many digits"
    );
  }

  pushBack ( c );
  buf[j++] = '\0';

  i = strtol( buf, &end, 0 );

  if ( *end != '\0' )
  {
    throw ParseException( JEM_FUNC, "badly formed integer" );
  }

  return i;
}


//-----------------------------------------------------------------------
//   parseFloat
//-----------------------------------------------------------------------


float Reader::parseFloat ()
{
  double  x = parseDouble ();

  int     sign;

  if ( x < 0.0 )
  {
    x    = -x;
    sign = -1;
  }
  else
  {
    sign = 1;
  }

  if ( x > 0.0 && x < minOf<float>() )
  {
    throw IllegalInputException (
      JEM_FUNC,
      String::format (
        "floating point underflow; smallest value is %e",
        minOf<float> ()
      )
    );
  }
  if ( x > maxOf<float>() )
  {
    throw IllegalInputException (
      JEM_FUNC,
      String::format (
        "floating point overflow; largest value is %e",
        maxOf<float> ()
      )
    );
  }

  return (float) (sign * x);
}


//-----------------------------------------------------------------------
//   parseDouble
//-----------------------------------------------------------------------


double Reader::parseDouble ()
{
  char    buf[BUFSIZE_ + 1];
  char*   end;

  double  x;
  int     c;
  int     i;

  skipWhite ();

  x = 0.0;
  i = 0;
  c = read ();

  if ( c < 0 )
  {
    throw ParseException ( JEM_FUNC, "unexpected end of input" );
  }

  if ( c == '+' || c == '-' )
  {
    buf[i++] = (char) c;
    c        = read ();
  }

  while ( i < BUFSIZE_ && c >= 0 && std::isdigit(c) )
  {
    buf[i++] = (char) c;
    c        = read ();
  }

  if ( i < BUFSIZE_ && c == '.' )
  {
    buf[i++] = (char) c;
    c        = read ();

    while ( i < BUFSIZE_ && c >= 0 && std::isdigit(c) )
    {
      buf[i++] = (char) c;
      c        = read ();
    }
  }

  if ( i < BUFSIZE_ && (c == 'E' || c == 'e') )
  {
    buf[i++] = (char) c;
    c        = read ();

    if ( i < BUFSIZE_ && (c == '+' || c == '-') )
    {
      buf[i++] = (char) c;
      c        = read ();
    }

    while ( i < BUFSIZE_ && c >= 0 && std::isdigit(c) )
    {
      buf[i++] = (char) c;
      c        = read ();
    }
  }

  if ( i == BUFSIZE_ )
  {
    throw ParseException (
      JEM_FUNC,
      "badly formed floating point number: too many digits"
    );
  }

  pushBack( c );
  buf[i++] = '\0';

  x = strtod ( buf, &end );

  if ( *buf == '\0' || *end != '\0' )
  {
    throw ParseException (
      JEM_FUNC,
      "badly formed floating point number"
    );
  }

  return x;
}


//-----------------------------------------------------------------------
//   matchIgnoreCase_
//-----------------------------------------------------------------------


bool Reader::matchIgnoreCase_ ( const char* str )
{
  char  buf[BUFSIZE_];

  int   i = 0;

  while ( i < BUFSIZE_ && str[i] != '\0' )
  {
    int  c = read ();

    if ( c < 0 || toupper(c) != toupper(str[i]) )
    {
      pushBack ( c );

      while ( i > 0 )
      {
        pushBack ( buf[--i] );
      }

      return false;
    }

    buf[i++] = (char) c;
  }

  return true;
}


JEM_END_PACKAGE( io )
