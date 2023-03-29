
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
#include "Scanner.h"


JEM_BEGIN_UFUNC_IMP


//=======================================================================
//   class Scanner
//=======================================================================

//-----------------------------------------------------------------------
//   readSymbol
//-----------------------------------------------------------------------


void Scanner::readSymbol ()
{
  int  c = read ();

  while ( c >= 0 && (std::isalnum(c) ||
                     c == '_'   || c == '.' || c == ':' ||
                     c == '['   || c == ']') )
  {
    token += (char) c;
    c      = read  ();
  }

  unread ( c );
}


//-----------------------------------------------------------------------
//   readDollar
//-----------------------------------------------------------------------


void Scanner::readDollar ()
{
  JEM_ASSERT2 ( token.size() > 0 && token[0] == '$',
                "current token does not start with a "
                "`$\' character" );

  int  c = read ();

  if ( c < 0 || std::isspace( c ) )
  {
    parseError ( "parse error after `$\'" );
  }

  token += (char) c;

  switch ( c )
  {
  case '(':

    readDollar_ ( ')' );

    break;

  case '{':

    readDollar_ ( '}' );

    break;

  case '[':

    readDollar_ ( ']' );

    break;

  default:

    c = read ();

    while ( std::isalnum(c) || c == '_' )
    {
      token += (char) c;
      c      = read  ();
    }

    unread ( c );
  }
}


//-----------------------------------------------------------------------
//   readDollar_
//-----------------------------------------------------------------------


void Scanner::readDollar_ ( int delim )
{
  int  c = read ();

  // Skip leading whitespace.

  while ( c >= 0 && std::isspace(c) )
  {
    if ( c == '\n' )
    {
      lineno++;
    }

    c = read ();
  }

  // Read the symbol name.

  idx_t   n = 16;
  char*   t = token.xalloc ( n );
  idx_t   i = 0;

  while ( c >= 0 && c != delim )
  {
    if ( c == '\n' )
    {
      lineno++;
    }

    if ( i >= n )
    {
      n *= 2;
      t  = token.xalloc ( n );
    }

    t[i++] = (char) c;
    c      = read  ();
  }

  if ( c != delim )
  {
    parseError (
      String::format (
        "unterminated symbol (missing `%c\')", delim
      )
    );
  }

  // Strip trailing whitespace.

  while ( i > 0 && std::isspace(t[i - 1]) )
  {
    i--;
  }

  token.commit ( i );

  token += (char) c;
}


JEM_END_UFUNC_IMP
