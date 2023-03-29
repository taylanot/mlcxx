
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
#include <jem/base/assert.h>
#include <jem/base/StringBuffer.h>
#include <jem/util/StringUtils.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class StringUtils
//=======================================================================

//-----------------------------------------------------------------------
//   shorten
//-----------------------------------------------------------------------


String StringUtils::shorten ( const String& s, idx_t width )
{
  JEM_PRECHECK2 ( width >= 0, "invalid result width" );

  const int    DOT_COUNT = 3;
  const char*  DOTS      = "...";

  const idx_t  n = s.size ();


  if ( n <= width )
  {
    return s;
  }

  idx_t  i = max ( 1_idx, (width - DOT_COUNT + 1) / 2 );
  idx_t  j = max ( i + DOT_COUNT, n - (width - DOT_COUNT - i) );

  if ( n <= j )
  {
    return s;
  }
  else
  {
    return ( s[slice(BEGIN,i)] + String( DOTS, DOTS + DOT_COUNT ) +
             s[slice(j,END)  ] );
  }
}


//-----------------------------------------------------------------------
//   split (at whitespace )
//-----------------------------------------------------------------------


Array<String> StringUtils::split ( const String& s )
{
  const idx_t     n = s.size();

  Array<String>  words;

  idx_t          wordCount;
  idx_t          i, j;


  wordCount = 0;

  for ( int phase = 0; phase < 2; phase++ )
  {
    i = j = wordCount = 0;

    while ( j < n )
    {
      while ( j < n && std::isspace( s[j] ) )
      {
        j++;
      }

      i = j;

      while ( j < n && ! std::isspace( s[j] ) )
      {
        j++;
      }

      if ( j > i )
      {
        if ( phase > 0 )
        {
          words[wordCount] = s[slice(i,j)];
        }

        wordCount++;
      }

      j++;
    }

    if ( phase == 0 )
    {
      words.ref ( Array<String>( wordCount ) );
    }

  }

  return words;
}


//-----------------------------------------------------------------------
//   split (at a character)
//-----------------------------------------------------------------------


Array<String> StringUtils::split ( const String& s, char c )
{
  const idx_t    n = s.size();

  Array<String>  words;
  idx_t          wordCount;
  idx_t          i, j;


  wordCount = 0;

  for ( j = 0; j < n; j++ )
  {
    if ( s[j] == c )
    {
      wordCount++;
    }
  }

  words.ref ( Array<String>( wordCount + 1 ) );

  i = wordCount = 0;

  for ( j = 0; j < n; j++ )
  {
    if ( s[j] == c )
    {
      words[wordCount++] = s[slice(i,j)];
      i = j + 1;
    }
  }

  words[wordCount] = s[slice(i,j)];

  return words;
}


//-----------------------------------------------------------------------
//   join
//-----------------------------------------------------------------------


String StringUtils::join ( const Array<String>&  words )
{
  const idx_t  n = words.size ();

  if ( n == 0 )
  {
    return String ();
  }

  if ( n == 1 )
  {
    return words[0];
  }

  StringBuffer  buf;

  buf.reserve ( max( 64_idx, words.size() ) );

  buf += words[0];

  for ( idx_t i = 1; i < n; i++ )
  {
    buf += ' ';
    buf += words[i];
  }

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   join (with a separator string)
//-----------------------------------------------------------------------


String StringUtils::join

  ( const Array<String>&  words,
    const String&         sep )

{
  const idx_t  n = words.size ();

  if ( n == 0 )
  {
    return String ();
  }

  if ( n == 1 )
  {
    return words[0];
  }

  StringBuffer  buf;

  buf.reserve ( max( 64_idx, words.size() ) );

  buf += words[0];

  for ( idx_t i = 1; i < n; i++ )
  {
    buf += sep;
    buf += words[i];
  }

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   replace
//-----------------------------------------------------------------------


String StringUtils::replace

  ( const String&  str,
    const String&  word,
    const String&  rep )

{
  idx_t  j = str.find ( word );


  if ( j < 0 )
  {
    return str;
  }
  else
  {
    const idx_t   m = word.size ();
    const idx_t   n = str .size ();

    StringBuffer  buf;


    buf.reserve  ( n + 32 );

    buf += str[slice(BEGIN,j)];
    buf += rep;

    idx_t  i = j + m;

    while ( i < n )
    {
      j = str.find ( word, i );

      if ( j < 0 )
      {
        buf += str[slice(i,END)];
        break;
      }

      buf += str[slice(i,j)];
      buf += rep;

      i = j + m;
    }

    return buf.toString ();
  }
}


//-----------------------------------------------------------------------
//   expand
//-----------------------------------------------------------------------


String StringUtils::expand

  ( const String&  str,
    idx_t           count )

{
  if ( str.find( '%' ) < 0 )
  {
    return str;
  }
  else
  {
    const char*   s = str.addr ();
    const idx_t   n = str.size ();

    StringBuffer  buf;

    idx_t          i, j, k;


    buf.reserve ( n + 32 );

    i = 0;

    while ( i < n )
    {
      if ( s[i] != '%' )
      {
        buf += s[i++];
        continue;
      }

      j = i;

      if ( (++i) == n )
      {
        goto bad_format;
      }

      if ( s[i] == '%' )
      {
        buf += '%';
        continue;
      }

      k = 1;

      if ( std::isdigit(s[i]) )
      {
        k = s[i] - '0';

        if ( (++i) == n )
        {
          goto bad_format;
        }
      }

      if ( s[i] == 'i' )
      {
        buf += String::format ( "%.*d", k, count );
      }
      else
      {
        goto bad_format;
      }

      i++;
      continue;

     bad_format:

      for ( ; j < i; i++ )
      {
        buf += s[j];
      }
    }

    return buf.toString ();
  }
}


JEM_END_PACKAGE( util )
