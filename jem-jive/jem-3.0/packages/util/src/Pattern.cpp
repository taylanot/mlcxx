
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


#include <jem/base/IllegalInputException.h>
#include <jem/util/Pattern.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class Pattern::Utils_
//=======================================================================


class Pattern::Utils_
{
 public:

  static const int    STACK_SIZE = 16;


  static void         missingBracketError

    ( const String&     pattern );

  static void         stackOverflowError

    ( const String&     pattern );

  static bool         inCharset

    ( idx_t&            i,
      const char*       p,
      idx_t             psize,
      char              c );

  static bool         matchChar

    ( idx_t&            i,
      const char*       p,
      idx_t             psize,
      idx_t&            j,
      const char*       w,
      idx_t             wsize );

  static bool         matchShortest

    ( idx_t&            i,
      const char*       p,
      idx_t             psize,
      idx_t&            j,
      const char*       w,
      idx_t             wsize );

  static bool         matchLongest

    ( idx_t&            i,
      const char*       p,
      idx_t             psize,
      idx_t&            j,
      const char*       w,
      idx_t             wsize );

};


//-----------------------------------------------------------------------
//   missingBracketError
//-----------------------------------------------------------------------


void Pattern::Utils_::missingBracketError

  ( const String&  pattern )

{
  throw IllegalInputException (
    CLASS_NAME_,
    String::format (
      "invalid pattern: `%s\'; missing `]\'", pattern
    )
  );
}


//-----------------------------------------------------------------------
//   stackOverflowError
//-----------------------------------------------------------------------


void Pattern::Utils_::stackOverflowError

  ( const String&  pattern )

{
   throw IllegalInputException (
    CLASS_NAME_,
    String::format (
      "invalid pattern: `%s\'; too many `*\' characters", pattern
    )
  );
}


//-----------------------------------------------------------------------
//   inCharset
//-----------------------------------------------------------------------


bool Pattern::Utils_::inCharset

  ( idx_t&       i,
    const char*  p,
    idx_t        psize,
    char         c )

{
  JEM_ASSERT ( p[i] == '[' );

  const idx_t  pend   = psize - 1;
  bool         negate = false;
  bool         match  = false;
  idx_t        j      = i;


  j++;

  if ( j == psize )
  {
    return false;
  }

  if ( p[j] == '^' )
  {
    negate = true;
    j++;
  }
  else if ( p[j] == ']' )
  {
    match = (c == ']');
    j++;
  }
  else if ( p[j] == '-' )
  {
    match = (c == '-');
    j++;
  }

  while ( j < pend && p[j] != ']' && ! match )
  {
    if ( p[j] == '-' && p[j + 1] != ']' )
    {
      match = (p[j - 1] <= c && c <= p[j + 1]);
      j++;
    }
    else if ( p[j] == '\\' )
    {
      j++;

      if ( p[j] == 'n' )
      {
        match = (c == '\n');
      }
      else if ( p[j] == 't' )
      {
        match = (c == '\t');
      }
      else
      {
        match = (c == p[j]);
      }
    }
    else
    {
      match = (c == p[j]);
    }

    j++;
  }

  while ( j < psize && p[j] != ']' )
  {
    j++;
  }

  if ( j == psize )
  {
    missingBracketError ( String( p, p + psize ) );
  }

  j++;

  if ( negate )
  {
    match = ! match;
  }

  if ( match )
  {
    i = j;
  }

  return match;
}


//-----------------------------------------------------------------------
//   matchChar
//-----------------------------------------------------------------------


bool Pattern::Utils_::matchChar

  ( idx_t& i, const char* p, idx_t psize,
    idx_t& j, const char* w, idx_t wsize )

{
  if ( p[i] == '^' && j == 0 )
  {
    i++;
    return true;
  }

  if ( p[i] == '$' && j == wsize )
  {
    i++;
    return true;
  }

  if ( j == wsize )
  {
    return false;
  }

  if ( p[i] == '?' )
  {
    i++;
    j++;
    return true;
  }

  if ( p[i] == '\\' && i + 1 < psize )
  {
    i++;

    if ( (p[i] == 'n' && w[j] == '\n') ||
         (p[i] == 't' && w[j] == '\t') ||
         (p[i] == w[j]) )
    {
      i++;
      j++;
      return true;
    }

    i--;
  }

  if ( p[i] == '[' )
  {
    if ( inCharset( i, p, psize, w[j] ) )
    {
      j++;
      return true;
    }
    else
    {
      return false;
    }
  }

  if ( p[i] == w[j] )
  {
    i++;
    j++;
    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   matchShortest
//-----------------------------------------------------------------------


bool Pattern::Utils_::matchShortest

  ( idx_t& i, const char* p, idx_t psize,
    idx_t& j, const char* w, idx_t wsize )

{
  idx_t  stack[2][STACK_SIZE];
  int    top;


  top = -1;

  while ( true )
  {
    if ( i == psize )
    {
      return true;
    }

    if ( p[i] == '*' )
    {
      i++;
      top++;

      if ( top >= STACK_SIZE )
      {
        stackOverflowError ( String( p, p + psize ) );
      }

      stack[0][top] = i;
      stack[1][top] = j;

      continue;
    }

    if ( matchChar( i, p, psize, j, w, wsize ) )
    {
      continue;
    }

    while ( top >= 0 && stack[1][top] == wsize )
    {
      top--;
    }

    if ( top < 0 )
    {
      return false;
    }

    stack[1][top]++;

    i = stack[0][top];
    j = stack[1][top];
  }

  // This statement should never be reached.

  JEM_ASSERT ( false );

  return false;
}


//-----------------------------------------------------------------------
//   matchLongest
//-----------------------------------------------------------------------


bool Pattern::Utils_::matchLongest

  ( idx_t& i, const char* p, idx_t psize,
    idx_t& j, const char* w, idx_t wsize )

{
  idx_t  stack[3][STACK_SIZE];
  int    top;


  top = -1;

  while ( true )
  {
    if ( i == psize )
    {
      return true;
    }

    if ( p[i] == '*' )
    {
      i++;
      top++;

      if ( top >= STACK_SIZE )
      {
        stackOverflowError ( String( p, p + psize ) );
      }

      stack[0][top] = i;
      stack[1][top] = wsize;
      stack[2][top] = j;

      j = wsize;

      continue;
    }

    if ( matchChar( i, p, psize, j, w, wsize ) )
    {
      continue;
    }

    while ( top >= 0 && stack[1][top] == stack[2][top] )
    {
      top--;
    }

    if ( top < 0 )
    {
      return false;
    }

    stack[1][top]--;

    i = stack[0][top];
    j = stack[1][top];
  }

  return false;
}


//=======================================================================
//   class Pattern
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Pattern::CLASS_NAME_ = "jem::util::Pattern";


//-----------------------------------------------------------------------
//   check
//-----------------------------------------------------------------------


void Pattern::check

  ( const String&  pattern )

{
  const char*  p     = pattern.addr ();
  const idx_t  psize = pattern.size ();

  idx_t  starCount = 0;
  idx_t  i         = 0;


  while ( i < psize )
  {
    if ( p[i] == '*' )
    {
      starCount++;
    }
    else if ( p[i] == '[' )
    {
      while ( i < psize && p[i] != ']' )
      {
        i++;
      }

      if ( i == psize )
      {
        Utils_::missingBracketError ( pattern );
      }
    }

    i++;
  }

  if ( starCount >= Utils_::STACK_SIZE )
  {
    Utils_::stackOverflowError ( pattern );
  }
}


//-----------------------------------------------------------------------
//   matches
//-----------------------------------------------------------------------


bool Pattern::matches

  ( const String&  pattern,
    const String&  word )

{
  const char*  p     = pattern.addr ();
  const idx_t  psize = pattern.size ();
  const char*  w     = word   .addr ();
  const idx_t  wsize = word   .size ();

  idx_t  i = 0;
  idx_t  j = 0;


  if ( Utils_::matchLongest( i, p, psize, j, w, wsize ) )
  {
    return (j == wsize);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool Pattern::contains

  ( const String&  pattern,
    const String&  word )

{
  Span  s = findLongestMatch ( pattern, 0,
                               word.addr(), word.size() );

  return s.notEmpty ();
}


//-----------------------------------------------------------------------
//   findShortestMatch
//-----------------------------------------------------------------------


Range<idx_t> Pattern::findShortestMatch

  ( const String&  pattern,
    idx_t          istart,
    const char*    w,
    idx_t          wsize )

{
  const char*  p     = pattern.addr ();
  const idx_t  psize = pattern.size ();

  idx_t  i, j;


  if ( psize == 0 )
  {
    return Span ( 0, 0 );
  }

  if ( istart < 0 )
  {
    istart = 0;
  }

  for ( ; istart < wsize; istart++ )
  {
    i = 0;
    j = istart;

    if ( p[0] == '*' ||
         Utils_::matchChar( i, p, psize, j, w, wsize ) )
    {
      i = 0;
      j = istart;

      if ( Utils_::matchShortest( i, p, psize, j, w, wsize ) ) {
        return Span ( istart, j );
      }
    }
  }

  return Span ( 0, 0 );
}


//-----------------------------------------------------------------------
//   rfindShortestMatch
//-----------------------------------------------------------------------


Range<idx_t> Pattern::rfindShortestMatch

  ( const String&  pattern,
    idx_t          istart,
    const char*    w,
    idx_t          wsize )

{
  const char*  p     = pattern.addr ();
  const idx_t  psize = pattern.size ();

  idx_t  i, j;


  if ( psize == 0 )
  {
    return Span ( 0, 0 );
  }

  if ( istart >= wsize )
  {
    istart = wsize - 1;
  }

  for ( ; istart >= 0; istart-- )
  {
    i = 0;
    j = istart;

    if ( p[0] == '*' ||
         Utils_::matchChar( i, p, psize, j, w, wsize ) )
    {
      i = 0;
      j = istart;

      if ( Utils_::matchShortest( i, p, psize, j, w, wsize ) )
      {
        return Span ( istart, j );
      }
    }
  }

  return Span ( 0, 0 );
}


//-----------------------------------------------------------------------
//   findLongestMatch
//-----------------------------------------------------------------------


Range<idx_t> Pattern::findLongestMatch

  ( const String&  pattern,
    idx_t          istart,
    const char*    w,
    idx_t          wsize )

{
  const char*  p     = pattern.addr ();
  const idx_t  psize = pattern.size ();

  idx_t  i, j;


  if ( psize == 0 )
  {
    return Span ( 0, 0 );
  }

  if ( istart < 0 )
  {
    istart = 0;
  }

  for ( ; istart < wsize; istart++ )
  {
    i = 0;
    j = istart;

    if ( p[0] == '*' ||
         Utils_::matchChar( i, p, psize, j, w, wsize ) )
    {
      i = 0;
      j = istart;

      if ( Utils_::matchLongest( i, p, psize, j, w, wsize ) )
      {
        return Span ( istart, j );
      }
    }
  }

  return Span ( 0, 0 );
}



//-----------------------------------------------------------------------
//   rfindLongestMatch
//-----------------------------------------------------------------------


Range<idx_t> Pattern::rfindLongestMatch

  ( const String&  pattern,
    idx_t          istart,
    const char*    w,
    idx_t          wsize )

{
  const char*  p     = pattern.addr ();
  const idx_t  psize = pattern.size ();

  idx_t  i, j;


  if ( psize == 0 )
  {
    return Span ( 0, 0 );
  }

  if ( istart >= wsize )
  {
    istart = wsize - 1;
  }

  for ( ; istart >= 0; istart-- )
  {
    i = 0;
    j = istart;

    if ( p[0] == '*' ||
         Utils_::matchChar( i, p, psize, j, w, wsize ) )
    {
      i = 0;
      j = istart;

      if ( Utils_::matchLongest( i, p, psize, j, w, wsize ) )
      {
        return Span ( istart, j );
      }
    }
  }

  return Span ( 0, 0 );
}


JEM_END_PACKAGE( util )
