
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <cctype>
#include <cstring>
#include <jem/base/limits.h>
#include <jem/base/RuntimeException.h>
#include <jem/base/IllegalInputException.h>
#include <jive/geom/IntegrationSchemes.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::BEGIN;
using jem::slice;
using jem::IllegalInputException;


//=======================================================================
//   class IntegrationSchemes
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  IntegrationSchemes::GAUSS        = "Gauss";
const char*  IntegrationSchemes::UNIFORM      = "Uniform";
const char*  IntegrationSchemes::TRAPEZOIDAL  = "Trapezoidal";
const char*  IntegrationSchemes::NEWTON_COTES = "NewtonCotes";


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void IntegrationSchemes::parse

  ( String&        name,
    idx_t&         pcount,
    const String&  ischeme )

{
  const char*  s = ischeme.addr ();

  idx_t        scale;
  idx_t        i, j;


  for ( j = ischeme.size() - 1; j >= 0 && std::isspace(s[j]); j-- ) ;

  pcount = 0;
  scale  = 1;

  for ( ; j >= 0 && std::isdigit(s[j]); j-- )
  {
    pcount += scale * (s[j] - '0');

    if ( scale > jem::maxOf( scale ) / 10 )
    {
      throw IllegalInputException (
        JEM_FUNC,
        String::format (
          "invalid integration scheme: `%s\': too many integration "
          "points specified",
          ischeme
        )
      );
    }

    scale *= 10;
  }

  if ( pcount == 0 )
  {
     throw IllegalInputException (
       JEM_FUNC,
       String::format (
         "invalid integration scheme: `%s\': no integration "
         "points specified",
         ischeme
       )
     );
  }

  for ( ; j >= 0 && std::isspace(s[j]); j-- ) ;

  j++;

  for ( i = 0; i < j && std::isspace(s[i]); i++ ) ;

  if ( i == j )
  {
    name = String ();
  }
  else
  {
    name = ischeme[slice(i,j)];
  }
}


//-----------------------------------------------------------------------
//   split
//-----------------------------------------------------------------------


void IntegrationSchemes::split

  ( String&        part0,
    String&        part1,
    const String&  ischeme )

{
  String*  parts[2];

  parts[0] = & part0;
  parts[1] = & part1;

  split_ ( 2, parts, ischeme );
}


void IntegrationSchemes::split

  ( String&        part0,
    String&        part1,
    String&        part2,
    const String&  ischeme )

{
  String*  parts[3];

  parts[0] = & part0;
  parts[1] = & part1;
  parts[2] = & part2;

  split_ ( 3, parts, ischeme );
}


void IntegrationSchemes::split

  ( String&        part0,
    String&        part1,
    String&        part2,
    String&        part3,
    const String&  ischeme )

{
  String*  parts[4];

  parts[0] = & part0;
  parts[1] = & part1;
  parts[2] = & part2;
  parts[3] = & part3;

  split_ ( 4, parts, ischeme );
}


//-----------------------------------------------------------------------
//   join
//-----------------------------------------------------------------------


String IntegrationSchemes::join

  ( const String&  part0,
    const String&  part1 )

{
  const idx_t  m = part0.size ();
  const idx_t  n = part1.size ();

  if ( m + n <= 128 )
  {
    char  buf[128 + 1];

    std::memcpy ( buf,         part0.addr(), (size_t) m );

    buf[m] = '*';

    std::memcpy ( buf + m + 1, part1.addr(), (size_t) n );

    return String ( buf, buf + (m + n + 1) );
  }
  else
  {
    return String::format ( "%s*%s", part0, part1 );
  }
}


String IntegrationSchemes::join

  ( const String&  part0,
    const String&  part1,
    const String&  part2 )

{
  return String::format ( "%s*%s*%s", part0, part1, part2 );
}


String IntegrationSchemes::join

  ( const String&  part0,
    const String&  part1,
    const String&  part2,
    const String&  part3 )

{
  return String::format ( "%s*%s*%s*%s",
                          part0, part1, part2, part3 );
}


//-----------------------------------------------------------------------
//   split_
//-----------------------------------------------------------------------


void IntegrationSchemes::split_

  ( idx_t          n,
    String**       parts,
    const String&  ischeme )

{
  idx_t  i, j, k, m;


  m = ischeme.size ();
  k = n - 1;

  while ( k > 0 )
  {
    i = ischeme.rfind ( '*', m - 1 );

    if ( i < 0 )
    {
      throw IllegalInputException (
        JEM_FUNC,
        String::format (
          "invalid integration scheme: `%s\'; expected a compound "
          "scheme comprising %d sub-schemes",
          ischeme,
          n
        )
      );
    }

    j = i + 1;

    while ( j < m && std::isspace(ischeme[j])     ) j++;
    while ( i > 0 && std::isspace(ischeme[i - 1]) ) i--;

    if ( i == 0 || j == m )
    {
      throw IllegalInputException (
        JEM_FUNC,
        String::format (
          "invalid compound integration scheme: `%s\': "
          "empty sub-scheme",
          ischeme
        )
      );
    }

    *parts[k] = ischeme[slice(j,m)];

    m = i;
    k--;
  }

  *parts[0] = ischeme[slice(BEGIN,m)];
}


JIVE_END_PACKAGE( geom )
