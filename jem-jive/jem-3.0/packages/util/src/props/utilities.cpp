
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
#include <cstring>
#include <jem/base/rtti.h>
#include <jem/base/StringBuffer.h>
#include "utilities.h"


JEM_BEGIN_PACKAGE( util )
JEM_BEGIN_NAMESPACE( props )


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   joinNames
//-----------------------------------------------------------------------


String            joinNames

  ( const String&   first,
    const String&   second )

{
  const idx_t  len1 = first .size ();
  const idx_t  len2 = second.size ();

  if      ( len1 == 0 )
  {
    return second;
  }
  else if ( len2 == 0 )
  {
    return first;
  }
  else
  {
    StringBuffer  buf ( len1 + len2 + 1 );
    char*         ptr = buf.addr ();

    std::memcpy ( ptr, first.addr(),  (size_t) len1 );

    ptr[len1] = '.';
    ptr      += len1 + 1;

    std::memcpy ( ptr, second.addr(), (size_t) len2 );

    return buf.toString ();
  }
}


//-----------------------------------------------------------------------
//   printValue (char)
//-----------------------------------------------------------------------


void              printValue

  ( PrintWriter&    out,
    char            val )

{
  if ( std::isalnum( (int) val ) )
  {
    print      ( out, '\'', val, '\'' );
  }
  else
  {
    printValue ( out, String( val ) );
  }
}


//-----------------------------------------------------------------------
//   printValue (String)
//-----------------------------------------------------------------------


void              printValue

  ( PrintWriter&    out,
    const String&   val )

{
  const PrintWriter
    ::Options       options = out.getOptions ();
  const char*       str     = val.addr       ();
  const idx_t       len     = val.size       ();

  char              c1, c2;
  idx_t             i,  j;


  c1 = c2 = '\0';

  out.setOptions ( options | PrintWriter::NO_WORD_WRAP );

  try
  {
    print ( out, '\"' );

    for ( i = j = 0; j < len; j++ )
    {
      if ( std::isalnum( (int) str[j] ) )
      {
        continue;
      }

      switch ( str[j] )
      {
      case '\"':

        c1 = '\\'; c2 = '\"'; break;

      case '\b':

        c1 = '\\'; c2 = 'b';  break;

      case '\f':

        c1 = '\\'; c2 = 'f';  break;

      case '\r':

        c1 = '\\'; c2 = 'r';  break;

      case '\t':

        c1 = '\\'; c2 = 't';  break;

      case '\v':

        c1 = '\\'; c2 = 'v';  break;

      case '\\':

        c1 = c2 = '\\';       break;

      case '$':

        c1 = c2 = '$';        break;

      default:

        continue;
      }

      if ( j > i )
      {
        out.write ( str + i, (j - i) );
      }

      print ( out, c1, c2 );

      i = j + 1;
    }

    if ( j > i )
    {
      out.write ( str + i, (j - i) );
    }

    print ( out, '\"' );
  }
  catch ( ... )
  {
    out.setOptions ( options );
    throw;
  }

  out.setOptions ( options );
}


JEM_END_NAMESPACE( props )
JEM_END_PACKAGE( util )
