
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


#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/StringBuffer.h>
#include <jem/io/PrintWriter.h>
#include <jive/util/utilities.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   joinNames
//-----------------------------------------------------------------------


String            joinNames

  ( const String&   first,
    const String&   second )

{
  using jem::StringBuffer;

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
//   isGzipFile
//-----------------------------------------------------------------------


bool isGzipFile ( const String& fname )
{
  const idx_t  n = fname.size ();

  if ( n > 3 )
  {
    const String  ext = fname[slice(n - 3,END)];

    if ( ext.equalsIgnoreCase( ".gz" ) )
    {
      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   printIntList
//-----------------------------------------------------------------------


void                printIntList

  ( PrintWriter&      out,
    const IdxVector&  ilist,
    const char*       sep )

{
  const idx_t  listSize = ilist.size ();

  idx_t        stride;
  idx_t        i, j;


  i = 0;
  j = i + 1;

  while ( j < listSize )
  {
    stride = ilist[j] - ilist[i];

    if ( stride > 0 )
    {
      do {
        j++;
      }
      while ( j < listSize &&
              ilist[j] == (ilist[j - 1] + stride) );
    }

    if ( i > 0 )
    {
      print ( out, sep );
    }

    if ( (j - i) < 5 )
    {
      print ( out, ilist[i] );

      for ( i++; i < j; i++ )
      {
        print ( out, sep, ilist[i] );
      }
    }
    else
    {
      print ( out, '[', ilist[i], ':', ilist[j - 1] + 1 );

      if ( stride > 1 )
      {
        print ( out, ';', stride );
      }

      print ( out, ']' );
    }

    i = j;
    j = i + 1;
  }

  if ( i < listSize )
  {
    if ( i > 0 )
    {
      print ( out, sep );
    }

    print ( out, ilist[i] );
  }
}


JIVE_END_PACKAGE( util )
