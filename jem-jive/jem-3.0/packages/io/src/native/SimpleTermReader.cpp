
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


#include <cstdio>
#include <jem/base/StringBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/native/System.h>
#include <jem/io/File.h>
#include <jem/io/InputStream.h>
#include <jem/io/EOFException.h>
#include "native/SimpleTermReader.h"


JEM_DEFINE_CLASS( jem::io::StdTermReader );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class StdTermReader
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdTermReader::StdTermReader ()
{}


StdTermReader::~StdTermReader ()
{}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t StdTermReader::poll ( const Time& timeout )
{
  return File::getStdin()->poll ( timeout );
}


//-----------------------------------------------------------------------
//   readLine
//-----------------------------------------------------------------------


String StdTermReader::readLine

  ( const String&   prompt,
    WordCompleter*  wcomp )

{
  const idx_t  n = prompt.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( putchar( prompt[i] ) == EOF )
    {
      String  details = NativeSystem::strerror ();

      throw IOException (
        JEM_FUNC,
        String::format (
          "error writing to the terminal (%s)", details
        )
      );
    }
  }

  std::fflush ( stdout );

  StringBuffer  buf;

  buf.reserve ( 32 );

  while ( true )
  {
    int  c = getchar ();

    if ( c == '\n' )
    {
      break;
    }

    if ( c == EOF )
    {
      eofError_ ();
    }

    buf += (char) c;
  }

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> StdTermReader::clone () const
{
  return const_cast<Self*> ( this );
}


//-----------------------------------------------------------------------
//   eofError_
//-----------------------------------------------------------------------


void StdTermReader::eofError_ ()
{
  std::putchar ( '\n' );

  throw EOFException (
    "terminal",
    "end of file encountered while reading from the terminal"
  );
}


JEM_END_PACKAGE( io )
