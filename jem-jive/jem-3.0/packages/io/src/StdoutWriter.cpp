
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
#include <jem/base/Thread.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/IOException.h>
#include <jem/io/CancelledIOException.h>
#include <jem/io/StdoutWriter.h>

#undef JEM_MUST_FLUSH

#ifdef JEM_OS_WINDOWS
#  define JEM_MUST_FLUSH
#endif


JEM_DEFINE_CLASS( jem::io::StdoutWriter );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class StdoutWriter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdoutWriter::StdoutWriter ()
{}


StdoutWriter::StdoutWriter ( const Self& rhs )
{}


StdoutWriter::~StdoutWriter ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> StdoutWriter::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void StdoutWriter::close ()
{}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void StdoutWriter::flush ()
{
  if ( std::fflush( stdout ) )
  {
    if ( Thread::cancelled() )
    {
      throw CancelledIOException (
        "stdout",
        "flush of the standard output stream cancelled"
      );
    }
    else
    {
      throw IOException (
        "stdout",
        "failed to flush the standard output stream"
      );
    }
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void StdoutWriter::write ( int c )
{
  if ( std::putchar( c ) == EOF )
  {
    writeError_ ();
  }

#ifdef JEM_MUST_FLUSH

  if ( c == '\n' )
  {
    std::fflush ( stdout );
  }

#endif
}


void StdoutWriter::write

  ( int    c,
    idx_t  n )

{
  for ( idx_t i = 0; i < n; i++ )
  {
    if ( std::putchar( c ) == EOF )
    {
      writeError_ ();
    }
  }

#ifdef JEM_MUST_FLUSH

  if ( c == '\n' )
  {
    std::fflush ( stdout );
  }

#endif
}


void StdoutWriter::write

  ( const char*  buf,
    idx_t        n )

{
  for ( idx_t i = 0; i < n; i++ )
  {
    if ( std::putchar( buf[i] ) == EOF )
    {
      writeError_ ();
    }

#ifdef JEM_MUST_FLUSH

    if ( buf[i] == '\n' )
    {
      std::fflush ( stdout );
    }

#endif
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void StdoutWriter::writeNoThrow

  ( const char*   buf,
    idx_t         n ) noexcept

{
  for ( idx_t i = 0; i < n; i++ )
  {
    std::putchar ( buf[i] );

#ifdef JEM_MUST_FLUSH

    if ( buf[i] == '\n' )
    {
      std::fflush ( stdout );
    }

#endif
  }
}


//-----------------------------------------------------------------------
//   writeError_
//-----------------------------------------------------------------------


void StdoutWriter::writeError_ ()
{
  if ( Thread::cancelled() )
  {
    throw CancelledIOException (
      "stdout",
      "write to the standard output cancelled"
    );
  }
  else
  {
    throw IOException (
      "stdout",
      "error writing to the standard output"
    );
  }
}


JEM_END_PACKAGE( io )
