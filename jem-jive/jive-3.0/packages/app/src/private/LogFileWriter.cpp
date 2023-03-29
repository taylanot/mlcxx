
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


#include <jem/base/ClassTemplate.h>
#include <jem/io/File.h>
#include <jem/io/FileOutputStream.h>
#include "LogFileWriter.h"


JEM_DEFINE_CLASS( jive::app::LogFileWriter );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::io::File;
using jem::io::FileOutputStream;


//=======================================================================
//   class LogFileWriter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


LogFileWriter::LogFileWriter () :

  Super   ( File::getStdout() ),
  stdout_ ( true )

{
  atStart_ = (byte) 1;
}


LogFileWriter::LogFileWriter

  ( const String&  fname,
    OpenFlags      flags ) :

    Super   ( newInstance<FileOutputStream>( fname, flags ) ),
    stdout_ ( false )

{
  atStart_ = (byte) 1;
}


LogFileWriter::~LogFileWriter ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> LogFileWriter::dup ()
{
  // The LogFileWriter class is thread safe, so this method simply
  // returns the current object. If a new object would be returned,
  // then multiple newline characters written by different threads
  // would not be properly discarded.

  return this;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void LogFileWriter::close ()
{
  if ( ! stdout_ )
  {
    Super::close ();
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void LogFileWriter::write ( int c )
{
  if ( c == '\n' )
  {
    if ( atStart_ > 1 )
    {
      return;
    }

    atStart_++;
  }
  else
  {
    atStart_ = 0;
  }

  output_->write ( &c, 1 );
}


void LogFileWriter::write ( int c, idx_t n )
{
  if ( c == '\n' )
  {
    if ( atStart_ > 1 )
    {
      return;
    }

    n        = jem::min ( n, 2 - (idx_t) atStart_ );
    atStart_ = (byte) (atStart_ + (byte) n);
  }
  else
  {
    atStart_ = 0;
  }

  Super::write ( c, n );
}


void LogFileWriter::write ( const char* buf, idx_t n )
{
  idx_t  i, j;

  for ( i = j = 0; j < n; j++ )
  {
    if ( buf[j] == '\n' )
    {
      if ( atStart_ > 1 )
      {
        if ( j > i )
        {
          output_->write ( buf + i, j - i );
        }

        i = j + 1;
      }
      else
      {
        atStart_++;
      }
    }
    else
    {
      atStart_ = 0;
    }
  }

  if ( j > i )
  {
    output_->write ( buf + i, j - i );
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void LogFileWriter::writeNoThrow ( const char* buf, idx_t n ) noexcept
{
  idx_t  i, j;

  for ( i = j = 0; j < n; j++ )
  {
    if ( buf[j] == '\n' )
    {
      if ( atStart_ > 1 )
      {
        if ( j > i )
        {
          output_->writeNoThrow ( buf + i, j - i );
        }

        i = j + 1;
      }
      else
      {
        atStart_++;
      }
    }
    else
    {
      atStart_ = 0;
    }
  }

  if ( j > i )
  {
    output_->writeNoThrow ( buf + i, j - i );
  }
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void LogFileWriter::reset ()
{
  atStart_ = (byte) 1;
}


JIVE_END_PACKAGE( app )
