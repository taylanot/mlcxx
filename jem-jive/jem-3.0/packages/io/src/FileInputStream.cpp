
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


#include <jem/base/utilities.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/IOException.h>
#include <jem/io/File.h>
#include <jem/io/FileStream.h>
#include <jem/io/FileInputStream.h>


JEM_DEFINE_CLASS( jem::io::FileInputStream );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class FileInputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FileInputStream::FileInputStream

  ( const String&  name,
    OpenFlags      flags )

{
  flags   &= ~FileFlags::WRITE;
  flags   |=  FileFlags::READ;

  fstream_ = File::open ( name, flags );
}


FileInputStream::FileInputStream ( Ref<FileStream> fs ) :

  fstream_ ( fs )

{
  JEM_PRECHECK ( fs );
}


FileInputStream::FileInputStream ( const Self& rhs ) :

  fstream_ ( rhs.fstream_->dup() )

{}


FileInputStream::~FileInputStream ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<InputStream> FileInputStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void FileInputStream::close ()
{
  fstream_->close ();
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t FileInputStream::poll ( const Time& timeout )
{
  int  result = fstream_->poll ( timeout );

  if ( result & FileStream::POLL_READ )
  {
    return 1_idx;
  }
  else
  {
    return 0_idx;
  }
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t FileInputStream::read ( void* buf, idx_t n )
{
  return fstream_->read ( buf, n );
}


//-----------------------------------------------------------------------
//   skip
//-----------------------------------------------------------------------


idx_t FileInputStream::skip ( idx_t n )
{
  if ( n <= 0 )
  {
    return 0;
  }

  try
  {
    lint  oldPos = fstream_->getPosition ();
    lint  newPos = min ( oldPos + (lint) n, fstream_->getLength() );

    fstream_->setPosition ( newPos );

    return (idx_t) (newPos - oldPos);
  }
  catch ( const IOException& )
  {}

  return Super::skip ( n );
}


//-----------------------------------------------------------------------
//   getFileStream
//-----------------------------------------------------------------------


Ref<FileStream> FileInputStream::getFileStream () const
{
  return fstream_;
}


JEM_END_PACKAGE( io )
