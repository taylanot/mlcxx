
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


#include <jem/base/ClassTemplate.h>
#include <jem/io/params.h>
#include <jem/io/ZipException.h>
#include <jem/io/GzipInputStream.h>
#include <jem/io/Inflator.h>


JEM_DEFINE_CLASS( jem::io::GzipInputStream );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   private non-member functions
//=======================================================================

//-----------------------------------------------------------------------
//   readShort
//-----------------------------------------------------------------------


inline bool       readShort

  ( Inflator&       in,
    unsigned int&   num )

{
  byte  b[2];

  for ( int i = 0; i < 2; i++ )
  {
    if ( ! in.read( b[i] ) )
    {
      return false;
    }
  }

  num = 0;

  for ( int i = 1; i >= 0; i-- )
  {
    num = num << 8;
    num = num + (unsigned int) b[i];
  }

  return true;
}


//-----------------------------------------------------------------------
//   readLong
//-----------------------------------------------------------------------


inline bool       readLong

  ( Inflator&       in,
    ulint&          num )

{
  byte  b[4];

  for ( int i = 0; i < 4; i++ )
  {
    if ( ! in.read( b[i] ) )
    {
      return false;
    }
  }

  num = 0_ulint;

  for ( int i = 3; i >= 0; i-- )
  {
    num = num << 8;
    num = num + (ulint) b[i];
  }

  return true;
}


//=======================================================================
//   class GzipInputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GzipInputStream::GzipInputStream

  ( Ref<InputStream>  in,
    idx_t             bufsize ) :

    input_ ( in )

{
  JEM_PRECHECK ( in );

  if ( bufsize < 0 )
  {
    bufsize = DEFAULT_BUFSIZE;
  }

  inflator_ = newInstance<Inflator> ( in, RAW_ZLIB_MODE, bufsize );

  if ( ! readHeader_() )
  {
    throw ZipException (
      JEM_FUNC,
      "unzip error: invalid gzip header"
    );
  }
}


GzipInputStream::~GzipInputStream ()
{
  if ( inflator_ )
  {
    inflator_->finish ( NO_THROW );
  }
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void GzipInputStream::close ()
{
  if ( inflator_ )
  {
    inflator_->finish ();
    input_   ->close  ();

    inflator_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t GzipInputStream::poll ( const Time& timeout )
{
  if ( inflator_ )
  {
    return inflator_->poll ( timeout );
  }
  else
  {
    return 0_idx;
  }
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t GzipInputStream::read ( void* buf, idx_t n )
{
  idx_t  k = 0;

  if ( inflator_ )
  {
    Inflator&  inf = * inflator_;

    k = inf.inflate ( buf, n );

    if ( inf.atEnd() )
    {
      ulint  num;

      if ( ! readLong( inf, num ) )
      {
        throw ZipException (
          JEM_FUNC,
          "unzip error: unexpected end of stream (checksum missing)"
        );
      }

      if ( num != inf.getChecksum() )
      {
        throw ZipException (
          JEM_FUNC,
          "unzip error: corrupt input stream (checksum mismatch)"
        );
      }

      if ( ! readLong( inf, num ) )
      {
        throw ZipException (
          JEM_FUNC,
          "unzip error: unexpected end of stream "
          "(original stream size missing)"
        );
      }

      if ( num != inf.getBytesOut() )
      {
        throw ZipException (
          JEM_FUNC,
          "unzip error: corrupt input stream "
          "(stream size mismatch)"
        );
      }

      inf.finish ();

      inflator_ = nullptr;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   readHeader_
//-----------------------------------------------------------------------


bool GzipInputStream::readHeader_ ()
{
  const byte    FHCRC     = 1 << 1;
  const byte    FEXTRA    = 1 << 2;
  const byte    FNAME     = 1 << 3;
  const byte    FCOMMENT  = 1 << 4;
  const byte    FRESERVED = 0xe0;

  Inflator&     inf = * inflator_;

  ulint         lnum;
  unsigned int  inum;
  byte          flags;
  byte          b;


  b     = 0x0;
  flags = 0x0;

  // Read the gzip header (see www.gzip.org for the header format).
  // ... the magic bytes:

  for ( int i = 0; i < 2; i++ )
  {
    if ( ! inf.read( b ) || b != GZIP_MAGIC[i] )
    {
      return false;
    }
  }

  // ... the compression method:

  if ( ! inf.read( b ) || b != 8 )
  {
    return false;
  }

  // ... the flag byte:

  if ( ! inf.read( flags ) )
  {
    return false;
  }

  if ( flags & FRESERVED )
  {
    return false;
  }

  // ... the modification time

  if ( ! readLong( inf, lnum ) )
  {
    return false;
  }

  // ... the extra compressor flags

  if ( ! inf.read( b ) )
  {
    return false;
  }

  // ... the OS flag:

  if ( ! inf.read( b ) )
  {
    return false;
  }

  // Read optional fields

  if ( flags & FEXTRA )
  {
    if ( ! readShort( inf, inum ) )
    {
      return false;
    }

    int  n = (int) inum;

    for ( int i = 0; i < n; i++ )
    {
      if ( ! inf.read( b ) )
      {
        return false;
      }
    }
  }

  if ( flags & FNAME )
  {
    b = 1;

    while ( b )
    {
      if ( ! inf.read( b ) )
      {
        return false;
      }
    }
  }

  if ( flags & FCOMMENT )
  {
    b = 1;

    while ( b )
    {
      if ( ! inf.read( b ) )
      {
        return false;
      }
    }
  }

  if ( flags & FHCRC )
  {
    if ( ! readShort( inf, inum ) )
    {
      return false;
    }
  }

  return true;
}


JEM_END_PACKAGE( io )
