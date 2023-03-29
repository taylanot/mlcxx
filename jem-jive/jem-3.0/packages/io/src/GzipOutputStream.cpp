
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


#include <jem/base/Time.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/params.h>
#include <jem/io/IOException.h>
#include <jem/io/GzipOutputStream.h>
#include <jem/io/Deflator.h>


JEM_DEFINE_CLASS( jem::io::GzipOutputStream );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   private non-member functions
//=======================================================================

//-----------------------------------------------------------------------
//   writeLong
//-----------------------------------------------------------------------


inline void       writeLong

  ( Deflator&       out,
    ulint           num )

{
  for ( int i = 0; i < 4; i++ )
  {
    out.write ( (byte) (num & 255_ulint) );

    num = num >> 8;
  }
}


//=======================================================================
//   class GzipOutputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GzipOutputStream::GzipOutputStream

  ( Ref<OutputStream>  out,
    int                level,
    idx_t              bufsize ) :

    output_ ( out )

{
  JEM_PRECHECK ( out );

  ulint  t = (ulint) Time::now().sec ();

  if ( bufsize < 0 )
  {
    bufsize = DEFAULT_BUFSIZE;
  }

  deflator_  = newInstance<Deflator> ( out,   RAW_ZLIB_MODE,
                                       level, bufsize );

  Deflator&  def = * deflator_;

  // Write a simple gzip header (see www.gzip.org for the format).
  // ... magic bytes:

  def.write ( GZIP_MAGIC[0] );
  def.write ( GZIP_MAGIC[1] );

  // ... the compression method:

  def.write (   8 );

  // ... the flag byte:

  def.write (   0 );

  // ... the time stamp:

  writeLong ( def, t );

  // ... the extra compressor flags:

  def.write (   0 );

  // ... the OS flag:

#if defined(JEM_OS_POSIX)

  def.write (   3 );

#else

  def.write ( 255 );

#endif

}


GzipOutputStream::~GzipOutputStream ()
{
  if ( deflator_ )
  {
    finish_ ( NO_THROW );
  }
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void GzipOutputStream::close ()
{
  if ( deflator_ )
  {
    finish_        ();
    output_->close ();
  }
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void GzipOutputStream::flush ()
{
  if ( deflator_ )
  {
    deflator_->sync  ();
    output_  ->flush ();
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void GzipOutputStream::write ( const void* buf, idx_t n )
{
  if ( ! deflator_ )
  {
    throw IOException ( JEM_FUNC, "output stream not open" );
  }

  deflator_->deflate ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void GzipOutputStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  if ( deflator_ )
  {
    deflator_->deflate ( buf, n, NO_THROW );
  }
}


//-----------------------------------------------------------------------
//   finish_
//-----------------------------------------------------------------------


void GzipOutputStream::finish_ ( ThrowMode tm )
{
  deflator_->finish ( tm );

  writeLong ( *deflator_, deflator_->getChecksum() );
  writeLong ( *deflator_, deflator_->getBytesIn()  );

  deflator_->flush  ( tm );

  deflator_ = nullptr;
}


JEM_END_PACKAGE( io )
