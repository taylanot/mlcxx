
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

#ifndef JEM_IO_DEFLATOR_H
#define JEM_IO_DEFLATOR_H

#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/io/ZipMode.h>
#include <jem/io/ZipException.h>
#include <jem/io/OutputStream.h>

#include <zlib.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class Deflator
//-----------------------------------------------------------------------


class Deflator : public Collectable
{
 public:

                          Deflator

    ( Ref<OutputStream>     output,
      ZipMode               zmode,
      int                   level,
      idx_t                 bufsize );

  inline ulint            getBytesIn    () const;
  inline ulint            getBytesOut   () const;
  inline ulint            getChecksum   () const;

  inline void             sync

    ( ThrowMode             tm = CAN_THROW );

  inline void             finish

    ( ThrowMode             tm = CAN_THROW );

  inline idx_t            flush

    ( ThrowMode             tm = CAN_THROW );

  inline void             deflate

    ( const void*           buf,
      idx_t                 n,
      ThrowMode             tm = CAN_THROW );

  inline void             write

    ( byte                  b,
      ThrowMode             tm = CAN_THROW );


 protected:

  virtual                ~Deflator      ();


 private:

  inline void             makeSpace_    ();
  void                    reset_        ();

  void                    zipError_

    ( ThrowMode             tm );


 private:

  Ref<OutputStream>       output_;
  z_stream                zstream_;
  ulint                   crc_;
  idx_t                   bufsize_;
  byte*                   buffer_;
  bool                    synced_;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef Deflator          ZlibDeflator;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getBytesIn
//-----------------------------------------------------------------------


inline ulint Deflator::getBytesIn () const
{
  return (ulint) zstream_.total_in;
}


//-----------------------------------------------------------------------
//   getBytesOut
//-----------------------------------------------------------------------


inline ulint Deflator::getBytesOut () const
{
  return (ulint) zstream_.total_out;
}


//-----------------------------------------------------------------------
//   getChecksum
//-----------------------------------------------------------------------


inline ulint Deflator::getChecksum () const
{
  return crc_;
}


//-----------------------------------------------------------------------
//   sync
//-----------------------------------------------------------------------


inline void Deflator::sync ( ThrowMode tm )
{
  if ( ! synced_ )
  {
    zstream_.next_in  = 0;
    zstream_.avail_in = 0;

    do
    {
      makeSpace_ ();

      if ( ::deflate( &zstream_, Z_SYNC_FLUSH ) != Z_OK )
      {
        zipError_ ( tm );
        return;
      }
    }
    while ( ! zstream_.avail_out );

    synced_ = true;
  }
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


inline void Deflator::finish ( ThrowMode tm )
{
  int  result;

  zstream_.next_in  = 0;
  zstream_.avail_in = 0;

  do
  {
    makeSpace_ ();

    result = ::deflate ( &zstream_, Z_FINISH );

    if ( result != Z_OK && result != Z_STREAM_END )
    {
      zipError_ ( tm );
      return;
    }
  }
  while ( result != Z_STREAM_END );

  synced_ = true;
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


inline idx_t Deflator::flush ( ThrowMode tm )
{
  const idx_t  n = bufsize_ - (idx_t) zstream_.avail_out;

  zstream_.next_out  = (Bytef*) buffer_;
  zstream_.avail_out = (uInt)   bufsize_;

  if ( tm == CAN_THROW )
  {
    output_->write        ( buffer_, n );
  }
  else
  {
    output_->writeNoThrow ( buffer_, n );
  }

  return n;
}


//-----------------------------------------------------------------------
//   deflate
//-----------------------------------------------------------------------


inline void Deflator::deflate

  ( const void*  buf,
    idx_t        n,
    ThrowMode    tm )

{
  while ( n > 0 )
  {
    int  k = (int) jem::min ( (idx_t) jem::maxOf<int>(), n );

    zstream_.next_in  = (Bytef*) buf;
    zstream_.avail_in = (uInt)   k;

    while ( zstream_.avail_in )
    {
      makeSpace_ ();

      if ( ::deflate( &zstream_, Z_NO_FLUSH ) != Z_OK )
      {
        zipError_ ( tm );
        return;
      }

      synced_ = false;
    }

    crc_ = (ulint) ::crc32 ( (uLong) crc_, (Bytef*) buf, (uInt) k );
    n   -= k;
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


inline void Deflator::write ( byte b, ThrowMode tm )
{
  makeSpace_ ();

  *zstream_.next_out = b;

  zstream_.next_out++;
  zstream_.avail_out--;
}


//-----------------------------------------------------------------------
//   makeSpace_
//-----------------------------------------------------------------------


inline void Deflator::makeSpace_ ()
{
  if ( zstream_.avail_out <= (uInt) (bufsize_ >> 3) )
  {
    flush ();
  }
}


JEM_END_PACKAGE( io )

#endif
