
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

#ifndef JEM_IO_INFLATOR_H
#define JEM_IO_INFLATOR_H

#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/io/ZipMode.h>
#include <jem/io/ZipException.h>
#include <jem/io/InputStream.h>

#include <zlib.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class Inflator
//-----------------------------------------------------------------------


class Inflator : public Collectable
{
 public:

                          Inflator

    ( Ref<InputStream>      input,
      ZipMode               zmode,
      idx_t                 bufsize );

  inline bool             atEnd         () const;
  inline ulint            getBytesIn    () const;
  inline ulint            getBytesOut   () const;
  inline ulint            getChecksum   () const;

  inline idx_t            poll

    ( const Time&           timeout );

  inline void             finish

    ( ThrowMode             tm = CAN_THROW );

  inline idx_t            inflate

    ( void*                 buf,
      idx_t                 n );

  inline bool             read

    ( byte&                 b );


 protected:

  virtual                ~Inflator      ();


 private:

  void                    read_         ();


 private:

  Ref<InputStream>        input_;
  z_stream                zstream_;
  ulint                   crc_;
  bool                    eof_;
  bool                    zeof_;
  idx_t                   bufsize_;
  byte*                   buffer_;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef Inflator          ZlibInflator;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


inline bool Inflator::atEnd () const
{
  return zeof_;
}


//-----------------------------------------------------------------------
//   getBytesIn
//-----------------------------------------------------------------------


inline ulint Inflator::getBytesIn () const
{
  return (ulint) zstream_.total_in;
}


//-----------------------------------------------------------------------
//   getBytesOut
//-----------------------------------------------------------------------


inline ulint Inflator::getBytesOut () const
{
  return (ulint) zstream_.total_out;
}


//-----------------------------------------------------------------------
//   getChecksum
//-----------------------------------------------------------------------


inline ulint Inflator::getChecksum () const
{
  return crc_;
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


inline idx_t Inflator::poll ( const Time& timeout )
{
  if ( zstream_.avail_in > 0 )
  {
    return (idx_t) zstream_.avail_in;
  }
  else
  {
    return input_->poll ( timeout );
  }
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


inline void Inflator::finish ( ThrowMode tm )
{
  zstream_.next_in   = 0;
  zstream_.avail_in  = 0;
  zstream_.next_out  = 0;
  zstream_.avail_out = 0;
}


//-----------------------------------------------------------------------
//   inflate
//-----------------------------------------------------------------------


inline idx_t Inflator::inflate ( void* buf, idx_t n )
{
  int    result;
  idx_t  k;


  if ( zeof_ || n == 0 )
  {
    return 0;
  }

  zstream_.next_out  = (Bytef*) buf;
  zstream_.avail_out = (uInt)
    jem::min ( (idx_t) jem::maxOf<int>(), n );

  while ( true )
  {
    if ( ! zstream_.avail_in )
    {
      read_ ();
    }

    result = ::inflate ( &zstream_, Z_SYNC_FLUSH );

    if ( result == Z_STREAM_END )
    {
      zeof_ = true;
      break;
    }

    if ( result != Z_OK )
    {
      zstream_.next_out  = 0;
      zstream_.avail_out = 0;

      if ( result == Z_BUF_ERROR )
      {
        throw ZipException (
          JEM_FUNC,
          "unzip error: unexpected end of input stream"
        );
      }
      else
      {
        throw ZipException (
          JEM_FUNC,
          String::format ( "unzip error: %s", zstream_.msg )
        );
      }
    }

    if ( ! zstream_.avail_out )
    {
      break;
    }
  }

  k    = n - (idx_t) zstream_.avail_out;
  crc_ = (ulint) ::crc32 ( (uLong) crc_, (Bytef*) buf, (uInt) k );

  return k;
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


inline bool Inflator::read ( byte& b )
{
  if ( ! zstream_.avail_in )
  {
    read_ ();

    if ( eof_ )
    {
      return false;
    }
  }

  b = (byte) *zstream_.next_in++;

  zstream_.avail_in--;

  return true;
}


JEM_END_PACKAGE( io )

#endif
