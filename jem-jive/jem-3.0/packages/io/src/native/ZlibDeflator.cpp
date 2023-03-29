
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


#include <cstring>
#include <jem/base/MemCache.h>
#include "native/ZlibDeflator.h"


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class Deflator
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Deflator::Deflator

  ( Ref<OutputStream>  out,
    ZipMode            zmode,
    int                level,
    idx_t              bufsize ) :

    output_ ( out )

{
  int  result;


  if ( level < 0 )
  {
    level = Z_DEFAULT_COMPRESSION;
  }

  std::memset ( &zstream_, 0x0, sizeof(z_stream) );

  zstream_.next_in   = 0;
  zstream_.avail_in  = 0;
  zstream_.next_out  = 0;
  zstream_.avail_out = 0;
  zstream_.zalloc    = Z_NULL;
  zstream_.zfree     = Z_NULL;
  zstream_.opaque    = 0;

  if ( zmode == RAW_ZLIB_MODE )
  {
    // This gets rid of the zlib header

    result = deflateInit2 ( &zstream_, level, Z_DEFLATED,
                            -15,       8,     Z_DEFAULT_STRATEGY );
  }
  else
  {
    result = deflateInit  ( &zstream_, level );
  }

  if ( result != Z_OK )
  {
    throw ZipException (
      JEM_FUNC,
      String::format ( "zip error: %s", zstream_.msg )
    );
  }

  crc_     = (ulint) ::crc32  ( 0UL,  Z_NULL, 0 );
  bufsize_ = jem::max ( 1024_idx, bufsize );
  buffer_  = (byte*) MemCache::alloc ( (size_t) bufsize_ );

  reset_ ();
}


Deflator::~Deflator ()
{
  deflateEnd ( &zstream_ );

  MemCache::dealloc ( buffer_, (size_t) bufsize_ );

  buffer_ = 0;
}


//-----------------------------------------------------------------------
//   reset_
//-----------------------------------------------------------------------


void Deflator::reset_ ()
{
  zstream_.next_in   = 0;
  zstream_.avail_in  = 0;
  zstream_.next_out  = (Bytef*) buffer_;
  zstream_.avail_out = (uInt)   bufsize_;
  synced_            = true;
}


//-----------------------------------------------------------------------
//   zipError_
//-----------------------------------------------------------------------


void Deflator::zipError_ ( ThrowMode tm )
{
  reset_ ();

  if ( tm == CAN_THROW )
  {
    throw ZipException (
      JEM_FUNC,
      String::format ( "zip error: %s", zstream_.msg )
    );
  }
}


JEM_END_PACKAGE( io )
