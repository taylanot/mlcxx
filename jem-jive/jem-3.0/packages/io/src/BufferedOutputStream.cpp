
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
#include <jem/base/assert.h>
#include <jem/base/MemCache.h>
#include <jem/base/MemoryError.h>
#include <jem/io/params.h>
#include <jem/io/BufferedOutputStream.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class BufferedOutputStream
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const byte  BufferedOutputStream::CANARY_VALUE_ = 123;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BufferedOutputStream::BufferedOutputStream

  ( Ref<OutputStream>  out,
    idx_t              bufsize ) :

    Super ( out )

{
  if ( bufsize <= 0_idx )
  {
    bufsize = DEFAULT_BUFSIZE;
  }

  init_ ( bufsize );
}


BufferedOutputStream::BufferedOutputStream ( const Self& rhs ) :

  Super ( rhs )

{
  init_ ( rhs.bufsize_ );
}


BufferedOutputStream::~BufferedOutputStream ()
{
  if ( last_ > 0_idx )
  {
    output_->writeNoThrow ( buffer_, last_ );
  }

  MemCache::dealloc ( buffer_, (size_t) bufsize_ + 1 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<OutputStream> BufferedOutputStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void BufferedOutputStream::close ()
{
  flushBuffer    ();
  output_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void BufferedOutputStream::flush ()
{
  flushBuffer    ();
  output_->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void BufferedOutputStream::write ( const void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  const byte*  src = (const byte*) buf;
  const idx_t  m   = bufsize_ - last_;

  if ( n <= m )
  {
    std::memcpy ( buffer_ + last_, src, (size_t) n );
    last_ += n;
  }
  else
  {
    std::memcpy ( buffer_ + last_, src, (size_t) m );

    last_ += m;
    src   += m;
    n     -= m;

    flushBuffer ();

    if ( n > bufsize_ )
    {
      output_->write ( src, n );
    }
    else
    {
      std::memcpy ( buffer_, src, (size_t) n );
      last_ = n;
    }
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void BufferedOutputStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  if ( last_ > 0_idx )
  {
    output_->writeNoThrow ( buffer_, last_ );
    last_ = 0_idx;
  }

  if ( n > 0_idx )
  {
    output_->writeNoThrow ( buf, n );
  }
}


//-----------------------------------------------------------------------
//   flushBuffer
//-----------------------------------------------------------------------


void BufferedOutputStream::flushBuffer ()
{
  idx_t  n = last_;

  if ( buffer_[bufsize_] != CANARY_VALUE_ )
  {
    throw MemoryError ( JEM_FUNC );
  }

  last_ = 0_idx;

  output_->write ( buffer_, n );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void BufferedOutputStream::init_ ( idx_t bufsize )
{
  bufsize_ = bufsize;
  buffer_  = (byte*) MemCache::alloc ( (size_t) bufsize_ + 1 );
  last_    = 0_idx;

  buffer_[bufsize_] = CANARY_VALUE_;
}


JEM_END_PACKAGE( io )
