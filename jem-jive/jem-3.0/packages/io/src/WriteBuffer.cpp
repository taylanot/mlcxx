
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


#include <jem/base/assert.h>
#include <jem/base/MemCache.h>
#include <jem/base/MemoryError.h>
#include <jem/io/params.h>
#include <jem/io/WriteBuffer.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class WriteBuffer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char  WriteBuffer::CANARY_VALUE_ = 123;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


WriteBuffer::WriteBuffer ( idx_t sz )
{
  if ( sz <= 0 )
  {
    sz = DEFAULT_BUFSIZE;
  }

  buffer_     = (char*) MemCache::alloc ( (size_t) sz + 1 );
  buffer_[sz] = CANARY_VALUE_;
  bufsize_    = sz;
  last_       = 0;
}


WriteBuffer::~WriteBuffer ()
{
  MemCache::dealloc ( buffer_, (size_t) bufsize_ + 1 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void WriteBuffer::reset ()
{
  last_ = 0;
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void WriteBuffer::flush ( Writer& out )
{
  idx_t  n = last_;

  if ( buffer_[bufsize_] != CANARY_VALUE_ )
  {
    throw MemoryError ( JEM_FUNC );
  }

  last_ = 0;

  out.write ( buffer_, n );
}


//-----------------------------------------------------------------------
//   flushNoThrow
//-----------------------------------------------------------------------


void WriteBuffer::flushNoThrow ( Writer& out ) noexcept
{
  idx_t  n = last_;

  last_ = 0;

  out.writeNoThrow ( buffer_, n );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void WriteBuffer::write ( Writer& out, int c, idx_t n )
{
  while ( n > 0 )
  {
    if ( last_ == bufsize_ )
    {
      flush ( out );
    }

    buffer_[last_++] = (char) c;
    n--;
  }
}


void WriteBuffer::write

  ( Writer&      out,
    const char*  src,
    idx_t        n )

{
  JEM_PRECHECK ( n >= 0 );

  idx_t  k = bufsize_ - last_;

  if ( n <= k )
  {
    std::memcpy ( buffer_ + last_, src, (size_t) n );
    last_ += n;
  }
  else
  {
    std::memcpy ( buffer_ + last_, src, (size_t) k );

    last_ += k;

    flush ( out );

    src += k;
    n   -= k;

    if ( n > bufsize_ )
    {
      out.write ( src, n );
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


void WriteBuffer::writeNoThrow

  ( Writer&      out,
    const char*  src,
    idx_t        n ) noexcept

{
  if ( last_ > 0 )
  {
    out.writeNoThrow ( buffer_, last_ );
    last_ = 0;
  }

  if ( n > 0 )
  {
    out.writeNoThrow ( src, n );
  }
}


JEM_END_PACKAGE( io )
