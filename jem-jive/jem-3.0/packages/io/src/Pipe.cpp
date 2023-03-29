
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
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/base/MemoryError.h>
#include <jem/io/IOException.h>
#include <jem/io/CancelledIOException.h>
#include <jem/io/Pipe.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class Pipe
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Pipe::Pipe ( idx_t bufsize )
{
  JEM_PRECHECK ( bufsize > 0 );

  readerCount_     = 0;
  writerCount_     = 0;
  buffer_          = (byte*) MemCache::alloc ( (size_t) bufsize + 1 );
  buffer_[bufsize] = CANARY_VALUE_;
  bufsize_         = bufsize;
  first_           = 0;
  last_            = 0;
  cycle_           = true;
}


Pipe::~Pipe ()
{
  MemCache::dealloc ( buffer_ , (size_t) bufsize_ + 1 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   addReader
//-----------------------------------------------------------------------


void Pipe::addReader ()
{
  Lock<Mutex>  lock ( mutex_ );

  readerCount_++;
}


//-----------------------------------------------------------------------
//   addWriter
//-----------------------------------------------------------------------


void Pipe::addWriter ()
{
  Lock<Mutex>  lock ( mutex_ );

  writerCount_++;
}


//-----------------------------------------------------------------------
//   deleteReader
//-----------------------------------------------------------------------


void Pipe::deleteReader ()
{
  Lock<Mutex>  lock ( mutex_ );

  readerCount_--;

  if ( readerCount_ == 0 )
  {
    writers_.notifyAll ( mutex_ );
  }
}


//-----------------------------------------------------------------------
//   deleteWriter
//-----------------------------------------------------------------------


void Pipe::deleteWriter ()
{
  Lock<Mutex>  lock ( mutex_ );

  writerCount_--;

  if ( writerCount_ == 0 )
  {
    readers_.notifyAll ( mutex_ );
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t Pipe::poll ( const Time& timeout )
{
  Lock<Mutex>  lock ( mutex_ );
  idx_t        n;

  if ( cycle_ )
  {
    n = last_ - first_;
  }
  else
  {
    n = bufsize_ - first_ + last_;
  }

  if ( n > 0 )
  {
    return n;
  }

  if ( writerCount_ == 0 )
  {
    return 0;
  }

  readers_.wait ( mutex_, timeout );

  if ( cycle_ )
  {
    n = last_ - first_;
  }
  else
  {
    n = bufsize_ - first_ + last_;
  }

  return n;
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t Pipe::read ( void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  Lock<Mutex>  lock ( mutex_ );

  byte*        dest;
  idx_t        k, m;


  if ( buffer_[bufsize_] != CANARY_VALUE_ )
  {
    throw MemoryError ( JEM_FUNC );
  }

  dest = (byte*) buf;
  k    = 0;

  try
  {
    while ( k < n )
    {
      if ( cycle_ )
      {
        m = min( n - k, last_ - first_ );
      }
      else
      {
        m = min( n - k, bufsize_ - first_ );
      }

      if ( m > 0 )
      {
        std::memcpy ( dest, ( buffer_ + first_ ), (size_t) m );

        dest   += m;
        k      += m;
        first_ += m;

        if ( first_ == bufsize_ )
        {
          first_ = 0;
          cycle_ = ! cycle_;
        }
      }
      else if ( writerCount_ ==  0 )
      {
        // End of stream condition

        break;
      }
      else
      {
        // Wake up all waiting writers before we go to sleep or
        // before this thread is cancelled.

        writers_.notifyAll ( mutex_ );
        readers_.wait      ( mutex_ );
      }
    }
  }
  catch ( const CancelledException& )
  {
    throw CancelledIOException ( k, JEM_FUNC, "read cancelled" );
  }

  writers_.notifyAll ( mutex_ );

  return k;
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void Pipe::write ( const void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  Lock<Mutex>  lock ( mutex_ );

  const byte*  src;
  idx_t        k, m;


  if ( buffer_[bufsize_] != CANARY_VALUE_ )
  {
    throw MemoryError ( JEM_FUNC );
  }

  src = (const byte*) buf;
  k   = 0;

  try
  {
    while ( k < n )
    {
      if ( cycle_ )
      {
        m = min( n - k, bufsize_ - last_ );
      }
      else
      {
        m = min( n - k, first_ - last_ );
      }

      if ( m > 0 )
      {
        std::memcpy( ( buffer_ + last_ ), src, (size_t) m );

        src   += m;
        k     += m;
        last_ += m;

        if ( last_ == bufsize_ )
        {
          last_  = 0;
          cycle_ = ! cycle_;
        }
      }
      else if ( readerCount_ == 0 )
      {
        throw IOException ( JEM_FUNC, "broken pipe" );
      }
      else
      {
        // Wake up all waiting readers before we go to sleep
        // or before this thread is cancelled.

        readers_.notifyAll ( mutex_ );
        writers_.wait      ( mutex_ );
      }
    }
  }
  catch ( const CancelledException& )
  {
    throw CancelledIOException ( k, JEM_FUNC, "write cancelled" );
  }

  readers_.notifyAll ( mutex_ );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void Pipe::writeNoThrow ( const void* buf, idx_t n ) noexcept
{
  Lock<Mutex>  lock ( mutex_ );

  const byte*  src;
  idx_t        k, m;


  src = (const byte*) buf;
  k   = 0;

  while ( k < n )
  {
    if ( cycle_ )
    {
      m = min( n - k, bufsize_ - last_ );
    }
    else
    {
      m = min( n - k, first_ - last_ );
    }

    if ( m > 0 )
    {
      std::memcpy( ( buffer_ + last_ ), src, (size_t) m );

      src   += m;
      k     += m;
      last_ += m;

      if ( last_ == bufsize_ )
      {
        last_  = 0;
        cycle_ = ! cycle_;
      }
    }
    else if ( readerCount_ == 0 )
    {
      return;
    }
    else
    {
      readers_.notifyAll    ( mutex_ );
      writers_.waitNoCancel ( mutex_ );
    }
  }

  readers_.notifyAll ( mutex_ );
}


JEM_END_PACKAGE( io )
