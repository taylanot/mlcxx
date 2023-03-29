
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
#include <jem/base/MemCache.h>
#include <jem/base/MemoryError.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/OutOfMemoryException.h>
#include <jem/io/IOException.h>
#include <jem/mp/ChannelConstants.h>
#include <jem/mp/OutputChannel.h>


JEM_DEFINE_CLASS ( jem::mp::OutputChannel );


JEM_BEGIN_PACKAGE( mp )


using jem::io::IOException;


//=======================================================================
//   class OutputChannel
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


OutputChannel::OutputChannel ( idx_t bufsize )
{
  if ( bufsize <= 0 )
  {
    allocBuffer_ ();
  }
  else
  {
    capacity_ = jem::max ( 2_idx, bufsize );
    buffer_   = (byte*) MemCache::alloc ( (size_t) capacity_ + 1 );
  }

  bufsize_ = 0;
  last_    = 0;
  isOpen_  = false;

  buffer_[capacity_] = ChannelConstants::CANARY_VALUE;
}


OutputChannel::~OutputChannel ()
{
  MemCache::dealloc ( buffer_, (size_t) capacity_ + 1 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   open
//-----------------------------------------------------------------------


void OutputChannel::open ( const Port& p )
{
  idx_t  n;

  if ( isOpen_ )
  {
    close ();
  }

  port_ = p;

  port_.send ( capacity_ );
  port_.recv ( n );

  bufsize_   = jem::min ( capacity_, n );
  buffer_[0] = ChannelConstants::OK;
  last_      = 1;
  isOpen_    = true;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void OutputChannel::close ()
{
  if ( isOpen_ )
  {
    buffer_[0] = ChannelConstants::CLOSE;

    flushBuffer_ ();

    bufsize_ = 0;
    last_    = 0;
    isOpen_  = false;
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t OutputChannel::poll ()
{
  return (bufsize_ - last_);
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void OutputChannel::flush ()
{
  if ( isOpen_ && last_ > 1 )
  {
    flushBuffer_ ();
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void OutputChannel::write ( const void* buf, idx_t n )
{
  const byte*  src = (const byte*) buf;
  idx_t        k   = 0;
  idx_t        m;


  while ( isOpen_ && k < n )
  {
    if ( last_ == bufsize_ )
    {
      flushBuffer_ ();
    }

    m = jem::min ( n - k, bufsize_ - last_ );
    std::memcpy  ( buffer_ + last_, src, (size_t) m );

    k     += m;
    last_ += m;
    src   += m;
  }

  if ( k != n )
  {
    throw IOException (
      JEM_FUNC,
      "error sending data: broken connection"
    );
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void OutputChannel::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  const byte*  src = (const byte*) buf;
  idx_t        k   = 0;
  idx_t        m;


  while ( isOpen_ && k < n )
  {
    if ( last_ == bufsize_ )
    {
      flushBuffer_ ( false );
    }

    m = jem::min ( n - k, bufsize_ - last_ );
    std::memcpy  ( buffer_ + last_, src, (size_t) m );

    k     += m;
    last_ += m;
    src   += m;
  }
}


//-----------------------------------------------------------------------
//   allocBuffer_
//-----------------------------------------------------------------------


void OutputChannel::allocBuffer_ ()
{
  buffer_   = nullptr;
  capacity_ = ChannelConstants::LARGE_BUFSIZE;

  do
  {
    try
    {
      buffer_ = (byte*) MemCache::alloc ( (size_t) capacity_ + 1 );
    }
    catch ( const OutOfMemoryException& )
    {
      if ( capacity_ < ChannelConstants::TINY_BUFSIZE )
      {
        throw;
      }

      buffer_    = nullptr;
      capacity_ /= 2_idx;
    }
  }
  while ( ! buffer_ );
}


//-----------------------------------------------------------------------
//   flushBuffer_
//-----------------------------------------------------------------------


void OutputChannel::flushBuffer_ ( bool throwOnError )
{
  byte  response;

  if ( throwOnError &&
       buffer_[capacity_] != ChannelConstants::CANARY_VALUE )
  {
    throw MemoryError ( JEM_FUNC );
  }

  port_.send ( buffer_, last_ );
  port_.recv ( response );

  if ( response == ChannelConstants::CLOSE )
  {
    bufsize_ = 0;
    last_    = 0;
    isOpen_  = false;
  }
  else
  {
    buffer_[0] = ChannelConstants::OK;
    last_      = 1_idx;
  }
}


JEM_END_PACKAGE( mp )
