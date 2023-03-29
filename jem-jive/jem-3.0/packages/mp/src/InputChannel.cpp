
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
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/base/MemoryError.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/OutOfMemoryException.h>
#include <jem/io/IOException.h>
#include <jem/io/OutputStream.h>
#include <jem/mp/Status.h>
#include <jem/mp/ChannelConstants.h>
#include <jem/mp/InputChannel.h>


JEM_DEFINE_CLASS( jem::mp::InputChannel );


JEM_BEGIN_PACKAGE( mp )


using jem::io::IOException;


//=======================================================================
//   class InputChannel
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


InputChannel::InputChannel ( idx_t bufsize )
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
  first_   = 0;
  last_    = 0;
  isOpen_  = false;

  buffer_[capacity_] = ChannelConstants::CANARY_VALUE;
}


InputChannel::~InputChannel ()
{
  MemCache::dealloc ( buffer_, (size_t) capacity_ + 1 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   open
//-----------------------------------------------------------------------


void InputChannel::open ( const Port& p )
{
  idx_t  n;

  if ( isOpen_ )
  {
    close ();
  }

  port_ = p;

  port_.recv ( n );
  port_.send ( capacity_ );

  bufsize_ = jem::min ( capacity_, n );
  isOpen_  = true;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void InputChannel::close ()
{
  if ( isOpen_ )
  {
    const idx_t  n = bufsize_;

    bufsize_ = 0;
    first_   = 0;
    last_    = 0;
    isOpen_  = false;

    port_.recv ( buffer_, n );
    port_.send ( ChannelConstants::CLOSE );
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t InputChannel::poll ( const Time& timeout )
{
  return (last_ - first_);
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t InputChannel::read ( void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  byte*  dest = (byte*) buf;
  idx_t  k    = 0;
  idx_t  m;


  if ( first_ < last_ )
  {
    k = jem::min ( n, last_ - first_ );
    std::memcpy  ( dest, buffer_ + first_, (size_t) k );

    first_ += k;
    dest   += k;
  }

  while ( isOpen_ && k < n )
  {
    if ( first_ == last_ )
    {
      fillBuffer_ ();
    }

    m = jem::min ( n - k, last_ - first_ );
    std::memcpy  ( dest,  buffer_ + first_, (size_t) m );

    k      += m;
    first_ += m;
    dest   += m;
  }

  return k;
}


//-----------------------------------------------------------------------
//   pipeTo
//-----------------------------------------------------------------------


void InputChannel::pipeTo ( io::OutputStream& out )
{
  idx_t  k;

  while ( isOpen_ )
  {
    if ( first_ == last_ )
    {
      fillBuffer_ ();
    }

    k      = first_;
    first_ = last_;

    out.write ( buffer_ + k, last_ - k );
  }
}


//-----------------------------------------------------------------------
//   allocBuffer_
//-----------------------------------------------------------------------


void InputChannel::allocBuffer_ ()
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
//   fillBuffer_
//-----------------------------------------------------------------------


void InputChannel::fillBuffer_ ()
{
  Status  status;

  if ( buffer_[capacity_] != ChannelConstants::CANARY_VALUE )
  {
    throw MemoryError ( JEM_FUNC );
  }

  first_ = last_ = 0;

  port_.recv ( buffer_, bufsize_, & status );
  port_.send ( ChannelConstants::OK );

  last_  = status.size;
  first_ = 1_idx;

  if ( buffer_[0] == ChannelConstants::CLOSE )
  {
    isOpen_ = false;
  }
}


JEM_END_PACKAGE( mp )
