
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
#include <jem/io/params.h>
#include <jem/io/BufferedInputStream.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class BufferedInputStream
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const byte  BufferedInputStream::CANARY_VALUE_ = 123;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BufferedInputStream::BufferedInputStream

  ( Ref<InputStream>  in,
    idx_t             bufsize ) :

    Super ( in )

{
  if ( bufsize <= 0_idx )
  {
    bufsize = DEFAULT_BUFSIZE;
  }

  init_ ( bufsize );
}


BufferedInputStream::BufferedInputStream ( const Self& rhs ) :

  Super ( rhs )

{
  init_ ( rhs.bufsize_ );
}


BufferedInputStream::~BufferedInputStream ()
{
  MemCache::dealloc ( buffer_, (size_t) bufsize_ + 1 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<InputStream> BufferedInputStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t BufferedInputStream::poll ( const Time& timeout )
{
  if ( first_ < last_ )
  {
    return (last_ - first_);
  }
  else
  {
    return input_->poll ( timeout );
  }
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t BufferedInputStream::read ( void* dest, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  if ( first_ == last_ )
  {
    if ( buffer_[bufsize_] != CANARY_VALUE_ )
    {
      throw MemoryError ( JEM_FUNC );
    }

    if ( n > bufsize_ )
    {
      return input_->read ( dest, n );
    }
    else
    {
      first_ = last_ = 0;
      last_  = input_->read ( buffer_, bufsize_ );
    }
  }

  n = min     ( last_ - first_, n );
  std::memcpy ( dest, buffer_ + first_, (size_t) n );

  first_ += n;

  return n;
}


//-----------------------------------------------------------------------
//   skip
//-----------------------------------------------------------------------


idx_t BufferedInputStream::skip ( idx_t n )
{
  idx_t  k;

  if ( n <= last_ - first_ )
  {
    k       = n;
    first_ += n;
  }
  else
  {
    k      = last_ - first_;
    first_ = last_;
    k     += input_->skip ( n - k );
  }

  return k;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void BufferedInputStream::init_ ( idx_t bufsize )
{
  bufsize_ = bufsize;
  buffer_  = (byte*) MemCache::alloc ( (size_t) bufsize_ + 1 );
  first_   = 0_idx;
  last_    = 0_idx;

  buffer_[bufsize_] = CANARY_VALUE_;
}


JEM_END_PACKAGE( io )
