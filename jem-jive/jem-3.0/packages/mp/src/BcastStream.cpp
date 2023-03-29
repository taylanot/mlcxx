
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
#include <jem/base/MemoryError.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/OutOfMemoryException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/FileOpenException.h>
#include <jem/mp/utilities.h>
#include <jem/mp/ChannelConstants.h>
#include <jem/mp/BcastStream.h>


JEM_DEFINE_CLASS( jem::mp::BcastStream );


JEM_BEGIN_PACKAGE( mp )


using jem::io::IOException;


//=======================================================================
//   class BcastStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


BcastStream::BcastStream

  ( int                  root,
    const Ref<Context>&  ctx,
    const Ref<Input>&    in,
    idx_t                bufsize )

{
  JEM_PRECHECK ( ctx != nullptr && root >= 0 && root < ctx->size() );

  int  minRoot;


  root_      = root;
  myRank_    = ctx->myRank ();
  procCount_ = ctx->size   ();
  context_   = ctx->clone  ();
  minRoot    = allmin ( *ctx, root_ );

  if ( minRoot != root )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "inconsistent root rank (%d) specified", root_
      )
    );
  }

  if ( myRank_ == root_ )
  {
    if ( ! in )
    {
      throw IllegalArgumentException (
        JEM_FUNC,
        String::format (
          "NIL input at the root process (rank = %d)", myRank_
        )
      );
    }

    input_ = in;
  }
  else
  {
    if ( in )
    {
      throw IllegalArgumentException (
        JEM_FUNC,
        String::format (
          "non-NIL input at a non-root process "
          "(rank = %d, root = %d)", myRank_, root_
        )
      );
    }
  }

  if ( bufsize <= 0 )
  {
    allocBuffer_ ( *ctx );
  }
  else
  {
    capacity_ = allmin ( *ctx, bufsize );
    buffer_   = (byte*) MemCache::alloc ( (size_t) capacity_ + 1 );
  }

  first_  = 0;
  last_   = 0;
  isOpen_ = true;

  buffer_[capacity_] = ChannelConstants::CANARY_VALUE;
}


BcastStream::~BcastStream ()
{
  MemCache::dealloc ( buffer_, (size_t) capacity_ + 1 );

  buffer_ = nullptr;
}


//-----------------------------------------------------------------------
//   open
//-----------------------------------------------------------------------


Ref<BcastStream> BcastStream::open

  ( int                  root,
    const Ref<Context>&  ctx,
    const String&        fname,
    OpenFlags            flags,
    idx_t                bufsize )

{
  JEM_PRECHECK ( ctx );

  Ref<Input>  fin;
  int         ok = 0;


  if ( ctx->myRank() == root )
  {
    try
    {
      fin = newInstance<io::FileInputStream> ( fname, flags );
    }
    catch ( const IOException& )
    {
      ctx->broadcast ( SendBuffer( &ok, 1 ) );

      throw;
    }

    ok = 1;

    ctx->broadcast ( SendBuffer( &ok, 1 ) );
  }
  else
  {
    ctx->broadcast ( RecvBuffer( &ok, 1 ), root );

    if ( ! ok )
    {
      throw io::FileOpenException (
        JEM_FUNC,
        String::format ( "failed to open the file `%s\'", fname )
      );
    }
  }

  return newInstance<Self> ( root, ctx, fin, bufsize );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void BcastStream::close ()
{
  first_  = 0;
  last_   = 0;
  isOpen_ = false;

  if ( input_ )
  {
    input_->close ();
    input_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t BcastStream::poll ( const Time& timeout )
{
  return (last_ - first_);
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t BcastStream::read ( void* buf, idx_t n )
{
  if ( first_ == last_ )
  {
    fillBuffer_ ();
  }

  n = jem::min ( n,   last_   - first_ );
  std::memcpy  ( buf, buffer_ + first_, (size_t) n );

  first_ += n;

  return n;
}


//-----------------------------------------------------------------------
//   allocBuffer_
//-----------------------------------------------------------------------


void BcastStream::allocBuffer_ ( Context& ctx )
{
  idx_t  size = getStreamBufsize ( ctx );

  buffer_ = nullptr;

  do
  {
    capacity_ = allmin ( ctx, size );

    try
    {
      buffer_ = (byte*) MemCache::alloc ( (size_t) capacity_ + 1 );
    }
    catch ( const OutOfMemoryException& )
    {
      if ( size < ChannelConstants::TINY_BUFSIZE )
      {
        throw;
      }

      // Try again with a smaller buffer size.

      size   /= 2;
      buffer_ = nullptr;
    }
  }
  while ( ! buffer_ );
}


//-----------------------------------------------------------------------
//   fillBuffer_
//-----------------------------------------------------------------------


void BcastStream::fillBuffer_ ()
{
  idx_t  stat[2];


  if ( buffer_[capacity_] != ChannelConstants::CANARY_VALUE )
  {
    throw MemoryError ( JEM_FUNC );
  }

  first_ = last_ = 0;

  if ( ! isOpen_ )
  {
    return;
  }

  stat[0] = 1_idx;
  stat[1] = 0;

  if ( myRank_ == root_ )
  {
    try
    {
      stat[1] = input_->read ( buffer_, capacity_ );
    }
    catch ( const IOException& )
    {
      stat[0] = 0;
    }

    if ( procCount_ > 1 )
    {
      context_->broadcast ( SendBuffer( stat, 2 ) );

      if ( stat[0] )
      {
        context_->broadcast ( SendBuffer( buffer_, stat[1] ) );
      }
    }
  }
  else
  {
    context_->broadcast ( RecvBuffer( stat, 2 ), root_ );

    if ( stat[0] )
    {
      context_->broadcast ( RecvBuffer( buffer_, stat[1] ), root_ );
    }
  }

  if ( ! stat[0] )
  {
    throw IOException ( JEM_FUNC, "input error" );
  }

  last_ = stat[1];
}


JEM_END_PACKAGE( mp )
