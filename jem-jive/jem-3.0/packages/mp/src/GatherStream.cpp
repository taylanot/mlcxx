
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/NullOutputStream.h>
#include <jem/io/FileOutputStream.h>
#include <jem/io/FileOpenException.h>
#include <jem/mp/utilities.h>
#include <jem/mp/Port.h>
#include <jem/mp/InputChannel.h>
#include <jem/mp/OutputChannel.h>
#include <jem/mp/ChannelConstants.h>
#include <jem/mp/GatherStream.h>


JEM_DEFINE_CLASS( jem::mp::GatherStream );


JEM_BEGIN_PACKAGE( mp )


//=======================================================================
//   class GatherStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructors
//-----------------------------------------------------------------------


GatherStream::GatherStream

  ( int                  root,
    const Ref<Context>&  ctx,
    Ref<OutputStream>    out,
    idx_t                bufsize )

{
  JEM_PRECHECK ( ctx  != nullptr && root >= 0 && root < ctx->size() );

  int  minRoot;

  root_    = root;
  myRank_  = ctx->myRank ();
  context_ = ctx->clone  ();
  minRoot  = allmin ( *ctx, root_ );

  if ( minRoot != root )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "inconsistent root rank specified: %d", root_
      )
    );
  }

  if ( bufsize <= 0 )
  {
    bufsize = getStreamBufsize ( *ctx );
  }

  if ( myRank_ == root_ )
  {
    if ( ! out )
    {
      throw IllegalArgumentException (
        JEM_FUNC,
        String::format (
          "NIL output at the root process (rank = %d)", myRank_
        )
      );
    }

    output_    = out;
    inChannel_ = newInstance<InputChannel> ( bufsize );
  }
  else
  {
    if ( out )
    {
      throw IllegalArgumentException (
        JEM_FUNC,
        String::format (
          "non-NIL output at a non-root process "
          " (rank = %d, root = %d)", myRank_, root_
        )
      );
    }

    bitBucket_  = newInstance<io::NullOutputStream> ();
    outChannel_ = newInstance<OutputChannel> ( bufsize );
    output_     = bitBucket_;
  }
}


GatherStream::~GatherStream ()
{}


//-----------------------------------------------------------------------
//   open
//-----------------------------------------------------------------------


Ref<GatherStream> GatherStream::open

  ( int                  root,
    const Ref<Context>&  ctx,
    const String&        fname,
    OpenFlags            flags,
    idx_t                bufsize )

{
  JEM_PRECHECK ( ctx );

  Ref<OutputStream>  fout;

  int  ok = 0;


  if ( ctx->myRank() == root )
  {
    try
    {
      fout = newInstance<io::FileOutputStream> ( fname, flags );
    }
    catch ( const io::IOException& )
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

  return newInstance<Self> ( root, ctx, fout, bufsize );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void GatherStream::close ()
{
  output_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void GatherStream::flush ()
{
  output_->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void GatherStream::write ( const void* buf, idx_t n )
{
  output_->write ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void GatherStream::writeNoThrow ( const void* buf, idx_t n ) noexcept
{
  output_->write ( buf, n );
}


//-----------------------------------------------------------------------
//   beginSession
//-----------------------------------------------------------------------


void GatherStream::beginSession ()
{
  if ( myRank_ != root_ )
  {
    outChannel_->open ( Port( context_, root_ ) );

    output_ = outChannel_;
  }
}


//-----------------------------------------------------------------------
//   endSession
//-----------------------------------------------------------------------


void GatherStream::endSession ()
{
  if ( myRank_ == root_ )
  {
    const int  procCount = context_->size ();

    for ( int i = 1; i < procCount; i++ )
    {
      inChannel_->open   ( Port( context_, i ) );
      inChannel_->pipeTo ( * output_ );
      inChannel_->close  ();
    }
  }
  else
  {
    outChannel_->close ();
    output_ = bitBucket_;
  }
}


JEM_END_PACKAGE( mp )
