
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


#include <cstdlib>
#include <cstring>
#include <jem/mp/error.h>
#include <jem/mp/UniContext.h>
#include "uni/error.h"
#include "uni/MessagePool.h"
#include "uni/Request.h"
#include "uni/RequestList.h"


JEM_BEGIN_PACKAGE( mp )


//=======================================================================
//   class UniContext
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


UniContext::UniContext ()
{
  mpool_ = newInstance<uni::MessagePool> ();
}


UniContext::~UniContext ()
{}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


int UniContext::size () const
{
  return 1;
}


//-----------------------------------------------------------------------
//   myRank
//-----------------------------------------------------------------------


int UniContext::myRank () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   abort
//-----------------------------------------------------------------------


void UniContext::abort ( int err )
{
  std::exit ( err );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Context> UniContext::clone ()
{
  return newInstance<Self> ();
}


String UniContext::getErrorString ( int err ) const
{
  return uni::makeErrorString ( err );
}


//-----------------------------------------------------------------------
//   newRequestList
//-----------------------------------------------------------------------


Ref<RequestList> UniContext::newRequestList ()
{
  return newInstance<uni::RequestList> ( mpool_ );
}


//-----------------------------------------------------------------------
//   send
//-----------------------------------------------------------------------


void UniContext::send

  ( const SendBuffer&  buf,
    int                dest,
    int                tag )

{
  if ( dest != 0 )
  {
    sendRankError ( JEM_FUNC, dest, 1 );
  }

  if ( tag < 0 )
  {
    sendTagError ( JEM_FUNC, tag );
  }

  uni::Message  msg ( buf, tag );


  msg.status.error = 0;

  mpool_->queueSend ( msg );

  msg.wait ();

  if ( msg.status.error )
  {
    uni::raiseError ( JEM_FUNC, msg.status.error );
  }
}


//-----------------------------------------------------------------------
//   recv
//-----------------------------------------------------------------------


void UniContext::recv

  ( const RecvBuffer&  buf,
    int                src,
    int                tag,
    Status*            stat )

{
  if ( src != 0 && src != ANY_SOURCE )
  {
    recvRankError ( JEM_FUNC, src, 1 );
  }

  if ( tag < 0 && tag != ANY_TAG )
  {
    recvTagError ( JEM_FUNC, tag );
  }

  uni::Message  msg ( buf, tag );

  msg.status.error = 0;

  mpool_->queueRecv ( msg );

  msg.wait ();

  if ( stat )
  {
    *stat = msg.status;
  }

  if ( msg.status.error )
  {
    uni::raiseError ( JEM_FUNC, msg.status.error );
  }
}


//-----------------------------------------------------------------------
//   initSend
//-----------------------------------------------------------------------


Ref<Request> UniContext::initSend

  ( const SendBuffer&  buf,
    int                dest,
    int                tag )

{
  if ( dest != 0 )
  {
    sendRankError ( JEM_FUNC, dest, 1 );
  }

  if ( tag < 0 )
  {
    sendTagError ( JEM_FUNC, tag );
  }

  return newInstance<uni::Request> ( buf, tag, mpool_ );
}


//-----------------------------------------------------------------------
//   initRecv
//-----------------------------------------------------------------------


Ref<Request> UniContext::initRecv

  ( const RecvBuffer&  buf,
    int                src,
    int                tag )

{
  if ( src != 0 && src != ANY_SOURCE )
  {
    recvRankError ( JEM_FUNC, src, 1 );
  }

  if ( tag < 0 && tag != ANY_TAG )
  {
    recvTagError ( JEM_FUNC, tag );
  }

  return newInstance<uni::Request> ( buf, tag, mpool_ );
}


//-----------------------------------------------------------------------
//   barrier
//-----------------------------------------------------------------------


void UniContext::barrier ()
{}


//-----------------------------------------------------------------------
//   broadcast
//-----------------------------------------------------------------------


void UniContext::broadcast ( const SendBuffer& buf )
{}


void UniContext::broadcast

  ( const RecvBuffer&  buf,
    int                root )

{
  uni::raiseError ( JEM_FUNC, uni::DEADLOCK_ERROR );
}


//-----------------------------------------------------------------------
//   reduce
//-----------------------------------------------------------------------


void UniContext::reduce

  ( const RecvBuffer&  rbuf,
    const SendBuffer&  sbuf,
    int                root,
    Opcode             op )

{
  if ( root != 0 )
  {
    sendRankError ( JEM_FUNC, root, 1 );
  }

  if ( rbuf.size() != sbuf.size() )
  {
    uni::raiseError ( JEM_FUNC, uni::BUFFER_SIZE_ERROR );
  }

  if ( rbuf.type() != sbuf.type() )
  {
    uni::raiseError ( JEM_FUNC, uni::BUFFER_TYPE_ERROR );
  }

  std::memcpy (
    rbuf.addr (),
    sbuf.addr (),
    (size_t) (sizeOf(sbuf.type()) * sbuf.size())
  );
}


//-----------------------------------------------------------------------
//   allreduce
//-----------------------------------------------------------------------


void  UniContext::allreduce

    ( const RecvBuffer&  rbuf,
      const SendBuffer&  sbuf,
      Opcode             op )

{
  if ( rbuf.size() != sbuf.size() )
  {
    uni::raiseError ( JEM_FUNC, uni::BUFFER_SIZE_ERROR );
  }

  if ( rbuf.type() != sbuf.type() )
  {
    uni::raiseError ( JEM_FUNC, uni::BUFFER_TYPE_ERROR );
  }

  std::memcpy (
    rbuf.addr (),
    sbuf.addr (),
    (size_t) (sizeOf(sbuf.type()) * sbuf.size())
  );
}


JEM_END_PACKAGE( mp )
