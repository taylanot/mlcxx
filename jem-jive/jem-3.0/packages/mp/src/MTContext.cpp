
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
#include <jem/base/Once.h>
#include <jem/base/Thread.h>
#include <jem/base/MemCache.h>
#include <jem/base/RangeException.h>
#include <jem/base/thread/SigThread.h>
#include <jem/base/thread/ThreadLib.h>
#include <jem/mp/error.h>
#include <jem/mp/AbortException.h>
#include <jem/mp/MTContext.h>
#include "mt/error.h"
#include "mt/Arena.h"
#include "mt/Message.h"
#include "mt/Optable.h"
#include "mt/ReduceBuffer.h"
#include "mt/RecvRequest.h"
#include "mt/SendRequest.h"
#include "mt/RequestList.h"


JEM_BEGIN_PACKAGE( mp )


using jem::mp::mt::Arena;
using jem::mp::mt::Message;
using jem::mp::mt::Optable;
using jem::mp::mt::ReduceNode;
using jem::mp::mt::ReduceTree;
using jem::mp::mt::ReduceAction;
using jem::mp::mt::ReduceBuffer;


//=======================================================================
//   class MTContext
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MTContext::MTContext ( int n ) :

  myRank_ ( 0 )

{
  JEM_PRECHECK ( n > 0 );

  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, & initOnce_ );

  ThreadLib::init ();

  arena_  = newInstance<Arena> ( n );
  rstack_ = 0;
  rdepth_ = 0;
}


MTContext::MTContext

  ( int          rank,
    const Self&  rhs ) :

    myRank_ ( rank )

{
  JEM_PRECHECK ( rank >= 0 && rank < rhs.size() );

  arena_  = rhs.arena_;
  rstack_ = 0;
  rdepth_ = 0;
}


MTContext::MTContext

  ( int                rank,
    const Ref<Arena>&  arena ) :

    arena_  ( arena ),
    myRank_ ( rank  )

{
  rstack_ = 0;
  rdepth_ = 0;
}


MTContext::~MTContext ()
{
  if ( rstack_ )
  {
    MemCache::dealloc ( rstack_, (size_t) (rdepth_ + 1) *
                                 sizeof(void*) );

    rstack_ = 0;
  }
}


//-----------------------------------------------------------------------
//   getErrorString
//-----------------------------------------------------------------------


String MTContext::getErrorString ( int err ) const
{
  return mt::makeErrorString ( err );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


int MTContext::size () const
{
  return arena_->size;
}


//-----------------------------------------------------------------------
//   myRank
//-----------------------------------------------------------------------


int MTContext::myRank () const
{
  return myRank_;
}


//-----------------------------------------------------------------------
//   isShared
//-----------------------------------------------------------------------


bool MTContext::isShared () const
{
  return true;
}


//-----------------------------------------------------------------------
//   abort
//-----------------------------------------------------------------------


void MTContext::abort ( int err )
{
  throw AbortException ( JEM_FUNC, err );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Context> MTContext::clone ()
{
  Ref<Arena>  arena;

  if ( myRank_ == 0 )
  {
    arena         = newInstance<Arena> ( arena_->size );
    arena_->clone = arena;

    barrier ();
    barrier ();

    arena_->clone = nullptr;
  }
  else
  {
    barrier ();

    arena = arena_->clone;

    barrier ();
  }

  return newInstance<Self> ( myRank_, arena );
}


//-----------------------------------------------------------------------
//   newRequestList
//-----------------------------------------------------------------------


Ref<RequestList> MTContext::newRequestList ()
{
  return newInstance<mt::RequestList> ( myRank_, arena_ );
}


//-----------------------------------------------------------------------
//   send
//-----------------------------------------------------------------------


void MTContext::send

  ( const SendBuffer&  buf,
    int                dest,
    int                tag )

{
  if ( dest < 0 || dest >= arena_->size )
  {
    sendRankError ( JEM_FUNC, dest, arena_->size );
  }

  if ( tag < 0 )
  {
    sendTagError  ( JEM_FUNC, tag );
  }


  Message  msg ( buf, myRank_, tag );


  msg.status.error = 0;

  if ( arena_->mpools[dest]->queueSend( msg ) )
  {
    byte  sig = msg.wait ();

    if ( sig & SigThread::CANCEL )
    {
      arena_->mpools[dest]->cancel ( msg );
      throw CancelledException ( JEM_FUNC, "send cancelled" );
    }
  }

  if ( msg.status.error )
  {
    mt::raiseError ( JEM_FUNC, msg.status.error );
  }
}


//-----------------------------------------------------------------------
//   recv
//-----------------------------------------------------------------------


void MTContext::recv

  ( const RecvBuffer&  buf,
    int                src,
    int                tag,
    Status*            stat )

{
  if ( src >= arena_->size || (src < 0 && src != ANY_SOURCE) )
  {
    recvRankError ( JEM_FUNC, src, arena_->size );
  }

  if ( tag < 0 && tag != ANY_TAG )
  {
    recvTagError  ( JEM_FUNC, tag );
  }


  Message  msg ( buf, src, tag );
  bool     enqueued;


  msg.status.error = 0;

  if ( src == ANY_SOURCE )
  {
    enqueued = arena_->mpools[myRank_]->queueAnyRecv ( msg );
  }
  else
  {
    enqueued = arena_->mpools[myRank_]->queueRecv ( msg );
  }

  if ( enqueued )
  {
    byte  sig = msg.wait ();

    if ( sig & SigThread::CANCEL )
    {
      arena_->mpools[myRank_]->cancel ( msg );
      throw CancelledException ( JEM_FUNC, "recv cancelled" );
    }
  }

  if ( msg.status.error )
  {
    mt::raiseError ( JEM_FUNC, msg.status.error );
  }

  if ( stat )
  {
    *stat = msg.status;
  }
}


//-----------------------------------------------------------------------
//   initSend
//-----------------------------------------------------------------------


Ref<jem::mp::Request> MTContext::initSend

  ( const SendBuffer&  buf,
    int                dest,
    int                tag )

{
  if ( dest < 0 || dest >= arena_->size )
  {
    sendRankError ( JEM_FUNC, dest, arena_->size );
  }

  if ( tag < 0 )
  {
    sendTagError  ( JEM_FUNC, tag );
  }

  return newInstance<mt::SendRequest> ( arena_,  buf,
                                        myRank_, dest, tag );
}


//-----------------------------------------------------------------------
//   initRecv
//-----------------------------------------------------------------------


Ref<jem::mp::Request> MTContext::initRecv

  ( const RecvBuffer&  buf,
    int                src,
    int                tag )

{
  if ( src >= arena_->size || (src < 0 && src != ANY_SOURCE) )
  {
    recvRankError ( JEM_FUNC, src, arena_->size );
  }

  if ( tag < 0 && tag != ANY_TAG )
  {
    recvTagError  ( JEM_FUNC, tag );
  }

  return newInstance<mt::RecvRequest> ( arena_,  buf,
                                        myRank_, src, tag );
}


//-----------------------------------------------------------------------
//   barrier
//-----------------------------------------------------------------------


void MTContext::barrier ()
{
  const int    ACTION = static_cast<int> (ReduceAction::BARRIER);

  ReduceNode*  node;
  void**       stack;

  int          err = 0;
  int          top = 0;


  if ( ! rstack_ )
  {
    initStack_ ();
  }

  node  = arena_->firstTree->getNode ( myRank_ );
  stack = rstack_;

  while ( node )
  {
    node->monitor.lock ();

    if ( node->action )
    {
      err |= node->error;

      if ( node->action != ACTION )
      {
        err |= mt::REDUCE_ACTION_ERROR;
      }

      stack[top++] = node;
      node         = node->parent;
    }
    else
    {
      bool  cancelled;

      node->action = ACTION;
      node->error  = err;

      node->monitor.wait ( cancelled );

      err = node->error;

      if ( cancelled )
      {
        err         |= mt::CANCELLED_ERROR;
        node->action = static_cast<int> (ReduceAction::NONE);
      }

      node->monitor.unlock ();

      break;
    }
  }

  if ( ! node )
  {
    arena_->firstTree.swap ( arena_->secondTree );

    if ( Thread::cancelled() )
    {
      err |= mt::CANCELLED_ERROR;
    }
  }

  while ( top )
  {
    node = static_cast<ReduceNode*> ( stack[--top] );

#ifndef NDEBUG

    stack[top]   = 0;

#endif

    node->error  = err;
    node->action = static_cast<int> (ReduceAction::NONE);

    node->monitor.notify ();
    node->monitor.unlock ();
  }

  JEM_ASSERT_NOTHROW ( checkStack_() );

  if ( err )
  {
    mt::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   broadcast
//-----------------------------------------------------------------------


void MTContext::broadcast ( const SendBuffer& buf )
{
  if ( arena_->size > 1 )
  {
    broadcast_ ( ReduceBuffer( buf, myRank_ ), myRank_ );
  }
}


void MTContext::broadcast ( const RecvBuffer& buf, int root )
{
  if ( root < 0 || root >= arena_->size )
  {
    sendRankError ( JEM_FUNC, root, arena_->size );
  }

  if ( arena_->size > 1 )
  {
    broadcast_ ( ReduceBuffer( buf, myRank_ ), root );
  }
}


//-----------------------------------------------------------------------
//   reduce
//-----------------------------------------------------------------------


void MTContext::reduce

  ( const RecvBuffer&  in,
    const SendBuffer&  out,
    int                root,
    Opcode             opcode )

{
  if ( root < 0 || root >= arena_->size )
  {
    sendRankError ( JEM_FUNC, root, arena_->size );
  }

  const int            ACTION =

    static_cast<int> (ReduceAction::REDUCE);

  ReduceBuffer         myBuf  ( in, myRank_ );

  ReduceNode*          node;
  void**               stack;
  const ReduceBuffer*  rbuf;
  Optable::Handler     op;

  int                  err = 0;
  int                  top = 0;


  if ( ! rstack_ )
  {
    initStack_ ();
  }

  op = Optable::get ( opcode, myBuf.type() );

  if ( ! op )
  {
    err = mt::REDUCE_TYPE_ERROR;
  }

  err |= myBuf.check ( out );

  if ( ! err )
  {
    myBuf.copy ( out );
  }

  rbuf  = & myBuf;
  node  =   arena_->firstTree->getNode ( myRank_ );
  stack =   rstack_;

  while ( node )
  {
    node->monitor.lock ();

    if ( node->action )
    {
      err |= node->error;

      if ( node->action != ACTION )
      {
        err |= mt::REDUCE_ACTION_ERROR;
      }

      if ( err )
      {
        op = & Optable::dummyHandler;
      }

      if ( rbuf->rank == root )
      {
        err |= op ( *rbuf, *node->rbuf );
      }
      else
      {
        err |= op ( *node->rbuf, *rbuf );
        rbuf = node->rbuf;
      }

      stack[top++] = node;
      node         = node->parent;
    }
    else
    {
      bool  cancelled;

      node->rbuf   = rbuf;
      node->error  = err;
      node->action = ACTION;

      node->monitor.wait ( cancelled );

      err = node->error;

      if ( cancelled )
      {
        err         |= mt::CANCELLED_ERROR;
        node->action = static_cast<int> (ReduceAction::NONE);
        node->rbuf   = 0;
      }

      node->monitor.unlock ();

      break;
    }
  }

  if ( ! node )
  {
    arena_->firstTree.swap ( arena_->secondTree );

    if ( Thread::cancelled() )
    {
      err |= mt::CANCELLED_ERROR;
    }
  }

  while ( top )
  {
    node = static_cast<ReduceNode*> ( stack[--top] );

#ifndef NDEBUG

    stack[top]   = 0;

#endif

    node->error  = err;
    node->action = static_cast<int> (ReduceAction::NONE);
    node->rbuf   = 0;

    node->monitor.notify ();
    node->monitor.unlock ();
  }

  JEM_ASSERT_NOTHROW ( checkStack_() );

  if ( err )
  {
    mt::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   allreduce
//-----------------------------------------------------------------------


void MTContext::allreduce

  ( const RecvBuffer&  in,
    const SendBuffer&  out,
    Opcode             opcode )

{
  const int          ACTION =

    static_cast<int> (ReduceAction::ALLREDUCE);

  ReduceBuffer       myBuf  ( in, myRank_ );

  ReduceNode*        node;
  void**             stack;
  Optable::Handler   op;

  int                err = 0;
  int                top = 0;


  if ( ! rstack_ )
  {
    initStack_ ();
  }

  op = Optable::get ( opcode, myBuf.type() );

  if ( ! op )
  {
    err = mt::REDUCE_TYPE_ERROR;
  }

  err |= myBuf.check ( out );

  if ( ! err )
  {
    myBuf.copy ( out );
  }

  node  = arena_->firstTree->getNode ( myRank_ );
  stack = rstack_;

  while ( node )
  {
    node->monitor.lock ();

    if ( node->action )
    {
      err |= node->error;

      if ( node->action != ACTION )
      {
        err |= mt::REDUCE_ACTION_ERROR;
      }

      if ( err )
      {
        op = & Optable::dummyHandler;
      }

      err         |= op ( myBuf, *node->rbuf );
      stack[top++] = node;
      node         = node->parent;
    }
    else
    {
      bool  cancelled;

      node->rbuf   = & myBuf;
      node->error  =   err;
      node->action =   ACTION;

      node->monitor.wait ( cancelled );

      err = node->error;

      if ( cancelled )
      {
        err         |= mt::CANCELLED_ERROR;
        node->action = static_cast<int> (ReduceAction::NONE);
        node->rbuf   = 0;
      }

      node->monitor.unlock ();

      break;
    }
  }

  if ( ! node )
  {
    arena_->firstTree.swap ( arena_->secondTree );

    if ( Thread::cancelled() )
    {
      err |= mt::CANCELLED_ERROR;
    }
  }

  while ( top )
  {
    node = static_cast<ReduceNode*> ( stack[--top] );

#ifndef NDEBUG

    stack[top]   = 0;

#endif

    if ( ! err )
    {
      node->rbuf->copy ( myBuf );
    }

    node->error  = err;
    node->action = static_cast<int> (ReduceAction::NONE);
    node->rbuf   = 0;

    node->monitor.notify ();
    node->monitor.unlock ();
  }

  JEM_ASSERT_NOTHROW ( checkStack_() );

  if ( err )
  {
    mt::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   broadcast_
//-----------------------------------------------------------------------


void MTContext::broadcast_

  ( const ReduceBuffer&  myBuf,
    int                  root )

{
  const int            ACTION =

    static_cast<int> (ReduceAction::BCAST);

  ReduceNode*          node;
  void**               stack;
  const ReduceBuffer*  rbuf;

  int                  err = 0;
  int                  top = 0;


  if ( ! rstack_ )
  {
    initStack_ ();
  }

  rbuf  = & myBuf;
  node  =   arena_->firstTree->getNode ( myBuf.rank );
  stack =   rstack_;

  while ( node )
  {
    node->monitor.lock ();

    if ( node->action )
    {
      err = node->error;
      err|= node->rbuf->check ( *rbuf );

      if ( node->action != ACTION )
      {
        err |= mt::REDUCE_ACTION_ERROR;
      }

      if ( node->rbuf->rank == root )
      {
        rbuf       = node->rbuf;
        node->rbuf = 0;

        if ( ! err )
        {
          myBuf.copy ( *rbuf );
        }
      }

      stack[top++] = node;
      node         = node->parent;
    }
    else
    {
      bool  cancelled;

      node->error  = err;
      node->action = ACTION;
      node->rbuf   = rbuf;

      node->monitor.wait ( cancelled );

      err = node->error;

      if ( cancelled )
      {
        err         |= mt::CANCELLED_ERROR;
        node->action = static_cast<int> (ReduceAction::NONE);
        node->rbuf   = 0;
      }

      node->monitor.unlock ();

      break;
    }
  }

  if ( ! node )
  {
    arena_->firstTree.swap ( arena_->secondTree );

    if ( Thread::cancelled() )
    {
      err |= mt::CANCELLED_ERROR;
    }
  }

  while ( top )
  {
    node = static_cast<ReduceNode*> ( stack[--top] );

#ifndef NDEBUG

    stack[top]   = 0;

#endif

    if ( ! err && node->rbuf )
    {
      node->rbuf->copy ( myBuf );
    }

    node->error  = err;
    node->action = static_cast<int> (ReduceAction::NONE);
    node->rbuf   = 0;

    node->monitor.notify ();
    node->monitor.unlock ();
  }

  JEM_ASSERT_NOTHROW ( checkStack_() );

  if ( err )
  {
    mt::raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   initStack_
//-----------------------------------------------------------------------


void MTContext::initStack_ ()
{
  JEM_PRECHECK ( ! rstack_ );

  rdepth_ = arena_->firstTree->depth ();
  rstack_ = (void**)

    MemCache::alloc ( (size_t) (rdepth_ + 1) * sizeof(void*) );

  for ( int i = 0; i <= rdepth_; i++ )
  {
    rstack_[i] = 0;
  }
}


//-----------------------------------------------------------------------
//   checkStack_
//-----------------------------------------------------------------------


inline bool MTContext::checkStack_ () const
{
  return (rstack_[rdepth_] == 0);
}


//-----------------------------------------------------------------------
//   initOnce_
//-----------------------------------------------------------------------


void MTContext::initOnce_ ()
{
  Optable::init ();
}


JEM_END_PACKAGE( mp )
