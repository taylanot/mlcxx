
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
#include <jem/base/CancelledException.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/base/thread/spin.h>
#include <jem/base/thread/SigThread.h>
#include <jem/base/thread/ThreadLib.h>
#include <jem/mp/error.h>
#include "Arena.h"
#include "Rendezvous.h"
#include "RecvRequest.h"
#include "SendRequest.h"
#include "RequestList.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


//=======================================================================
//   class RequestList::Utils_
//=======================================================================


class RequestList::Utils_
{
 public:

  static inline void    activateRequest

    ( RequestList&        rlist,
      idx_t               ireq );

  static inline void    deactivateRequest

    ( RequestList&        rlist,
      idx_t               ireq );

  static inline void    startRequest

    ( Request&            req );

  static inline bool    testRequest

    ( Request&            req );

  static void           setStats

    ( RequestList&        rlist,
      Status*             stats );

  static void           clearWaitQueue

    ( RequestList&        rlist,
      Rendezvous&         rz,
      idx_t               waitCount );

  static void           indexError

    ( const String&       where,
      idx_t               ireq );

};


//-----------------------------------------------------------------------
//   activateRequest
//-----------------------------------------------------------------------


inline void RequestList::Utils_::activateRequest

  ( RequestList&  rlist,
    idx_t         ireq )

{
  const idx_t  i = rlist.iperm_[ireq];
  const idx_t  j = rlist.activeCount_;
  const idx_t  k = rlist.jperm_[j];

  JEM_ASSERT_NOTHROW ( i >= j );

  rlist.iperm_[ireq] = j;
  rlist.iperm_[k]    = i;
  rlist.jperm_[j]    = ireq;
  rlist.jperm_[i]    = k;

  rlist.activeCount_++;
}


//-----------------------------------------------------------------------
//   deactivateRequest
//-----------------------------------------------------------------------


inline void RequestList::Utils_::deactivateRequest

  ( RequestList&  rlist,
    idx_t         ireq )

{
  const idx_t  i = rlist.iperm_[ireq];
  const idx_t  j = rlist.activeCount_ - 1;
  const idx_t  k = rlist.jperm_[j];

  JEM_ASSERT_NOTHROW ( i <= j );

  rlist.iperm_[ireq] = j;
  rlist.iperm_[k]    = i;
  rlist.jperm_[j]    = ireq;
  rlist.jperm_[i]    = k;

  rlist.activeCount_ = j;
}


//-----------------------------------------------------------------------
//   startRequest
//-----------------------------------------------------------------------


inline void RequestList::Utils_::startRequest ( Request& req )
{
  req.message.status.error = 0;

  if ( req.mode == Request::SEND_MODE )
  {
    req.enqueued = req.mpool->queueSend ( req.message );
  }
  else
  {
    if ( req.message.rank == ANY_SOURCE )
    {
      req.enqueued = req.mpool->queueAnyRecv ( req.message );
    }
    else
    {
      req.enqueued = req.mpool->queueRecv    ( req.message );
    }
  }
}


//-----------------------------------------------------------------------
//   testRequest
//-----------------------------------------------------------------------


inline bool RequestList::Utils_::testRequest ( Request& req )
{
  if ( req.enqueued )
  {
    Message&  msg = req.message;

    msg.spinlock.lock ();
    req.enqueued = msg.enqueued;
    msg.spinlock.unlock ();
  }

  return ! req.enqueued;
}


//-----------------------------------------------------------------------
//   setStats
//-----------------------------------------------------------------------


void RequestList::Utils_::setStats

  ( RequestList&  rlist,
    Status*       stats )

{
  const idx_t  m = rlist.activeCount_;
  const idx_t  n = rlist.reqs_.size ();

  for ( idx_t i = 0; i < m; i++ )
  {
    idx_t  j = rlist.jperm_[i];

    if ( rlist.reqs_[j]->enqueued )
    {
      stats[j].error = -1;
    }
    else
    {
      stats[j] = rlist.reqs_[j]->message.status;
    }
  }

  for ( idx_t i = m; i < n; i++ )
  {
    idx_t  j = rlist.jperm_[i];
    stats[j] = EMPTY_STATUS;
  }
}


//-----------------------------------------------------------------------
//   clearWaitQueue
//-----------------------------------------------------------------------


void RequestList::Utils_::clearWaitQueue

  ( RequestList&  rlist,
    Rendezvous&   rz,
    idx_t         waitCount )

{
  Message*  msg;
  idx_t     k = 0;

  for ( idx_t i = 0; i < waitCount; i++ )
  {
    msg = & ( rlist.reqs_[rlist.waitQueue_[i]]->message );

    msg->spinlock.lock ();

    bool  enqueued = msg->enqueued;

    msg->notifyHandler = 0;
    msg->notifyData    = 0;

    msg->spinlock.unlock ();

    if ( ! enqueued )
    {
      k++;
    }
  }

  rz.waitNoCancel ( k, 0 );
}


//-----------------------------------------------------------------------
//   indexError
//-----------------------------------------------------------------------


void RequestList::Utils_::indexError

  ( const String&  where,
    idx_t          ireq )

{
  throw IllegalIndexException (
    where,
    String::format ( "illegal request index: %d", ireq )
  );
}


//=======================================================================
//   class RequestList
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


RequestList::RequestList

  ( int                rank,
    const Ref<Arena>&  arena ) :

    rank_        (  rank ),
    arena_       ( arena ),
    activeCount_ (     0 )

{
  JEM_PRECHECK ( arena );
}


RequestList::~RequestList ()
{}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t RequestList::size () const
{
  return reqs_.size ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void RequestList::clear ( Status* )
{
  reqs_     .clear ();
  iperm_    .clear ();
  jperm_    .clear ();
  waitQueue_.clear ();

  activeCount_ = 0;
}


//-----------------------------------------------------------------------
//   addRequest (send mode)
//-----------------------------------------------------------------------


idx_t RequestList::addRequest

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

  const idx_t   ireq = reqs_.size ();

  Ref<Request>  req  =

    newInstance<SendRequest> ( arena_, buf, rank_, dest, tag );

  reqs_     .pushBack ( std::move( req ) );
  iperm_    .pushBack ( ireq );
  jperm_    .pushBack ( ireq );
  waitQueue_.pushBack (   -1 );

  return ireq;
}


//-----------------------------------------------------------------------
//   addRequest (recv mode)
//-----------------------------------------------------------------------


idx_t RequestList::addRequest

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

  const idx_t   ireq = reqs_.size ();

  Ref<Request>  req  =

    newInstance<RecvRequest> ( arena_, buf, rank_, src, tag );

  reqs_     .pushBack ( std::move( req ) );
  iperm_    .pushBack ( ireq );
  jperm_    .pushBack ( ireq );
  waitQueue_.pushBack (   -1 );

  return ireq;
}


//-----------------------------------------------------------------------
//   startOne
//-----------------------------------------------------------------------


void RequestList::startOne ( idx_t ireq )
{
  if ( ireq < 0 || ireq >= reqs_.size() )
  {
    Utils_::indexError ( JEM_FUNC, ireq );
  }

  if ( iperm_[ireq] >= activeCount_ )
  {
    Utils_::startRequest    ( *reqs_[ireq] );
    Utils_::activateRequest ( *this, ireq  );
  }
}


//-----------------------------------------------------------------------
//   startAll
//-----------------------------------------------------------------------


void RequestList::startAll ()
{
  const idx_t  n     = reqs_.size  ();
  idx_t*       jperm = jperm_.addr ();

  for ( idx_t i = activeCount_; i < n; i++ )
  {
    Utils_::startRequest ( * reqs_[jperm[i]] );
  }

  activeCount_ = n;
}


//-----------------------------------------------------------------------
//   testOne
//-----------------------------------------------------------------------


bool RequestList::testOne ( idx_t ireq, Status* stat )
{
  if ( ireq < 0 || ireq >= reqs_.size() )
  {
    Utils_::indexError ( JEM_FUNC, ireq );
  }

  if ( iperm_[ireq] < activeCount_ )
  {
    Request&  req = * reqs_[ireq];

    if ( Utils_::testRequest( req ) )
    {
      if ( stat )
      {
        *stat = req.message.status;
      }

      Utils_::deactivateRequest ( *this, ireq );

      if ( req.message.status.error )
      {
        raiseError ( JEM_FUNC, req.message.status.error );
      }
    }
  }
  else if ( stat )
  {
    *stat = EMPTY_STATUS;
  }

  return (iperm_[ireq] >= activeCount_);
}


//-----------------------------------------------------------------------
//   testSome
//-----------------------------------------------------------------------


void RequestList::testSome

  ( idx_t&   count,
    idx_t*   ireqs,
    Status*  stats )

{
  Request*  req;
  int       err;
  idx_t     i, j;


  count = 0;
  err   = 0;
  i     = 0;

  while ( i < activeCount_ )
  {
    j   = jperm_[i];
    req = reqs_ [j].get ();

    if ( Utils_::testRequest( *req ) )
    {
      ireqs[count++] = j;
      err           |= req->message.status.error;

      Utils_::deactivateRequest ( *this, j );
    }
    else
    {
      i++;
    }
  }

  if ( stats )
  {
    for ( i = 0; i < count; i++ )
    {
      stats[i] = reqs_[ireqs[i]]->message.status;
    }
  }

  if ( err )
  {
    raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   testAll
//-----------------------------------------------------------------------


bool RequestList::testAll ( Status* stats )
{
  Request*  req;
  int       err;
  idx_t     count;
  idx_t     i, j;


  err   = 0;
  count = 0;

  for ( i = 0; i < activeCount_; i++ )
  {
    req = reqs_[jperm_[i]].get ();

    if ( Utils_::testRequest( *req ) )
    {
      err |= req->message.status.error;
      count++;
    }
  }

  if ( err )
  {
    if ( stats )
    {
      Utils_::setStats ( *this, stats );
    }

    // Remove all completed and failed requests.

    i = 0;

    while ( i < activeCount_ )
    {
      j   = jperm_[i];
      req = reqs_ [j].get ();

      if ( ! req->enqueued )
      {
        Utils_::deactivateRequest ( *this, j );
      }
      else
      {
        i++;
      }
    }

    raiseError ( JEM_FUNC, err );
  }

  if ( count == activeCount_ )
  {
    if ( stats )
    {
      Utils_::setStats ( *this, stats );
    }

    activeCount_ = 0;
  }

  return ( activeCount_ == 0 );
}


//-----------------------------------------------------------------------
//   waitOne
//-----------------------------------------------------------------------


void RequestList::waitOne ( idx_t ireq, Status* stat )
{
  if ( ireq < 0 || ireq >= reqs_.size() )
  {
    Utils_::indexError ( JEM_FUNC, ireq );
  }

  Request&  req = * reqs_[ireq];

  byte      sig;

  if ( iperm_[ireq] < activeCount_ )
  {
    if ( req.enqueued )
    {
      sig          = req.message.wait ();
      req.enqueued = ! (sig & SigThread::NOTIFY);

      if ( sig & SigThread::CANCEL )
      {
        throw CancelledException ( JEM_FUNC, "wait cancelled" );
      }
    }

    Utils_::deactivateRequest ( *this, ireq );

    if ( stat )
    {
      *stat = req.message.status;
    }

    if ( req.message.status.error )
    {
      raiseError ( JEM_FUNC, req.message.status.error );
    }
  }
  else if ( stat )
  {
    *stat = EMPTY_STATUS;
  }
}


//-----------------------------------------------------------------------
//   waitSome
//-----------------------------------------------------------------------


void RequestList::waitSome

  ( idx_t&   readyCount,
    idx_t*   ireqs,
    Status*  stats )

{
  Rendezvous   rz;
  Request*     req;
  Message*     msg;
  byte         sig;
  idx_t        waitCount;
  int          err;
  idx_t        i, j, k;


  readyCount = 0;
  waitCount  = 0;
  err        = 0;

  for ( i = 0; i < activeCount_; i++ )
  {
    j   =   jperm_[i];
    req =   reqs_ [j].get ();
    msg = & req->message;

    if ( ! req->enqueued )
    {
      ireqs[readyCount++] = j;
      err                |= msg->status.error;
      continue;
    }

    msg->spinlock.lock ();

    if ( ! msg->enqueued )
    {
      msg->spinlock.unlock ();

      req->enqueued       = false;
      ireqs[readyCount++] = j;
      err                |= msg->status.error;

      continue;
    }

    if ( readyCount == 0 )
    {
      msg->notifyHandler = & Rendezvous::notify;
      msg->notifyData    = & rz;

      msg->spinlock.unlock ();

      waitQueue_[waitCount++] = j;
    }
    else
    {
      msg->spinlock.unlock ();
    }
  }

  if ( readyCount == 0 && waitCount > 0 )
  {
    sig = rz.wait ( 1, ThreadLib::SPIN_COUNT );

    if ( sig & SigThread::CANCEL )
    {
      Utils_::clearWaitQueue   ( *this, rz, waitCount );
      throw CancelledException ( JEM_FUNC,
                                 "waitSome cancelled" );
    }
  }

  k = 0;

  for ( i = 0; i < waitCount; i++ )
  {
    j   =   waitQueue_[i];
    req =   reqs_ [j].get ();
    msg = & req->message;

    msg->spinlock.lock ();

    req->enqueued      = msg->enqueued;
    msg->notifyHandler = 0;
    msg->notifyData    = 0;

    msg->spinlock.unlock ();

    if ( ! req->enqueued )
    {
      ireqs[readyCount++] = j;
      err                |= msg->status.error;
      k++;
    }
  }

  if ( k > 0 )
  {
    rz.waitNoCancel ( k, ThreadLib::SPIN_COUNT );
  }

  if ( readyCount == activeCount_ )
  {
    activeCount_ = 0;
  }
  else
  {
    for ( i = 0; i < readyCount; i++ )
    {
      Utils_::deactivateRequest ( *this, ireqs[i] );
    }
  }

  if ( stats )
  {
    for ( i = 0; i < readyCount; i++ )
    {
      stats[i] = reqs_[ireqs[i]]->message.status;
    }
  }

  if ( err )
  {
    raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   waitAll
//-----------------------------------------------------------------------


void RequestList::waitAll ( Status* stats )
{
  Rendezvous  rz;
  Request*    req;
  Message*    msg;
  byte        sig;
  lint        spinCount;
  idx_t       waitCount;
  int         err;
  idx_t       i, j;
  lint        k;


  spinCount = ThreadLib::SPIN_COUNT;
  waitCount = 0;
  err       = 0;

  for ( i = 0; i < activeCount_; i++ )
  {
    j   =   jperm_[i];
    req =   reqs_ [j].get ();
    msg = & req->message;

    if ( ! req->enqueued )
    {
      err |= msg->status.error;
      continue;
    }

    JEM_SPIN_WHILE ( msg->enqueued, k, spinCount );

    spinCount -= k;

    msg->spinlock.lock ();

    if ( ! msg->enqueued )
    {
      msg->spinlock.unlock ();

      req->enqueued = false;
      err          |= msg->status.error;
    }
    else
    {
      msg->notifyHandler = & Rendezvous::notify;
      msg->notifyData    = & rz;

      msg->spinlock.unlock ();

      waitQueue_[waitCount++] = j;
    }
  }

  if ( waitCount > 0 )
  {
    sig = rz.wait ( waitCount, spinCount );

    if ( sig & SigThread::CANCEL )
    {
      Utils_::clearWaitQueue   ( *this, rz, waitCount );
      throw CancelledException ( JEM_FUNC,
                                 "waitAll cancelled" );
    }

    for ( i = 0; i < waitCount; i++ )
    {
      req           = reqs_[waitQueue_[i]].get ();
      req->enqueued = false;
      err          |= req->message.status.error;
    }
  }

  if ( stats )
  {
    Utils_::setStats ( *this, stats );
  }

  activeCount_ = 0;

  if ( err )
  {
    raiseError ( JEM_FUNC, err );
  }
}


//-----------------------------------------------------------------------
//   cancelOne
//-----------------------------------------------------------------------


void RequestList::cancelOne ( idx_t ireq )
{
  if ( ireq < 0 || ireq >= reqs_.size() )
  {
    Utils_::indexError ( JEM_FUNC, ireq );
  }

  if ( iperm_[ireq] < activeCount_ )
  {
    Request&  req = * reqs_[ireq];

    if ( req.enqueued )
    {
      req.mpool->cancel ( req.message );
      req.enqueued = false;
    }

    Utils_::deactivateRequest ( *this, ireq );
  }
}


//-----------------------------------------------------------------------
//   cancelAll
//-----------------------------------------------------------------------


void RequestList::cancelAll ( Status* stats )
{
  for ( idx_t  i = 0; i < activeCount_; i++ )
  {
    idx_t     j   = jperm_[i];
    Request*  req = reqs_[j].get ();

    if ( req->enqueued )
    {
      req->mpool->cancel ( req->message );
      req->enqueued = false;
    }
  }

  activeCount_ = 0;
}


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )
