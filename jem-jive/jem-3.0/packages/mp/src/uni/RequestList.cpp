
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
#include <jem/base/IllegalIndexException.h>
#include <jem/mp/error.h>
#include "error.h"
#include "Request.h"
#include "MessagePool.h"
#include "RequestList.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( uni )


//=======================================================================
//   class RequestList
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


RequestList::RequestList ( const Ref<MPool>&  mpool ) :

  mpool_ ( mpool )

{
  JEM_PRECHECK ( mpool );
}


RequestList::~RequestList ()
{}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t RequestList::size () const
{
  return (int) reqs_.size ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void RequestList::clear ( Status* )
{
  reqs_.clear ();
}


//-----------------------------------------------------------------------
//   addRequest (send mode)
//-----------------------------------------------------------------------


idx_t RequestList::addRequest

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

  const idx_t  ireq = reqs_.size ();

  reqs_.pushBack ( newInstance<Request>( buf, tag, mpool_ ) );

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
  if ( src != 0 && src != ANY_SOURCE )
  {
    recvRankError ( JEM_FUNC, src, 1 );
  }

  if ( tag < 0 || tag != ANY_TAG )
  {
    recvTagError ( JEM_FUNC, tag );
  }

  const idx_t  ireq = reqs_.size ();

  reqs_.pushBack ( newInstance<Request>( buf, tag, mpool_ ) );

  return ireq;
}


//-----------------------------------------------------------------------
//   startOne
//-----------------------------------------------------------------------


void RequestList::startOne ( idx_t ireq )
{
  if ( ireq < 0 || ireq >= reqs_.size() )
  {
    indexError_ ( JEM_FUNC, ireq );
  }

  reqs_[ireq]->start ();
}


//-----------------------------------------------------------------------
//   startAll
//-----------------------------------------------------------------------


void RequestList::startAll ()
{
  const idx_t  n = reqs_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    reqs_[i]->start ();
  }
}


//-----------------------------------------------------------------------
//   testOne
//-----------------------------------------------------------------------


bool RequestList::testOne ( idx_t ireq, Status* stat )
{
  if ( ireq < 0 || ireq >= reqs_.size() )
  {
    indexError_ ( JEM_FUNC, ireq );
  }

  return reqs_[ireq]->test ( stat );
}


//-----------------------------------------------------------------------
//   testSome
//-----------------------------------------------------------------------


void RequestList::testSome

  ( idx_t&   count,
    idx_t*   ireqs,
    Status*  stats )

{
  const idx_t  n = reqs_.size ();

  Request*     req;
  int          err;


  count = 0;
  err   = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    req = reqs_[i].get ();

    if ( req->active && ! req->message.enqueued )
    {
      req->active  = false;
      err         |= req->message.status.error;
      ireqs[count] = i;
      count++;
    }
  }

  if ( stats )
  {
    for ( idx_t i = 0; i < count; i++ )
    {
      stats[i] = reqs_[ireqs[i]]->message.status;
    }
  }

  if ( err )
  {
    raiseError ( JEM_FUNC, getError_( ireqs, count ) );
  }
}


//-----------------------------------------------------------------------
//   testAll
//-----------------------------------------------------------------------


bool RequestList::testAll ( Status* stats )
{
  const idx_t  n = reqs_.size ();

  Request*     req;
  int          err;
  idx_t        count;


  err   = 0;
  count = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    req = reqs_[i].get ();

    if ( ! req->active )
    {
      count++;
      continue;
    }

    if ( ! req->message.enqueued )
    {
      err |= req->message.status.error;
      count++;
    }
  }

  if ( err )
  {
    if ( stats )
    {
      setStats_ ( stats );
    }

    // Remove all completed and failed requests.

    for ( idx_t i = 0; i < n; i++ )
    {
      req = reqs_[i].get ();

      if ( ! req->message.enqueued )
      {
        req->active = false;

        if ( req->message.status.error )
        {
          err = req->message.status.error;
        }
      }
    }

    raiseError ( JEM_FUNC, err );
  }

  if ( count == n )
  {
    if ( stats )
    {
      setStats_ ( stats );
    }

    for ( idx_t i = 0; i < n; i++ )
    {
      reqs_[i]->active = false;
    }
  }

  return (count == n);
}


//-----------------------------------------------------------------------
//   waitOne
//-----------------------------------------------------------------------


void RequestList::waitOne ( idx_t ireq, Status* stat )
{
  if ( ireq < 0 || ireq >= reqs_.size() )
  {
    indexError_ ( JEM_FUNC, ireq );
  }

  reqs_[ireq]->wait ( stat );
}


//-----------------------------------------------------------------------
//   waitSome
//-----------------------------------------------------------------------


void RequestList::waitSome

  ( idx_t&   count,
    idx_t*   ireqs,
    Status*  stats )

{
  const idx_t  n = reqs_.size ();

  Request*     req;
  int          err;


  count = 0;
  err   = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    req = reqs_[i].get ();

    if ( req->active && ! req->message.enqueued )
    {
      req->active  = false;
      ireqs[count] = i;
      err         |= req->message.status.error;
      count++;
    }
  }

  if ( count == 0 )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      req = reqs_[i].get ();

      if ( req->active )
      {
        req->message.wait ();

        req->active  = false;
        ireqs[count] = i;
        err         |= req->message.status.error;
        count++;
      }
    }
  }

  if ( stats )
  {
    for ( idx_t i = 0; i < count; i++ )
    {
      stats[i] = reqs_[ireqs[i]]->message.status;
    }
  }

  if ( err )
  {
    raiseError ( JEM_FUNC, getError_( ireqs, count ) );
  }
}


//-----------------------------------------------------------------------
//   waitAll
//-----------------------------------------------------------------------


void RequestList::waitAll ( Status* stats )
{
  const idx_t   n = reqs_.size ();

  Request*      req;
  int           err;


  for ( idx_t i = 0; i < n; i++ )
  {
    req = reqs_[i].get ();

    if ( req->active )
    {
      req->message.wait ();
    }
  }

  if ( stats )
  {
    setStats_ ( stats );
  }

  err = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    req = reqs_[i].get ();

    req->active = false;

    if ( req->message.status.error )
    {
      err = req->message.status.error;
    }
  }

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
    indexError_ ( JEM_FUNC, ireq );
  }

  reqs_[ireq]->cancel ();
}


//-----------------------------------------------------------------------
//   cancelAll
//-----------------------------------------------------------------------


void RequestList::cancelAll ( Status* stats )
{
  const idx_t  n = reqs_.size ();

  Request*     req;


  for ( idx_t i = 0; i < n; i++ )
  {
    req = reqs_[i].get ();

    if ( req->message.enqueued )
    {
      req->message.cancel ();
    }

    req->active = false;
  }
}


//-----------------------------------------------------------------------
//   setStats_
//-----------------------------------------------------------------------


void RequestList::setStats_ ( Status* stats ) const
{
  const idx_t  n = reqs_.size ();

  Request*     req;


  for ( idx_t i = 0; i < n; i++ )
  {
    req = reqs_[i].get ();

    if ( req->active )
    {
      if ( req->message.enqueued )
      {
        stats[i].error = -1;
      }
      else
      {
        stats[i] = req->message.status;
      }
    }
    else
    {
      stats[i] = EMPTY_STATUS;
    }
  }
}


//-----------------------------------------------------------------------
//   getError_
//-----------------------------------------------------------------------


int RequestList::getError_

  ( const idx_t*  ireqs,
    idx_t         count ) const

{
  Request*  req;

  for ( idx_t i = 0; i < count; i++ )
  {
    req = reqs_[ireqs[i]].get ();

    if ( req->message.status.error )
    {
      return req->message.status.error;
    }
  }

  return 0;
}


//-----------------------------------------------------------------------
//   indexError
//-----------------------------------------------------------------------


void RequestList::indexError_

  ( const String&  where,
    idx_t          ireq ) const

{
  throw IllegalIndexException (
    where,
    String::format ( "illegal request index: %d", ireq )
  );
}


JEM_END_NAMESPACE ( uni )
JEM_END_PACKAGE   ( mp )
