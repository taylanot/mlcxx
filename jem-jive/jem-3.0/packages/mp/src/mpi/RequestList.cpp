
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


#include <jem/mp/config.h>

#ifdef JEM_USE_MPI

#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/RangeException.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/MPIContext.h>
#include "error.h"
#include "utilities.h"
#include "RequestList.h"


using jem::util::Flex;


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mpi )


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

  static void           deactivateRequests

    ( RequestList&        rlist );

  static void           setStats

    ( RequestList&        rlist,
      Status*             stats );

  static void           indexError

    ( const String&       context,
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

  rlist.iperm_[ireq] = j;
  rlist.iperm_[k]    = i;
  rlist.jperm_[i]    = k;
  rlist.jperm_[j]    = ireq;

  jem::swap ( rlist.reqs_[i], rlist.reqs_[j] );

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

  rlist.iperm_[ireq] = j;
  rlist.iperm_[k]    = i;
  rlist.jperm_[i]    = k;
  rlist.jperm_[j]    = ireq;

  jem::swap ( rlist.reqs_[i], rlist.reqs_[j] );

  rlist.activeCount_ = j;
}


//-----------------------------------------------------------------------
//   deactivateRequests
//-----------------------------------------------------------------------


void RequestList::Utils_::deactivateRequests

  ( RequestList&  rlist )

{
  const idx_t  n = rlist.activeCount_;

  Flex<idx_t>  ireqs ( n );

  idx_t        k = 0;


  for ( idx_t i = 0; i < n; i++ )
  {
    if ( rlist.stats_[i].MPI_ERROR != MPI_ERR_PENDING )
    {
      ireqs[k++] = rlist.jperm_[i];
    }
  }

  for ( idx_t i = 0; i < k; i++ )
  {
    deactivateRequest ( rlist, ireqs[i] );
  }
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

    if ( rlist.stats_[i].MPI_ERROR == MPI_ERR_PENDING )
    {
      stats[j].error = -1;
    }
    else
    {
      setStatus ( stats[j], rlist.stats_[i], rlist.types_[j] );
    }
  }

  for ( idx_t i = m; i < n; i++ )
  {
    stats[rlist.jperm_[i]] = EMPTY_STATUS;
  }
}


//-----------------------------------------------------------------------
//   indexError
//-----------------------------------------------------------------------


void RequestList::Utils_::indexError

  ( const String&  context,
    idx_t          ireq )

{
  throw IllegalIndexException (
    context,
    String::format ( "illegal request index: %d", ireq )
  );
}


//=======================================================================
//   class RequestList
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


RequestList::RequestList ( const Ref<MPIContext>& ctx ) :

  context_     ( ctx ),
  activeCount_ (   0 )

{
  JEM_PRECHECK ( ctx );
}


RequestList::~RequestList ()
{
  if ( ! MPIContext::finalized() )
  {
    const idx_t  n = reqs_.size ();

    for ( idx_t i = 0; i < activeCount_; i++ )
    {
      MPI_Cancel ( &reqs_[i] );
    }

    for ( idx_t i = 0; i < n; i++ )
    {
      MPI_Request_free ( & reqs_[i] );
    }
  }
}


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


void RequestList::clear ( Status* stats )
{
  const idx_t  n = reqs_.size ();

  if ( activeCount_ > 0 )
  {
    cancelAll ( stats );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    MPI_Request_free ( & reqs_[i] );
  }

  reqs_ .clear ();
  types_.clear ();
  stats_.clear ();
  iperm_.clear ();
  jperm_.clear ();

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
  const idx_t   ireq = reqs_.size ();

  MPI_Request   req;
  MPI_Datatype  type;
  MPI_Status    stat;
  int           err;


  if ( ireq == maxOf<int>() )
  {
    throw RangeException (
      JEM_FUNC,
      "maximum request list size exceeded"
    );
  }

  if ( buf.size() > maxOf<int>() )
  {
    sizeOverflowError ( JEM_FUNC, buf.size() );
  }

  type = convertType   ( buf.type() );

  err  = MPI_Send_init ( buf.addr (),
                         (int) buf.size (),
                         type,
                         dest,
                         tag,
                         context_->getComm (),
                         & req );

  if ( err )
  {
    raiseError ( JEM_FUNC, err );
  }

  std::memset ( &stat, 0x0, sizeof(stat) );

  reqs_ .pushBack ( req  );
  types_.pushBack ( type );
  stats_.pushBack ( stat );
  iperm_.pushBack ( ireq );
  jperm_.pushBack ( ireq );

  return ireq;
}


//-----------------------------------------------------------------------
//   addRequest (recvmode)
//-----------------------------------------------------------------------


idx_t RequestList::addRequest

  ( const RecvBuffer&  buf,
    int                src,
    int                tag )

{
  const idx_t   ireq = reqs_.size ();

  MPI_Request   req;
  MPI_Datatype  type;
  MPI_Status    stat;
  int           err;


  if ( src == ANY_SOURCE )
  {
    src = MPI_ANY_SOURCE;
  }

  if ( tag == ANY_TAG )
  {
    tag = MPI_ANY_TAG;
  }

  if ( ireq == maxOf<int>() )
  {
    throw RangeException (
      JEM_FUNC,
      "maximum request list size exceeded"
    );
  }

  if ( buf.size() > maxOf<int>() )
  {
    sizeOverflowError ( JEM_FUNC, buf.size() );
  }

  type = convertType   ( buf.type() );

  err  = MPI_Recv_init ( buf.addr (),
                         (int) buf.size (),
                         type,
                         src,
                         tag,
                         context_->getComm (),
                         & req );

  if ( err )
  {
    raiseError ( JEM_FUNC, err );
  }

  std::memset ( &stat, 0x0, sizeof(stat) );

  reqs_ .pushBack ( req  );
  types_.pushBack ( type );
  stats_.pushBack ( stat );
  iperm_.pushBack ( ireq );
  jperm_.pushBack ( ireq );

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

  idx_t  i = iperm_[ireq];

  if ( i >= activeCount_ )
  {
    int  err = MPI_Start ( & reqs_[i] );

    if ( err )
    {
      raiseError ( JEM_FUNC, err );
    }

    Utils_::activateRequest ( *this, ireq );
  }
}


//-----------------------------------------------------------------------
//   startAll
//-----------------------------------------------------------------------


void RequestList::startAll ()
{
  const idx_t  n = reqs_.size ();

  if ( activeCount_ < n )
  {
    int  err = MPI_Startall ( (int) (n - activeCount_),
                              reqs_.addr() + activeCount_ );

    if ( err )
    {
      raiseError ( JEM_FUNC, err );
    }

    activeCount_ = n;
  }
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

  int    flag = 0;
  idx_t  i    = iperm_[ireq];

  if ( i < activeCount_ )
  {
    int  err = MPI_Test ( & reqs_[i],
                          & flag,
                          & stats_[i] );

    if ( err )
    {
      if ( stat )
      {
        stat->error = stats_[i].MPI_ERROR;
      }

      Utils_::deactivateRequest ( *this, ireq );

      raiseError ( JEM_FUNC, err );
    }

    if ( flag )
    {
      if ( stat )
      {
        setStatus ( *stat, stats_[i], types_[ireq] );
      }

      Utils_::deactivateRequest ( *this, ireq );
    }
  }
  else if ( stat )
  {
    *stat = EMPTY_STATUS;
  }

  return (bool) flag;
}


//-----------------------------------------------------------------------
//   testSome
//-----------------------------------------------------------------------


void RequestList::testSome

  ( idx_t&   count,
    idx_t*   ireqs,
    Status*  stats )

{
  if ( activeCount_ == 0 )
  {
    count = 0; return;
  }

  if ( itemp_.size() < activeCount_ )
  {
    itemp_.resize ( activeCount_ );
  }

  int  n;
  int  err = MPI_Testsome ( (int) activeCount_,
                            reqs_.addr(),
                            & n,
                            itemp_.addr(),
                            stats_.addr() );

  count = (idx_t) n;

  for ( idx_t i = 0; i < count; i++ )
  {
    ireqs[i] = jperm_[itemp_[i]];
  }

  if ( stats )
  {
    for ( idx_t i = 0; i < count; i++ )
    {
      setStatus ( stats[i], stats_[i], types_[ireqs[i]] );
    }
  }

  for ( idx_t i = 0; i < count; i++ )
  {
    Utils_::deactivateRequest ( *this, ireqs[i] );
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
  int  flag;
  int  err = MPI_Testall ( (int) activeCount_,
                           reqs_.addr(),
                           & flag,
                           stats_.addr() );

  if ( err )
  {
    if ( stats )
    {
      Utils_::setStats ( *this, stats );
    }

    Utils_::deactivateRequests ( *this );

    raiseError ( JEM_FUNC, err );
  }

  if ( flag )
  {
    if ( stats )
    {
      Utils_::setStats ( *this, stats );
    }

    activeCount_ = 0;

    return true;
  }
  else
  {
    return false;
  }
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

  idx_t  i = iperm_[ireq];

  if ( i < activeCount_ )
  {
    int  err = MPI_Wait ( & reqs_[i], & stats_[i] );

    Utils_::deactivateRequest ( *this, ireq );

    if ( err )
    {
      if ( stat )
      {
        stat->error = stats_[i].MPI_ERROR;
      }

      raiseError ( JEM_FUNC, err );
    }

    if ( stat )
    {
      setStatus ( *stat, stats_[i], types_[ireq] );
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

  ( idx_t&   count,
    idx_t*   ireqs,
    Status*  stats )

{
  if ( activeCount_ == 0 )
  {
    count = 0; return;
  }

  if ( itemp_.size() < activeCount_ )
  {
    itemp_.resize ( activeCount_ );
  }

  int  n;
  int  err = MPI_Waitsome ( (int) activeCount_,
                            reqs_.addr(),
                            & n,
                            itemp_.addr(),
                            stats_.addr() );

  count = (idx_t) n;

  for ( idx_t i = 0; i < count; i++ )
  {
    ireqs[i] = jperm_[itemp_[i]];
  }

  if ( stats )
  {
    for ( idx_t i = 0; i < count; i++ )
    {
      setStatus ( stats[i], stats_[i], types_[ireqs[i]] );
    }
  }

  for ( idx_t i = 0; i < count; i++ )
  {
    Utils_::deactivateRequest ( *this, ireqs[i] );
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
  int  err = MPI_Waitall ( (int) activeCount_,
                           reqs_ .addr(),
                           stats_.addr() );

  if ( err )
  {
    if ( stats )
    {
      Utils_::setStats ( *this, stats );
    }

    Utils_::deactivateRequests ( *this );

    raiseError ( JEM_FUNC, err );
  }

  if ( stats )
  {
    Utils_::setStats ( *this, stats );
  }

  activeCount_ = 0;
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

  idx_t  i = iperm_[ireq];

  if ( i < activeCount_ )
  {
    int  err = MPI_Cancel ( & reqs_[i] );

    if ( err )
    {
      raiseError ( JEM_FUNC, err );
    }

    err = MPI_Wait ( & reqs_[i], & stats_[i] );

    Utils_::deactivateRequest ( *this, ireq );

    if ( err )
    {
      raiseError ( JEM_FUNC, err );
    }
  }
}


//-----------------------------------------------------------------------
//   cancelAll
//-----------------------------------------------------------------------


void RequestList::cancelAll ( Status* stats )
{
  const idx_t  n = reqs_.size ();

  int    lastErr = 0;


  for ( idx_t ireq = 0; ireq < n; ireq++ )
  {
    stats_[ireq].MPI_ERROR = 0;

    idx_t  i = iperm_[ireq];

    if ( i >= activeCount_ )
    {
      continue;
    }

    int  err = MPI_Cancel ( & reqs_[i] );

    if ( err )
    {
      stats_[ireq].MPI_ERROR = err;
      lastErr = err;
      continue;
    }

    err = MPI_Wait ( & reqs_[i], & stats_[ireq] );

    Utils_::deactivateRequest ( *this, ireq );

    if ( err )
    {
      lastErr = err;
      continue;
    }
  }

  if ( lastErr )
  {
    for ( idx_t ireq = 0; ireq < n; ireq++ )
    {
      stats[ireq].error = stats_[ireq].MPI_ERROR;
    }

    raiseError ( JEM_FUNC, lastErr );
  }
}


JEM_END_NAMESPACE ( mpi )
JEM_END_PACKAGE   ( mp )

#endif
