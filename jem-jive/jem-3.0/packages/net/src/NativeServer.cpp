
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


#include <jem/base/limits.h>
#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include "Errors.h"
#include "NativeServer.h"
#include "NativeSocket.h"

extern "C"
{
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <unistd.h>
  #include <errno.h>
}


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class NativeServer
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NativeServer::NativeServer ()
{
  openc_ = 0;
}


NativeServer::~NativeServer ()
{
  for ( idx_t i = 0; i < socks_.size(); i++ )
  {
    if ( socks_[i].fd >= 0 )
    {
      ::close ( socks_[i].fd );
    }
  }

  openc_ = 0;
}


//-----------------------------------------------------------------------
//   listen
//-----------------------------------------------------------------------


idx_t NativeServer::listen

  ( const NativeAddress&  addr,
    int                   backlog )

{
  const idx_t  n   = socks_.size ();
  idx_t        idx = -1;

  int          sock;
  int          ierr;


  sock = NativeSocket::newSocket ( addr );
  ierr = ::bind ( sock, addr.addr(), addr.size() );

  if ( ierr )
  {
    ::close            ( sock );
    Errors::bindFailed ( JEM_FUNC, addr.toString() );
  }

  ierr = ::listen ( sock, backlog );

  if ( ierr )
  {
    ::close                 ( sock );
    Errors::newSocketFailed ( JEM_FUNC );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( socks_[i].fd < 0 )
    {
      socks_[i].fd = sock;
      idx          = i;

      break;
    }
  }

  if ( idx < 0 )
  {
    idx = n;

    try
    {
      pollfd_t  pfd;

      pfd.fd      = sock;
      pfd.events  = 0;
      pfd.revents = 0;

      socks_.pushBack ( pfd );
      iperm_.pushBack ( idx );
    }
    catch ( ... )
    {
      ::close ( sock );

      if ( socks_.size() > n )
      {
        socks_.popBack ();
      }

      if ( iperm_.size() > n )
      {
        iperm_.popBack ();
      }

      throw;
    }
  }

  openc_++;

  return idx;
}


//-----------------------------------------------------------------------
//   accept
//-----------------------------------------------------------------------


Ref<Socket> NativeServer::accept ( idx_t idx )
{
  const idx_t              n = socks_.size ();

  Ref<NativeHandle>        handle;
  struct sockaddr_storage  addr;
  socklen_t                size;
  int                      sock;


  if ( Thread::cancelled() )
  {
    Errors::threadCancelled ( JEM_FUNC );
  }

  if ( (idx < 0) || (idx >= n) )
  {
    Errors::invalidSocket   ( JEM_FUNC );
  }

  if ( socks_[idx].fd < 0 )
  {
    Errors::closedSocket    ( JEM_FUNC );
  }

  size = (socklen_t) sizeof(addr);
  sock = ::accept ( socks_[idx].fd,
                    (sockaddr_t*) &addr, &size );

  if ( (sock < 0) || (size > (socklen_t) sizeof(addr)) )
  {
    if ( (errno == EINTR) && Thread::cancelled() )
    {
      Errors::threadCancelled ( JEM_FUNC );
    }
    else
    {
      Errors::acceptFailed    ( JEM_FUNC );
    }
  }

  try
  {
    handle = newInstance<NativeHandle> ( sock );
  }
  catch ( ... )
  {
    NativeHandle::closeNoThrow ( sock );
    throw;
  }

  return newInstance<NativeSocket> (
    newInstance<NativeAddress> ( (const sockaddr_t*) &addr, size ),
    handle
  );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void NativeServer::close ( idx_t idx )
{
  const idx_t  n = socks_.size ();

  if ( (idx >= 0) && (idx < n) && (socks_[idx].fd >= 0) )
  {
    if ( ::close( socks_[idx].fd ) )
    {
      Errors::closeFailed ( JEM_FUNC );
    }

    socks_[idx].fd = -1;

    openc_--;

    JEM_ASSERT ( openc_ >= 0 );
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t NativeServer::poll ( const Time& timeout )
{
  const idx_t  n = socks_.size ();

  int          ierr;
  int          msec;


  if ( Thread::cancelled() )
  {
    Errors::threadCancelled ( JEM_FUNC );
  }

  if ( openc_ <= 0 )
  {
    return -1_idx;
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    socks_[i].revents = 0;

    if ( socks_[i].fd >= 0 )
    {
      socks_[i].events = POLLIN;
    }
    else
    {
      socks_[i].events = 0;
    }
  }

  if ( timeout < Time::zero() )
  {
    msec = -1;
  }
  else
  {
    lint  sec =

      jem::min (
        timeout.sec (),
        (lint) maxOf<int>() / 1000_lint - 1000_lint
      );

    msec = (int) (sec            * 1000_lint +
                  timeout.nsec() / 1000000_lint);
  }

  ierr = ::poll ( socks_.addr(), (nfds_t) n, msec );

  if      ( ierr > 0 )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idx = iperm_[i];

      if ( socks_[idx].revents & POLLIN )
      {
        // Change the permutation list to make sure that another,
        // ready socket is picked the next time.

        jem::swap ( iperm_[i], iperm_[n - 1] );

        return idx;
      }
    }
  }
  else if ( ierr )
  {
    if ( (errno == EINTR) && Thread::cancelled() )
    {
      Errors::threadCancelled ( JEM_FUNC );
    }
    else
    {
      Errors::pollFailed      ( JEM_FUNC );
    }
  }

  return -1_idx;
}


//-----------------------------------------------------------------------
//   getLocalAddress
//-----------------------------------------------------------------------


Ref<SocketAddress> NativeServer::getLocalAddress ( idx_t idx ) const
{
  const idx_t  n = socks_.size ();

  if ( (idx < 0) || (idx >= n) )
  {
    Errors::invalidSocket ( JEM_FUNC );
  }

  if ( socks_[idx].fd < 0 )
  {
    Errors::closedSocket  ( JEM_FUNC );
  }

  return NativeSocket::getLocalAddress ( socks_[idx].fd );
}


JEM_END_PACKAGE( net )

