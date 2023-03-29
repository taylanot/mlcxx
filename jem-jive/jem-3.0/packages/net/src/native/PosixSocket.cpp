
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
#include <jem/base/limits.h>
#include <jem/base/Once.h>
#include <jem/base/Thread.h>
#include "Errors.h"
#include "NativeSocket.h"

extern "C"
{
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <signal.h>
  #include <errno.h>
}


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class NativeSocket
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NativeSocket::NativeSocket

  ( Ref<NativeAddress>  peer,
    Ref<NativeHandle>   handle ) :

    peer_   ( peer ),
    handle_ ( handle )

{
  JEM_PRECHECK ( peer && handle );
}


NativeSocket::~NativeSocket ()
{}


//-----------------------------------------------------------------------
//   connect
//-----------------------------------------------------------------------


Ref<NativeSocket> NativeSocket::connect

  ( const NativeAddress&  addr,
    const Time&           timeout )

{
  Ref<NativeHandle>  handle;
  int                sock;
  int                ierr;


  if ( Thread::cancelled() )
  {
    Errors::threadCancelled ( JEM_FUNC );
  }

  sock = newSocket ( addr );

  try
  {
    handle = newInstance<NativeHandle> ( sock );
  }
  catch ( ... )
  {
    NativeHandle::closeNoThrow ( sock );
    throw;
  }

  ierr = ::connect ( sock, addr.addr(), addr.size() );

  if ( ierr )
  {
    if ( errno == EINPROGRESS )
    {
      int  wait = handle->poll ( POLL_WRITE, timeout );

      if ( ! (wait & POLL_WRITE) )
      {
        Errors::connectTimedOut ( JEM_FUNC, addr.toString() );
      }
    }
    else
    {
      if ( (errno == EINTR) && Thread::cancelled() )
      {
        Errors::threadCancelled ( JEM_FUNC );
      }
      else
      {
        Errors::connectFailed   ( JEM_FUNC, addr.toString() );
      }
    }
  }

  return newInstance<NativeSocket> (
    const_cast<NativeAddress*> ( &addr ),
    handle
  );
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Socket> NativeSocket::dup ()
{
  return newInstance<Self> ( peer_, handle_ );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void NativeSocket::close ( const Time& timeout )
{
  handle_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void NativeSocket::flush ( const Time& timeout )
{
  // Nothing to be done.
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


int NativeSocket::poll

  ( int          what,
    const Time&  timeout )

{
  return handle_->poll ( what, timeout );
}


//-----------------------------------------------------------------------
//   peek
//-----------------------------------------------------------------------


idx_t NativeSocket::peek

  ( void*        buf,
    idx_t        len,
    const Time&  timeout )

{
  JEM_PRECHECK ( len >= 0 );

  return read_ ( buf, len, timeout, MSG_PEEK );
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t NativeSocket::read

  ( void*        buf,
    idx_t        len,
    const Time&  timeout )

{
  JEM_PRECHECK ( len >= 0 );

  return read_ ( buf, len, timeout );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


idx_t NativeSocket::write

  ( const void*  buf,
    idx_t        len,
    const Time&  timeout )

{
  JEM_PRECHECK ( len >= 0 );

  size_t   wsize  = (size_t) len;
  int      sock   = handle_->getFD ();
  ssize_t  result = 0;


  if ( ! wsize )
  {
    return 0;
  }

  result = ::send ( sock, buf, wsize, 0 );

  if ( (result < 0) && ((errno == EAGAIN) ||
                        (errno == EWOULDBLOCK)) )
  {
    int  wait = handle_->poll ( POLL_WRITE, timeout );

    if ( wait & POLL_WRITE )
    {
      result = ::send ( sock, buf, wsize, 0 );
    }
    else
    {
      // Timeout.

      return -1_idx;
    }
  }

  if ( result < 0 )
  {
    if ( (errno == EINTR) && Thread::cancelled() )
    {
      Errors::threadCancelled ( JEM_FUNC );
    }
    else
    {
      Errors::sendFailed      ( toString() );
    }
  }

  return (idx_t) result;
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


idx_t NativeSocket::writeNoThrow

  ( const void*  buf,
    idx_t        len ) noexcept

{
  size_t   wsize  = (size_t) len;
  int      sock   = handle_->getFD ();
  ssize_t  result = 0;


  if ( len <= 0 )
  {
    return 0;
  }

  result = ::send ( sock, buf, wsize, 0 );

  if ( result < 0 )
  {
    return 0;
  }
  else
  {
    return (idx_t) result;
  }
}


//-----------------------------------------------------------------------
//   getNoDelay
//-----------------------------------------------------------------------


bool NativeSocket::getNoDelay () const
{
  socklen_t  size;
  int        state;
  int        ierr;

  size = (socklen_t) sizeof(state);
  ierr = ::getsockopt ( handle_->getFD (),
                        IPPROTO_TCP,
                        TCP_NODELAY,
                        (void*) &state,
                        &size );

  if ( ierr )
  {
    Errors::getNoDelayFailed ( toString() );
  }

  return (state ? true : false);
}


//-----------------------------------------------------------------------
//   setNoDelay
//-----------------------------------------------------------------------


void NativeSocket::setNoDelay ( bool choice )
{
  int  state;
  int  ierr;

  state = choice ? 1 : 0;
  ierr  = ::setsockopt ( handle_->getFD(),
                         IPPROTO_TCP,
                         TCP_NODELAY,
                         (const void*) &state,
                         (socklen_t) sizeof(state) );

  if ( ierr )
  {
    Errors::setNoDelayFailed ( toString() );
  }
}


//-----------------------------------------------------------------------
//   getPeerAddress
//-----------------------------------------------------------------------


Ref<SocketAddress> NativeSocket::getPeerAddress () const
{
  return peer_;
}


//-----------------------------------------------------------------------
//   setPeerAddress
//-----------------------------------------------------------------------


void NativeSocket::setPeerAddress ( const Ref<Address>& addr )
{
  JEM_PRECHECK ( addr );

  const NativeAddress&  peer = toNativeAddress ( *addr );

  peer_ = const_cast<NativeAddress*> ( &peer );
}


//-----------------------------------------------------------------------
//   getLocalAddress
//-----------------------------------------------------------------------


Ref<SocketAddress> NativeSocket::getLocalAddress () const
{
  if ( ! local_ )
  {
    const_cast<Self*> ( this ) -> local_ =

      getLocalAddress ( handle_->getFD() );
  }

  return local_;
}


Ref<NativeAddress> NativeSocket::getLocalAddress ( int sock )
{
  struct sockaddr_storage  addr;
  socklen_t                size;
  int                      ierr;

  std::memset ( &addr, 0x0, sizeof(addr) );

  size = (socklen_t) sizeof(addr);
  ierr = ::getsockname ( sock, (sockaddr_t*) &addr, &size );

  if ( ierr || (size > (socklen_t) sizeof(addr)) )
  {
    Errors::getLocalAddrFailed ( JEM_FUNC );
  }

  return newInstance<NativeAddress> ( (sockaddr_t*) &addr, size );
}


//-----------------------------------------------------------------------
//   getSocketHandle
//-----------------------------------------------------------------------


Ref<SocketHandle> NativeSocket::getSocketHandle () const
{
  return handle_;
}


//-----------------------------------------------------------------------
//   newSocket
//-----------------------------------------------------------------------


int NativeSocket::newSocket ( const NativeAddress& addr )
{
  static Once  once = JEM_ONCE_INITIALIZER;

  int          sock;

  runOnce ( once, setSigHandler_ );

  sock = ::socket ( addr.addr()->sa_family, SOCK_STREAM, 0 );

  if ( sock < 0 )
  {
    Errors::newSocketFailed ( JEM_FUNC );
  }

  return sock;
}


//-----------------------------------------------------------------------
//   read_
//-----------------------------------------------------------------------


inline idx_t NativeSocket::read_

  ( void*        buf,
    idx_t        len,
    const Time&  timeout,
    int          flags )

{
  size_t   rsize  = (size_t) len;
  int      sock   = handle_->getFD ();
  ssize_t  result = 0;


  if ( ! rsize )
  {
    return 0;
  }

  result = ::recv ( sock, buf, rsize, flags );

  if ( (result < 0) && ((errno == EAGAIN) ||
                        (errno == EWOULDBLOCK)) )
  {
    int  wait = handle_->poll ( POLL_READ, timeout );

    if ( wait & POLL_READ )
    {
      result = ::recv ( sock, buf, rsize, flags );
    }
    else
    {
      // Timeout.

      return -1_idx;
    }
  }

  if ( result < 0 )
  {
    if ( (errno == EINTR) && Thread::cancelled() )
    {
      Errors::threadCancelled ( JEM_FUNC );
    }
    else
    {
      Errors::recvFailed      ( toString() );
    }
  }

  return (idx_t) result;
}


//-----------------------------------------------------------------------
//   setSigHandler_
//-----------------------------------------------------------------------


void NativeSocket::setSigHandler_ ()
{
  struct sigaction  oldHandler;
  struct sigaction  newHandler;

  int               ierr = 0;

  ierr = sigaction ( SIGPIPE, nullptr, &oldHandler );

  if ( ierr )
  {
    return;
  }

  // If a handler has already been set, then leave it in place.

  if ( oldHandler.sa_handler != nullptr )
  {
    return;
  }

  newHandler.sa_handler = SIG_IGN;
  newHandler.sa_flags   = 0;

  ierr = sigemptyset ( &newHandler.sa_mask );

  if ( ierr )
  {
    return;
  }

  ierr = sigaction ( SIGPIPE, &newHandler, nullptr );
}


JEM_END_PACKAGE( net )

