
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
#include <jem/base/Thread.h>
#include <jem/net/Socket.h>
#include "Errors.h"
#include "NativeHandle.h"

extern "C"
{
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <unistd.h>
  #include <errno.h>
  #include <fcntl.h>
  #include <poll.h>
}


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class NativeHandle
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NativeHandle::NativeHandle ( int sock )
{
  int  flags;
  int  ierr;


  sock_ = sock;
  flags = ::fcntl ( sock, F_GETFL, 0 );

  if ( flags == -1 )
  {
    ierr = 1;
  }
  else
  {
    flags |= O_NONBLOCK;
    ierr   = ::fcntl ( sock, F_SETFL, flags );
  }

  if ( ierr )
  {
    Errors::setNoBlockFailed ( JEM_FUNC );
  }
}


NativeHandle::~NativeHandle ()
{
  closeNoThrow ( sock_ );

  sock_ = -1;
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


int NativeHandle::poll

  ( int          what,
    const Time&  timeout )

{
  struct pollfd  pfd;
  int            ierr;
  int            msec;


  if ( Thread::cancelled() )
  {
    Errors::threadCancelled ( JEM_FUNC );
  }

  if ( ! what )
  {
    return 0;
  }

  if ( sock_ < 0 )
  {
    Errors::closedSocket ( JEM_FUNC );
  }

  pfd.fd      = sock_;
  pfd.events  = 0;
  pfd.revents = 0;

  if ( what & Socket::POLL_READ )
  {
    pfd.events |= POLLIN;
  }

  if ( what & Socket::POLL_WRITE )
  {
    pfd.events |= POLLOUT;
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

  ierr = ::poll ( &pfd, 1, msec );
  what = 0;

  if ( ierr < 0 )
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

  if ( ierr > 0 )
  {
    if ( pfd.revents & POLLIN )
    {
      what |= Socket::POLL_READ;
    }

    if ( pfd.revents & POLLOUT )
    {
      what |= Socket::POLL_WRITE;
    }
  }

  return what;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void NativeHandle::close ()
{
  close ( sock_ );
}


void NativeHandle::close ( int& sock )
{
  if ( sock >= 0 )
  {
    int  ierr;

    // To avoid raising relatively harmless errors, the return
    // value of shutdown() is deliberately not checked.

    ::shutdown ( sock, SHUT_RDWR );

    ierr = ::close ( sock );

    if ( ierr )
    {
      Errors::closeFailed ( JEM_FUNC );
    }

    sock = -1;
  }
}


//-----------------------------------------------------------------------
//   closeNoThrow
//-----------------------------------------------------------------------


void NativeHandle::closeNoThrow ( int& sock )
{
  if ( sock >= 0 )
  {
    ::shutdown ( sock, SHUT_RDWR );
    ::close    ( sock );

    sock = -1;
  }
}


//-----------------------------------------------------------------------
//   toInteger
//-----------------------------------------------------------------------


lint NativeHandle::toInteger () const
{
  return (lint) sock_;
}


JEM_END_PACKAGE( net )

