
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


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class NativeHandle
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NativeHandle::NativeHandle ( SOCKET sock )
{
  unsigned long  mode = 1;
  int            ierr = 0;

  sock_ = sock;
  ierr  = ::ioctlsocket ( sock, FIONBIO, &mode );

  if ( ierr )
  {
    Errors::setNoBlockFailed ( JEM_FUNC );
  }
}


NativeHandle::~NativeHandle ()
{
  closeNoThrow ( sock_ );

  sock_ = INVALID_SOCKET;
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


int NativeHandle::poll

  ( int          what,
    const Time&  timeout )

{
  fd_set*  rfdp = nullptr;
  fd_set*  wfdp = nullptr;

  fd_set   rfds;
  fd_set   wfds;
  int      ierr;


  if ( Thread::cancelled() )
  {
    Errors::threadCancelled ( JEM_FUNC );
  }

  if ( ! what )
  {
    return 0;
  }

  if ( sock_ == INVALID_SOCKET )
  {
    Errors::closedSocket ( JEM_FUNC );
  }

  if ( what & Socket::POLL_READ )
  {
    rfdp = &rfds;

    FD_ZERO ( rfdp );
    FD_SET  ( sock_, rfdp );
  }

  if ( what & Socket::POLL_WRITE )
  {
    wfdp = &wfds;

    FD_ZERO ( wfdp );
    FD_SET  ( sock_, wfdp );
  }

  if ( timeout < Time::zero() )
  {
    ierr = ::select ( 1, rfdp, wfdp, nullptr, nullptr );
  }
  else
  {
    struct timeval  tv;

    tv.tv_sec  = (int) jem::min ( (lint) maxOf<int>(),
                                  timeout.sec() );
    tv.tv_usec = (int) (timeout.nsec() / 1000_lint);

    ierr = ::select ( 1, rfdp, wfdp, nullptr, &tv );
  }

  what = 0;

  if      ( ierr > 0 )
  {
    if ( rfdp && FD_ISSET( sock_, rfdp ) )
    {
      what |= Socket::POLL_READ;
    }

    if ( wfdp && FD_ISSET( sock_, wfdp ) )
    {
      what |= Socket::POLL_WRITE;
    }
  }
  else if ( ierr )
  {
    if ( (WSAGetLastError() == WSAEINTR) && Thread::cancelled() )
    {
      Errors::threadCancelled ( JEM_FUNC );
    }
    else
    {
      Errors::pollFailed      ( JEM_FUNC );
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


void NativeHandle::close ( SOCKET& sock )
{
  if ( sock != INVALID_SOCKET )
  {
    int  ierr;

    // To avoid raising relatively harmless errors, the return
    // value of shutdown() is deliberately not checked.

    ::shutdown ( sock, SD_BOTH );

    ierr = ::closesocket ( sock );

    if ( ierr )
    {
      Errors::closeFailed ( JEM_FUNC );
    }

    sock = INVALID_SOCKET;
  }
}


//-----------------------------------------------------------------------
//   closeNoThrow
//-----------------------------------------------------------------------


void NativeHandle::closeNoThrow ( SOCKET& sock )
{
  if ( sock != INVALID_SOCKET )
  {
    ::shutdown    ( sock, SD_BOTH );
    ::closesocket ( sock );

    sock = INVALID_SOCKET;
  }
}


//-----------------------------------------------------------------------
//   toInteger
//-----------------------------------------------------------------------


lint NativeHandle::toInteger () const
{
  if ( sock_ == INVALID_SOCKET )
  {
    return -1_lint;
  }
  else
  {
    return (lint) sock_;
  }
}


JEM_END_PACKAGE( net )

