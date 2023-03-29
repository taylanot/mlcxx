
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


#include <jem/net/Socket.h>
#include <jem/net/SocketInputStream.h>
#include "Errors.h"


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class SocketInputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SocketInputStream::SocketInputStream ( const Ref<Socket>& sock ) :

  socket_  ( sock ),
  timeout_ ( -1_lint )

{
  JEM_PRECHECK ( sock );
}


SocketInputStream::SocketInputStream ( const Self& rhs ) :

  socket_  ( rhs.socket_ ),
  timeout_ ( rhs.timeout_ )

{}


SocketInputStream::~SocketInputStream ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<io::InputStream> SocketInputStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void SocketInputStream::close ()
{
  return socket_->close ( timeout_ );
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t SocketInputStream::poll ( const Time& timeout )
{
  int  wait = socket_->poll ( Socket::POLL_READ, timeout );

  if ( wait & Socket::POLL_READ )
  {
    return 1_idx;
  }
  else
  {
    return 0_idx;
  }
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t SocketInputStream::read

  ( void*  buf,
    idx_t  len )

{
  idx_t  n = socket_->read ( buf, len, timeout_ );

  if ( n < 0 )
  {
    Errors::readTimedOut ( JEM_FUNC );
  }

  return n;
}


//-----------------------------------------------------------------------
//   readAll
//-----------------------------------------------------------------------


void SocketInputStream::readAll

  ( void*  buf,
    idx_t  len )

{
  socket_->readAll ( buf, len, timeout_ );
}


//-----------------------------------------------------------------------
//   getSocket
//-----------------------------------------------------------------------


Ref<Socket> SocketInputStream::getSocket () const
{
  return socket_;
}


//-----------------------------------------------------------------------
//   getTimeout
//-----------------------------------------------------------------------


Time SocketInputStream::getTimeout () const
{
  return timeout_;
}


//-----------------------------------------------------------------------
//   setTimeout
//-----------------------------------------------------------------------


void SocketInputStream::setTimeout ( const Time& timeout )
{
  timeout_ = timeout;
}


JEM_END_PACKAGE( net )

