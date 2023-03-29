
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
#include <jem/net/SocketOutputStream.h>


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class SocketOutputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SocketOutputStream::SocketOutputStream ( const Ref<Socket>& sock ) :

  socket_  ( sock ),
  timeout_ ( -1_lint )

{
  JEM_PRECHECK ( sock );
}


SocketOutputStream::SocketOutputStream ( const Self& rhs ) :

  socket_  ( rhs.socket_ ),
  timeout_ ( rhs.timeout_ )

{}


SocketOutputStream::~SocketOutputStream ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<io::OutputStream> SocketOutputStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void SocketOutputStream::close ()
{
  return socket_->close ( timeout_ );
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void SocketOutputStream::flush ()
{
  socket_->flush ( timeout_ );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void SocketOutputStream::write

  ( const void*  buf,
    idx_t        len )

{
  socket_->writeAll ( buf, len, timeout_ );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void SocketOutputStream::writeNoThrow

  ( const void*  buf,
    idx_t        len ) noexcept

{
  while ( len > 0 )
  {
    idx_t  k = socket_->writeNoThrow ( buf, len );

    if ( k <= 0 )
    {
      break;
    }

    len -= k;
    buf  = (const byte*) buf + k;
  }
}


//-----------------------------------------------------------------------
//   getSocket
//-----------------------------------------------------------------------


Ref<Socket> SocketOutputStream::getSocket () const
{
  return socket_;
}


//-----------------------------------------------------------------------
//   getTimeout
//-----------------------------------------------------------------------


Time SocketOutputStream::getTimeout () const
{
  return timeout_;
}


//-----------------------------------------------------------------------
//   setTimeout
//-----------------------------------------------------------------------


void SocketOutputStream::setTimeout ( const Time& timeout )
{
  timeout_ = timeout;
}


JEM_END_PACKAGE( net )

