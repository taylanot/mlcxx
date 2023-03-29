
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


#include <jem/base/Time.h>
#include <jem/net/Socket.h>
#include <jem/net/SocketAddress.h>
#include "Errors.h"


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class Socket
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Socket::POLL_READ  = 1 << 0;
const int  Socket::POLL_WRITE = 1 << 1;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Socket::~Socket ()
{}


//-----------------------------------------------------------------------
//   readAll
//-----------------------------------------------------------------------


void Socket::readAll

  ( void*        buf,
    idx_t        len,
    const Time&  timeout )

{
  if ( timeout < Time::zero() )
  {
    while ( len > 0 )
    {
      idx_t  k = read ( buf, len, timeout );

      if ( k <= 0 )
      {
        Errors::readTruncated ( JEM_FUNC, len );
      }

      len -= k;
      buf  = (byte*) buf + k;
    }
  }
  else
  {
    Time  now = Time::now ();
    Time  end = now + timeout;

    while ( (len > 0) && (now <= end) )
    {
      idx_t  k = read ( buf, len, end - now );

      if ( k < 0 )
      {
        break;
      }

      if ( k == 0 )
      {
        Errors::readTruncated ( JEM_FUNC, len );
      }

      len -= k;
      buf  = (byte*) buf + k;

      if ( len > 0 )
      {
        now = Time::now ();
      }
    }

    if ( len > 0 )
    {
      Errors::readTimedOut ( JEM_FUNC );
    }
  }
}


//-----------------------------------------------------------------------
//   writeAll
//-----------------------------------------------------------------------


void Socket::writeAll

  ( const void*  buf,
    idx_t        len,
    const Time&  timeout )

{
  if ( timeout < Time::zero() )
  {
    while ( len > 0 )
    {
      idx_t  k = write ( buf, len, timeout );

      if ( k <= 0 )
      {
        Errors::writeTruncated ( JEM_FUNC, len );
      }

      len -= k;
      buf  = (const byte*) buf + k;
    }
  }
  else
  {
    Time  now = Time::now ();
    Time  end = now + timeout;

    while ( (len > 0) && (now <= end) )
    {
      idx_t  k = write ( buf, len, end - now );

      if ( k < 0 )
      {
        break;
      }

      if ( k == 0 )
      {
        Errors::writeTruncated ( JEM_FUNC, len );
      }

      len -= k;
      buf  = (const byte*) buf + k;

      if ( len > 0 )
      {
        now = Time::now ();
      }
    }

    if ( len > 0 )
    {
      Errors::writeTimedOut ( JEM_FUNC );
    }
  }
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Socket::toString () const
{
  String  local = "<unknown address>";
  String  peer  = local;

  try
  {
    local = getLocalAddress()->toString ();
    peer  = getPeerAddress ()->toString ();
  }
  catch ( ... )
  {}

  if      ( (local.size() == 0) && (peer.size() == 0) )
  {
    return String::format ( "jem::net::Socket@%p",
                            (const void*) this );
  }
  else if ( local.size() == 0 )
  {
    return String::format ( "jem::net::Socket -> %s", peer );
  }
  else if ( peer.size() == 0 )
  {
    return String::format ( "jem::net::Socket @ %s", local );
  }
  else
  {
    return String::format ( "jem::net::Socket @ %s -> %s",
                            local, peer );

  }
}


JEM_END_PACKAGE( net )

