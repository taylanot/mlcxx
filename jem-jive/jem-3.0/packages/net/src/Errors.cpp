
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


#include <jem/base/Error.h>
#include <jem/base/CancelledException.h>
#include <jem/base/native/System.h>
#include <jem/net/TimeoutException.h>
#include "Errors.h"


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class Errors
//=======================================================================

//-----------------------------------------------------------------------
//   invalidDomain
//-----------------------------------------------------------------------


void Errors::invalidDomain

  ( const String&  where,
    int            type )

{
  throw Error (
    where,
    "invalid network domain type: " + String ( type )
  );
}


//-----------------------------------------------------------------------
//   unknownDomain
//-----------------------------------------------------------------------


void Errors::unknownDomain ( const String& where )
{
  throw NetworkException ( where, "unknown network domain type" );
}


//-----------------------------------------------------------------------
//   notInetDomain
//-----------------------------------------------------------------------


void Errors::notInetDomain ( const String& where )
{
  throw NetworkException (
    where,
    "invalid network domain; should be IPv4 or IPv6"
  );
}


//-----------------------------------------------------------------------
//   closeFailed
//-----------------------------------------------------------------------


void Errors::closeFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "error closing network socket",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   closedSocket
//-----------------------------------------------------------------------


void Errors::closedSocket ( const String& where )
{
  throw NetworkException (
    JEM_FUNC,
    "network operation failed: closed socket"
  );
}


//-----------------------------------------------------------------------
//   invalidSocket
//-----------------------------------------------------------------------


void Errors::invalidSocket ( const String& where )
{
  throw NetworkException ( where, "invalid socket specified" );
}


//-----------------------------------------------------------------------
//   recvFailed
//-----------------------------------------------------------------------


void Errors::recvFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "network receive operation failed",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   sendFailed
//-----------------------------------------------------------------------


void Errors::sendFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "network send operation failed",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   pollFailed
//-----------------------------------------------------------------------


void Errors::pollFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "error waiting for a network socket",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   bindFailed
//-----------------------------------------------------------------------


void Errors::bindFailed

  ( const String&  where,
    const String&  addr )

{
  throw NetworkException (
    where,
    makeMessage_ (
      String::format (
        "failed to bind a server socket to the address `%s\'",
        addr
      ),
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   acceptFailed
//-----------------------------------------------------------------------


void Errors::acceptFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "failed to accept an incoming network connection",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   connectFailed
//-----------------------------------------------------------------------


void Errors::connectFailed

  ( const String&  where,
    const String&  addr )

{
  throw NetworkException (
    where,
    makeMessage_ (
      String::format (
        "failed to connect to `%s\'",
        addr
      ),
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   connectTimedOut
//-----------------------------------------------------------------------


void Errors::connectTimedOut

  ( const String&  where,
    const String&  addr )

{
  throw TimeoutException (
    where,
    String::format (
      "timeout while trying to connect to `%s\'",
      addr
    )
  );
}


//-----------------------------------------------------------------------
//   threadCancelled
//-----------------------------------------------------------------------


void Errors::threadCancelled ( const String& where )
{
  throw CancelledException (
    where,
    "network operation cancelled"
  );
}


//-----------------------------------------------------------------------
//   readTimedOut
//-----------------------------------------------------------------------


void Errors::readTimedOut ( const String& where )
{
  throw TimeoutException (
    JEM_FUNC,
    "network receive operation timed out"
  );
}


//-----------------------------------------------------------------------
//   writeTimedOut
//-----------------------------------------------------------------------


void Errors::writeTimedOut ( const String& where )
{
  throw TimeoutException (
    JEM_FUNC,
    "network send operation timed out"
  );
}


//-----------------------------------------------------------------------
//   readTruncated
//-----------------------------------------------------------------------


void Errors::readTruncated

  ( const String&  where,
    idx_t          count )

{
  throw NetworkException (
    where,
    String::format (
      "unexpected end of input stream; expected %d more bytes",
      count
    )
  );
}


//-----------------------------------------------------------------------
//   writeTruncated
//-----------------------------------------------------------------------


void Errors::writeTruncated

  ( const String&  where,
    idx_t          count )

{
  throw NetworkException (
    where,
    String::format (
      "network send operation terminated; "
      "expected to send %d more bytes",
      count
    )
  );
}


//-----------------------------------------------------------------------
//   lookupFailed
//-----------------------------------------------------------------------


void Errors::lookupFailed

  ( const String&  where,
    const String&  addr,
    const String&  what )

{
  throw NetworkException (
    where,
    makeMessage_ (
      String::format (
        "failed to resolve the network address `%s\'",
        addr
      ),
      what
    )
  );
}


//-----------------------------------------------------------------------
//   portLookupFailed
//-----------------------------------------------------------------------


void Errors::portLookupFailed ( const String& where )
{
  throw NetworkException ( where, "network port lookup failed" );
}


//-----------------------------------------------------------------------
//   hostLookupFailed
//-----------------------------------------------------------------------


void Errors::hostLookupFailed ( const String& where )
{
  throw NetworkException ( where, "network host lookup failed" );
}


//-----------------------------------------------------------------------
//   newSocketFailed
//-----------------------------------------------------------------------


void Errors::newSocketFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "failed to create a network socket",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   setNoBlockFailed
//-----------------------------------------------------------------------


void Errors::setNoBlockFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "error creating a non-blocking network socket",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   setNoDelayFailed
//-----------------------------------------------------------------------


void Errors::setNoDelayFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "failed to set the NO_DELAY socket option",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   getNoDelayFailed
//-----------------------------------------------------------------------


void Errors::getNoDelayFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "failed to query the NO_DELAY socket option",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   getLocalAddrFailed
//-----------------------------------------------------------------------


void Errors::getLocalAddrFailed ( const String& where )
{
  throw NetworkException (
    where,
    makeMessage_ (
      "failed to determine the local socket address",
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   makeMessage_
//-----------------------------------------------------------------------


String Errors::makeMessage_

  ( const String&  what,
    const String&  details )

{
  if      ( (what   .size() == 0) &&
            (details.size() == 0) )
  {
    return String ();
  }
  else if ( details.size() == 0 )
  {
    return what;
  }
  else if ( what   .size() == 0 )
  {
    return details;
  }
  else
  {
    return String::format ( "%s (%s)", what, details );
  }
}


JEM_END_PACKAGE( net )

