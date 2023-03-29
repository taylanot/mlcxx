
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
#include <jem/base/Error.h>
#include <jem/base/Thread.h>
#include <jem/base/CString.h>
#include <jem/base/Integer.h>
#include <jem/base/native/System.h>
#include "Errors.h"
#include "NativeAddress.h"

#ifndef JEM_OS_WINDOWS

extern "C"
{
  #include <netinet/in.h>
  #include <sys/types.h>
  #include <netdb.h>
}

#endif


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class SocketAddress
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  SocketAddress::LOCAL_ONLY_ = 1 << 0;


//-----------------------------------------------------------------------
//   lookup
//-----------------------------------------------------------------------


Ref<SocketAddress> SocketAddress::lookup ( Domain::Type type )
{
  if ( type == Domain::IPv4 )
  {
    struct sockaddr_in  addr4;

    std::memset ( &addr4, 0x0, sizeof(addr4) );

    addr4.sin_family = AF_INET;

    return newInstance<NativeAddress> (
      (sockaddr_t*) &addr4,
      (socklen_t)   sizeof(addr4)
    );
  }

  if ( type == Domain::IPv6 )
  {
    struct sockaddr_in6  addr6;

    std::memset ( &addr6, 0x0, sizeof(addr6) );

    addr6.sin6_family = AF_INET6;

    return newInstance<NativeAddress> (
      (sockaddr_t*) &addr6,
      (socklen_t)   sizeof(addr6)
    );
  }

  Errors::notInetDomain ( JEM_FUNC );

  return nullptr;
}


Ref<SocketAddress> SocketAddress::lookup

  ( Kind           kind,
    Domain::Type   type,
    const String&  addr,
    int            port )

{
  return lookup_ ( kind, type, addr, port );
}


//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


Ref<SocketAddress> SocketAddress::convert

  ( Kind           kind,
    Domain::Type   type,
    const String&  addr,
    int            port )

{
  return lookup_ ( kind, type, addr, port, LOCAL_ONLY_ );
}


//-----------------------------------------------------------------------
//   lookup_
//-----------------------------------------------------------------------


Ref<SocketAddress> SocketAddress::lookup_

  ( Kind           kind,
    Domain::Type   type,
    const String&  addr,
    int            port,
    int            flags )

{
  const char*         pserv = nullptr;
  const char*         pnode = nullptr;
  struct addrinfo*    list  = nullptr;
  struct addrinfo*    info  = nullptr;
  int                 ierr  = 0;

  Ref<SocketAddress>  saddr;
  struct addrinfo     hints;
  CString             service;
  CString             node;


  // Handle special addresses.

  if ( (kind  == SERVER)      &&
       (type  != Domain::ANY) &&
       (addr  == ANY_HOST)    &&
       (port  <= 0) )
  {
    return lookup ( type );
  }

  if ( addr == LOCAL_HOST )
  {
    port = jem::max ( 0, port );

    if ( type == Domain::IPv4 )
    {
      struct sockaddr_in  addr4;

      std::memset ( &addr4, 0x0, sizeof(addr4) );

      addr4.sin_family      = AF_INET;
      addr4.sin_port        = (unsigned short) port;
      addr4.sin_addr.s_addr = INADDR_LOOPBACK;

      return newInstance<NativeAddress> (
        (sockaddr_t*) &addr4,
        (socklen_t)   sizeof(addr4)
      );
    }

    if ( type == Domain::IPv6 )
    {
      struct sockaddr_in6  addr6;

      std::memset ( &addr6, 0x0, sizeof(addr6) );

      addr6.sin6_family           = AF_INET6;
      addr6.sin6_port             = (unsigned short) port;
      addr6.sin6_addr.s6_addr[15] = 1;

      return newInstance<NativeAddress> (
        (sockaddr_t*) &addr6,
        (socklen_t)   sizeof(addr6)
      );
    }

    Errors::notInetDomain ( JEM_FUNC );
  }

  if ( Thread::cancelled() )
  {
    Errors::threadCancelled ( JEM_FUNC );
  }

#ifdef JEM_OS_WINDOWS

  WinSockLib::init ();

#endif

  std::memset ( &hints, 0x0, sizeof(hints) );

  hints.ai_flags    = 0;
  hints.ai_socktype = SOCK_STREAM;

  if ( flags & LOCAL_ONLY_ )
  {
    hints.ai_flags |= AI_NUMERICHOST;
  }

  if ( kind == SERVER )
  {
    hints.ai_flags |= AI_PASSIVE;
  }

  if ( addr.size() )
  {
    node  = CString ( addr );
    pnode = node;
  }

  if ( port > 0 )
  {
    service         = CString ( String( port ) );
    pserv           = service;
    hints.ai_flags |= AI_NUMERICSERV;
  }

  if      ( type == Domain::ANY )
  {
    hints.ai_family = AF_UNSPEC;
  }
  else if ( type == Domain::IPv4 )
  {
    hints.ai_family = AF_INET;
  }
  else if ( type == Domain::IPv6 )
  {
    hints.ai_family = AF_INET6;
  }
  else
  {
    Errors::invalidDomain ( JEM_FUNC, (int) type );
  }

  ierr = ::getaddrinfo ( pnode, pserv, &hints, &list );

  if ( ierr )
  {
    String  details;

    ::freeaddrinfo ( list );

    if      ( ierr == EAI_NONAME )
    {
      details = "host not found";
    }
    else if ( ierr == EAI_SERVICE )
    {
      details = "invalid port";
    }
    else
    {
      details = NativeSystem::strerror ();
    }

    Errors::lookupFailed ( JEM_FUNC,
                           toString ( addr, port ),
                           details );
  }

  for ( info = list; info; info = info->ai_next )
  {
    if ( (info->ai_socktype == SOCK_STREAM) &&
         ((info->ai_family == AF_INET)      ||
          (info->ai_family == AF_INET6)) )
    {
      break;
    }
  }

  if ( ! info )
  {
    ::freeaddrinfo       ( list );
    Errors::lookupFailed ( JEM_FUNC, toString( addr, port ) );
  }

  if ( info->ai_family == AF_INET )
  {
    type = Domain::IPv4;
  }
  else
  {
    type = Domain::IPv6;
  }

  try
  {
    saddr = newInstance<NativeAddress> (
      info->ai_addr,
      info->ai_addrlen
    );
  }
  catch ( ... )
  {
    ::freeaddrinfo ( list );
    throw;
  }

  ::freeaddrinfo ( list );

  return saddr;
}


//=======================================================================
//   class NativeAddress
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NativeAddress::NativeAddress

  ( const sockaddr_t*  addr,
    socklen_t          size )

{
  JEM_PRECHECK ( size >  0 );
  JEM_PRECHECK ( size <= (socklen_t)
                 sizeof(struct sockaddr_storage) );

#ifdef JEM_OS_WINDOWS

  WinSockLib::init ();

#endif

  size_ = size;
  type_ = Domain::ANY;
  port_ = ANY_PORT;

  std::memset ( &addr_, 0x0, sizeof(addr_) );
  std::memcpy ( &addr_, addr, size );
}


NativeAddress::~NativeAddress ()
{
  std::memset ( &addr_, 0x0, sizeof(addr_) );

  size_ = 0;
  port_ = ANY_PORT;
}


//-----------------------------------------------------------------------
//   getPort
//-----------------------------------------------------------------------


int NativeAddress::getPort () const
{
  if ( port_ == ANY_PORT )
  {
    const_cast<Self*> ( this ) -> getNameInfo_ ();
  }

  return port_;
}


//-----------------------------------------------------------------------
//   getDomain
//-----------------------------------------------------------------------


Domain::Type NativeAddress::getDomain () const
{
  Domain::Type  type = type_;

  if ( type == Domain::ANY )
  {
    const sockaddr_t*  ap = (const sockaddr_t*) &addr_;

    if      ( ap->sa_family == AF_INET )
    {
      type = Domain::IPv4;
    }
    else if ( ap->sa_family == AF_INET6 )
    {
      type = Domain::IPv6;
    }
    else
    {
      Errors::unknownDomain ( JEM_FUNC );
    }

    const_cast<Self*> ( this ) -> type_ = type;
  }

  return type;
}


//-----------------------------------------------------------------------
//   getHostName
//-----------------------------------------------------------------------


String NativeAddress::getHostName () const
{
  if ( ! host_.size() )
  {
    const_cast<Self*> ( this ) -> getNameInfo_ ();
  }

  return host_;
}


//-----------------------------------------------------------------------
//   getNameInfo_
//-----------------------------------------------------------------------


void NativeAddress::getNameInfo_ ()
{
  char  name[256];
  char  serv[64];
  int   port;
  int   ierr;


  std::memset ( name, 0x0, sizeof(name) );
  std::memset ( serv, 0x0, sizeof(serv) );

  ierr = ::getnameinfo ( addr(),     size_,
                         name, (int) sizeof(name),
                         serv, (int) sizeof(serv),
                         NI_NUMERICHOST | NI_NUMERICSERV );

  if ( ierr )
  {
    Errors::hostLookupFailed ( JEM_FUNC );
  }

  if ( ! Integer::parse( port, String( serv ) ) )
  {
    Errors::portLookupFailed ( JEM_FUNC );
  }

  host_ = jem::clone ( name );
  port_ = port;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   toNativeAddress
//-----------------------------------------------------------------------


const NativeAddress& toNativeAddress ( const SocketAddress& addr )
{
  const NativeAddress*  na =

    dynamic_cast<const NativeAddress*> ( &addr );

  if ( ! na )
  {
    throw Error (
      JEM_FUNC,
      "invalid SocketAddress type; expected an instance of "
      "class jem::net::NativeAddress"
    );
  }

  return *na;
}


JEM_END_PACKAGE( net )

