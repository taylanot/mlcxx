
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
#include <jem/base/Time.h>
#include <jem/base/Integer.h>
#include <jem/net/Socket.h>
#include <jem/net/ProxyProtocol.h>
#include <jem/net/NetworkException.h>
#include "NativeAddress.h"

#ifndef JEM_OS_WINDOWS

extern "C"
{
  #include <netinet/in.h>
  #include <sys/types.h>
}

#endif


JEM_BEGIN_PACKAGE( net )


// The ProxyProtocol class parses a PROXY header, if present, in a
// socket input stream. The socket peer address is set to the
// source address provided by the PROXY header.


//=======================================================================
//   class ProxyProtocol::HeaderV1_
//=======================================================================


class ProxyProtocol::HeaderV1_
{
 public:

  char                    line[108];

};


//=======================================================================
//   class ProxyProtocol::HeaderV2_
//=======================================================================


class ProxyProtocol::HeaderV2_
{
 public:

  struct                  Ip4
  {
    byte                    srcAddr[4];
    byte                    dstAddr[4];
    byte                    srcPort[2];
    byte                    dstPort[2];
  };

  struct                  Ip6
  {
    byte                    srcAddr[16];
    byte                    dstAddr[16];
    byte                    srcPort[2];
    byte                    dstPort[2];
  };

  struct                  Unx
  {
    byte                    srcAddr[108];
    byte                    dstAddr[108];
  };


 public:

  byte                    sig[12];
  byte                    cmd;
  byte                    fam;
  byte                    len[2];

  union
  {
    Ip4                   ip4;
    Ip6                   ip6;
    Unx                   unx;
  };

};


//=======================================================================
//   class ProxyProtocol::Header_
//=======================================================================


class ProxyProtocol::Header_
{
 public:

  union
  {
    HeaderV1_             v1;
    HeaderV2_             v2;
  };

};


//=======================================================================
//   class ProxyProtocol
//=======================================================================

//-----------------------------------------------------------------------
//   recvHeader
//-----------------------------------------------------------------------


bool ProxyProtocol::recvHeader ( Socket& sock )
{
  return recvHeader ( sock, -1_sec );
}


bool ProxyProtocol::recvHeader

  ( Socket&      sock,
    const Time&  timeout )

{
  using std::memcmp;
  using std::memchr;

  const byte    v2sig[12] = { 0x0D, 0x0A, 0x0D, 0x0A, 0x00, 0x0D,
                              0x0A, 0x51, 0x55, 0x49, 0x54, 0x0A };
  Ref<Address>  addr;
  Header_       hdr;
  idx_t         k, n;


  n = sock.peek ( (void*) &hdr, (idx_t) sizeof(hdr), timeout );

  if      ( (n >= 16)                              &&
            (memcmp( hdr.v2.sig, v2sig, 12 ) == 0) &&
            ((hdr.v2.cmd & 0xF0) == 0x20) )
  {
    int  size = 16 + (hdr.v2.len[0] << 8) + (int) hdr.v2.len[1];

    if ( n < (idx_t) size )
    {
      badHeader_ ( "truncated version 2 header" );
    }

    n    = (idx_t) size;
    addr = parseHeader2_ ( hdr.v2 );
  }
  else if ( (n >= 8) &&
            (memcmp( hdr.v1.line, "PROXY", 5 ) == 0) )
  {
    char*  end = (char*) memchr ( hdr.v1.line, '\r', (size_t) n - 1 );

    if ( ! end || end[1] != '\n')
    {
      badHeader_ ( "truncated version 1 header" );
    }

    k    = (idx_t) (end - hdr.v1.line);
    n    = k + 2;
    addr = parseHeader1_ ( String( hdr.v1.line, k ) );
  }
  else
  {
    return false;
  }

  // Remove the header data from the socket input stream.

  k = sock.read ( (void*) &hdr, n, timeout );

  if ( k < n )
  {
    throw NetworkException (
      JEM_FUNC,
      "failed to extract the PROXY header from the "
      "socket input stream"
    );
  }

  if ( addr )
  {
    sock.setPeerAddress ( addr );
  }

  return true;
}


//-----------------------------------------------------------------------
//   badHeader_
//-----------------------------------------------------------------------


void ProxyProtocol::badHeader_ ( const String& what )
{
  String  msg = "invalid PROXY header";

  if ( what.size() )
  {
    msg = String::format ( "%s: %s", msg, what );
  }

  throw NetworkException ( JEM_FUNC, msg );
}


//-----------------------------------------------------------------------
//   parseHeader1_
//-----------------------------------------------------------------------


Ref<SocketAddress>
  ProxyProtocol::parseHeader1_ ( const String& hdr )
{
  const idx_t   len  = hdr.size ();

  Domain::Type  type = Domain::ANY;
  idx_t         ipos = 5;
  idx_t         jpos = ipos;
  int           port = -1;

  String        fam;
  String        addr;


  // Extract the protocol family.

  if ( (ipos >= len) || (hdr[ipos] != ' ' ) )
  {
    badHeader_ ( "missing protocol family name" );
  }

  ipos++;

  for ( jpos = ipos; jpos < len; jpos++ )
  {
    if ( hdr[jpos] == ' ' )
    {
      break;
    }
  }

  fam = hdr[slice(ipos,jpos)];

  if      ( fam == "TCP4" )
  {
    type = Domain::IPv4;
  }
  else if ( fam == "TCP6" )
  {
    type = Domain::IPv6;
  }
  else if ( fam == "UNKNOWN" )
  {
    // Keep the current peer address.

    return nullptr;
  }
  else
  {
    badHeader_ ( "invalid protocol family name: " + fam );
  }

  // Extract the source address.

  ipos = jpos + 1;

  if ( ipos >= len )
  {
    badHeader_ ( "missing source address" );
  }

  for ( jpos = ipos; jpos < len; jpos++ )
  {
    if ( hdr[jpos] == ' ' )
    {
      break;
    }
  }

  addr = hdr[slice(ipos,jpos)];

  // Skip over the destination address.

  ipos = jpos + 1;

  if ( ipos >= len )
  {
    badHeader_ ( "missing destination address" );
  }

  for ( jpos = ipos; jpos < len; jpos++ )
  {
    if ( hdr[jpos] == ' ' )
    {
      break;
    }
  }

  // Extract the port number.

  ipos = jpos + 1;

  if ( ipos >= len )
  {
    badHeader_ ( "missing source port number" );
  }

  for ( jpos = ipos; jpos < len; jpos++ )
  {
    if ( hdr[jpos] == ' ' )
    {
      break;
    }
  }

  if ( ! Integer::parse( port, hdr[slice(ipos,jpos)] ) ||
       (port < 0) || (port > 65535) )
  {
    badHeader_ ( "invalid source port number: " +
                 hdr[slice(ipos,jpos)] );
  }

  return Address::convert ( Address::CLIENT, type, addr, port );
}


//-----------------------------------------------------------------------
//   parseHeader2_
//-----------------------------------------------------------------------


Ref<SocketAddress>
  ProxyProtocol::parseHeader2_ ( const HeaderV2_& hdr )
{
  using std::memcpy;
  using std::memset;

  const byte  cmd = hdr.cmd & 0xF;

  if ( cmd == 0x01 )
  {
    // PROXY command.

    if ( hdr.fam == 0x11 )
    {
      // TCPv4 protocol family.

      struct sockaddr_in  addr;

      memset ( &addr, 0x0, sizeof(addr) );

      addr.sin_family = AF_INET;

      memcpy ( &addr.sin_addr, hdr.ip4.srcAddr, 4 );
      memcpy ( &addr.sin_port, hdr.ip4.srcPort, 2 );

      return newInstance<NativeAddress> (
        (sockaddr_t*) &addr,
        (socklen_t)   sizeof(addr)
      );
    }

    if ( hdr.fam == 0x21 )
    {
      // TCPv6 protocol family.

      struct sockaddr_in6  addr;

      memset ( &addr, 0x0, sizeof(addr) );

      addr.sin6_family = AF_INET6;

      memcpy ( &addr.sin6_addr, hdr.ip6.srcAddr, 16 );
      memcpy ( &addr.sin6_port, hdr.ip6.srcPort,  2 );

      return newInstance<NativeAddress> (
        (sockaddr_t*) &addr,
        (socklen_t)   sizeof(addr)
      );
    }

    // Unsupported protocol; keep the current address.

    return nullptr;
  }

  if ( cmd == 0x00 )
  {
    // LOCAL command. Keep the current address.

    return nullptr;
  }

  badHeader_ ( "invalid version 2 command" );

  return nullptr;
}


JEM_END_PACKAGE( net )

