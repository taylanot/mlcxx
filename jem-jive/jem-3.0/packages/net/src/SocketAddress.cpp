
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


#include <jem/base/Integer.h>
#include <jem/net/SocketAddress.h>


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class SocketAddress
//=======================================================================

// Note that the function lookup() is implemented by the
// NativeSockAddr class.

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int    SocketAddress::ANY_PORT   = -1;
const char*  SocketAddress::ANY_HOST   = "";
const char*  SocketAddress::LOCAL_HOST = "$";


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


SocketAddress::~SocketAddress ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String SocketAddress::toString () const
{
  String  s;

  try
  {
    s = toString ( getHostName(), getPort() );
  }
  catch ( ... )
  {
    s = "<unknown address>";
  }

  return s;
}


String SocketAddress::toString

  ( const String&  host,
    int            port )

{
  if       ( (host.size() == 0) && (port <= 0) )
  {
    return String ( "*:*" );
  }
  else if ( host.size() == 0 )
  {
    return ("*:" + String( port ) );
  }
  else if ( port <= 0 )
  {
    return (host + ":*");
  }
  else
  {
    return String::format ( "%s:%d", host, port );
  }
}


//-----------------------------------------------------------------------
//   splitString
//-----------------------------------------------------------------------


bool SocketAddress::splitString

  (  String&        host,
     int&           port,
     const String&  addr )

{
  idx_t  j = addr.rfind ( ':' );

  port = 0;

  if ( j >= 0 )
  {
    String  ps = addr[slice(j + 1,END)].stripWhite ();

    host = addr[slice(BEGIN,j)].stripWhite ();

    if ( ps.size() > 0 )
    {
      return Integer::parse ( port, ps );
    }
  }
  else
  {
    host = addr;
  }

  return true;
}


JEM_END_PACKAGE ( net )

