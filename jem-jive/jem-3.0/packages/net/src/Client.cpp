
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
#include <jem/net/Client.h>
#include <jem/net/Resolver.h>
#include "NativeSocket.h"
#include "NativeAddress.h"


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class Client
//=======================================================================

//-----------------------------------------------------------------------
//   connect
//-----------------------------------------------------------------------


Ref<Socket> Client::connect ( const Address& addr )
{
  return NativeSocket::connect (
    toNativeAddress ( addr ), -1_sec
  );
}


Ref<Socket> Client::connect

  ( const Address&  addr,
    const Time&     timeout )

{
  return NativeSocket::connect (
    toNativeAddress ( addr ), timeout
  );
}


Ref<Socket> Client::connect

  ( Domain::Type   type,
    const String&  host,
    int            port )

{
  return connect ( type, host, port, -1_sec );
}


Ref<Socket> Client::connect

  ( Domain::Type   type,
    const String&  host,
    int            port,
    const Time&    timeout )

{
  Ref<Address>  addr;
  Time          left;

  if ( timeout < Time::zero() )
  {
    addr = Resolver::lookup ( Address::CLIENT, type, host, port );
    left = timeout;
  }
  else
  {
    Time  end = timeout + Time::now ();

    addr = Resolver::lookup ( Address::CLIENT, type,
                              host, port, timeout );
    left = end - Time::now ();

    if ( left < Time::zero() )
    {
      left = Time::zero ();
    }
  }

  return NativeSocket::connect ( toNativeAddress( *addr ), left );
}


JEM_END_PACKAGE( net )

