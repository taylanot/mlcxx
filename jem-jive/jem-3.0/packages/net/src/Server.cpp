
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
#include <jem/net/Server.h>
#include <jem/net/Resolver.h>
#include "NativeServer.h"
#include "NativeAddress.h"


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class Server
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


Server::Server ()
{
  server_ = newInstance<NativeServer> ();
}


//-----------------------------------------------------------------------
//   listen
//-----------------------------------------------------------------------


idx_t Server::listen

  ( const Address&  addr,
    int             backlog )

{
  return server_->listen ( toNativeAddress( addr ), backlog );
}


idx_t Server::listen

  ( Domain::Type  type,
    int           backlog )

{
  Ref<Address>  addr = Address::lookup ( type );

  return server_->listen ( toNativeAddress( *addr ), backlog );
}


idx_t Server::listen

  ( Domain::Type   type,
    const String&  host,
    int            port,
    int            backlog )

{
  Ref<Address>  addr =

    Resolver::lookup ( Address::SERVER, type, host, port );

  return server_->listen ( toNativeAddress( *addr ), backlog );
}


idx_t Server::listen

  ( Domain::Type   type,
    const String&  host,
    int            port,
    int            backlog,
    const Time&    timeout )

{
  Ref<Address>  addr =

    Resolver::lookup ( Address::SERVER, type, host, port,
                       timeout );

  return server_->listen ( toNativeAddress( *addr ), backlog );
}


//-----------------------------------------------------------------------
//   accept
//-----------------------------------------------------------------------


Ref<Socket> Server::accept ( idx_t idx )
{
  return server_->accept ( idx );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void Server::close ( idx_t idx )
{
  server_->close ( idx );
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t Server::poll ( const Time& timeout )
{
  return server_->poll ( timeout );
}


//-----------------------------------------------------------------------
//   isListening
//-----------------------------------------------------------------------


bool Server::isListening () const
{
  return server_->isListening ();
}


bool Server::isListening ( idx_t idx ) const
{
  return server_->isListening ( idx );
}


//-----------------------------------------------------------------------
//   getLocalAddress
//-----------------------------------------------------------------------


Ref<SocketAddress> Server::getLocalAddress ( idx_t idx ) const
{
  return server_->getLocalAddress ( idx );
}


JEM_END_PACKAGE( net )

