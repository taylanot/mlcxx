
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

#ifndef JEM_SSL_SSLCLIENT_H
#define JEM_SSL_SSLCLIENT_H

#include <jem/net/Client.h>
#include <jem/ssl/SslSocket.h>
#include <jem/ssl/ClientContext.h>


JEM_BEGIN_PACKAGE( ssl )


//-----------------------------------------------------------------------
//   class SslClient
//-----------------------------------------------------------------------


class SslClient : public Client
{
 public:

  typedef Client          Super;
  typedef SslClient       Self;

  static Ref<SslSocket>   connect

    ( const Address&        addr,
      Ref<ClientContext>    context );

  static Ref<SslSocket>   connect

    ( const Address&        addr,
      Ref<ClientContext>    context,
      const Time&           timeout );

  static Ref<SslSocket>   connect

    ( Domain::Type          type,
      const String&         host,
      int                   port,
      Ref<ClientContext>    context );

  static Ref<SslSocket>   connect

    ( Domain::Type          type,
      const String&         host,
      int                   port,
      Ref<ClientContext>    context,
      const Time&           timeout );

};


JEM_END_PACKAGE( ssl )

#endif

