
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

#ifndef JEM_NET_SOCKETADDRESS_H
#define JEM_NET_SOCKETADDRESS_H

#include <jem/base/Object.h>
#include <jem/net/Domain.h>


JEM_BEGIN_PACKAGE( net )


//-----------------------------------------------------------------------
//   class SocketAddress
//-----------------------------------------------------------------------


class SocketAddress : public Object
{
 public:

  typedef Object          Super;
  typedef SocketAddress   Self;

  static const int        ANY_PORT;
  static const char*      ANY_HOST;
  static const char*      LOCAL_HOST;

  enum                    Kind
  {
                            CLIENT,
                            SERVER
  };

  static Ref<Self>        lookup

    ( Domain::Type          type );

  static Ref<Self>        lookup

    ( Kind                  kind,
      Domain::Type          type,
      const String&         host,
      int                   port );

  static Ref<Self>        convert

    ( Kind                  kind,
      Domain::Type          type,
      const String&         host,
      int                   port );

  virtual int             getPort        () const = 0;
  virtual Domain::Type    getDomain      () const = 0;
  virtual String          getHostName    () const = 0;

  virtual String          toString       () const override;

  static String           toString

    ( const String&         host,
      int                   port );

  static bool             splitString

    ( String&               host,
      int&                  port,
      const String&         addr );


 protected:

  virtual                ~SocketAddress  ();


 private:

  static Ref<Self>        lookup_

    ( Kind                  kind,
      Domain::Type          type,
      const String&         host,
      int                   port,
      int                   flags = 0 );


 private:

  static const int        LOCAL_ONLY_;

};


typedef SocketAddress     SockAddr;


JEM_END_PACKAGE( net )

#endif
