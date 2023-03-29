
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

#ifndef JEM_NET_NATIVE_NATIVEADDRESS_H
#define JEM_NET_NATIVE_NATIVEADDRESS_H

#include <jem/net/SocketAddress.h>

#ifdef JEM_OS_WINDOWS

#include "WinSockLib.h"

#else

extern "C"
{
  #include <sys/socket.h>
}

#endif


JEM_BEGIN_PACKAGE( net )


typedef struct sockaddr   sockaddr_t;


//-----------------------------------------------------------------------
//   class NativeAddress
//-----------------------------------------------------------------------


class NativeAddress : public SocketAddress
{
 public:

  typedef SocketAddress   Super;
  typedef NativeAddress   Self;

                          NativeAddress

    ( const sockaddr_t*     addr,
      socklen_t             size );

  inline socklen_t        size            () const noexcept;
  inline
    const sockaddr_t*     addr            () const noexcept;

  virtual int             getPort         () const override;
  virtual Domain::Type    getDomain       () const override;
  virtual String          getHostName     () const override;


 protected:

  virtual                ~NativeAddress  ();


 private:

  void                    getNameInfo_   ();


 private:

  struct
    sockaddr_storage      addr_;
  socklen_t               size_;
  Domain::Type            type_;
  int                     port_;
  String                  host_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


const NativeAddress&      toNativeAddress

  ( const SocketAddress&    addr );


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline socklen_t NativeAddress::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


inline const sockaddr_t* NativeAddress::addr () const noexcept
{
  return (const sockaddr_t*) &addr_;
}


JEM_END_PACKAGE( net )

#endif
