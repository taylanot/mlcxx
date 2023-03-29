
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

#ifndef JEM_NET_SOCKET_H
#define JEM_NET_SOCKET_H

#include <jem/base/forward.h>
#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE( net )


class SocketHandle;
class SocketAddress;


//-----------------------------------------------------------------------
//   class Socket
//-----------------------------------------------------------------------


class Socket : public Object
{
 public:

  typedef Object          Super;
  typedef Socket          Self;

  typedef SocketHandle    Handle;
  typedef SocketAddress   Address;

  static const int        POLL_READ;
  static const int        POLL_WRITE;


  virtual Ref<Socket>     dup              ()          = 0;

  virtual void            close

    ( const Time&           timeout )                  = 0;

  virtual void            flush

    ( const Time&           timeout )                  = 0;

  virtual int             poll

    ( int                   what,
      const Time&           timeout )                  = 0;

  virtual idx_t           peek

    ( void*                 buf,
      idx_t                 len,
      const Time&           timeout )                  = 0;

  virtual idx_t           read

    ( void*                 buf,
      idx_t                 len,
      const Time&           timeout )                  = 0;

  void                    readAll

    ( void*                 buf,
      idx_t                 len,
      const Time&           timeout );

  virtual idx_t           write

    ( const void*           buf,
      idx_t                 len,
      const Time&           timeout )                  = 0;

  void                    writeAll

    ( const void*           buf,
      idx_t                 len,
      const Time&           timeout );

  virtual idx_t           writeNoThrow

    ( const void*           buf,
      idx_t                 len )             noexcept = 0;

  virtual bool            getNoDelay       () const    = 0;

  virtual void            setNoDelay

    ( bool                  choice )                   = 0;

  virtual Ref<Address>    getPeerAddress   () const    = 0;

  virtual void            setPeerAddress

    ( const Ref<Address>&   addr )                     = 0;

  virtual Ref<Address>    getLocalAddress  () const    = 0;
  virtual Ref<Handle>     getSocketHandle  () const    = 0;

  virtual String          toString         () const override;


 protected:

  virtual                ~Socket          ();

};


JEM_END_PACKAGE( net )

#endif

