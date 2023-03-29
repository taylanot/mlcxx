
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

#ifndef JEM_SSL_SSLSOCKET_H
#define JEM_SSL_SSLSOCKET_H

#include <jem/base/array/Array.h>
#include <jem/net/Socket.h>
#include <jem/ssl/import.h>


JEM_BEGIN_PACKAGE( ssl )


class SslStream;
class SslContext;
class SslSession;


//-----------------------------------------------------------------------
//   class SslSocket
//-----------------------------------------------------------------------


class SslSocket : public Socket
{
 public:

  typedef Socket          Super;
  typedef SslSocket       Self;

                          SslSocket

    ( Ref<Socket>           sock,
      Ref<SslContext>       context );

  virtual Ref<Socket>     dup              ()          override;

  virtual void            close

    ( const Time&           timeout )                  override;

  virtual void            flush

    ( const Time&           timeout )                  override;

  virtual int             poll

    ( int                   what,
      const Time&           timeout )                  override;

  virtual idx_t           peek

    ( void*                 buf,
      idx_t                 len,
      const Time&           timeout )                  override;

  virtual idx_t           read

    ( void*                 buf,
      idx_t                 len,
      const Time&           timeout )                  override;

  virtual idx_t           write

    ( const void*           buf,
      idx_t                 len,
      const Time&           timeout )                  override;

  virtual idx_t           writeNoThrow

    ( const void*           buf,
      idx_t                 len )             noexcept override;

  virtual bool            getNoDelay       () const    override;

  virtual void            setNoDelay

    ( bool                  choice )                   override;

  virtual Ref<Address>    getPeerAddress   () const    override;

  virtual void            setPeerAddress

    ( const Ref<Address>&   addr )                     override;

  virtual Ref<Address>    getLocalAddress  () const    override;
  virtual Ref<Handle>     getSocketHandle  () const    override;

  Ref<SslContext>         getContext       () const;
  Ref<SslSession>         getSession       () const;
  Ref<Certificate>        getPeerCert      () const;

  void                    doHandshake

    ( const Time&           timeout,
      Ref<SslSession>       session = nullptr );

  bool                    isEncrypted      () const;

  void                    getKeyMaterial

    ( const Array<byte>&    buf,
      const String&         label,
      const Array<byte>&    context )         const;

  String                  getCipherName    () const;


 protected:

  virtual                ~SslSocket        ();


 private:

  Ref<Socket>             socket_;
  Ref<SslContext>         context_;
  Ref<SslStream>          sstream_;

};


JEM_END_PACKAGE( ssl )

#endif

