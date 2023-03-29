
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

#ifndef JEM_SSL_SSLCONTEXT_H
#define JEM_SSL_SSLCONTEXT_H

#include <jem/base/Object.h>
#include <jem/ssl/import.h>


JEM_BEGIN_PACKAGE( ssl )


//-----------------------------------------------------------------------
//   class SslContext
//-----------------------------------------------------------------------


class SslContext : public Object
{
 public:

  typedef Object          Super;
  typedef SslContext      Self;

  enum                    SslVersion
  {
                            TLS_1,
                            TLS_1_1,
                            TLS_1_2,
                            TLS_1_3
  };

  enum                    VerifyMode
  {
                            NO_VERIFY,
                            QUERY_PEER,
                            VERIFY_PEER
  };

  inline bool             isClientContext  () const noexcept;
  inline bool             isServerContext  () const noexcept;

  void                    useCertificate

    ( Ref<Certificate>      cert,
      Ref<PrivateKey>       pkey );

  Ref<Certificate>        getCertificate   () const;

  void                    setVerifyMode

    ( VerifyMode            mode );

  VerifyMode              getVerifyMode    () const;

  void                    setMinVersion

    ( SslVersion            vers );

  void                    setMaxVersion

    ( SslVersion            vers );

  SslVersion              getMinVersion    () const;
  SslVersion              getMaxVersion    () const;

  void                    setCaCertDir

    ( const String&         path );

  void                    loadCaCertFile

    ( const String&         path );

  void                    addCaCertificate

    ( Ref<Certificate>      cert );

  void                    addClientCaCert

    ( Ref<Certificate>      cert );

  void                    loadClientCaCerts

    ( const String&         path );

  static bool             isSupported

    ( SslVersion            vers );


 protected:

  enum                    Role_
  {
                            NO_ROLE_,
                            CLIENT_ROLE_,
                            SERVER_ROLE_
  };

                          SslContext       ();
  virtual                ~SslContext       ();

  void                    setContext_

    ( void*                 context,
      Role_                 role );

  inline void*            getContext_      () const noexcept;


 private:

  void*                   context_;
  VerifyMode              verify_;
  Role_                   role_;
  Ref<Certificate>        cert_;

};


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isClientContext
//-----------------------------------------------------------------------


inline bool SslContext::isClientContext () const noexcept
{
  return (role_ == CLIENT_ROLE_);
}


//-----------------------------------------------------------------------
//   isServerContext
//-----------------------------------------------------------------------


inline bool SslContext::isServerContext () const noexcept
{
  return (role_ == SERVER_ROLE_);
}


//-----------------------------------------------------------------------
//   getContext_
//-----------------------------------------------------------------------


inline void* SslContext::getContext_ () const noexcept
{
  return context_;
}


JEM_END_PACKAGE( ssl )

#endif

