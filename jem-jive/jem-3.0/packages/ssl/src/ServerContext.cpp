
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


#include <jem/ssl/ServerContext.h>
#include "SslLib.h"
#include "NativeContext.h"

extern "C"
{
  #include <openssl/ssl.h>
}


JEM_BEGIN_PACKAGE( ssl )


//=======================================================================
//   class ServerCtxImp
//=======================================================================


class ServerCtxImp : public ServerContext,
                     public NativeContext
{
 public:

  typedef ServerContext   Super;
  typedef ServerCtxImp    Self;


                          ServerCtxImp   ();

  virtual SSL_CTX*        getSslContext  () const override;


 protected:

  virtual                ~ServerCtxImp   ();

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ServerCtxImp::ServerCtxImp ()
{}


ServerCtxImp::~ServerCtxImp ()
{}


//-----------------------------------------------------------------------
//   getSslContext
//-----------------------------------------------------------------------


SSL_CTX* ServerCtxImp::getSslContext () const
{
  return reinterpret_cast<SSL_CTX*> ( getContext_() );
}


//=======================================================================
//   class ServerContext
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ServerContext::ServerContext ()
{
  SSL_CTX*  ctx = SSL_CTX_new ( TLS_server_method() );

  if ( ! ctx )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to create an SSL server context"
    );
  }

  setContext_ ( ctx, SERVER_ROLE_ );
}


ServerContext::~ServerContext ()
{}


//-----------------------------------------------------------------------
//   setName
//-----------------------------------------------------------------------


void ServerContext::setName ( const String& name )
{
  JEM_PRECHECK ( name.size() <= maxNameSize() );

  SSL_CTX*  ctx = reinterpret_cast<SSL_CTX*> ( getContext_() );

  SSL_CTX_set_session_id_context (
    ctx,
    (const unsigned char*) name.addr (),
    (unsigned int)         name.size ()
  );
}


//-----------------------------------------------------------------------
//   maxNameSize
//-----------------------------------------------------------------------


idx_t ServerContext::maxNameSize () const
{
  return (idx_t) SSL_MAX_SSL_SESSION_ID_LENGTH;
}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


Ref<ServerContext> ServerContext::newInstance

  ( Ref<Certificate>  cert,
    Ref<PrivateKey>   pkey )

{
  JEM_PRECHECK ( cert && pkey );

  Ref<Self>  context = jem::newInstance<ServerCtxImp> ();

  context->useCertificate ( cert, pkey );

  return context;
}


JEM_END_PACKAGE( ssl )

