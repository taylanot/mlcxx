
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


#include <jem/ssl/ClientContext.h>
#include "SslLib.h"
#include "NativeContext.h"


JEM_BEGIN_PACKAGE( ssl )


//=======================================================================
//   class ClientCtxImp
//=======================================================================


class ClientCtxImp : public ClientContext,
                     public NativeContext
{
 public:

  typedef ClientContext   Super;
  typedef ClientCtxImp    Self;


                          ClientCtxImp   ();

  virtual SSL_CTX*        getSslContext  () const override;


 protected:

  virtual                ~ClientCtxImp   ();

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ClientCtxImp::ClientCtxImp ()
{}


ClientCtxImp::~ClientCtxImp ()
{}


//-----------------------------------------------------------------------
//   getSslContext
//-----------------------------------------------------------------------


SSL_CTX* ClientCtxImp::getSslContext () const
{
  return reinterpret_cast<SSL_CTX*> ( getContext_() );
}


//=======================================================================
//   class ClientContext
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ClientContext::ClientContext ()
{
  SSL_CTX*  ctx = SSL_CTX_new ( TLS_client_method() );

  if ( ! ctx )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to create an SSL client context"
    );
  }

  setContext_ ( ctx, CLIENT_ROLE_ );
}


ClientContext::~ClientContext ()
{}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


Ref<ClientContext> ClientContext::newInstance ()
{
  return jem::newInstance<ClientCtxImp> ();
}


Ref<ClientContext> ClientContext::newInstance

  ( Ref<Certificate>  cert,
    Ref<PrivateKey>   pkey )

{
  JEM_PRECHECK ( cert && pkey );

  Ref<Self>  context = jem::newInstance<ClientCtxImp> ();

  context->useCertificate ( cert, pkey );

  return context;
}


JEM_END_PACKAGE( ssl )

