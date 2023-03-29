
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


#include <jem/base/Error.h>
#include <jem/base/Thread.h>
#include <jem/base/CString.h>
#include <jem/base/CancelledException.h>
#include <jem/crypto/native/PrivateKey.h>
#include <jem/crypto/native/Certificate.h>
#include <jem/ssl/SslContext.h>
#include <jem/ssl/SslException.h>
#include "SslLib.h"

extern "C"
{
  #include <openssl/ssl.h>
}


#ifdef JEM_SSL_CTX_CAST
#  undef JEM_SSL_CTX_CAST
#endif

#define JEM_SSL_CTX_CAST( ctx )      \
                                     \
  reinterpret_cast<SSL_CTX*> ( ctx )


//=======================================================================
//   non-member utility functions
//=======================================================================


extern "C"
{
  static int              jem_ssl_no_verify

    ( int                   preverify_ok,
      X509_STORE_CTX*       x509_ctx )

  {
    return 1;
  }
}


JEM_BEGIN_PACKAGE( ssl )


//=======================================================================
//   class SslContext
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SslContext::SslContext ()
{
  context_ = nullptr;
  verify_  = VERIFY_PEER;
  role_    = NO_ROLE_;
}


SslContext::~SslContext ()
{
  if ( context_ )
  {
    SSL_CTX_free ( JEM_SSL_CTX_CAST( context_ ) );

    context_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   useCertificate
//-----------------------------------------------------------------------


void SslContext::useCertificate

  ( Ref<Certificate>  cert,
    Ref<PrivateKey>   key )

{
  JEM_PRECHECK ( cert && key );

  int  ierr;

  ierr = SSL_CTX_use_certificate (
    JEM_SSL_CTX_CAST    ( context_ ),
    crypto::getX509Cert ( *cert )
  );

  if ( 1 != ierr )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to associate an X509 certificate with an SSL context"
    );
  }

  ierr = SSL_CTX_use_PrivateKey (
    JEM_SSL_CTX_CAST  ( context_ ),
    crypto::getEvpKey ( *key )
  );

  if ( 1 != ierr )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to associate a private key with an SSL context"
    );
  }

  ierr = SSL_CTX_check_private_key ( JEM_SSL_CTX_CAST( context_ ) );

  if ( 1 != ierr )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "mismatch between the specified X509 certificate "
      "and its private key"
    );
  }

  cert_ = cert;
}


//-----------------------------------------------------------------------
//   getCertificate
//-----------------------------------------------------------------------


Ref<Certificate> SslContext::getCertificate () const
{
  return cert_;
}


//-----------------------------------------------------------------------
//   setVerifyMode
//-----------------------------------------------------------------------


void SslContext::setVerifyMode ( VerifyMode mode )
{
  int  flags = 0;

  switch ( mode )
  {
  case NO_VERIFY:

    flags = SSL_VERIFY_NONE;
    break;

  case QUERY_PEER:

    flags = SSL_VERIFY_PEER;

    if ( role_ == SERVER_ROLE_ )
    {
      flags |= SSL_VERIFY_CLIENT_ONCE;
    }

    break;

  case VERIFY_PEER:

    flags = SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT;
    break;

  default:

    throw Error (
      JEM_FUNC,
      "unexpected verify mode: " + String ( (int) mode )
    );
  }

  if ( mode == VERIFY_PEER )
  {
    SSL_CTX_set_verify (
      JEM_SSL_CTX_CAST ( context_ ),
      flags,
      nullptr
    );
  }
  else
  {
    SSL_CTX_set_verify (
      JEM_SSL_CTX_CAST ( context_ ),
      flags,
      jem_ssl_no_verify
    );
  }

  verify_ = mode;
}


//-----------------------------------------------------------------------
//   getVerifyMode
//-----------------------------------------------------------------------


SslContext::VerifyMode SslContext::getVerifyMode () const
{
  return verify_;
}


//-----------------------------------------------------------------------
//   setMinVersion
//-----------------------------------------------------------------------


void SslContext::setMinVersion ( SslVersion vers )
{
  int  level = 0;
  int  ierr  = 0;

  switch ( vers )
  {
  case TLS_1:   level = TLS1_VERSION;   break;
  case TLS_1_1: level = TLS1_1_VERSION; break;
  case TLS_1_2: level = TLS1_2_VERSION; break;

  case TLS_1_3:

#ifdef TLS1_3_VERSION
                level = TLS1_3_VERSION; break;
#else
    throw SslException (
      JEM_FUNC,
      "TLS 1.3 is not supported by the OpenSSL library that "
      "is being used; consider upgrading to a new release of "
      "the OpenSSL library"
    );
#endif

  default:

    throw Error (
      JEM_FUNC,
      "unexpected SSL protocol version: " + String ( (int) vers )
    );
  }

  ierr = (int) SSL_CTX_set_min_proto_version (
    JEM_SSL_CTX_CAST ( context_ ),
    level
  );

  if ( 1 != ierr )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to set the minimum SSL protocol version"
    );
  }
}


//-----------------------------------------------------------------------
//   setMaxVersion
//-----------------------------------------------------------------------


void SslContext::setMaxVersion ( SslVersion vers )
{
  int  level = 0;
  int  ierr  = 0;

  switch ( vers )
  {
  case TLS_1:   level = TLS1_VERSION;   break;
  case TLS_1_1: level = TLS1_1_VERSION; break;
  case TLS_1_2: level = TLS1_2_VERSION; break;

  case TLS_1_3:

#ifdef TLS1_3_VERSION
                level = TLS1_3_VERSION; break;
#else
    return;
#endif

  default:

    throw Error (
      JEM_FUNC,
      "unexpected SSL protocol version: " + String ( (int) vers )
    );
  }

  ierr = (int) SSL_CTX_set_max_proto_version (
    JEM_SSL_CTX_CAST ( context_ ),
    level
  );

  if ( 1 != ierr )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to set the maximum SSL protocol version"
    );
  }
}


//-----------------------------------------------------------------------
//   getMinVersion
//-----------------------------------------------------------------------


SslContext::SslVersion SslContext::getMinVersion () const
{
  SslVersion  vers  = TLS_1;
  int         level = (int)

    SSL_CTX_get_min_proto_version (
      JEM_SSL_CTX_CAST ( context_ )
    );

  switch ( level )
  {
  case TLS1_VERSION:   vers = TLS_1;   break;
  case TLS1_1_VERSION: vers = TLS_1_1; break;
  case TLS1_2_VERSION: vers = TLS_1_2; break;

#ifdef TLS1_3_VERSION
  case TLS1_3_VERSION: vers = TLS_1_3; break;
#endif

  default:

    throw Error (
      JEM_FUNC,
      "unexpected SSL protocol version: " + String ( level )
    );
  }

  return vers;
}


//-----------------------------------------------------------------------
//   getMaxVersion
//-----------------------------------------------------------------------


SslContext::SslVersion SslContext::getMaxVersion () const
{
  SslVersion  vers  = TLS_1;
  int         level = (int)

    SSL_CTX_get_max_proto_version (
      JEM_SSL_CTX_CAST ( context_ )
    );

  switch ( level )
  {
  case TLS1_VERSION:   vers = TLS_1;   break;
  case TLS1_1_VERSION: vers = TLS_1_1; break;
  case TLS1_2_VERSION: vers = TLS_1_2; break;

#ifdef TLS1_3_VERSION
  case TLS1_3_VERSION: vers = TLS_1_3; break;
#endif

  default:

    throw Error (
      JEM_FUNC,
      "unexpected SSL protocol version: " + String ( level )
    );
  }

  return vers;
}


//-----------------------------------------------------------------------
//   setCaCertDir
//-----------------------------------------------------------------------


void SslContext::setCaCertDir ( const String& path )
{
  SSL_CTX*  ctx  = JEM_SSL_CTX_CAST ( context_ );
  int       ierr = 0;


  if ( Thread::cancelled() )
  {
    throw CancelledException ( JEM_FUNC );
  }

  if ( ! path.size() )
  {
    ierr = SSL_CTX_set_default_verify_dir ( ctx );
  }
  else
  {
    ierr = SSL_CTX_load_verify_locations (
      ctx,
      nullptr,
      makeCString ( path )
    );
  }

  if ( 1 != ierr )
  {
    String  msg;

    if ( ! path.size() )
    {
      msg = "failed to set the default CA certificate directory";
    }
    else
    {
      msg = String::format (
        "failed to set the CA certificate directory `%s\'",
        path
      );
    }

    SslLib::raiseError ( JEM_FUNC, msg );
  }
}


//-----------------------------------------------------------------------
//   loadCaCertFile
//-----------------------------------------------------------------------


void SslContext::loadCaCertFile ( const String& path )
{
  SSL_CTX*  ctx  = JEM_SSL_CTX_CAST ( context_ );
  int       ierr = 0;


  if ( Thread::cancelled() )
  {
    throw CancelledException ( JEM_FUNC );
  }

  if ( ! path.size() )
  {
    ierr = SSL_CTX_set_default_verify_file ( ctx );
  }
  else
  {
    ierr = SSL_CTX_load_verify_locations (
      ctx,
      makeCString ( path ),
      nullptr
    );
  }

  if ( 1 != ierr )
  {
    String  msg;

    if ( ! path.size() )
    {
      msg = "failed to load the default CA certificate file";
    }
    else
    {
      msg = String::format (
        "failed to load the CA certificate file `%s\'",
        path
      );
    }

    SslLib::raiseError ( JEM_FUNC, msg );
  }
}


//-----------------------------------------------------------------------
//   addCaCertificate
//-----------------------------------------------------------------------


void SslContext::addCaCertificate ( Ref<Certificate> cert )
{
  JEM_PRECHECK ( cert );

  X509*        xcert = getX509Cert ( *cert );

  X509_STORE*  store =

    SSL_CTX_get_cert_store ( JEM_SSL_CTX_CAST( context_ ) );


  if ( ! store )
  {
    throw SslException (
      JEM_FUNC,
      "failed to get the certificate store"
    );
  }

  if ( 1 != X509_up_ref( xcert ) )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to increase the reference count of an X509 "
      "certificate object"
    );
  }

  if ( 1 != X509_STORE_add_cert( store, xcert ) )
  {
    X509_free ( xcert );

    SslLib::raiseError (
      JEM_FUNC,
      "failed to add a CA certificate to an SSL context"
    );
  }
}


//-----------------------------------------------------------------------
//   addClientCaCert
//-----------------------------------------------------------------------


void SslContext::addClientCaCert ( Ref<Certificate> cert )
{
  JEM_PRECHECK ( cert );

  X509*  xcert = getX509Cert ( *cert );
  int    ierr  =

    SSL_CTX_add_client_CA ( JEM_SSL_CTX_CAST( context_ ), xcert );

  if ( 1 != ierr )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to add a client CA certificate to an SSL context"
    );
  }
}


//-----------------------------------------------------------------------
//   loadClientCaCerts
//-----------------------------------------------------------------------


void SslContext::loadClientCaCerts ( const String& path )
{
  SSL_CTX*              ctx  = JEM_SSL_CTX_CAST ( context_ );
  STACK_OF(X509_NAME)*  list = nullptr;


  if ( Thread::cancelled() )
  {
    throw CancelledException ( JEM_FUNC );
  }

  list = SSL_load_client_CA_file ( makeCString( path ) );

  if ( ! list )
  {
    SslLib::raiseError (
      JEM_FUNC,
      String::format (
        "failed to load client CA certificates from the "
        "file `%s\'",
        path
      )
    );
  }

  SSL_CTX_set_client_CA_list ( ctx, list );
}


//-----------------------------------------------------------------------
//   isSupported
//-----------------------------------------------------------------------


bool SslContext::isSupported ( SslVersion vers )
{
#ifndef TLS1_3_VERSION

  if ( vers == TLS_1_3 )
  {
    return false;
  }

#endif

  return true;
}


//-----------------------------------------------------------------------
//   setContext_
//-----------------------------------------------------------------------


void SslContext::setContext_

  ( void*             context,
    Role_             role )

{
  JEM_PRECHECK ( context );

  if ( context_ )
  {
    SSL_CTX_free ( JEM_SSL_CTX_CAST( context_ ) );

    context_ = nullptr;
  }

  context_ = context;
  role_    = role;

  try
  {
    setMinVersion ( TLS_1 );

    if ( role_ == CLIENT_ROLE_ )
    {
      setVerifyMode ( VERIFY_PEER );
    }
    else
    {
      setVerifyMode ( QUERY_PEER );
    }
  }
  catch ( ... )
  {
    SSL_CTX_free ( JEM_SSL_CTX_CAST( context_ ) );

    context_ = nullptr;

    throw;
  }
}


JEM_END_PACKAGE( ssl )

