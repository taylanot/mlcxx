
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


#include <jem/base/IllegalOperationException.h>
#include <jem/base/array/utilities.h>
#include <jem/ssl/SslSocket.h>
#include <jem/ssl/VerifyException.h>
#include "SslLib.h"
#include "SslStream.h"
#include "NativeSession.h"


JEM_BEGIN_PACKAGE( ssl )


//=======================================================================
//   class SslSocket
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SslSocket::SslSocket

  ( Ref<Socket>      sock,
    Ref<SslContext>  context ) :

    socket_  ( sock ),
    context_ ( context )

{
  JEM_PRECHECK ( sock && context );
}


SslSocket::~SslSocket ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Socket> SslSocket::dup ()
{
  throw IllegalOperationException (
    JEM_FUNC,
    "an SSL socket can not be duplicated"
  );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void SslSocket::close ( const Time& timeout )
{
  if ( sstream_ )
  {
    sstream_->close ( timeout );

    sstream_ = nullptr;
  }

  socket_->close ( timeout );
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void SslSocket::flush ( const Time& timeout )
{
  // Nothing to be done.
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


int SslSocket::poll

  ( int          what,
    const Time&  timeout )

{
  if ( sstream_ && (what & POLL_READ) )
  {
    if ( sstream_->pending() )
    {
      return POLL_READ;
    }
  }

  return socket_->poll ( what, timeout );
}


//-----------------------------------------------------------------------
//   peek
//-----------------------------------------------------------------------


idx_t SslSocket::peek

  ( void*        buf,
    idx_t        len,
    const Time&  timeout )

{
  if ( len <= 0 )
  {
    return 0;
  }
  else
  {
    if ( sstream_ )
    {
      return sstream_->peek ( buf, len, timeout );
    }
    else
    {
      return socket_ ->peek ( buf, len, timeout );
    }
  }
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t SslSocket::read

  ( void*        buf,
    idx_t        len,
    const Time&  timeout )

{
  if ( len <= 0 )
  {
    return 0;
  }
  else
  {
    if ( sstream_ )
    {
      return sstream_->read ( buf, len, timeout );
    }
    else
    {
      return socket_ ->read ( buf, len, timeout );
    }
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


idx_t SslSocket::write

  ( const void*  buf,
    idx_t        len,
    const Time&  timeout )

{
  if ( len <= 0 )
  {
    return 0;
  }
  else
  {
    if ( sstream_ )
    {
      return sstream_->write ( buf, len, timeout );
    }
    else
    {
      return socket_ ->write ( buf, len, timeout );
    }
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


idx_t SslSocket::writeNoThrow

  ( const void*  buf,
    idx_t        len ) noexcept

{
  if ( sstream_ )
  {
    return sstream_->writeNoThrow ( buf, len );
  }
  else
  {
    return socket_ ->writeNoThrow ( buf, len );
  }
}


//-----------------------------------------------------------------------
//   getNoDelay
//-----------------------------------------------------------------------


bool SslSocket::getNoDelay () const
{
  return socket_->getNoDelay ();
}


//-----------------------------------------------------------------------
//   setNoDelay
//-----------------------------------------------------------------------


void SslSocket::setNoDelay ( bool choice )
{
  socket_->setNoDelay ( choice );
}


//-----------------------------------------------------------------------
//   getPeerAddress
//-----------------------------------------------------------------------


Ref<SocketAddress> SslSocket::getPeerAddress () const
{
  return socket_->getPeerAddress ();
}



//-----------------------------------------------------------------------
//   setPeerAddress
//-----------------------------------------------------------------------


void SslSocket::setPeerAddress ( const Ref<Address>& addr )
{
  socket_->setPeerAddress ( addr );
}


//-----------------------------------------------------------------------
//   getLocalAddress
//-----------------------------------------------------------------------


Ref<SocketAddress> SslSocket::getLocalAddress () const
{
  return socket_->getLocalAddress ();
}


//-----------------------------------------------------------------------
//   getSocketHandle
//-----------------------------------------------------------------------


Ref<SocketHandle> SslSocket::getSocketHandle () const
{
  return socket_->getSocketHandle ();
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


Ref<SslContext> SslSocket::getContext () const
{
  return context_;
}


//-----------------------------------------------------------------------
//   getSession
//-----------------------------------------------------------------------


Ref<SslSession> SslSocket::getSession () const
{
  Ref<SslSession>  ret;

  if ( sstream_ )
  {
    SSL_SESSION*  ses = SSL_get1_session ( sstream_->getSSL() );

    if ( ses )
    {
      try
      {
        ret = newInstance<NativeSession> ( ses );
      }
      catch ( ... )
      {
        SSL_SESSION_free ( ses );
        throw;
      }
    }
  }

  return ret;
}


//-----------------------------------------------------------------------
//   getPeerCert
//-----------------------------------------------------------------------


Ref<Certificate> SslSocket::getPeerCert () const
{
  if ( sstream_ )
  {
    return sstream_->getPeerCert ();
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   doHandshake
//-----------------------------------------------------------------------


void SslSocket::doHandshake

  ( const Time&      timeout,
    Ref<SslSession>  session )

{
  if ( ! sstream_ )
  {
    Ref<SslStream>  stream =

      newInstance<SslStream> ( socket_, context_ );

    stream->connect ( timeout, session );

    if ( context_->getVerifyMode() == SslContext::VERIFY_PEER )
    {
      Ref<Certificate>  cert = stream->getPeerCert ();

      if ( ! cert )
      {
        throw VerifyException (
          JEM_FUNC,
          "SSL/TLS handshake failed (missing peer certificate)"
        );
      }

      if ( X509_V_OK != SSL_get_verify_result( stream->getSSL() ) )
      {
        String  details = "invalid peer certificate";
        String  error   = SslLib::errorString ();

        if ( error.size() )
        {
          details = details + "; ";
          details = details + error;
        }

        throw VerifyException (
          JEM_FUNC,
          String::format (
            "SSL/TLS handshake failed (%s)",
            details
          )
        );
      }
    }

    sstream_ = stream;
  }
}


//-----------------------------------------------------------------------
//   isEncrypted
//-----------------------------------------------------------------------


bool SslSocket::isEncrypted () const
{
  if ( sstream_ )
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getKeyMaterial
//-----------------------------------------------------------------------


void SslSocket::getKeyMaterial

  ( const Array<byte>&  buf,
    const String&       label,
    const Array<byte>&  context ) const

{
  if ( ! sstream_ )
  {
    throw SslException (
      JEM_FUNC,
      "failed to export key material (no SSL/TLS connection has "
      "been established)"
    );
  }
  else
  {
    Array<byte>  ctx = makeContiguous ( context );
    Array<byte>  out = makeContiguous ( buf );

    int          ierr;

    ierr = SSL_export_keying_material (
      sstream_->getSSL    (),
      out.addr            (),
      (size_t) out.size   (),
      label.addr          (),
      (size_t) label.size (),
      ctx.addr            (),
      (size_t) ctx.size   (),
      1
    );

    if ( 1 != ierr )
    {
      SslLib::raiseError (
        JEM_FUNC,
        "failed to export key material"
      );
    }

    if ( out.addr() != buf.addr() )
    {
      buf = out;
    }
  }
}


//-----------------------------------------------------------------------
//   getCipherName
//-----------------------------------------------------------------------


String SslSocket::getCipherName () const
{
  String  name;

  if ( sstream_ )
  {
    const SSL_CIPHER*  cipher =

      SSL_get_current_cipher ( sstream_->getSSL() );

    if ( cipher )
    {
      const char*  str = SSL_CIPHER_get_name ( cipher );

      if ( str )
      {
        name = jem::clone ( str );
      }
    }
  }

  return name;
}


JEM_END_PACKAGE( ssl )

