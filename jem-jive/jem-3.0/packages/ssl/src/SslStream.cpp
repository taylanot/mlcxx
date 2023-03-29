
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


#include <jem/base/Time.h>
#include <jem/net/SocketHandle.h>
#include <jem/net/TimeoutException.h>
#include <jem/crypto/native/Certificate.h>
#include "SslLib.h"
#include "SslStream.h"
#include "NativeContext.h"
#include "NativeSession.h"


JEM_BEGIN_PACKAGE( ssl )


//=======================================================================
//   class SslStream
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  SslStream::PEEK_MODE_ = 1;
const int  SslStream::READ_MODE_ = 2;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SslStream::SslStream

  ( Ref<Socket>      socket,
    Ref<SslContext>  context ) :

    ssl_       ( nullptr ),
    socket_    ( socket ),
    context_   ( context ),
    connected_ ( false )

{
  JEM_PRECHECK ( socket && context );

  ssl_ = SSL_new ( getSslContext( *context ) );

  if ( ! ssl_ )
  {
    SslLib::raiseError (
      JEM_FUNC,
      "failed to create a new SSL structure"
    );
  }

  SSL_set_mode ( ssl_, SSL_MODE_ENABLE_PARTIAL_WRITE );

  if ( context->isClientContext() )
  {
    SSL_set_connect_state ( ssl_ );
  }
  else
  {
    SSL_set_accept_state  ( ssl_ );
  }
}


SslStream::~SslStream ()
{
  if ( connected_ )
  {
    if ( ! SSL_shutdown( ssl_ ) )
    {
      SSL_shutdown ( ssl_ );
    }
  }

  SSL_free ( ssl_ );

  ssl_ = nullptr;
}


//-----------------------------------------------------------------------
//   connect
//-----------------------------------------------------------------------


void SslStream::connect

  ( const Time&      timeout,
    Ref<SslSession>  session )

{
  if ( ! connected_ )
  {
    Time  left = timeout;

    Time  end;
    int   ierr;

    ierr = SSL_set_fd (
      ssl_,
      (int) socket_->getSocketHandle()->toInteger ()
    );

    if ( 1 != ierr )
    {
      SslLib::raiseError (
        JEM_FUNC,
        "failed to bind an SSL connection to a network socket"
      );
    }

    if ( timeout >= Time::zero() )
    {
      end = timeout + Time::now ();
    }

    if ( session && context_->isClientContext() )
    {
      ierr = SSL_set_session ( ssl_, getSslSession( *session ) );

      if ( ierr == 0 )
      {
        SslLib::raiseError (
          JEM_FUNC,
          "failed to set an SSL session object"
        );
      }
    }

    while ( true )
    {
      ierr = SSL_do_handshake ( ssl_ );

      if ( ierr > 0 )
      {
        break;
      }

      if ( timeout >= Time::zero() )
      {
        left = end - Time::now ();

        if ( left <= Time::zero() )
        {
          throw net::TimeoutException (
            JEM_FUNC,
            "SSL/TLS handshake timed out"
          );
        }
      }

      ierr = SSL_get_error ( ssl_, ierr );

      if ( (ierr == SSL_ERROR_WANT_READ) ||
           (ierr == SSL_ERROR_WANT_WRITE) )
      {
        int  what = 0;

        if ( ierr == SSL_ERROR_WANT_READ )
        {
          what |= Socket::POLL_READ;
        }
        else
        {
          what |= Socket::POLL_WRITE;
        }

        socket_->poll ( what, left );
      }
      else
      {
        SslLib::raiseError (
          JEM_FUNC,
          "SSL/TLS handshake failed"
        );
      }
    }

    connected_ = true;
  }
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void SslStream::close ( const Time& timeout )
{
  if ( connected_ )
  {
    Time  left = timeout;

    Time  end;
    int   ierr;


    if ( timeout >= Time::zero() )
    {
      end = timeout + Time::now ();
    }

    while ( true )
    {
      ierr = SSL_shutdown ( ssl_ );

      if ( ierr > 0 )
      {
        return;
      }

      if ( timeout >= Time::zero() )
      {
        left = end - Time::now ();

        if ( left <= Time::zero() )
        {
          throw net::TimeoutException (
            JEM_FUNC,
            "timeout while closing an SSL connection"
          );
        }
      }

      if ( ierr == 0 )
      {
        continue;
      }

      ierr = SSL_get_error ( ssl_, ierr );

      if ( (ierr == SSL_ERROR_WANT_READ) ||
           (ierr == SSL_ERROR_WANT_WRITE) )
      {
        int  what = 0;

        if ( ierr == SSL_ERROR_WANT_READ )
        {
          what |= Socket::POLL_READ;
        }
        else
        {
          what |= Socket::POLL_WRITE;
        }

        socket_->poll ( what, left );
      }
      else
      {
        SslLib::raiseError (
          JEM_FUNC,
          "failed to close an SSL connection"
        );
      }
    }
  }

  connected_ = false;
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


idx_t SslStream::writeNoThrow

  ( const void*  buf,
    idx_t        len ) noexcept

{
  idx_t  out = 0;

  if ( connected_ )
  {
    int  ierr;

    while ( len > 0 )
    {
      int  k = (int) jem::min ( (idx_t) maxOf<int>(), len );

      ierr = SSL_write ( ssl_, buf, k );

      if ( ierr <= 0 )
      {
        break;
      }

      out += (idx_t) k;
      len -= (idx_t) k;
      buf  = (const byte*) buf + k;
    }
  }

  return out;
}


//-----------------------------------------------------------------------
//   getPeerCert
//-----------------------------------------------------------------------


Ref<Certificate> SslStream::getPeerCert () const
{
  Ref<Certificate>  cert;

  if ( connected_ )
  {
    X509*  x = SSL_get_peer_certificate ( ssl_ );

    if ( x )
    {
      try
      {
        cert = newInstance<crypto::CertImp> ( x );
      }
      catch ( ... )
      {
        X509_free ( x );
        throw;
      }

      X509_free ( x );
    }
  }

  return cert;
}


//-----------------------------------------------------------------------
//   read_
//-----------------------------------------------------------------------


idx_t SslStream::read_

  ( void*        buf,
    int          len,
    const Time&  timeout,
    int          mode )

{
  Time  left = timeout;

  Time  end;
  int   ierr;


  for ( bool first = true; true; first = false )
  {
    if ( mode == PEEK_MODE_ )
    {
      ierr = SSL_peek ( ssl_, buf, len );
    }
    else
    {
      ierr = SSL_read ( ssl_, buf, len );
    }

    if ( ierr > 0 )
    {
      return (idx_t) ierr;
    }

    ierr = SSL_get_error ( ssl_, ierr );

    if ( ierr == SSL_ERROR_ZERO_RETURN )
    {
      // Connection closed by peer.

      return 0;
    }

    if ( (ierr == SSL_ERROR_WANT_READ) ||
         (ierr == SSL_ERROR_WANT_WRITE) )
    {
      int  what = 0;

      if ( ierr == SSL_ERROR_WANT_READ )
      {
        what |= Socket::POLL_READ;
      }
      else
      {
        what |= Socket::POLL_WRITE;
      }

      if ( timeout >= Time::zero() )
      {
        Time  now = Time::now ();

        if ( first )
        {
          end  = timeout + now;
        }
        else
        {
          left = end - now;
        }

        if ( left <= Time::zero() )
        {
          return -1_idx;
        }
      }

      if ( ! socket_->poll( what, left ) )
      {
        // Timeout.

        return -1_idx;
      }
    }
    else
    {
      SslLib::raiseError (
        JEM_FUNC,
        "SSL receive operation failed"
      );
    }
  }

  return 0;
}


//-----------------------------------------------------------------------
//   write_
//-----------------------------------------------------------------------


idx_t SslStream::write_

  ( const void*  buf,
    int          len,
    const Time&  timeout )

{
  Time  left = timeout;

  Time  end;
  int   ierr;


  for ( bool first = true; true; first = false )
  {
    ierr = SSL_write ( ssl_, buf, len );

    if ( ierr > 0 )
    {
      return (idx_t) ierr;
    }

    ierr = SSL_get_error ( ssl_, ierr );

    if ( (ierr == SSL_ERROR_WANT_READ) ||
         (ierr == SSL_ERROR_WANT_WRITE) )
    {
      int  what = 0;

      if ( ierr == SSL_ERROR_WANT_READ )
      {
        what |= Socket::POLL_READ;
      }
      else
      {
        what |= Socket::POLL_WRITE;
      }

      if ( timeout >= Time::zero() )
      {
        Time  now = Time::now ();

        if ( first )
        {
          end  = timeout + now;
        }
        else
        {
          left = end - now;
        }

        if ( left <= Time::zero() )
        {
          return -1_idx;
        }
      }

      if ( ! socket_->poll( what, left ) )
      {
        // Timeout.

        return -1_idx;
      }
    }
    else
    {
      SslLib::raiseError (
        JEM_FUNC,
        "SSL send operation failed"
      );
    }
  }

  return 0;
}


JEM_END_PACKAGE( ssl )

