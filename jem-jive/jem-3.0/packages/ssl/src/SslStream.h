
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

#ifndef JEM_SSL_SSLSTREAM_H
#define JEM_SSL_SSLSTREAM_H

#include <jem/base/limits.h>
#include <jem/net/Socket.h>
#include <jem/ssl/SslContext.h>
#include <jem/ssl/SslSession.h>

extern "C"
{
  #include <openssl/ssl.h>
}


JEM_BEGIN_PACKAGE( ssl )


//-----------------------------------------------------------------------
//   class SslStream
//-----------------------------------------------------------------------


class SslStream : public Object
{
 public:

  typedef Object          Super;
  typedef SslStream       Self;

                          SslStream

    ( Ref<Socket>           socket,
      Ref<SslContext>       context );

  inline SSL*             getSSL       () const noexcept;

  void                    connect

    ( const Time&           timeout,
      Ref<SslSession>       session = nullptr );

  inline bool             connected    () const noexcept;

  void                    close

    ( const Time&           timeout );

  inline idx_t            pending      () const;

  inline idx_t            peek

    ( void*                 buf,
      idx_t                 len,
      const Time&           timeout );

  inline idx_t            read

    ( void*                 buf,
      idx_t                 len,
      const Time&           timeout );

  inline idx_t            write

    ( const void*           buf,
      idx_t                 len,
      const Time&           timeout );

  idx_t                   writeNoThrow

    ( const void*           buf,
      idx_t                 len )         noexcept;

  Ref<Certificate>        getPeerCert  () const;


 protected:

  virtual                ~SslStream    ();


 private:

  idx_t                   read_

    ( void*                 buf,
      int                   len,
      const Time&           timeout,
      int                   mode );

  idx_t                   write_

    ( const void*           buf,
      int                   len,
      const Time&           timeout );


 private:

  static const int        PEEK_MODE_;
  static const int        READ_MODE_;

  SSL*                    ssl_;
  Ref<Socket>             socket_;
  Ref<SslContext>         context_;
  bool                    connected_;

};


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getSSL
//-----------------------------------------------------------------------


inline SSL* SslStream::getSSL () const noexcept
{
  return ssl_;
}


//-----------------------------------------------------------------------
//   connected
//-----------------------------------------------------------------------


inline bool SslStream::connected () const noexcept
{
  return connected_;
}


//-----------------------------------------------------------------------
//   pending
//-----------------------------------------------------------------------


idx_t SslStream::pending () const
{
  return (idx_t) SSL_pending ( ssl_ );
}


//-----------------------------------------------------------------------
//   peek
//-----------------------------------------------------------------------


inline idx_t SslStream::peek

  ( void*        buf,
    idx_t        len,
    const Time&  timeout )

{
  JEM_ASSERT ( len > 0 );
  JEM_ASSERT ( connected() );

  int  rlen = (int) jem::min ( (idx_t) maxOf<int>(), len );
  int  ierr = SSL_peek ( ssl_, buf, rlen );

  if ( ierr > 0 )
  {
    return (idx_t) ierr;
  }
  else
  {
    return read_ ( buf, rlen, timeout, PEEK_MODE_ );
  }
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


inline idx_t SslStream::read

  ( void*        buf,
    idx_t        len,
    const Time&  timeout )

{
  JEM_ASSERT ( len > 0 );
  JEM_ASSERT ( connected() );

  int  rlen = (int) jem::min ( (idx_t) maxOf<int>(), len );
  int  ierr = SSL_read ( ssl_, buf, rlen );

  if ( ierr > 0 )
  {
    return (idx_t) ierr;
  }
  else
  {
    return read_ ( buf, rlen, timeout, READ_MODE_ );
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


inline idx_t SslStream::write

  ( const void*  buf,
    idx_t        len,
    const Time&  timeout )

{
  JEM_ASSERT ( len > 0 );
  JEM_ASSERT ( connected() );

  int  wlen = (int) jem::min ( (idx_t) maxOf<int>(), len );
  int  ierr = SSL_write ( ssl_, buf, wlen );

  if ( ierr > 0 )
  {
    return (idx_t) ierr;
  }
  else
  {
    return write_ ( buf, wlen, timeout );
  }
}


JEM_END_PACKAGE( ssl )

#endif
