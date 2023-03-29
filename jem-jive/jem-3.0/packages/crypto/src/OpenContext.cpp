
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


#include <jem/base/limits.h>
#include <jem/crypto/CryptKey.h>
#include <jem/crypto/OpenContext.h>
#include <jem/crypto/native/CryptoLib.h>
#include <jem/crypto/native/PrivateKey.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class OpenContext
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


OpenContext::~OpenContext ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void OpenContext::start

  ( Cipher::Type       type,
    const PrivateKey&  pkey,
    const CryptKey&    ckey )

{
  JEM_PRECHECK ( ckey.isSuitable( type ) );

  ByteVector  ek  = ckey.getKey ();
  ByteVector  iv  = ckey.getIV  ();
  int         rv  = 0;

  int         ekl;

  ekl = (int) jem::min ( ek.size(), (idx_t) maxOf( ekl ) );

  rv  = EVP_OpenInit (
    JEM_CIPHER_CTX_CAST  ( context_ ),
    CryptoLib::getCipher ( type ),
    ek.addr              (),
    ekl,
    iv.addr              (),
    getEvpKey            ( pkey )
  );

  if ( rv <= 0 )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "envelope decryption start failed"
    );
  }

  started_ = true;
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


idx_t OpenContext::update

  ( void*        dest,
    const void*  data,
    idx_t        size )

{
  JEM_PRECHECK ( started() );
  JEM_PRECHECK ( size >= 0 );

  EVP_CIPHER_CTX*  ctx = JEM_CIPHER_CTX_CAST ( context_ );
  idx_t            len = 0;


  while ( size )
  {
    int  k = (int) jem::min ( (idx_t) maxOf<int>(), size );
    int  n = 0;

    if ( 1 != EVP_OpenUpdate( ctx, (byte*)  dest, &n,
                              (const byte*) data,  k ) )
    {
      CryptoLib::raiseError (
        JEM_FUNC,
        "envelope decryption update failed"
      );
    }

    size -= (idx_t) k;
    len  += (idx_t) n;
    dest  = (byte*)       dest + n;
    data  = (const byte*) data + k;
  }

  return len;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


idx_t OpenContext::finish

  ( void*        dest,
    const void*  data,
    idx_t        size )

{
  EVP_CIPHER_CTX*  ctx   = JEM_CIPHER_CTX_CAST ( context_ );
  idx_t            count = update ( dest, data, size );
  int              len   = 0;

  dest = (byte*) dest + count;

  if ( 1 != EVP_OpenFinal( ctx, (byte*) dest, &len ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "envelope decryption finalization failed"
    );
  }

  started_ = false;

  return (count + (idx_t) len);
}


//-----------------------------------------------------------------------
//   open_
//-----------------------------------------------------------------------


void OpenContext::open_

  ( ByteVector&        out,
    Cipher::Type       type,
    const PrivateKey&  pkey,
    const CryptKey&    ckey,
    const void*        data,
    idx_t              size )

{
  Ref<OpenContext>  ctx = newInstance<OpenContext> ();

  ctx->start ( type, pkey, ckey );
  out.ref    ( ctx->finish<ByteVector>( data, size ) );
}


void OpenContext::open_

  ( String&            out,
    Cipher::Type       type,
    const PrivateKey&  pkey,
    const CryptKey&    ckey,
    const void*        data,
    idx_t              size )

{
  Ref<OpenContext>  ctx = newInstance<OpenContext> ();

  ctx->start ( type, pkey, ckey );

  out = ctx->finish<String> ( data, size );
}


JEM_END_PACKAGE( crypto )

