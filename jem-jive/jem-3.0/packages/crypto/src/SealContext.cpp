
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


#include <utility>
#include <jem/base/limits.h>
#include <jem/base/array/utilities.h>
#include <jem/crypto/CryptKey.h>
#include <jem/crypto/SealContext.h>
#include <jem/crypto/native/CryptoLib.h>
#include <jem/crypto/native/PublicKey.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class SealContext
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


SealContext::~SealContext ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


CryptKey SealContext::start

  ( Cipher::Type      type,
    const PublicKey&  key )

{
  EVP_PKEY*          pkey   = getEvpKey            ( key );
  const EVP_CIPHER*  cipher = CryptoLib::getCipher ( type );
  byte*              ekp    = nullptr;
  int                ekl    = 0;
  int                rv     = 0;

  ByteVector         ek;
  ByteVector         iv;


  ek.resize ( (idx_t) EVP_PKEY_size( pkey ) );
  iv.resize ( (idx_t) EVP_CIPHER_iv_length( cipher ) );

  ekp = ek.addr ();
  rv  = EVP_SealInit (
    JEM_CIPHER_CTX_CAST ( context_ ),
    cipher,
    &ekp,
    &ekl,
    iv.addr (),
    &pkey, 1
  );

  if ( ! rv )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "envelope encryption start failed"
    );
  }

  started_ = true;

  JEM_PRECHECK ( (ekl >= 0) && (ekl <= ek.size()) );

  ek.reshape ( (idx_t) ekl );

  return CryptKey ( std::move( ek ), std::move( iv ) );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


idx_t SealContext::update

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

    if ( 1 != EVP_SealUpdate( ctx, (byte*)  dest, &n,
                              (const byte*) data,  k ) )
    {
      CryptoLib::raiseError (
        JEM_FUNC,
        "envelope encryption update failed"
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


idx_t SealContext::finish

  ( void*        dest,
    const void*  data,
    idx_t        size )

{
  EVP_CIPHER_CTX*  ctx   = JEM_CIPHER_CTX_CAST ( context_ );
  idx_t            count = update ( dest, data, size );
  int              len   = 0;

  dest = (byte*) dest + count;

  if ( 1 != EVP_SealFinal( ctx, (byte*) dest, &len ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "envelope encryption finalization failed"
    );
  }

  started_ = false;

  return (count + (idx_t) len);
}


//-----------------------------------------------------------------------
//   seal
//-----------------------------------------------------------------------


ByteVector SealContext::seal

  ( Cipher::Type      type,
    CryptKey&         ckey,
    const PublicKey&  pkey,
    const void*       data,
    idx_t             size )

{
  Ref<SealContext>  ctx = newInstance<SealContext> ();

  ckey = ctx->start ( type, pkey );

  return ctx->finish ( data, size );
}


ByteVector SealContext::seal

  ( Cipher::Type       type,
    CryptKey&          ckey,
    const PublicKey&   pkey,
    const ByteVector&  data )

{
  Ref<SealContext>  ctx = newInstance<SealContext> ();

  ckey = ctx->start ( type, pkey );

  return ctx->finish ( makeContiguous( data ).addr(), data.size() );
}


ByteVector SealContext::seal

  ( Cipher::Type      type,
    CryptKey&         ckey,
    const PublicKey&  pkey,
    const String&     data )

{
  Ref<SealContext>  ctx = newInstance<SealContext> ();

  ckey = ctx->start ( type, pkey );

  return ctx->finish ( data.addr(), data.size() );
}


JEM_END_PACKAGE( crypto )


