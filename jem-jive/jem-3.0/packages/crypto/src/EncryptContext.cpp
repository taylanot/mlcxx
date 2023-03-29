
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
#include <jem/base/array/utilities.h>
#include <jem/crypto/CryptKey.h>
#include <jem/crypto/EncryptContext.h>
#include <jem/crypto/native/CryptoLib.h>
#include <jem/crypto/native/PublicKey.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class EncryptContext
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


EncryptContext::~EncryptContext ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void EncryptContext::start

  ( Cipher::Type     type,
    const CryptKey&  key )

{
  JEM_PRECHECK ( key.isSuitable( type ) );

  int  rv;

  rv = EVP_EncryptInit (
    JEM_CIPHER_CTX_CAST  ( context_ ),
    CryptoLib::getCipher ( type ),
    key.getKeyAddr       (),
    key.getIVAddr        ()
  );

  if ( ! rv )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "encryption start failed"
    );
  }

  started_ = true;
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


idx_t EncryptContext::update

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

    if ( 1 != EVP_EncryptUpdate( ctx, (byte*)  dest, &n,
                                 (const byte*) data,  k ) )
    {
      CryptoLib::raiseError (
        JEM_FUNC,
        "encryption update failed"
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


idx_t EncryptContext::finish

  ( void*        dest,
    const void*  data,
    idx_t        size )

{
  EVP_CIPHER_CTX*  ctx   = JEM_CIPHER_CTX_CAST ( context_ );
  idx_t            count = update ( dest, data, size );
  int              len   = 0;

  dest = (byte*) dest + count;

  if ( 1 != EVP_EncryptFinal( ctx, (byte*) dest, &len ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "encryption finalization failed"
    );
  }

  started_ = false;

  return (count + (idx_t) len);
}


//-----------------------------------------------------------------------
//   encrypt
//-----------------------------------------------------------------------


ByteVector EncryptContext::encrypt

  ( Cipher::Type     type,
    const CryptKey&  key,
    const void*      data,
    idx_t            size )

{
  Ref<EncryptContext>  ctx = newInstance<EncryptContext> ();

  ctx->start ( type, key );

  return ctx->finish ( data, size );
}


ByteVector EncryptContext::encrypt

  ( Cipher::Type       type,
    const CryptKey&    key,
    const ByteVector&  data )

{
  Ref<EncryptContext>  ctx = newInstance<EncryptContext> ();

  ctx->start ( type, key );

  return ctx->finish ( makeContiguous( data ).addr(), data.size() );
}


ByteVector EncryptContext::encrypt

  ( Cipher::Type     type,
    const CryptKey&  key,
    const String&    data )

{
  Ref<EncryptContext>  ctx = newInstance<EncryptContext> ();

  ctx->start ( type, key );

  return ctx->finish ( data.addr(), data.size() );
}


JEM_END_PACKAGE( crypto )

