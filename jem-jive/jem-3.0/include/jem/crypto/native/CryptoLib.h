
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

#ifndef JEM_CRYPTO_NATIVE_CRYPTOLIB_H
#define JEM_CRYPTO_NATIVE_CRYPTOLIB_H

#include <jem/base/String.h>
#include <jem/crypto/Cipher.h>
#include <jem/crypto/Digest.h>

extern "C"
{
  #include <openssl/evp.h>
}


JEM_BEGIN_PACKAGE( crypto )


//-----------------------------------------------------------------------
//   class CryptoLib
//-----------------------------------------------------------------------


class CryptoLib
{
 public:

  static const
    EVP_CIPHER*           getCipher

    ( Cipher::Type          type );

  static const EVP_MD*    getDigest

    ( Digest::Type          type );

  static void             raiseError

    ( const String&         where,
      const String&         what = "" );

  static String           errorString    ();

  static String           errorString

    ( const String&         what );

};


//-----------------------------------------------------------------------
//   macros
//-----------------------------------------------------------------------


#ifdef  JEM_CIPHER_CTX_CAST
# undef JEM_CIPHER_CTX_CAST
#endif
#define JEM_CIPHER_CTX_CAST( ptr )          \
                                            \
  reinterpret_cast<EVP_CIPHER_CTX*>( ptr )


JEM_END_PACKAGE( crypto )

#endif

