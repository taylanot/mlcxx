
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
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/crypto/CryptKey.h>
#include <jem/crypto/CryptoException.h>
#include <jem/crypto/native/CryptoLib.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class CryptKey
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


CryptKey::CryptKey ()
{}


CryptKey::CryptKey

  ( const ByteVector&  key,
    const ByteVector&  iv ) :

    key_ ( key.clone() ),
    iv_  ( iv .clone() )

{}


CryptKey::CryptKey

  ( ByteVector&&  key,
    ByteVector&&  iv ) :

    key_ ( std::move( key ) ),
    iv_  ( std::move( iv  ) )

{}


CryptKey::CryptKey ( const CryptKey& rhs ) :

  key_ ( rhs.key_ ),
  iv_  ( rhs.iv_ )

{}


CryptKey::CryptKey ( CryptKey&& rhs ) noexcept :

  key_ ( std::move( rhs.key_ ) ),
  iv_  ( std::move( rhs.iv_ ) )

{}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


CryptKey& CryptKey::operator = ( const CryptKey& rhs )
{
  key_.ref ( rhs.key_ );
  iv_ .ref ( rhs.iv_ );

  return *this;
}


CryptKey& CryptKey::operator = ( CryptKey&& rhs ) noexcept
{
  key_.ref ( rhs.key_ );
  iv_ .ref ( rhs.iv_ );

  return *this;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void CryptKey::readFrom ( DataInput& in )
{
  decode ( in, key_, iv_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void CryptKey::writeTo ( DataOutput& out ) const
{
  encode ( out, key_, iv_ );
}


//-----------------------------------------------------------------------
//   isEmpty
//-----------------------------------------------------------------------


bool CryptKey::isEmpty () const noexcept
{
  return (key_.size() == 0);
}


//-----------------------------------------------------------------------
//   isSuitable
//-----------------------------------------------------------------------


bool CryptKey::isSuitable ( Cipher::Type type ) const
{
  return ((key_.size() >= Cipher::keySize( type )) &&
          (iv_ .size() >= Cipher::ivSize ( type )));
}


//-----------------------------------------------------------------------
//   getKey
//-----------------------------------------------------------------------


ByteVector CryptKey::getKey () const
{
  return key_.clone ();
}


//-----------------------------------------------------------------------
//   getIV
//-----------------------------------------------------------------------


ByteVector CryptKey::getIV () const
{
  return iv_.clone ();
}


//-----------------------------------------------------------------------
//   getKeySize
//-----------------------------------------------------------------------


idx_t CryptKey::getKeySize () const noexcept
{
  return key_.size ();
}


//-----------------------------------------------------------------------
//   getIVSize
//-----------------------------------------------------------------------


idx_t CryptKey::getIVSize () const noexcept
{
  return iv_.size ();
}


//-----------------------------------------------------------------------
//   getKeyAddr
//-----------------------------------------------------------------------


const byte* CryptKey::getKeyAddr () const noexcept
{
  return key_.addr ();
}


//-----------------------------------------------------------------------
//   getIVAddr
//-----------------------------------------------------------------------


const byte* CryptKey::getIVAddr () const noexcept
{
  return iv_.addr ();
}


//-----------------------------------------------------------------------
//   makeKey
//-----------------------------------------------------------------------


CryptKey CryptKey::makeKey

  ( Cipher::Type       type,
    const String&      passwd,
    const ByteVector&  salt )

{
  // Note that a large hash is used so that the size of the hash
  // is likely to be larger than the combined length of the
  // encryption key and initialization vector.

  const int          iter   = 55555;
  const EVP_CIPHER*  cipher = CryptoLib::getCipher ( type );
  const EVP_MD*      digest = EVP_sha512           ();
  const byte*        saltp  = nullptr;
  int                saltl  = 0;
  int                passl  = 0;
  int                ekl    = 0;
  int                ivl    = 0;
  int                dkl    = 0;
  int                rv     = 0;

  ByteVector         dk;
  ByteVector         ek;
  ByteVector         iv;


  if ( passwd.size() > (idx_t) maxOf<int>() )
  {
    throw CryptoException (
      JEM_FUNC,
      "password too long (integer overflow)"
    );
  }

  passl = (int) passwd.size ();

  if ( salt.size() > 0 )
  {
    saltp = salt.addr ();
    saltl = (int) jem::min ( (idx_t) maxOf( saltl ), salt.size() );
  }

  ekl = EVP_CIPHER_key_length ( cipher );
  ivl = EVP_CIPHER_iv_length  ( cipher );
  dkl = ekl + ivl;

  dk.resize ( (idx_t) dkl );

  rv = PKCS5_PBKDF2_HMAC (
    passwd.addr (),
    passl,
    saltp,
    saltl,
    iter,
    digest,
    dkl,
    dk.addr ()
  );

  if ( 1 != rv )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "encryption key generation failed"
    );
  }

  ek.resize ( (idx_t) ekl );
  iv.resize ( (idx_t) ivl );

  // Take the encryption key and the initialization vector from the
  // entire derived key to increase the computational costs for an
  // adversary in the case that the derived key is smaller than the
  // hash size.

  for ( int i = 0, j = 0, k = 0; i < dkl; )
  {
    if ( j < ekl )
    {
      ek[j++] = dk[i++];
    }

    if ( k < ivl )
    {
      iv[k++] = dk[i++];
    }
  }

  return CryptKey ( std::move( ek ), std::move( iv ) );
}


//-----------------------------------------------------------------------
//   runPBKDF2
//-----------------------------------------------------------------------


void CryptKey::runPBKDF2

  ( const ByteVector&  key,
    Digest::Type       type,
    int                iter,
    const String&      passwd,
    const ByteVector&  salt )

{
  JEM_PRECHECK ( iter > 0 );

  const EVP_MD*  digest = CryptoLib::getDigest ( type );
  const byte*    saltp  = nullptr;
  int            saltl  = 0;
  int            passl  = 0;
  int            dkl    = 0;
  int            rv     = 0;

  ByteVector     dk;


  if ( salt.size() > 0 )
  {
    saltp = salt.addr ();
    saltl = (int) jem::min ( (idx_t) maxOf( saltl ), salt.size() );
  }

  if ( passwd.size() > (idx_t) maxOf<int>() )
  {
    throw CryptoException (
      JEM_FUNC,
      "password too long (integer overflow)"
    );
  }


  if ( key.size() > (idx_t) maxOf<int>() )
  {
    throw CryptoException (
      JEM_FUNC,
      "too many PBKDF2 output bytes requested (integer overflow)"
    );
  }

  passl = (int) passwd.size ();
  dkl   = (int) key   .size ();

  if ( key.isContiguous() )
  {
    dk.ref    ( key );
  }
  else
  {
    dk.resize ( key.size() );
  }

  rv = PKCS5_PBKDF2_HMAC (
    passwd.addr (),
    passl,
    saltp,
    saltl,
    iter,
    digest,
    dkl,
    dk.addr ()
  );

  if ( 1 != rv )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "PBKDF2 execution failed"
    );
  }

  if ( dk.addr() != key.addr() )
  {
    key = dk;
  }
}


JEM_END_PACKAGE( crypto )

