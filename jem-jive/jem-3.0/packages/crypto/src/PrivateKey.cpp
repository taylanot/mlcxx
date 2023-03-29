
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
#include <jem/base/CString.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/crypto/native/CryptoLib.h>
#include <jem/crypto/native/PrivateKey.h>
#include <jem/crypto/native/BioInputStream.h>
#include <jem/crypto/native/BioOutputStream.h>

extern "C"
{
  #include <openssl/x509.h>
  #include <openssl/pem.h>
}


//=======================================================================
//   non-member utility functions
//=======================================================================


extern "C"
{
  static int              jem_crypt_no_passwd_cb

    ( char*                 buf,
      int                   size,
      int                   rwflag,
      void*                 u )

  {
    return 0;
  }
}


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class PrivateKey
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


PrivateKey::~PrivateKey ()
{}


//-----------------------------------------------------------------------
//   fromPEM
//-----------------------------------------------------------------------


Ref<PrivateKey> PrivateKey::fromPEM

  ( const String&  pem,
    const String&  passwd )

{
  Ref<BioOutputStream>  bio  = newInstance<BioOutputStream> ();
  EVP_PKEY*             pkey = nullptr;

  Ref<PrivateKey>       rkey;


  bio->write ( pem.addr(), pem.size() );

  if ( passwd.size() )
  {
    CString  pw ( passwd );

    pkey = PEM_read_bio_PrivateKey (
      bio->getBIO (),
      nullptr,
      nullptr,
      (void*) pw.addr ()
    );

  }
  else
  {
    pkey = PEM_read_bio_PrivateKey (
      bio->getBIO (),
      nullptr,
      jem_crypt_no_passwd_cb,
      nullptr
    );
  }

  if ( ! pkey )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "private key conversion from PEM format failed"
    );
  }

  try
  {
    rkey = newInstance<PrivateKeyImp> ( pkey );
  }
  catch ( ... )
  {
    EVP_PKEY_free ( pkey );
    throw;
  }

  EVP_PKEY_free ( pkey );

  return rkey;
}


//-----------------------------------------------------------------------
//   fromDER
//-----------------------------------------------------------------------


Ref<PrivateKey> PrivateKey::fromDER ( const ByteVector& der )
{
  Ref<BioOutputStream>  bio  = newInstance<BioOutputStream> ();
  EVP_PKEY*             pkey = nullptr;

  Ref<PrivateKey>       rkey;


  bio->write ( makeContiguous( der ).addr(), der.size() );

  pkey = d2i_PrivateKey_bio (
    bio->getBIO (),
    nullptr
  );

  if ( ! pkey )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "private key conversion from DER format failed"
    );
  }

  try
  {
    rkey = newInstance<PrivateKeyImp> ( pkey );
  }
  catch ( ... )
  {
    EVP_PKEY_free ( pkey );
    throw;
  }

  EVP_PKEY_free ( pkey );

  return rkey;
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool PrivateKey::equals ( const PrivateKey& rhs ) const
{
  ByteVector  left  = this->toDER ();
  ByteVector  right = rhs  .toDER ();

  return ((left.size() == right.size()) && testall( left == right ));
}


//=======================================================================
//   class PrivateKeyImp
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PrivateKeyImp::PrivateKeyImp ( EVP_PKEY* pkey )
{
  JEM_PRECHECK ( pkey );

  pkey_ = pkey;

  if ( 1 != EVP_PKEY_up_ref( pkey_ ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "PKEY ref count increment failed"
    );
  }
}


PrivateKeyImp::~PrivateKeyImp ()
{
  EVP_PKEY_free ( pkey_ );

  pkey_ = nullptr;
}


//-----------------------------------------------------------------------
//   toPEM
//-----------------------------------------------------------------------


String PrivateKeyImp::toPEM ( const String& passwd ) const
{
  Ref<BioInputStream>  bio = newInstance<BioInputStream> ();
  int                  rv  = 0;

  if ( passwd.size() )
  {
    CString  pw ( passwd );

    rv = PEM_write_bio_PrivateKey (
      bio->getBIO (),
      pkey_,
      EVP_des_ede3_cbc (),
      nullptr, 0,
      nullptr,
      (void*) pw.addr ()
    );
  }
  else
  {
    rv = PEM_write_bio_PrivateKey (
      bio->getBIO (),
      pkey_,
      nullptr,
      nullptr, 0,
      nullptr,
      nullptr
    );
  }

  if ( 1 != rv )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "private key conversion to PEM format failed"
    );
  }

  return bio->readString().stripWhite ();
}


//-----------------------------------------------------------------------
//   toDER
//-----------------------------------------------------------------------


ByteVector PrivateKeyImp::toDER () const
{
  Ref<BioInputStream>  bio = newInstance<BioInputStream> ();
  int                  rv  = 0;

  rv = i2d_PKCS8PrivateKey_bio (
    bio->getBIO (),
    pkey_,
    nullptr,
    nullptr, 0,
    nullptr,
    nullptr
  );

  if ( 1 != rv )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "private key conversion to DER format failed"
    );
  }

  return bio->readBytes ();
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   getEvpKey
//-----------------------------------------------------------------------


EVP_PKEY* getEvpKey ( const PrivateKey& key )
{
  const PrivateKeyImp*  imp =

    dynamic_cast<const PrivateKeyImp*> ( &key );

  if ( ! imp )
  {
    throw Error (
      JEM_FUNC,
      "invalid PrivateKey type (dynamic cast failed)"
    );
  }

  return imp->getEvpKey ();
}


JEM_END_PACKAGE( crypto )

