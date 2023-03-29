
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
#include <jem/base/StringBuffer.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/io/FileReader.h>
#include <jem/io/BufferedReader.h>
#include <jem/crypto/native/CryptoLib.h>
#include <jem/crypto/native/PublicKey.h>
#include <jem/crypto/native/Certificate.h>
#include <jem/crypto/native/BioInputStream.h>
#include <jem/crypto/native/BioOutputStream.h>

extern "C"
{
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
//   class Certificate
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Certificate::~Certificate ()
{}


//-----------------------------------------------------------------------
//   fromPEM
//-----------------------------------------------------------------------


Ref<Certificate> Certificate::fromPEM

  ( const String&  pem,
    const String&  passwd )

{
  Ref<BioOutputStream>  bio = newInstance<BioOutputStream> ();
  X509*                 xc  = nullptr;

  Ref<Certificate>      cert;


  bio->write ( pem.addr(), pem.size() );

  if ( passwd.size() )
  {
    CString  pw ( passwd );

    xc = PEM_read_bio_X509 (
      bio->getBIO (),
      nullptr,
      nullptr,
      (void*) pw.addr ()
    );

  }
  else
  {
    xc = PEM_read_bio_X509 (
      bio->getBIO (),
      nullptr,
      jem_crypt_no_passwd_cb,
      nullptr
    );
  }

  if ( ! xc )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "X509 certificate conversion from PEM format failed"
    );
  }

  try
  {
    cert = newInstance<CertImp> ( xc );
  }
  catch ( ... )
  {
    X509_free ( xc );
    throw;
  }

  X509_free ( xc );

  return cert;
}


//-----------------------------------------------------------------------
//   fromDER
//-----------------------------------------------------------------------


Ref<Certificate> Certificate::fromDER ( const ByteVector& der )
{
  Ref<BioOutputStream>  bio = newInstance<BioOutputStream> ();
  X509*                 xc  = nullptr;

  Ref<Certificate>      cert;


  bio->write ( makeContiguous( der ).addr(), der.size() );

  xc = d2i_X509_bio (
    bio->getBIO (),
    nullptr
  );

  if ( ! xc )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "X509 certificate conversion from DER format failed"
    );
  }

  try
  {
    cert = newInstance<CertImp> ( xc );
  }
  catch ( ... )
  {
    X509_free ( xc );
    throw;
  }

  X509_free ( xc );

  return cert;
}


//-----------------------------------------------------------------------
//   fromFile
//-----------------------------------------------------------------------


Ref<Certificate> Certificate::fromFile

  ( const String&  name,
    const String&  passwd )

{
  using jem::io::Reader;
  using jem::io::FileReader;
  using jem::io::BufferedReader;

  Ref<Reader>   input;
  StringBuffer  buffer;


  input = newInstance<BufferedReader> (
    newInstance<FileReader> ( name )
  );

  while ( true )
  {
    int  ch = input->read ();

    if ( ch < 0 )
    {
      break;
    }

    buffer += (char) ch;
  }

  return fromPEM ( name, passwd );
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Certificate::equals ( const Certificate& rhs ) const
{
  ByteVector  left  = this->toDER ();
  ByteVector  right = rhs  .toDER ();

  return ((left.size() == right.size()) && testall( left == right ));
}


//=======================================================================
//   class CertImp
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


CertImp::CertImp ( X509* cert )
{
  JEM_PRECHECK ( cert );

  cert_ = cert;

  if ( 1 != X509_up_ref( cert_ ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "certificate ref count increment failed"
    );
  }
}


CertImp::~CertImp ()
{
  X509_free ( cert_ );
}


//-----------------------------------------------------------------------
//   toPEM
//-----------------------------------------------------------------------


String CertImp::toPEM () const
{
  Ref<BioInputStream>  bio = newInstance<BioInputStream> ();

  if ( 1 != PEM_write_bio_X509( bio->getBIO(), cert_ ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "X509 certificate conversion to PEM format failed"
    );
  }

  return bio->readString().stripWhite ();
}


//-----------------------------------------------------------------------
//   toDER
//-----------------------------------------------------------------------


ByteVector CertImp::toDER () const
{
  Ref<BioInputStream>  bio = newInstance<BioInputStream> ();

  if ( 1 != i2d_X509_bio( bio->getBIO(), cert_ ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "X509 certificate conversion to DER format failed"
    );
  }

  return bio->readBytes ();
}


//-----------------------------------------------------------------------
//   toText
//-----------------------------------------------------------------------


String CertImp::toText () const
{
  const int            indent = 2;

  Ref<BioInputStream>  bio;
  StringBuffer         buffer;


  buffer += "Subject:\n";
  bio     = newInstance<BioInputStream> ();

  X509_NAME_print_ex (
    bio->getBIO           (),
    X509_get_subject_name ( cert_ ),
    indent,
    XN_FLAG_MULTILINE
  );

  buffer += bio->readString ();
  buffer += "\nIssuer:\n";
  bio     = newInstance<BioInputStream> ();

  X509_NAME_print_ex (
    bio->getBIO          (),
    X509_get_issuer_name ( cert_ ),
    indent,
    XN_FLAG_MULTILINE
  );

  buffer += bio->readString ();

  return buffer.toString ();
}


//-----------------------------------------------------------------------
//   getPublicKey
//-----------------------------------------------------------------------


Ref<PublicKey> CertImp::getPublicKey () const
{
  Ref<PublicKey>  rkey;
  EVP_PKEY*       pkey;

  pkey = X509_get_pubkey ( cert_ );

  if ( ! pkey )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "failed to retrieve the public key from an X509 certificate"
    );
  }

  try
  {
    rkey = newInstance<PublicKeyImp> ( pkey );
  }
  catch ( ... )
  {
    EVP_PKEY_free ( pkey );
    throw;
  }

  EVP_PKEY_free ( pkey );

  return rkey;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   getX509Cert
//-----------------------------------------------------------------------


X509* getX509Cert ( const Certificate& cert )
{
  const CertImp*  imp =

    dynamic_cast<const CertImp*> ( &cert );

  if ( ! imp )
  {
    throw Error (
      JEM_FUNC,
      "invalid Certificate type (dynamic cast failed)"
    );
  }

  return imp->getX509 ();
}


JEM_END_PACKAGE( crypto )

