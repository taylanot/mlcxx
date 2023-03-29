
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
#include <jem/base/StringBuffer.h>
#include <jem/crypto/CryptoException.h>
#include <jem/crypto/native/CryptoLib.h>

extern "C"
{
  #include <openssl/err.h>
}


#if defined(_MSC_VER)
#  pragma comment(lib,"ws2_32.lib")
#  pragma comment(lib,"gdi32.lib")
#  pragma comment(lib,"advapi32.lib")
#  pragma comment(lib,"crypt32.lib")
#  pragma comment(lib,"user32.lib")
#endif


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class CryptoLib
//=======================================================================

//-----------------------------------------------------------------------
//   getCipher
//-----------------------------------------------------------------------


const EVP_CIPHER* CryptoLib::getCipher ( Cipher::Type type )
{
  switch ( type )
  {
  case Cipher::AES_256_CBC:

    return EVP_aes_256_cbc ();
  }

  throw Error (
    JEM_FUNC,
    "unexpected cipher type: " + String( (int) type )
  );

  return nullptr;
}


//-----------------------------------------------------------------------
//   getDigest
//-----------------------------------------------------------------------


const EVP_MD* CryptoLib::getDigest ( Digest::Type type )
{
  switch ( type )
  {
  case Digest::SHA256:

    return EVP_sha256 ();

  case Digest::SHA512:

    return EVP_sha512 ();
  }

  throw Error (
    JEM_FUNC,
    "unexpected digest type: " + String( (int) type )
  );

  return nullptr;
}


//-----------------------------------------------------------------------
//   raiseError
//-----------------------------------------------------------------------


void CryptoLib::raiseError

  ( const String&  where,
    const String&  what )

{
  String  msg = what;

  if ( ! msg.size() )
  {
    msg = errorString ();
  }
  else
  {
    String  details = errorString ();

    if ( details.size() )
    {
      msg = String::format ( "%s (%s)", msg, details );
    }
  }

  ERR_clear_error ();

  throw CryptoException ( where, msg );
}


//-----------------------------------------------------------------------
//   errorString
//-----------------------------------------------------------------------


String CryptoLib::errorString ()
{
  unsigned long  ierr = 0;

  StringBuffer   buf;


  for ( int i = 0; i < 3; i++ )
  {
    ierr = ERR_get_error ();

    if ( ! ierr )
    {
      break;
    }

    if ( buf.size() )
    {
      buf += "; ";
    }

    buf += ERR_reason_error_string ( ierr );
  }

  ierr = ERR_peek_last_error ();

  if ( ierr )
  {
    if ( buf.size() )
    {
      buf += "; ";
    }

    buf += ERR_reason_error_string ( ierr );
  }

  ERR_clear_error ();

  return buf.toString ();
}


String CryptoLib::errorString ( const String& what )
{
  String  details = errorString ();

  if      ( ! what.size() )
  {
    return details;
  }
  else if ( ! details.size() )
  {
    return what;
  }
  else
  {
    return String::format ( "%s (%s)", what, details );
  }
}


JEM_END_PACKAGE( crypto )

