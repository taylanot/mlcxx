
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
#include <jem/crypto/Digest.h>
#include <jem/crypto/native/CryptoLib.h>


#ifdef  JEM_MD_CTX_CAST
# undef JEM_MD_CTX_CAST
#endif
#define JEM_MD_CTX_CAST( ptr )          \
                                        \
  reinterpret_cast<EVP_MD_CTX*>( ptr )


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class Digest
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Digest::Digest ()
{
  context_ = nullptr;
  started_ = false;
}


Digest::Digest ( const Digest& rhs )
{
  context_ = nullptr;
  started_ = rhs.started_;

  if ( rhs.context_ )
  {
    EVP_MD_CTX*  rtx = JEM_MD_CTX_CAST ( rhs.context_ );
    EVP_MD_CTX*  ltx = JEM_MD_CTX_CAST ( newContext_() );

    if ( 1 != EVP_MD_CTX_copy( ltx, rtx ) )
    {
      EVP_MD_CTX_free ( ltx );

      CryptoLib::raiseError (
        JEM_FUNC,
        "digest context copy failed"
      );
    }

    context_ = ltx;
  }
}


Digest::Digest ( Digest&& rhs ) noexcept
{
  context_     = rhs.context_;
  started_     = rhs.started_;
  rhs.context_ = nullptr;
  rhs.started_ = false;
}


Digest::~Digest ()
{
  if ( context_ )
  {
    EVP_MD_CTX_free ( JEM_MD_CTX_CAST( context_ ) );

    context_ = nullptr;
  }

  started_ = false;
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


Digest& Digest::operator = ( const Digest& rhs )
{
  return operator = ( Digest( rhs ) );
}


Digest& Digest::operator = ( Digest&& rhs ) noexcept
{
  jem::swap ( context_, rhs.context_ );
  jem::swap ( started_, rhs.started_ );

  return *this;
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void Digest::start ( Type type )
{
  const EVP_MD*  md = CryptoLib::getDigest ( type );

  if ( context_ )
  {
    if ( 1 != EVP_MD_CTX_reset( JEM_MD_CTX_CAST( context_ ) ) )
    {
      CryptoLib::raiseError (
        JEM_FUNC,
        "digest context reset failed"
      );
    }
  }
  else
  {
    context_ = newContext_ ();
  }

  if ( 1 != EVP_DigestInit_ex( JEM_MD_CTX_CAST( context_ ),
                               md, nullptr ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "digest initialization failed"
    );
  }

  started_ = true;
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void Digest::update

  ( const void*  data,
    idx_t        size )

{
  JEM_PRECHECK ( started() );
  JEM_PRECHECK ( size >= 0 );

  while ( size > 0 )
  {
    int  k = (int) jem::min ( (idx_t) maxOf<int>(), size );

    if ( 1 != EVP_DigestUpdate( JEM_MD_CTX_CAST( context_ ),
                                data, k ) )
    {
      CryptoLib::raiseError (
        JEM_FUNC,
        "digest update failed"
      );
    }

    size -= (idx_t) k;
    data  = (const byte*) data + k;
  }
}


void Digest::update ( const ByteVector& data )
{
  update ( makeContiguous( data ).addr(), data.size() );
}


void Digest::update ( const String& data )
{
  update ( data.addr(), data.size() );
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


idx_t Digest::finish ( byte* buf )
{
  JEM_PRECHECK ( started() );

  unsigned int  len;

  if ( 1 != EVP_DigestFinal_ex( JEM_MD_CTX_CAST( context_ ),
                                buf, &len ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "digest finalization failed"
    );
  }

  started_ = false;

  return (idx_t) len;
}


ByteVector Digest::finish ()
{
  JEM_PRECHECK ( started() );

  byte          result[EVP_MAX_MD_SIZE];
  unsigned int  size;


  if ( 1 != EVP_DigestFinal_ex( JEM_MD_CTX_CAST( context_ ),
                                result, &size ) )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "digest finalization failed"
    );
  }

  started_ = false;

  return ByteVector ( result, result + size );
}


ByteVector Digest::finish

  ( const void*  data,
    idx_t        size )

{
  update ( data, size );

  return finish ();
}


ByteVector Digest::finish ( const ByteVector& data )
{
  return finish ( makeContiguous( data ).addr(), data.size() );
}


ByteVector Digest::finish ( const String& data )
{
  return finish ( data.addr(), data.size() );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t Digest::size ( Type type )
{
  return (idx_t) EVP_MD_size ( CryptoLib::getDigest( type ) );
}


//-----------------------------------------------------------------------
//   compute
//-----------------------------------------------------------------------


ByteVector Digest::compute

  ( Type         type,
    const void*  data,
    idx_t        size )

{
  Digest  digest;

  digest.start ( type );

  return digest.finish ( data, size );
}


ByteVector Digest::compute

  ( Type               type,
    const ByteVector&  data )

{
  Digest  digest;

  digest.start ( type );

  return digest.finish ( makeContiguous( data ).addr(), data.size() );
}


ByteVector Digest::compute

  ( Type           type,
    const String&  data )

{
  Digest  digest;

  digest.start ( type );

  return digest.finish ( data.addr(), data.size() );
}


//-----------------------------------------------------------------------
//   newContext_
//-----------------------------------------------------------------------


void* Digest::newContext_ ()
{
  EVP_MD_CTX*  ctx = EVP_MD_CTX_new ();

  if ( ! ctx )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "digest context allocation failed"
    );
  }

  return ctx;
}


JEM_END_PACKAGE( crypto )

