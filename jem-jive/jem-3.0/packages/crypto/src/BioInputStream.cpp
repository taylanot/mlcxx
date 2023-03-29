
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
#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include <jem/base/StringBuffer.h>
#include <jem/io/IOException.h>
#include <jem/crypto/native/CryptoLib.h>
#include <jem/crypto/native/BioInputStream.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class BioInputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BioInputStream::BioInputStream ()
{
  bio_ = BIO_new ( BIO_s_mem() );

  if ( ! bio_ )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "BIO allocation failed"
    );
  }
}


BioInputStream::BioInputStream ( BIO* bio )
{
  bio_ = bio;

  if ( bio_ )
  {
    if ( 1 != BIO_up_ref( bio_ ) )
    {
      CryptoLib::raiseError (
        JEM_FUNC,
        "BIO ref count increment failed"
      );
    }
  }
}


BioInputStream::~BioInputStream ()
{
  if ( bio_ )
  {
    BIO_free_all ( bio_ );

    bio_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void BioInputStream::close ()
{
  if ( bio_ )
  {
    BIO_set_close ( bio_, BIO_CLOSE );
    BIO_free_all  ( bio_ );

    bio_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t BioInputStream::poll ( const jem::Time& timeout )
{
  idx_t  n = 0;

  if ( bio_ )
  {
    n = (idx_t) BIO_pending ( bio_ );

    if ( (n == 0) && (timeout > jem::Time::zero()) )
    {
      Time    deadline = Time::now() + timeout;
      double  pause    = jem::min ( 0.1, timeout.toDouble() );

      while ( true )
      {
        Thread::sleep ( Time( pause ) );

        pause += 0.1;
        pause  = jem::min ( 1.0, timeout.toDouble() );
        n      = (idx_t) BIO_pending ( bio_ );

        if ( n || (Time::now() > deadline) )
        {
          break;
        }
      }
    }
  }

  return n;
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t BioInputStream::read

  ( void*  buf,
    idx_t  n )

{
  JEM_PRECHECK ( n >= 0 );

  int  k;


  if ( ! bio_ )
  {
    throw io::IOException (
      JEM_FUNC,
      "illegal read operation performed on a closed input stream"
    );
  }

  k = BIO_read ( bio_, buf,
                 (int) jem::min( (idx_t) maxOf( k ), n ) );

  if ( (k == 0) && (n > 0) )
  {
    if ( ! BIO_should_retry( bio_ ) )
    {
      throw io::IOException (
        JEM_FUNC,
        CryptoLib::errorString (
          "unexpected end of input stream"
        )
      );
    }
  }

  return (idx_t) k;
}


//-----------------------------------------------------------------------
//   pending
//-----------------------------------------------------------------------


idx_t BioInputStream::pending () const
{
  idx_t  n = 0;

  if ( bio_ )
  {
    n = (idx_t) BIO_pending ( bio_ );
  }

  return n;
}


//-----------------------------------------------------------------------
//   readString
//-----------------------------------------------------------------------


String BioInputStream::readString ()
{
  StringBuffer  buffer;
  idx_t         count;
  idx_t         size;

  size  = pending ();
  count = read    ( buffer.xalloc( size ), size );

  if ( count != size )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "mismatch between the expected and actual number "
      "of bytes obtained from a BIO"
    );
  }

  buffer.commit ( count );

  return buffer.toString ();
}


//-----------------------------------------------------------------------
//   readBytes
//-----------------------------------------------------------------------


ByteVector BioInputStream::readBytes ()
{
  ByteVector  buffer;
  idx_t       count;
  idx_t       size;

  size  = pending ();

  buffer.resize   ( size );

  count = read    ( buffer.addr(), size );

  if ( count != size )
  {
    CryptoLib::raiseError (
      JEM_FUNC,
      "mismatch between the expected and actual number "
      "of bytes obtained from a BIO"
    );
  }

  return buffer;
}


JEM_END_PACKAGE( crypto )

