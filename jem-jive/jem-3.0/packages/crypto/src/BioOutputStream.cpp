
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
#include <jem/io/IOException.h>
#include <jem/crypto/native/CryptoLib.h>
#include <jem/crypto/native/BioOutputStream.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   class BioOutputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BioOutputStream::BioOutputStream ()
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


BioOutputStream::BioOutputStream ( BIO* bio )
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


BioOutputStream::~BioOutputStream ()
{
  if ( bio_ )
  {
    BIO_flush    ( bio_ );
    BIO_free_all ( bio_ );

    bio_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void BioOutputStream::close ()
{
  if ( bio_ )
  {
    BIO_set_close ( bio_, BIO_CLOSE );
    BIO_free_all  ( bio_ );

    bio_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void BioOutputStream::flush ()
{
  if ( bio_ )
  {
    if ( 1 != BIO_flush( bio_ ) )
    {
      if ( ! BIO_should_retry( bio_ ) )
      {
        throw io::IOException (
          JEM_FUNC,
          CryptoLib::errorString (
            "failed to flush buffered data"
          )
        );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void BioOutputStream::write

  ( const void*  buf,
    idx_t        n )

{
  JEM_PRECHECK ( n >= 0 );


  if ( ! bio_ )
  {
    throw io::IOException (
      JEM_FUNC,
      "illegal write operation performed on a closed output stream"
    );
  }

  while ( n )
  {
    int  k = (int) jem::min ( (idx_t) maxOf<int>(), n );

    k = BIO_write ( bio_, buf, k );

    if ( k <= 0 )
    {
      if ( ! BIO_should_retry( bio_ ) )
      {
        throw io::IOException (
          JEM_FUNC,
          CryptoLib::errorString (
            "write operation failed"
          )
        );
      }
    }

    n  -= (idx_t) k;
    buf = (const byte*) buf + k;
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void BioOutputStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  if ( bio_ )
  {
    while ( n )
    {
      int  k = (int) jem::min ( (idx_t) maxOf<int>(), n );

      k = BIO_write ( bio_, buf, k );

      if ( k <= 0 )
      {
        if ( ! BIO_should_retry( bio_ ) )
        {
          return;
        }
      }

      n  -= (idx_t) k;
      buf = (const byte*) buf + k;
    }
  }
}


JEM_END_PACKAGE( crypto )

