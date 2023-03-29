
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

#ifndef JEM_CRYPTO_NATIVE_BIOINPUTSTREAM_H
#define JEM_CRYPTO_NATIVE_BIOINPUTSTREAM_H

#include <jem/io/InputStream.h>
#include <jem/crypto/Array.h>

extern "C"
{
  #include <openssl/bio.h>
}


JEM_BEGIN_PACKAGE( crypto )


//-----------------------------------------------------------------------
//   class BioInputStream
//-----------------------------------------------------------------------


class BioInputStream : public io::InputStream
{
 public:

  typedef
    io::InputStream       Super;
  typedef
    BioInputStream        Self;


                          BioInputStream  ();

  explicit                BioInputStream

    ( BIO*                  bio );

  virtual void            close           ()       override;

  virtual idx_t           poll

    ( const Time&           timeout )              override;

  virtual idx_t           read

    ( void*                 buf,
      idx_t                 n )                    override;

  idx_t                   pending         () const;

  String                  readString      ();
  ByteVector              readBytes       ();

  inline BIO*             getBIO          () const noexcept;


 protected:

  virtual                ~BioInputStream  ();


 private:

  BIO*                    bio_;

};


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getBIO
//-----------------------------------------------------------------------


inline BIO* BioInputStream::getBIO () const noexcept
{
  return bio_;
}


JEM_END_PACKAGE( crypto )

#endif

