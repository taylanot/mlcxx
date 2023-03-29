
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

#ifndef JEM_CRYPTO_CRYPTKEY_H
#define JEM_CRYPTO_CRYPTKEY_H

#include <jem/crypto/import.h>
#include <jem/crypto/Array.h>
#include <jem/crypto/Cipher.h>
#include <jem/crypto/Digest.h>


JEM_BEGIN_PACKAGE( crypto )


//-----------------------------------------------------------------------
//   class CryptKey
//-----------------------------------------------------------------------


class CryptKey
{
 public:

                          CryptKey    ();

                          CryptKey

    ( const ByteVector&     key,
      const ByteVector&     iv );

                          CryptKey

    ( ByteVector&&          key,
      ByteVector&&          iv );

                          CryptKey

    ( const CryptKey&       rhs );

                          CryptKey

    ( CryptKey&&            rhs )              noexcept;

  CryptKey&               operator =

    ( const CryptKey&       rhs );

  CryptKey&               operator =

    ( CryptKey&&            rhs )              noexcept;

  void                    readFrom

    ( DataInput&            in );

  void                    writeTo

    ( DataOutput&           out )        const;

  bool                    isEmpty     () const noexcept;

  bool                    isSuitable

    ( Cipher::Type          type )       const;

  ByteVector              getKey      () const;
  ByteVector              getIV       () const;

  idx_t                   getKeySize  () const noexcept;
  idx_t                   getIVSize   () const noexcept;

  const byte*             getKeyAddr  () const noexcept;
  const byte*             getIVAddr   () const noexcept;

  static CryptKey         makeKey

    ( Cipher::Type          type,
      const String&         passwd,
      const ByteVector&     salt );

  static void             runPBKDF2

    ( const ByteVector&     key,
      Digest::Type          type,
      int                   iter,
      const String&         passwd,
      const ByteVector&     salt );


 private:

  ByteVector              key_;
  ByteVector              iv_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void             decode

  ( Input&                  in,
    CryptKey&               key );

template <class Output>

  inline void             encode

  ( Output&                 out,
    const CryptKey&         key );



//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void             decode

  ( Input&                  in,
    CryptKey&               key )

{
  key.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void             encode

  ( Output&                 out,
    const CryptKey&         key )

{
  key.writeTo ( out );
}


JEM_END_PACKAGE( crypto )

#endif

