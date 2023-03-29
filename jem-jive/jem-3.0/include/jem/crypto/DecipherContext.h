
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

#ifndef JEM_CRYPTO_DECIPHERCONTEXT_H
#define JEM_CRYPTO_DECIPHERCONTEXT_H

#include <jem/base/array/utilities.h>
#include <jem/crypto/Array.h>
#include <jem/crypto/CipherContext.h>


JEM_BEGIN_PACKAGE( crypto )


//-----------------------------------------------------------------------
//   class DecipherContext
//-----------------------------------------------------------------------


class DecipherContext : public CipherContext
{
 public:

  typedef CipherContext   Super;
  typedef
    DecipherContext       Self;


  virtual idx_t           update

    ( void*                 dest,
      const void*           data,
      idx_t                 size )            = 0;

  template <class T>
    T                     update

    ( const void*           data,
      idx_t                 size );

  template <class T>
    T                     update

    ( const ByteVector&     data );

  virtual idx_t           finish

    ( void*                 dest,
      const void*           data = nullptr,
      idx_t                 size = 0 )        = 0;

  template <class T>
    T                     finish           ();

  template <class T>
    T                     finish

    ( const void*           data,
      idx_t                 size );

  template <class T>
    T                     finish

    ( const ByteVector&     data );


 protected:

  virtual                ~DecipherContext  ();


 private:

  void                    update_

    ( ByteVector&           out,
      const void*           data,
      idx_t                 size );

  void                    update_

    ( String&               out,
      const void*           data,
      idx_t                 size );

  void                    finish_

    ( ByteVector&           out,
      const void*           data = nullptr,
      idx_t                 size = 0 );

  void                    finish_

    ( String&               out,
      const void*           data = nullptr,
      idx_t                 size = 0 );

};


typedef DecipherContext   DecipherCtx;


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


template <class T>

  T DecipherContext::update

  ( const void*  data,
    idx_t        size )

{
  T  out;

  update_ ( out, data, size );

  return out;
}


template <class T>

  T DecipherContext::update ( const ByteVector& data )

{
  T  out;

  update_ ( out, makeContiguous( data ).addr(), data.size() );

  return out;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


template <class T>

  T DecipherContext::finish ()

{
  T  out;

  finish_ ( out );

  return out;
}


template <class T>

  T DecipherContext::finish

  ( const void*  data,
    idx_t        size )

{
  T  out;

  finish_ ( out, data, size );

  return out;
}


template <class T>

  T DecipherContext::finish ( const ByteVector& data )

{
  T  out;

  finish_ ( out, makeContiguous( data ).addr(), data.size() );

  return out;
}


JEM_END_PACKAGE( crypto )

#endif

