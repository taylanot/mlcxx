
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

#ifndef JEM_IO_STREAMCODEC_H
#define JEM_IO_STREAMCODEC_H

#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/base/PrecheckException.h>
#include <jem/io/InputStream.h>
#include <jem/io/OutputStream.h>
#include <jem/io/DataStreamOptions.h>
#include <jem/io/IOverflowException.h>
#include <jem/io/xdr.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class StreamCodec
//-----------------------------------------------------------------------


class StreamCodec : public DataStreamOptions
{
 public:

  template <class T>
    static inline void    decode

    ( InputStream&          in,
      T&                    value,
      int                   sex );

  static inline void      decode

    ( InputStream&          in,
      idx_t&                value,
      int                   sex,
      Options               options );

  template <class T>
    static void           decode

    ( InputStream&          in,
      T*                    values,
      idx_t                 n,
      int                   sex );

  static inline void      decode

    ( InputStream&          in,
      idx_t*                values,
      idx_t                 n,
      int                   sex,
      Options               options );

  template <class T>
    static inline void    encode

    ( OutputStream&         out,
      const T&              value,
      int                   sex );

  static inline void      encode

    ( OutputStream&         out,
      const idx_t&          value,
      int                   sex,
      Options               options );

  template <class T>
    static void           encode

    ( OutputStream&         in,
      const T*              values,
      idx_t                 n,
      int                   sex );

  static inline void      encode

    ( OutputStream&         in,
      const idx_t*          values,
      idx_t                 n,
      int                   sex,
      Options               options );

  template <class T>
    static inline void    encodeNoThrow

    ( OutputStream&         out,
      const T&              value,
      int                   sex )  noexcept;


 private:

  static const int        BUFSIZE_ = 256;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class T>

  inline void StreamCodec::decode

  ( InputStream&   in,
    T&             value,
    int            sex )

{
  byte  buf[ xdr::TypeTraits<T>::XDR_SIZE ];

  in.readAll  ( buf, xdr::TypeTraits<T>::XDR_SIZE );
  xdr::decode ( value, buf, sex );
}


inline void StreamCodec::decode

  ( InputStream&   in,
    idx_t&         value,
    int            sex,
    Options        options )

{
  int  bytec = xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE;

  if      ( options & IDX_SIZE_4BYTES )
  {
    bytec = 4;
  }
  else if ( options & IDX_SIZE_8BYTES )
  {
    bytec = 8;
  }

  if ( bytec == 4 )
  {
    int  itmp;

    decode ( in, itmp, sex );

    value = (idx_t) itmp;
  }
  else
  {
    lint  itmp;

    decode ( in, itmp, sex );

    value = (idx_t) itmp;
  }
}


template <class T>

  void StreamCodec::decode

  ( InputStream&   in,
    T*             values,
    idx_t          n,
    int            sex )

{
  JEM_PRECHECK ( n >= 0 );

  const int  XDR_SIZE = xdr::TypeTraits<T>::XDR_SIZE;
  const int  N        = BUFSIZE_ / XDR_SIZE;

  byte       buf[BUFSIZE_];
  int        k;

  while ( n > 0 )
  {
    k = (int) min ( n, (idx_t) N );

    in.readAll  ( buf, k * XDR_SIZE );
    xdr::decode ( values, k, buf, sex );

    values += k;
    n      -= k;
  }
}


inline void StreamCodec::decode

  ( InputStream&   in,
    idx_t*         values,
    idx_t          n,
    int            sex,
    Options        options )

{
  int  bytec = xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE;

  if      ( options & IDX_SIZE_4BYTES )
  {
    bytec = 4;
  }
  else if ( options & IDX_SIZE_8BYTES )
  {
    bytec = 8;
  }

  if ( bytec == xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE )
  {
    decode ( in, (JEM_IDX_T*) values, n, sex );
  }
  else
  {
    if ( bytec == 4 )
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        int  itmp;

        decode ( in, itmp, sex );

        values[i] = (idx_t) itmp;
      }
    }
    else
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        lint  itmp;

        decode ( in, itmp, sex );

        values[i] = (idx_t) itmp;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class T>

  inline void StreamCodec::encode

  ( OutputStream&  out,
    const T&       value,
    int            sex )

{
  byte  buf[ xdr::TypeTraits<T>::XDR_SIZE ];

  xdr::encode ( buf, value, sex );
  out.write   ( buf, xdr::TypeTraits<T>::XDR_SIZE );
}


inline void StreamCodec::encode

  ( OutputStream&  out,
    const idx_t&   value,
    int            sex,
    Options        options )

{
  int  bytec = xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE;

  if      ( options & IDX_SIZE_4BYTES )
  {
    bytec = 4;
  }
  else if ( options & IDX_SIZE_8BYTES )
  {
    bytec = 8;
  }

  if ( bytec == xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE )
  {
    encode ( out, value.value, sex );
  }
  else
  {
    if ( bytec == 4 )
    {
      if ( (value < (idx_t) minOf<int>()) ||
           (value > (idx_t) maxOf<int>()) )
      {
        throw IOverflowException (
          JEM_FUNC,
          "value of type jem::idx_t does not fit into four bytes"
        );
      }

      encode ( out, (int)  value, sex );
    }
    else
    {
      encode ( out, (lint) value, sex );
    }
  }
}


template <class T>

  void StreamCodec::encode

  ( OutputStream&  out,
    const T*       values,
    idx_t          n,
    int            sex )

{
  const int  XDR_SIZE = xdr::TypeTraits<T>::XDR_SIZE;
  const int  N        = BUFSIZE_ / XDR_SIZE;

  byte       buf[BUFSIZE_];
  int        k;

  while ( n > 0 )
  {
    k = (int) min ( n, (idx_t) N );

    xdr::encode ( buf, values, k, sex );
    out.write   ( buf, k * XDR_SIZE );

    values += k;
    n      -= k;
  }
}


inline void StreamCodec::encode

  ( OutputStream&  out,
    const idx_t*   values,
    idx_t          n,
    int            sex,
    Options        options )

{
  int  bytec = xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE;

  if      ( options & IDX_SIZE_4BYTES )
  {
    bytec = 4;
  }
  else if ( options & IDX_SIZE_8BYTES )
  {
    bytec = 8;
  }

  if ( bytec == xdr::TypeTraits<JEM_IDX_T>::XDR_SIZE )
  {
    encode ( out, (const JEM_IDX_T*) values, n, sex );
  }
  else
  {
    if ( bytec == 8 )
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        encode ( out, (lint) values[i], sex );
      }
    }
    else
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        idx_t  k = values[i];

        if ( (k < (idx_t) minOf<int>()) ||
             (k > (idx_t) maxOf<int>()) )
        {
          throw IOverflowException (
            JEM_FUNC,
            "value of type jem::idx_t does not fit into four bytes"
          );
        }

        encode ( out, (int) k, sex );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   encodeNoThrow
//-----------------------------------------------------------------------


template <class T>

  inline void StreamCodec::encodeNoThrow

  ( OutputStream&  out,
    const T&       value,
    int            sex ) noexcept

{
  byte  buf[ xdr::TypeTraits<T>::XDR_SIZE ];

  xdr::encode      ( buf, value, sex );
  out.writeNoThrow ( buf, xdr::TypeTraits<T>::XDR_SIZE );
}


JEM_END_PACKAGE( io )

#endif
