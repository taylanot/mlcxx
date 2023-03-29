
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

#ifndef JEM_IO_DATAOUTPUT_H
#define JEM_IO_DATAOUTPUT_H

#include <jem/base/Interface.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class DataOutput
//-----------------------------------------------------------------------


class DataOutput : public Interface
{
 public:

  virtual void          encode

    ( bool                b ) = 0;

  virtual void          encode

    ( byte                b ) = 0;

  virtual void          encode

    ( char                c ) = 0;

  virtual void          encode

    ( short               s ) = 0;

  virtual void          encode

    ( int                 i ) = 0;

  virtual void          encode

    ( long                l ) = 0;

  virtual void          encode

    ( lint                l ) = 0;

  virtual void          encode

    ( idx_t               i ) = 0;

  virtual void          encode

    ( float               f ) = 0;

  virtual void          encode

    ( double              d ) = 0;

  virtual void          encode

    ( const bool*         buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const byte*         buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const char*         buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const short*        buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const int*          buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const long*         buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const lint*         buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const idx_t*        buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const float*        buf,
      idx_t               len ) = 0;

  virtual void          encode

    ( const double*       buf,
      idx_t               len ) = 0;


 protected:

  virtual              ~DataOutput  ();

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class Output>

  inline void           encode

  ( Output&               out,
    bool                  b );

template <class Output>

  inline void           encode

  ( Output&               out,
    byte                  b );

template <class Output>

  inline void           encode

  ( Output&               out,
    char                  c );

template <class Output>

  inline void           encode

  ( Output&               out,
    short                 i );

template <class Output>

  inline void           encode

  ( Output&               out,
    int                   i );

template <class Output>

  inline void           encode

  ( Output&               out,
    long                  i );

template <class Output>

  inline void           encode

  ( Output&               out,
    lint                  i );

template <class Output>

  inline void           encode

  ( Output&               out,
    idx_t                 i );

template <class Output>

  inline void           encode

  ( Output&               out,
    float                 x );

template <class Output>

  inline void           encode

  ( Output&               out,
    double                x );

template <class Output,
          class A, class B>

  inline void           encode

  ( Output&               out,
    const A&              a,
    const B&              b );

template <class Output,
          class A, class B, class... C>

  inline void           encode

  ( Output&               out,
    const A&              a,
    const B&              b,
    const C& ...          c );

template <class Output, class T>

  inline void           encodeArray

  ( Output&               out,
    const T*              buf,
    idx_t                 len );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void         encode

  ( Output&             out,
    bool                b )

{
  out.encode ( b );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    byte                b )

{
  out.encode ( b );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    char                c )

{
  out.encode ( c );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    short               i )

{
  out.encode ( i );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    int                 i )

{
  out.encode ( i );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    long                i )

{
  out.encode ( i );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    lint                i )

{
  out.encode ( i );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    idx_t               i )

{
  out.encode ( i );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    float               x )

{
  out.encode ( x );
}


template <class Output>

  inline void         encode

  ( Output&             out,
    double              x )

{
  out.encode ( x );
}


template <class Output,
          class A, class B>

  inline void         encode

  ( Output&             out,
    const A&            a,
    const B&            b )

{
  encode ( out, a );
  encode ( out, b );
}


template <class Output,
          class A, class B, class... C>

  inline void         encode

  ( Output&             out,
    const A&            a,
    const B&            b,
    const C& ...        c )

{
  encode ( out, a );
  encode ( out, b );
  encode ( out, c... );
}


//-----------------------------------------------------------------------
//   encodeArray
//-----------------------------------------------------------------------


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const bool*         buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const byte*         buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const char*         buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const short*        buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const int*          buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const long*         buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const lint*         buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const idx_t*        buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const float*        buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output>

  inline void         encodeArray

  ( Output&             out,
    const double*       buf,
    idx_t               n )

{
  out.encode ( buf, n );
}


template <class Output, class T>

  inline void         encodeArray

  ( Output&             out,
    const T*            buf,
    idx_t               n )

{
  for ( idx_t i = 0; i < n; i++ )
  {
    encode ( out, buf[i] );
  }
}


JEM_END_PACKAGE( io )

#endif
