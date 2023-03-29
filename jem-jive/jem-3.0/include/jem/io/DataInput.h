
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

#ifndef JEM_IO_DATAINPUT_H
#define JEM_IO_DATAINPUT_H

#include <jem/base/Interface.h>

namespace jem
{
  class String;
}


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class DataInput
//-----------------------------------------------------------------------


class DataInput : public Interface
{
 public:

  virtual bool          decodeBool    () = 0;
  virtual byte          decodeByte    () = 0;
  virtual char          decodeChar    () = 0;
  virtual short         decodeShort   () = 0;
  virtual int           decodeInt     () = 0;
  virtual long          decodeLong    () = 0;
  virtual lint          decodeLint    () = 0;
  virtual idx_t         decodeIdx     () = 0;
  virtual float         decodeFloat   () = 0;
  virtual double        decodeDouble  () = 0;

  virtual void          decode

    ( bool*               buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( byte*               buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( char*               buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( short*              buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( int*                buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( long*               buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( lint*               buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( idx_t*              buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( float*              buf,
      idx_t               len )           = 0;

  virtual void          decode

    ( double*             buf,
      idx_t               len )           = 0;


 protected:

  virtual              ~DataInput     ();

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void           decode

  ( Input&                in,
    bool&                 b );

template <class Input>

  inline void           decode

  ( Input&                in,
    byte&                 b );

template <class Input>

  inline void           decode

  ( Input&                in,
    char&                 c );

template <class Input>

  inline void           decode

  ( Input&                in,
    short&                i );

template <class Input>

  inline void           decode

  ( Input&                in,
    unsigned short&       i );

template <class Input>

  inline void           decode

  ( Input&                in,
    int&                  i );

template <class Input>

  inline void           decode

  ( Input&                in,
    unsigned int&         i );

template <class Input>

  inline void           decode

  ( Input&                in,
    long&                 i );

template <class Input>

  inline void           decode

  ( Input&                in,
    unsigned long&        i );

template <class Input>

  inline void           decode

  ( Input&                in,
    lint&                 i );

template <class Input>

  inline void           decode

  ( Input&                in,
    ulint&                i );

template <class Input>

  inline void           decode

  ( Input&                in,
    idx_t&                i );

template <class Input>

  inline void           decode

  ( Input&                in,
    uidx_t&               i );

template <class Input>

  inline void           decode

  ( Input&                in,
    float&                x );

template <class Input>

  inline void           decode

  ( Input&                in,
    double&               x );

template <class Input,
          class A, class B>

  inline void           decode

  ( Input&                in,
    A&                    a,
    B&                    b );

template <class Input,
          class A, class B, class... C>

  inline void           decode

  ( Input&                in,
    A&                    a,
    B&                    b,
    C& ...                c );

template <class Input, class T>

  inline void           decodeArray

  ( Input&                in,
    T*                    buf,
    idx_t                 len );

void                    decodeError

  ( const char*           where );

void                    decodeError

  ( const char*           where,
    const char*           what );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void         decode

  ( Input&              in,
    bool&               b )

{
  b = in.decodeBool ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    byte&               b )

{
  b = in.decodeByte ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    char&               c )

{
  c = in.decodeChar ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    short&              i )

{
  i = in.decodeShort ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    unsigned short&     i )

{
  short  tmp = in.decodeShort ();

  if ( tmp < 0 )
  {
    decodeError ( JEM_FUNC, "an unsigned short int" );
  }

  i = (unsigned short) tmp;
}


template <class Input>

  inline void        decode

  ( Input&              in,
    int&                i )

{
  i = in.decodeInt ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    unsigned int&       i )

{
  int  tmp = in.decodeInt ();

  if ( tmp < 0 )
  {
    decodeError ( JEM_FUNC, "an unsigned int" );
  }

  i = (unsigned int) tmp;
}


template <class Input>

  inline void        decode

  ( Input&              in,
    long&               i )

{
  i = in.decodeLong ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    unsigned long&      i )

{
  long  tmp = in.decodeLong ();

  if ( tmp < 0 )
  {
    decodeError ( JEM_FUNC, "an unsigned long" );
  }

  i = (unsigned long) tmp;
}


template <class Input>

  inline void         decode

  ( Input&              in,
    lint&               i )

{
  i = in.decodeLint ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    ulint&              i )

{
  lint  tmp = in.decodeLint ();

  if ( tmp < 0_lint )
  {
    decodeError ( JEM_FUNC, "an unsigned lint" );
  }

  i = (ulint) tmp;
}


template <class Input>

  inline void         decode

  ( Input&              in,
    idx_t&              i )

{
  i = in.decodeIdx ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    uidx_t&             i )

{
  idx_t  tmp = in.decodeIdx ();

  if ( tmp < 0_idx )
  {
    decodeError ( JEM_FUNC, "an unsigned idx_t" );
  }

  i = (uidx_t) tmp;
}


template <class Input>

  inline void         decode

  ( Input&              in,
    float&              x )

{
  x = in.decodeFloat ();
}


template <class Input>

  inline void         decode

  ( Input&              in,
    double&             x )

{
  x = in.decodeDouble ();
}


template <class Input,
          class A, class B>

  inline void         decode

  ( Input&              in,
    A&                  a,
    B&                  b )

{
  decode ( in, a );
  decode ( in, b );
}


template <class Input,
          class A, class B, class... C>

  inline void         decode

  ( Input&              in,
    A&                  a,
    B&                  b,
    C& ...              c )

{
  decode ( in, a );
  decode ( in, b );
  decode ( in, c... );
}


//-----------------------------------------------------------------------
//   decodeArray
//-----------------------------------------------------------------------


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    bool*               buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    byte*               buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    char*               buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    short*              buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    int*                buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    long*               buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    lint*               buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    idx_t*              buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    float*              buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input>

  inline void         decodeArray

  ( Input&              in,
    double*             buf,
    idx_t               n )

{
  in.decode ( buf, n );
}


template <class Input, class T>

  inline void         decodeArray

  ( Input&              in,
    T*                  buf,
    idx_t               n )

{
  for ( idx_t i = 0; i < n; i++ )
  {
    decode ( in, buf[i] );
  }
}


JEM_END_PACKAGE( io )

#endif
