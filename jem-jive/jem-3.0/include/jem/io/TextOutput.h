
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

#ifndef JEM_IO_TEXTOUTPUT_H
#define JEM_IO_TEXTOUTPUT_H

#include <cstring>
#include <jem/base/Interface.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class TextOutput
//-----------------------------------------------------------------------


class TextOutput : public Interface
{
 public:

  typedef TextOutput      Self;


  virtual void            write

    ( int                   c )          = 0;

  virtual void            write

    ( int                   c,
      idx_t                 n )          = 0;

  virtual void            write

    ( const char*           buf,
      idx_t                 len )        = 0;

  virtual void            printByte

    ( byte                  b )          = 0;

  virtual void            printBool

    ( bool                  b )          = 0;

  virtual void            printInt

    ( lint                  i )          = 0;

  virtual void            printFloat

    ( double                d )          = 0;

  template <class T>
    inline Self&          operator <<

    ( const T&               obj );


 protected:

  virtual                ~TextOutput  ();

};


//-----------------------------------------------------------------------
//   related classes
//-----------------------------------------------------------------------


struct                    Space       {};
extern const Space        space;

struct                    EndLine     {};
extern const EndLine      endl;


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Output>

  inline void             print

  ( Output&                 out,
    bool                    b );

template <class Output>

  inline void             print

  ( Output&                 out,
    char                    c );

template <class Output>

  inline void             print

  ( Output&                 out,
    byte                    b );

template <class Output>

  inline void             print

  ( Output&                 out,
    short                   i );

template <class Output>

  inline void             print

  ( Output&                 out,
    int                     i );

template <class Output>

  inline void             print

  ( Output&                 out,
    long                    i );

template <class Output>

  inline void             print

  ( Output&                 out,
    lint                    i );

template <class Output>

  inline void             print

  ( Output&                 out,
    idx_t                   i );

template <class Output>

  inline void             print

  ( Output&                 out,
    float                   x );

template <class Output>

  inline void             print

  ( Output&                 out,
    double                  x );

template <class Output>

  inline void             print

  ( Output&                 out,
    const char*             s );

template <class Output>

  inline void             print

  ( Output&                 out,
    const Space&            sp );

template <class Output>

  inline void             print

  ( Output&                 out,
    const EndLine&          eol );

template <class Output,
          class A, class B>

  inline void             print

  ( Output&                 in,
    const A&                a,
    const B&                b );

template <class Output,
          class A, class B, class... C>

  inline void             print

  ( Output&                 in,
    const A&                a,
    const B&                b,
    const C& ...            c );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   operator <<
//-----------------------------------------------------------------------


template <class T>

  inline TextOutput&  TextOutput::operator <<

  ( const T&            obj )

{
  print ( *this, obj );
  return *this;
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void         print

  ( Output&             out,
    bool                b )

{
  out.printBool ( b );
}


template <class Output>

  inline void         print

  ( Output&             out,
    char                c )

{
  out.write ( c );
}


template <class Output>

  inline void         print

  ( Output&             out,
    byte                b )

{
  out.printByte ( b );
}


template <class Output>

  inline void         print

  ( Output&             out,
    short               i )

{
  out.printInt ( (lint) i );
}


template <class Output>

  inline void         print

  ( Output&             out,
    int                 i )

{
  out.printInt ( (lint) i );
}


template <class Output>

  inline void         print

  ( Output&             out,
    long                i )

{
  out.printInt ( (lint) i );
}


template <class Output>

  inline void         print

  ( Output&             out,
    lint                i )

{
  out.printInt ( i );
}


template <class Output>

  inline void         print

  ( Output&             out,
    idx_t               i )

{
  out.printInt ( (lint) i );
}


template <class Output>

  inline void         print

  ( Output&             out,
    float               x )

{
  out.printFloat ( (double) x );
}


template <class Output>

  inline void         print

  ( Output&             out,
    double              x )

{
  out.printFloat ( x );
}


template <class Output>

  inline void         print

  ( Output&             out,
    const char*         str )

{
  if ( str )
  {
    out.write ( str, (idx_t) std::strlen( str ) );
  }
}


template <class Output>

  inline void         print

  ( Output&             out,
    const Space&        sp )

{
  out.write ( ' ' );
}


template <class Output>

  inline void         print

  ( Output&             out,
    const EndLine&      eol )

{
  out.write ( '\n' );
}


template <class Output,
          class A, class B>

  inline void         print

  ( Output&             out,
    const A&            a,
    const B&            b )

{
  print ( out, a );
  print ( out, b );
}


template <class Output,
          class A, class B, class... C>

  inline void         print

  ( Output&             out,
    const A&            a,
    const B&            b,
    const C& ...        c )

{
  print ( out, a );
  print ( out, b );
  print ( out, c... );
}


//-----------------------------------------------------------------------
//   printSequence
//-----------------------------------------------------------------------


template <class Output,
          class Iterator, class Sep>

  void                printSequence

  ( Output&             out,
    Iterator            first,
    Iterator            last,
    const Sep&          sep )

{
  while ( first != last )
  {
    print ( out, *first );

    ++first;

    if ( first != last )
    {
      print ( out, sep );
    }
  }
}


JEM_END_PACKAGE( io )

#endif
