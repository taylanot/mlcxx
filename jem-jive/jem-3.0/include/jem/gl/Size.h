
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

#ifndef JEM_GL_SIZE_H
#define JEM_GL_SIZE_H

#include <jem/defines.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Size2
//-----------------------------------------------------------------------


template <class T>

  class Size2

{
 public:

  inline                    Size2     ();

  explicit inline           Size2

    ( T                       s );

  inline                    Size2

    ( T                       w,
      T                       h );

  inline                    Size2

    ( const Size2&            rhs );

  template <class U>
    inline                  Size2

    ( const Size2<U>&         rhs );

  inline Size2&             operator =

    ( const Size2&            rhs );

  template <class U>
    inline Size2&           operator =

    ( const Size2<U>&         rhs );

  inline void               swap

    ( Size2&                  rhs );

  inline bool               isValid   () const;


 public:

  T                         width;
  T                         height;

};



//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef Size2<GLsizei>      Size2i;
typedef Size2<GLfloat>      Size2f;
typedef Size2<GLdouble>     Size2d;


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T>

  inline void               swap

  ( Size2<T>&                 lhs,
    Size2<T>&                 rhs );

template <class T>

  inline Size2<T>           makeSize

  ( T                         width,
    T                         height );

template <class Output, class T>

  inline void               print

  ( Output&                   out,
    const Size2<T>&           size );







//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Size2
//=======================================================================


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T>

  inline Size2<T>::Size2 ()

{
  width = height = (T) 0;
}


template <class T>

  inline Size2<T>::Size2 ( T s )

{
  width = height = s;
}


template <class T>

  inline Size2<T>::Size2 ( T w, T h )

{
  width  = w;
  height = h;
}


template <class T>

  inline Size2<T>::Size2 ( const Size2& rhs )

{
  width  = rhs.width;
  height = rhs.height;
}


template <class T>
  template <class U>

  inline Size2<T>::Size2 ( const Size2<U>& rhs )

{
  width  = rhs.width;
  height = rhs.height;
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T>

  inline Size2<T>& Size2<T>::operator = ( const Size2& rhs )

{
  width  = rhs.width;
  height = rhs.height;

  return *this;
}


template <class T>
  template <class U>

  inline Size2<T>& Size2<T>::operator = ( const Size2<U>& rhs )

{
  width  = rhs.width;
  height = rhs.height;

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void Size2<T>::swap ( Size2& rhs )

{
  T  tmp;

  tmp        = width;
  width      = rhs.width;
  rhs.width  = tmp;
  tmp        = height;
  height     = rhs.height;
  rhs.height = tmp;
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


template <class T>

  inline bool Size2<T>::isValid () const

{
  return (width >= (T) 0 && height >= (T) 0);
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void swap ( Size2<T>& lhs, Size2<T>& rhs )

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   makeSize
//-----------------------------------------------------------------------


template <class T>

  inline Size2<T> makeSize ( T width, T height )

{
  return Size2<T> ( width, height );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output, class T>

  inline void               print

  ( Output&                   out,
    const Size2<T>&           size )

{
  print ( out, size.width, " x ", size.height );
}


JEM_END_PACKAGE( gl )

#endif
