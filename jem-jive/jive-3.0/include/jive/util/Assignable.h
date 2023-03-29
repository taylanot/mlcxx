
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_UTIL_ASSIGNABLE_H
#define JIVE_UTIL_ASSIGNABLE_H

#include <utility>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Assignable
//-----------------------------------------------------------------------


template <class T>

  class Assignable : public T

{
 public:

  typedef Assignable    Self;
  typedef T             Super;


  inline                Assignable  ();
  inline                Assignable

    ( const T&            rhs );

  inline                Assignable

    ( T&&                 rhs );

  inline                Assignable

    ( const Self&         rhs );

  inline                Assignable

    ( Self&&              rhs );

  inline Assignable&    operator =

    ( const T&            rhs );

  inline Assignable&    operator =

    ( const Self&         rhs );

  inline void           swap

    ( const Self&         rhs );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T>

  inline void           swap

  ( Assignable<T>&        lhs,
    Assignable<T>&        rhs );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Assignable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T>

  inline Assignable<T>::Assignable ()

{}


template <class T>

  inline Assignable<T>::Assignable ( const T& rhs ) :

    Super ( rhs )

{}


template <class T>

  inline Assignable<T>::Assignable ( T&& rhs ) :

    Super ( std::move( rhs ) )

{}


template <class T>

  inline Assignable<T>::Assignable ( const Self& rhs ) :

    Super ( rhs )

{}


template <class T>

  inline Assignable<T>::Assignable ( Self&& rhs ) :

    Super ( std::move( rhs ) )

{}


//-----------------------------------------------------------------------
//   operator =
//-----------------------------------------------------------------------


template <class T>

  inline Assignable<T>& Assignable<T>::operator = ( const T& rhs )

{
  Super::assign_ ( rhs );

  return *this;
}


template <class T>

  inline Assignable<T>& Assignable<T>::operator = ( const Self& rhs )

{
  Super::assign_ ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void Assignable<T>::swap ( const Self& rhs )

{
  Super::swap_ ( rhs );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void     swap

  ( Assignable<T>&  lhs,
    Assignable<T>&  rhs )

{
  lhs.swap ( rhs );
}


JIVE_END_PACKAGE( util )

#endif
