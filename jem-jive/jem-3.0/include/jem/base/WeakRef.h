
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

#ifndef JEM_BASE_WEAKREF_H
#define JEM_BASE_WEAKREF_H

#include <utility>
#include <jem/base/Ref.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class WeakRefImp
//-----------------------------------------------------------------------


class WeakRefImp
{
 public:

  typedef WeakRefImp      Self;


                          WeakRefImp  () noexcept;

                          WeakRefImp

    ( Collectable*          obj );

                          WeakRefImp

    ( const Self&           rhs );

  inline                  WeakRefImp

    ( Self&&                rhs )        noexcept;

                         ~WeakRefImp  ();

  Self&                   operator =

    ( const Self&           rhs );

  inline Self&            operator =

    ( Self&&                rhs )        noexcept;

  Collectable*            get         () const;

  inline void             swap

    ( Self&                 rhs )        noexcept;


 private:

  WeakProxy*              proxy_;

};


//-----------------------------------------------------------------------
//   class WeakRef
//-----------------------------------------------------------------------


template <class T>

  class WeakRef

{
 public:

  typedef WeakRef         Self;


  inline                  WeakRef     () noexcept;

  inline                  WeakRef

    ( const Nil&            rhs )        noexcept;

  inline                  WeakRef

    ( std::nullptr_t        rhs )        noexcept;

  inline                  WeakRef

    ( const Ref<T>&         rhs );

  inline                  WeakRef

    ( const WeakRef&        rhs );

  inline                  WeakRef

    ( WeakRef&&             rhs )        noexcept;

  inline WeakRef&         operator =

    ( const WeakRef&        rhs );

  inline WeakRef&         operator =

    ( WeakRef&&              rhs )       noexcept;

  inline WeakRef&         operator =

    ( const Nil&            rhs );

  inline WeakRef&         operator =

    ( std::nullptr_t        rhs );

  inline Ref<T>           get         () const;

  inline void             swap

    ( WeakRef&              rhs )        noexcept;


 private:

  T*                      object_;
  WeakRefImp              weakref_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T>

  inline void             swap

  ( WeakRef<T>&             lhs,
    WeakRef<T>&             rhs ) noexcept;







//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class WeakRefImp
//=======================================================================


inline WeakRefImp::WeakRefImp ( Self&& rhs ) noexcept
{
  proxy_     = rhs.proxy_;
  rhs.proxy_ = nullptr;
}


inline WeakRefImp& WeakRefImp::operator = ( Self&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


inline void WeakRefImp::swap ( Self& rhs ) noexcept
{
  jem::swap ( proxy_, rhs.proxy_ );
}


//=======================================================================
//   class WeakRef
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T>

  inline WeakRef<T>::WeakRef () noexcept :

  object_ ( nullptr )

{}


template <class T>

  inline WeakRef<T>::WeakRef ( const Nil& ) noexcept :

  object_ ( nullptr )

{}


template <class T>

  inline WeakRef<T>::WeakRef ( std::nullptr_t ) noexcept :

  object_ ( nullptr )

{}


template <class T>

  inline WeakRef<T>::WeakRef ( const Ref<T>& rhs ) :

    object_  ( rhs.get()     ),
    weakref_ ( rhs.getBase() )

{}


template <class T>

  inline WeakRef<T>::WeakRef ( const Self& rhs ) :

    object_  ( rhs.object_  ),
    weakref_ ( rhs.weakref_ )

{}


template <class T>

  inline WeakRef<T>::WeakRef ( Self&& rhs ) noexcept :

    object_  ( rhs.object_  ),
    weakref_ ( std::move( rhs.weakref_ ) )

{
  rhs.object_ ( nullptr );
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T>

  inline WeakRef<T>& WeakRef<T>::operator = ( const Self& rhs )

{
  object_  = rhs.object_;
  weakref_ = rhs.weakref_;

  return *this;
}


template <class T>

  inline WeakRef<T>& WeakRef<T>::operator = ( Self&& rhs ) noexcept

{
  swap ( rhs );

  return *this;
}


template <class T>

  inline WeakRef<T>& WeakRef<T>::operator = ( const Nil& )

{
  object_  = nullptr;
  weakref_ = WeakRefImp ();

  return *this;
}


template <class T>

  inline WeakRef<T>& WeakRef<T>::operator = ( std::nullptr_t )

{
  object_  = nullptr;
  weakref_ = WeakRefImp ();

  return *this;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


template <class T>

  inline Ref<T> WeakRef<T>::get () const

{
  Collectable*  base = weakref_.get ();

  if ( base )
  {
    return Ref<T> ( *object_, *base, RAW_REF );
  }
  else
  {
    const_cast<Self*> ( this ) -> object_ = nullptr;

    return Ref<T> ();
  }
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void WeakRef<T>::swap ( WeakRef& rhs ) noexcept

{
  jem    ::swap ( object_, rhs.object_  );
  weakref_.swap (          rhs.weakref_ );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void    swap

  ( WeakRef<T>&    lhs,
    WeakRef<T>&    rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE_BASE

#endif
