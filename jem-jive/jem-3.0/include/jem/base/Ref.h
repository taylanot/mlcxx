
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

#ifndef JEM_BASE_REF_H
#define JEM_BASE_REF_H

#include <utility>
#include <jem/base/Collectable.h>


JEM_BEGIN_PACKAGE_BASE


class Object;

template <class T> class  Ref;
template <>        class  Ref<Object>;


//-----------------------------------------------------------------------
//   class RawRef
//-----------------------------------------------------------------------


class                     RawRef {};

extern const RawRef       RAW_REF;


//-----------------------------------------------------------------------
//   class Ref
//-----------------------------------------------------------------------


template <class T>

  class Ref

{
 public:

  constexpr inline        Ref           () noexcept;

  constexpr inline        Ref

    ( const Nil&            rhs )          noexcept;

  constexpr inline        Ref

    ( std::nullptr_t        rhs )          noexcept;

  inline                  Ref

    ( T*                    rhs )          noexcept;

  inline                  Ref

    ( const Ref&            rhs )          noexcept;

  inline                  Ref

    ( Ref&&                 rhs )          noexcept;

  template <class U>
    inline                Ref

    ( const Ref<U>&         rhs )          noexcept;

  template <class U>
    inline                Ref

    ( Ref<U>&&              rhs )          noexcept;

  inline                  Ref

    ( T&                    rhs,
      Collectable&          base )         noexcept;

  constexpr inline        Ref

    ( T&                    rhs,
      Collectable&          base,
      const RawRef&         raw )          noexcept;

  inline                 ~Ref           () noexcept;

  Ref&                    operator  =

    ( const Ref&            rhs )          noexcept;

  inline Ref&             operator  =

    ( Ref&&                 rhs )          noexcept;

  template <class U>
    Ref&                  operator  =

    ( const Ref<U>&         rhs )          noexcept;

  template <class U>
    Ref&                  operator  =

    ( Ref<U>&&              rhs )          noexcept;

  Ref&                    operator  =

    ( T*                    rhs )          noexcept;

  inline Ref&             operator  =

    ( const Nil&            rhs )          noexcept;

  inline Ref&             operator  =

    ( std::nullptr_t        rhs )          noexcept;

  template <class U>
    constexpr
    inline bool           operator ==

    ( const Ref<U>&         rhs )          const noexcept;

  template <class U>
    constexpr
    inline bool           operator ==

    ( const U*              rhs )          const noexcept;

  constexpr inline bool   operator ==

    ( const Nil&            rhs )          const noexcept;

  constexpr inline bool   operator ==

    ( std::nullptr_t        rhs )          const noexcept;

  inline T*               operator ->   () const;
  inline T&               operator  *   () const;
  constexpr
    explicit inline       operator bool () const noexcept;
  constexpr inline bool   operator  !   () const noexcept;

  inline void             swap

    ( Ref&                  rhs )                noexcept;

  constexpr inline T*     get           () const noexcept;
  constexpr inline
    Collectable*          getBase       () const noexcept;

  inline void             release       ()       noexcept;


 private:

  T*                      obj_;
  Collectable*            base_;

};


//-----------------------------------------------------------------------
//   class RefCaster
//-----------------------------------------------------------------------


template <class T, class U>

  class RefCaster

{
 public:

  typedef Ref<T>          ReturnType;


  static inline Ref<T>    staticCast

    ( const Ref<U>&         rhs );

  static inline Ref<T>    dynamicCast

    ( const Ref<U>&         rhs );

  static Ref<T>           checkedCast

    ( const Ref<U>&         rhs );

};


//-----------------------------------------------------------------------
//   class RefCaster<T*,U>
//-----------------------------------------------------------------------


template <class T, class U>

  class RefCaster<T*,U>

{
 public:

  typedef T*              ReturnType;


  static inline T*        staticCast

    ( const Ref<U>&         rhs );

  static inline T*        dynamicCast

    ( const Ref<U>&         rhs );

  static T*               checkedCast

    ( const Ref<U>&         rhs );

};


//-----------------------------------------------------------------------
//   class RefCaster<T&,U>
//-----------------------------------------------------------------------


template <class T, class U>

  class RefCaster<T&,U>

{
 public:

  typedef T&              ReturnType;


  static inline T&        staticCast

    ( const Ref<U>&         rhs );

  static T&               dynamicCast

    ( const Ref<U>&         rhs );

  static T&               checkedCast

    ( const Ref<U>&         rhs );

};


//-----------------------------------------------------------------------
//   class DynRefCaster<T,U>
//-----------------------------------------------------------------------


template <class T, class U>

  class DynRefCaster

{
 public:

  static inline T*        dynamicCast

    ( const Ref<U>&         rhs );

};

//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class T>

  inline void             swap

  ( Ref<T>&                 lhs,
    Ref<T>&                 rhs ) noexcept;

template <class T>

  Ref<T>                  newInstance ();

template <class T, class... Args>

  Ref<T>                  newInstance

  ( Args&& ...              a );

template <class T, class U>

  constexpr inline bool   operator ==

  ( const T*                lhs,
    const Ref<U>&           rhs ) noexcept;

template <class T>

  constexpr inline bool   operator ==

  ( const Nil&              lhs,
    const Ref<T>&           rhs ) noexcept;

template <class T>

  constexpr inline bool   operator ==

  ( std::nullptr_t          lhs,
    const Ref<T>&           rhs ) noexcept;

template <class T, class U>

  constexpr inline bool   operator !=

  ( const Ref<T>&           lhs,
    const Ref<U>&           rhs ) noexcept;

template <class T, class U>

  constexpr inline bool   operator !=

  ( const Ref<T>&           lhs,
    const U*                rhs ) noexcept;

template <class T, class U>

  constexpr inline bool   operator !=

  ( const T*                lhs,
    const Ref<U>&           rhs ) noexcept;

template <class T>

  constexpr inline bool   operator !=

  ( const Ref<T>&           lhs,
    const Nil&              rhs ) noexcept;

template <class T>

  constexpr inline bool   operator !=

  ( const Nil&              lhs,
    const Ref<T>&           rhs ) noexcept;

template <class T>

  constexpr inline bool   operator !=

  ( const Ref<T>&           lhs,
    std::nullptr_t          rhs ) noexcept;

template <class T>

  constexpr inline bool   operator !=

  ( std::nullptr_t          lhs,
    const Ref<T>&           rhs ) noexcept;

template <class T, class U>

  inline
  typename RefCaster<T,U>::ReturnType

                          staticCast

  ( const Ref<U>&           rhs );

template <class T, class U>

  inline
  typename RefCaster<T,U>::ReturnType

                          dynamicCast

  ( const Ref<U>&           rhs );

template <class T, class U>

  inline
  typename RefCaster<T,U>::ReturnType

                          checkedCast

  ( const Ref<U>&           rhs );

template <class T>

  inline void             castError

  ( const char*             where,
    const Ref<T>&           rhs );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//  class Ref
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T>

  constexpr inline Ref<T>::Ref () noexcept :

    obj_  ( nullptr ),
    base_ ( nullptr )

{}


template <class T>

  constexpr inline Ref<T>::Ref ( const Nil& ) noexcept :

    obj_  ( nullptr ),
    base_ ( nullptr )

{}


template <class T>

  constexpr inline Ref<T>::Ref ( std::nullptr_t ) noexcept :

    obj_  ( nullptr ),
    base_ ( nullptr )

{}


template <class T>

  inline Ref<T>::Ref ( T* rhs ) noexcept

{
  obj_  = rhs;
  base_ = rhs;

  Collectable::addRef ( base_ );
}


template <class T>

  inline Ref<T>::Ref ( const Ref& rhs ) noexcept

{
  obj_  = rhs.obj_;
  base_ = rhs.base_;

  Collectable::addRef ( base_ );
}


template <class T>

  inline Ref<T>::Ref ( Ref&& rhs ) noexcept

{
  obj_  = rhs.get     ();
  base_ = rhs.getBase ();

  rhs.release ();
}


template <class T>
  template <class U>

  inline Ref<T>::Ref ( const Ref<U>& rhs ) noexcept

{
  obj_  = rhs.get     ();
  base_ = rhs.getBase ();

  Collectable::addRef ( base_ );
}


template <class T>
  template <class U>

  inline Ref<T>::Ref ( Ref<U>&& rhs ) noexcept

{
  obj_  = rhs.get     ();
  base_ = rhs.getBase ();

  rhs.release ();
}


template <class T>

  inline Ref<T>::Ref

  ( T&            rhs,
    Collectable&  base ) noexcept

{
  obj_  = & rhs;
  base_ = & base;

  Collectable::addRef ( base );
}


template <class T>

  constexpr inline Ref<T>::Ref

  ( T&             rhs,
    Collectable&   base,
    const RawRef&  raw ) noexcept :

    obj_  ( & rhs ),
    base_ ( & base )

{}


template <class T>

  inline Ref<T>::~Ref () noexcept

{
  Collectable::delRef ( base_ );
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T>

  Ref<T>& Ref<T>::operator = ( const Ref& rhs ) noexcept

{
  Collectable::addRef ( rhs.base_ );
  Collectable::delRef (     base_ );

  obj_  = rhs.obj_;
  base_ = rhs.base_;

  return *this;
}


template <class T>

  inline Ref<T>& Ref<T>::operator = ( Ref&& rhs ) noexcept

{
  swap ( rhs );

  return *this;
}


template <class T>
  template <class U>

  Ref<T>& Ref<T>::operator = ( const Ref<U>& rhs ) noexcept

{
  Collectable::addRef ( rhs.getBase() );
  Collectable::delRef ( base_ );

  obj_  = rhs.get     ();
  base_ = rhs.getBase ();

  return *this;
}


template <class T>
  template <class U>

  Ref<T>& Ref<T>::operator = ( Ref<U>&& rhs ) noexcept

{
  if ( base_ != rhs.getBase() )
  {
    Collectable::delRef ( base_ );

    obj_  = rhs.get     ();
    base_ = rhs.getBase ();

    rhs.release ();
  }

  return *this;
}


template <class T>

  Ref<T>& Ref<T>::operator = ( T* rhs ) noexcept

{
  Collectable::addRef ( rhs );
  Collectable::delRef ( base_ );

  obj_  = rhs;
  base_ = rhs;

  return *this;
}


template <class T>

  inline Ref<T>& Ref<T>::operator = ( const Nil& rhs ) noexcept

{
  Collectable::delRef ( base_ );

  obj_  = nullptr;
  base_ = nullptr;

  return *this;
}


template <class T>

  inline Ref<T>& Ref<T>::operator = ( std::nullptr_t rhs ) noexcept

{
  Collectable::delRef ( base_ );

  obj_  = nullptr;
  base_ = nullptr;

  return *this;
}


//-----------------------------------------------------------------------
//   equality operators
//-----------------------------------------------------------------------


template <class T>
  template <class U>

  constexpr inline bool
  Ref<T>::operator == ( const Ref<U>& rhs ) const noexcept

{
  return (base_ == rhs.getBase());
}


template <class T>
  template <class U>

  constexpr inline bool
  Ref<T>::operator == ( const U* rhs ) const noexcept

{
  return (obj_ == rhs);
}


template <class T>

  constexpr inline bool
  Ref<T>::operator == ( const Nil& rhs ) const noexcept

{
  return (obj_ == nullptr);
}


template <class T>

  constexpr inline bool
  Ref<T>::operator == ( std::nullptr_t rhs ) const noexcept

{
  return (obj_ == nullptr);
}


//-----------------------------------------------------------------------
//   dereferencing operators
//-----------------------------------------------------------------------


template <class T>

  inline T* Ref<T>::operator -> () const

{
  JEM_ASSERT2 ( get(), "NULL pointer" );

  return obj_;
}


template <class T>

  inline T& Ref<T>::operator * () const

{
  JEM_ASSERT2 ( get(), "NULL pointer" );

  return *obj_;
}


//-----------------------------------------------------------------------
//   bool operators
//-----------------------------------------------------------------------


template <class T>

  constexpr inline Ref<T>::operator bool () const noexcept

{
  return (obj_ != nullptr);
}


template <class T>

  constexpr inline bool Ref<T>::operator ! () const noexcept

{
  return obj_ == nullptr;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void Ref<T>::swap ( Ref& rhs ) noexcept

{
  jem::swap ( obj_,  rhs.obj_  );
  jem::swap ( base_, rhs.base_ );
}


//-----------------------------------------------------------------------
//   get & getBase
//-----------------------------------------------------------------------


template <class T>

  constexpr inline T* Ref<T>::get () const noexcept

{
  return obj_;
}


template <class T>

  constexpr inline Collectable* Ref<T>::getBase () const noexcept

{
  return base_;
}


//-----------------------------------------------------------------------
//   release
//-----------------------------------------------------------------------


template <class T>

  inline void Ref<T>::release () noexcept

{
  obj_  = nullptr;
  base_ = nullptr;
}


//=======================================================================
//   class RefCaster
//=======================================================================

//-----------------------------------------------------------------------
//   staticCast
//-----------------------------------------------------------------------


template <class T, class U>

  inline Ref<T> RefCaster<T,U>::staticCast ( const Ref<U>& rhs )

{
  if ( ! rhs )
  {
    return nullptr;
  }
  else
  {
    return Ref<T> ( static_cast<T&>( *rhs ), *(rhs.getBase()) );
  }
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


template <class T, class U>

  inline Ref<T> RefCaster<T,U>::dynamicCast ( const Ref<U>& rhs )

{
  T*  tmp = DynRefCaster<T,U>::dynamicCast ( rhs );

  if ( tmp )
  {
    return Ref<T> ( *tmp, *(rhs.getBase()) );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   checkedCast
//-----------------------------------------------------------------------


template <class T, class U>

  Ref<T> RefCaster<T,U>::checkedCast ( const Ref<U>& rhs )

{
  if ( ! rhs )
  {
    return nullptr;
  }
  else
  {
    T*  tmp = DynRefCaster<T,U>::dynamicCast ( rhs );

    if ( ! tmp )
    {
      castError ( JEM_FUNC, rhs );
    }

    return Ref<T> ( *tmp, *(rhs.getBase()) );
  }
}


//=======================================================================
//   class RefCaster<T*,U>
//=======================================================================

//-----------------------------------------------------------------------
//   staticCast
//-----------------------------------------------------------------------


template <class T, class U>

  inline T* RefCaster<T*,U>::staticCast ( const Ref<U>& rhs )

{
  return static_cast<T*> ( rhs.get() );
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


template <class T, class U>

  inline T* RefCaster<T*,U>::dynamicCast ( const Ref<U>& rhs )

{
  return DynRefCaster<T,U>::dynamicCast ( rhs );
}


//-----------------------------------------------------------------------
//   checkedCast
//-----------------------------------------------------------------------


template <class T, class U>

  T* RefCaster<T*,U>::checkedCast ( const Ref<U>& rhs )

{
  T*  tmp = DynRefCaster<T,U>::dynamicCast ( rhs );

  if ( ! tmp && rhs )
  {
    castError ( JEM_FUNC, rhs );
  }

  return tmp;
}


//=======================================================================
//   class RefCaster<T&,U>
//=======================================================================

//-----------------------------------------------------------------------
//   staticCast
//-----------------------------------------------------------------------


template <class T, class U>

  inline T& RefCaster<T&,U>::staticCast ( const Ref<U>& rhs )

{
  return static_cast<T&> ( *rhs );
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


template <class T, class U>

  T& RefCaster<T&,U>::dynamicCast ( const Ref<U>& rhs )

{
  T*  tmp = DynRefCaster<T,U>::dynamicCast ( rhs );

  if ( ! tmp )
  {
    castError ( JEM_FUNC, rhs );
  }

  return *tmp;
}


//-----------------------------------------------------------------------
//   checkedCast
//-----------------------------------------------------------------------


template <class T, class U>

  T& RefCaster<T&,U>::checkedCast ( const Ref<U>& rhs )

{
  T*  tmp = DynRefCaster<T,U>::dynamicCast ( rhs );

  if ( ! tmp )
  {
    castError ( JEM_FUNC, rhs );
  }

  return *tmp;
}


//=======================================================================
//   class DynRefCaster
//=======================================================================


template <class T, class U>

  inline T* DynRefCaster<T,U>::dynamicCast ( const Ref<U>& rhs )

{
  return dynamic_cast<T*> ( rhs.get() );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void   swap

  ( Ref<T>&       lhs,
    Ref<T>&       rhs ) noexcept

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   newInstance (no args)
//-----------------------------------------------------------------------


template <class T>

  Ref<T> newInstance ()

{
  void*  mem = MemCache::alloc ( sizeof(T) );
  T*     obj;

  try
  {
    obj = ::new (mem) T();
  }
  catch ( ... )
  {
    MemCache::dealloc ( mem, sizeof(T) );
    throw;
  }

  return Ref<T> ( *obj, *obj, RAW_REF );
}


//-----------------------------------------------------------------------
//   newInstance (1 or more args)
//-----------------------------------------------------------------------


template <class T, class... Args>

  Ref<T>  newInstance ( Args&& ... args )

{
  void*  mem = MemCache::alloc ( sizeof(T) );
  T*     obj;

  try
  {
    obj = ::new (mem) T( std::forward<Args>( args )... );
  }
  catch ( ... )
  {
    MemCache::dealloc ( mem, sizeof(T) );
    throw;
  }

  return Ref<T> ( *obj, *obj, RAW_REF );
}


//-----------------------------------------------------------------------
//   equality operators
//-----------------------------------------------------------------------


template <class T, class U>

  constexpr
  inline bool     operator ==

  ( const T*        lhs,
    const Ref<U>&   rhs ) noexcept

{
  return rhs.operator == ( lhs );
}


template <class T>

  constexpr
  inline bool     operator ==

  ( const Nil&      lhs,
    const Ref<T>&   rhs ) noexcept

{
  return rhs.operator == ( lhs );
}


template <class T>

  constexpr
  inline bool     operator ==

  ( std::nullptr_t  lhs,
    const Ref<T>&   rhs ) noexcept

{
  return rhs.operator == ( lhs );
}


template <class T, class U>

  constexpr
  inline bool     operator !=

  ( const Ref<T>&   lhs,
    const Ref<U>&   rhs ) noexcept

{
  return ! lhs.operator == ( rhs );
}


template <class T, class U>

  constexpr
  inline bool     operator !=

  ( const Ref<T>&   lhs,
    const U*        rhs ) noexcept

{
  return ! lhs.operator == ( rhs );
}


template <class T, class U>

  constexpr
  inline bool     operator !=

  ( const T*        lhs,
    const Ref<U>&   rhs ) noexcept

{
  return ! rhs.operator == ( lhs );
}


template <class T>

  constexpr
  inline bool     operator !=

  ( const Ref<T>&   lhs,
    const Nil&      rhs ) noexcept

{
  return ! lhs.operator == ( rhs );
}


template <class T>

  constexpr
  inline bool     operator !=

  ( const Nil&      lhs,
    const Ref<T>&   rhs ) noexcept

{
  return ! rhs.operator == ( lhs );
}

template <class T>

  constexpr
  inline bool     operator !=

  ( const Ref<T>&   lhs,
    std::nullptr_t  rhs ) noexcept

{
  return !lhs.operator == ( rhs );
}

template <class T>

  constexpr
  inline bool     operator !=

  ( std::nullptr_t  lhs,
    const Ref<T>&   rhs ) noexcept

{
  return ! rhs.operator == ( lhs );
}


//-----------------------------------------------------------------------
//   staticCast
//-----------------------------------------------------------------------


template <class T, class U>

  inline
  typename RefCaster<T,U>::ReturnType

                  staticCast

  ( const Ref<U>&   rhs )

{
  return RefCaster<T,U>::staticCast ( rhs );
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


template <class T, class U>

  inline
  typename RefCaster<T,U>::ReturnType

                  dynamicCast

  ( const Ref<U>&   rhs )

{
  return RefCaster<T,U>::dynamicCast ( rhs );
}


template <class T, class U>

  inline
  typename RefCaster<T,U>::ReturnType

                  checkedCast

  ( const Ref<U>&   rhs )

{
  return RefCaster<T,U>::checkedCast ( rhs );
}


//-----------------------------------------------------------------------
//   castError
//-----------------------------------------------------------------------


void              castError

  ( const char*     where,
    Collectable*    rhs );


template <class T>

  inline void     castError

  ( const char*     where,
    const Ref<T>&   rhs )

{
  castError ( where, rhs.getBase() );
}


JEM_END_PACKAGE_BASE

#endif
