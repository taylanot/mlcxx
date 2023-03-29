
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

#ifndef JEM_BASE_REFLECT_METHODTRAITS_H
#define JEM_BASE_REFLECT_METHODTRAITS_H

#include <jem/base/ObjectTraits.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class MethodTraitsBase
//-----------------------------------------------------------------------


template <class R, class A = Nil, class B = Nil, class C = Nil>

  class MethodTraitsBase

{
 public:

  static const int  ARG_COUNT = 3;


  static Class*     getReturnType ()
  {
    return ObjectTraits<R>::getType ();
  }

  static void       getArgTypes

    ( Class**         types )

  {
    types[0] = ObjectTraits<A>::getType ();
    types[1] = ObjectTraits<B>::getType ();
    types[2] = ObjectTraits<C>::getType ();
  }

};


//-----------------------------------------------------------------------
//   class MethodTraitsBase<R,A,B,Nil>
//-----------------------------------------------------------------------


template <class R, class A, class B>

  class MethodTraitsBase<R,A,B,Nil>

{
 public:

  static const int ARG_COUNT = 2;


  static Class*       getReturnType ()
  {
    return ObjectTraits<R>::getType ();
  }


  static void getArgTypes ( Class** types )
  {
    types[0] = ObjectTraits<A>::getType ();
    types[1] = ObjectTraits<B>::getType ();
  }

};


//-----------------------------------------------------------------------
//   class MethodTraitsBase<R,A,Nil,Nil>
//-----------------------------------------------------------------------


template <class R, class A>

  class MethodTraitsBase<R,A,Nil,Nil>

{
 public:

  static const int  ARG_COUNT = 1;


  static Class*     getReturnType ()
  {
    return ObjectTraits<R>::getType();
  }

  static void       getArgTypes

    ( Class**         types )

  {
    types[0] = ObjectTraits<A>::getType ();
  }

};


//-----------------------------------------------------------------------
//   class MethodTraitsBase<R,Nil,Nil,Nil>
//-----------------------------------------------------------------------


template <class R>

  class MethodTraitsBase<R,Nil,Nil,Nil>

{
 public:

  static const int  ARG_COUNT = 0;


  static Class*     getReturnType ()
  {
    return ObjectTraits<R>::getType ();
  }

  static void       getArgTypes

    ( Class** )

  {}

};


//-----------------------------------------------------------------------
//   class MethodTraits (primary template)
//-----------------------------------------------------------------------


template <class T>

  class MethodTraits {};


//-----------------------------------------------------------------------
//   class MethodTraits<void (*) ()>
//-----------------------------------------------------------------------


template <>

  class MethodTraits<void (*) ()> :
  public MethodTraitsBase<void>

{
 public:

  static const bool   IS_STATIC = true;


  static Ref<Object>  call ( void (*f) (),
                             const Ref<Object>* )

  {
    f();

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (T::*) ()>
//-----------------------------------------------------------------------


template <class T>

  class MethodTraits<void (T::*) ()> :
  public MethodTraitsBase<void,T>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( void (T::*m) (),
                             const Ref<Object>* args )
  {
    T& tmp = ObjectTraits<T&>::toValue ( args[0] );

    (tmp.*m) ();

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (T::*) () const>
//-----------------------------------------------------------------------


template <class T>

  class MethodTraits<void (T::*) () const> :
  public MethodTraitsBase<void,T>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( void (T::*m) () const,
                             const Ref<Object>* args )
  {
    const T& tmp = ObjectTraits<const T&>::toValue ( args[0] );

    (tmp.*m)();

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (*) ()>
//-----------------------------------------------------------------------


template <class R>

  class MethodTraits<R (*) ()> :
  public MethodTraitsBase<R>

{
 public:

  static const bool   IS_STATIC = true;

  static Ref<Object>  call ( R (*f) (),
                             const Ref<Object>* )
  {
    return ObjectTraits<R>::toObject ( f() );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (T::*) ()>
//-----------------------------------------------------------------------


template <class T, class R>

  class MethodTraits<R (T::*) ()> :
  public MethodTraitsBase<R,T>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( R (T::*m) (),
                             const Ref<Object>* args )
  {
    T& tmp = ObjectTraits<T&>::toValue ( args[0] );

    return ObjectTraits<R>::toObject ( (tmp.*m)() );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (T::*) () const>
//-----------------------------------------------------------------------


template <class T, class R>

  class MethodTraits<R (T::*) () const> :
  public MethodTraitsBase<R,T>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call( R (T::*m) () const,
                            const Ref<Object>* args )
  {
    const T& tmp = ObjectTraits<const T&>::toValue ( args[0] );

    return ObjectTraits<R>::toObject ( (tmp.*m)() );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (*) ( A )>
//-----------------------------------------------------------------------


template <class A>

  class MethodTraits<void (*) ( A )> :
  public MethodTraitsBase<void,A>

{
 public:

  static const bool   IS_STATIC = true;

  static Ref<Object>  call ( void (*f) ( A ),
                             const Ref<Object>* args )
  {
    f ( ObjectTraits<A>::toValue ( args[0] ) );

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (T::*) ( A )>
//-----------------------------------------------------------------------


template <class T, class A>

  class MethodTraits<void (T::*) ( A )> :
  public MethodTraitsBase<void,T,A>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( void (T::*m) ( A ),
                             const Ref<Object>* args )
  {
    T& tmp = ObjectTraits<T&>::toValue ( args[0] );

    (tmp.*m)( ObjectTraits<A>::toValue ( args[1] ) );

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (T::*) ( A ) const>
//-----------------------------------------------------------------------


template <class T, class A>

  class MethodTraits<void (T::*) ( A ) const> :
  public MethodTraitsBase<void,T,A>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( void (T::*m) ( A ) const,
                             const Ref<Object>* args )
  {
    const T& tmp = ObjectTraits<const T&>::toValue ( args[0] );

    (tmp.*m)( ObjectTraits<A>::toValue ( args[1] ) );

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (*) ( A )>
//-----------------------------------------------------------------------


template <class R, class A>

  class MethodTraits<R (*) ( A )> :
  public MethodTraitsBase<R,A>

{
 public:

  static const bool   IS_STATIC = true;

  static Ref<Object>  call ( R (*f) ( A ),
                             const Ref<Object>* args )
  {
    return ObjectTraits<R>::toObject (
      f ( ObjectTraits<A>::toValue ( args[0] ) )
    );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (T::*) ( A )>
//-----------------------------------------------------------------------


template <class T, class R, class A>

  class MethodTraits<R (T::*) ( A )> :
  public MethodTraitsBase<R,T,A>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( R (T::*m) ( A ),
                             const Ref<Object>* args )
  {
    T& tmp = ObjectTraits<T&>::toValue ( args[0] );

    return ObjectTraits<R>::toObject (
      (tmp.*m)( ObjectTraits<A>::toValue ( args[1] ) )
    );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (T::*) ( A ) const>
//-----------------------------------------------------------------------


template <class T, class R, class A>

  class MethodTraits<R (T::*) ( A ) const> :
  public MethodTraitsBase<R,T,A>

{
 public:

  static const bool   IS_STATIC = false;


  static Ref<Object>  call ( R (T::*m) ( A ) const,
                             const Ref<Object>* args )
  {
    const T& tmp = ObjectTraits<const T&>::toValue ( args[0] );

    return ObjectTraits<R>::toObject(
      (tmp.*m)( ObjectTraits<A>::toValue ( args[1] ) )
    );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (*) (A, B )>
//-----------------------------------------------------------------------


template <class A, class B>

  class MethodTraits<void (*) ( A, B )> :
  public MethodTraitsBase<void,A,B>

{
 public:

  static const bool   IS_STATIC = true;


  static Ref<Object>  call ( void (*f) ( A, B ),
                             const Ref<Object>* args )
  {
    f ( ObjectTraits<A>::toValue ( args[0] ),
        ObjectTraits<B>::toValue ( args[1] ) );

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (T::*) ( A, B )>
//-----------------------------------------------------------------------


template <class T, class A, class B>

  class MethodTraits<void (T::*) ( A, B )> :
  public MethodTraitsBase<void,T,A,B>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( void (T::*m) ( A, B ),
                             const Ref<Object>* args )
  {
    T& tmp = ObjectTraits<T&>::toValue ( args[0] );

    (tmp.*m)( ObjectTraits<A>::toValue ( args[1] ),
              ObjectTraits<B>::toValue ( args[2] ) );

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (T::*) ( A, B ) const>
//-----------------------------------------------------------------------


template <class T, class A, class B>

  class MethodTraits<void (T::*) ( A, B ) const> :
  public MethodTraitsBase<void,T,A,B>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( void (T::*m) ( A, B ) const,
                             const Ref<Object>* args )
  {
    const T& tmp = ObjectTraits<const T&>::toValue ( args[0] );

    (tmp.*m)( ObjectTraits<A>::toValue ( args[1] ),
              ObjectTraits<B>::toValue ( args[2] ) );

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (*) ( A, B )>
//-----------------------------------------------------------------------


template <class R, class A, class B>

  class MethodTraits<R (*) ( A, B )> :
  public MethodTraitsBase<R,A,B>

{
 public:

  static const bool   IS_STATIC = true;

  static Ref<Object>  call ( R (*f) ( A, B ),
                             const Ref<Object>* args )
  {
    return ObjectTraits<R>::toObject (
      f( ObjectTraits<A>::toValue ( args[0] ),
         ObjectTraits<B>::toValue ( args[1] ) )
    );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (T::*) ( A, B )>
//-----------------------------------------------------------------------


template <class T, class R, class A, class B>

  class MethodTraits<R (T::*) ( A, B )> :
  public MethodTraitsBase<R,T,A,B>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( R (T::*m) ( A, B ),
                             const Ref<Object>* args )
  {
    T& tmp = ObjectTraits<T&>::toValue ( args[0] );

    return ObjectTraits<R>::toObject (
      (tmp.*m)( ObjectTraits<A>::toValue ( args[1] ),
                ObjectTraits<B>::toValue ( args[2] ) )
    );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (T::*) ( A, B ) const>
//-----------------------------------------------------------------------


template <class T, class R, class A, class B>

  class MethodTraits<R (T::*) ( A, B ) const> :
  public MethodTraitsBase<R,T,A,B>

{
 public:

  static const bool   IS_STATIC = false;

  static Ref<Object>  call ( R (T::*m) ( A, B ) const,
                             const Ref<Object>* args )
  {
    const T& tmp = ObjectTraits<const T&>::toValue ( args[0] );

    return ObjectTraits<R>::toObject (
      (tmp.*m)( ObjectTraits<A>::toValue ( args[1] ),
                ObjectTraits<B>::toValue ( args[2] ) )
    );
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<void (*) ( A, B, C )
//-----------------------------------------------------------------------


template <class A, class B, class C>

  class MethodTraits<void (*) ( A, B, C )> :
  public MethodTraitsBase<void,A,B,C>

{
 public:

  static const bool   IS_STATIC = true;

  static Ref<Object>  call ( void (*f) ( A, B, C ),
                             const Ref<Object>* args )
  {
    f( ObjectTraits<A>::toValue ( args[0] ),
       ObjectTraits<B>::toValue ( args[1] ),
       ObjectTraits<C>::toValue ( args[2] ) );

    return nullptr;
  }

};


//-----------------------------------------------------------------------
//   class MethodTraits<R (*) ( A, B, C )>
//-----------------------------------------------------------------------


template <class R, class A, class B, class C>

  class MethodTraits<R (*) ( A, B, C )> :
  public MethodTraitsBase<R,A,B,C>

{
 public:

  static const bool   IS_STATIC = true;

  static Ref<Object>  call ( R (*f) ( A, B, C ),
                             const Ref<Object>* args )
  {
    return ObjectTraits<R>::toObject (
      f ( ObjectTraits<A>::toValue ( args[0] ),
          ObjectTraits<B>::toValue ( args[1] ),
          ObjectTraits<C>::toValue ( args[2] ) )
    );
  }

};


JEM_END_PACKAGE_BASE

#endif
