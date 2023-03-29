
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

#ifndef JEM_BASE_REFLECT_FIELDTRAITS_H
#define JEM_BASE_REFLECT_FIELDTRAITS_H

#include <jem/base/ObjectTraits.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class FieldGetterTraits (primary template)
//-----------------------------------------------------------------------


template <class T>

  class FieldGetterTraits {};


//-----------------------------------------------------------------------
//   class FieldGetterTraits<R (*) ()>
//-----------------------------------------------------------------------


template <class R>

  class FieldGetterTraits<R (*) ()>

{
 public:

  static const bool   IS_STATIC = true;

  static Class*       getFieldType ()
  {
    return ObjectTraits<R>::getType ();
  }

  static Ref<Object>  get ( const Ref<Object>&, R (*getter) () )
  {
    return ObjectTraits<R>::toObject ( getter() );
  }

};


//-----------------------------------------------------------------------
//   class FieldGetterTraits<R (*) ( T )>
//-----------------------------------------------------------------------


template <class R, class T>

  class FieldGetterTraits<R (*) ( T )>

{
 public:

  static const bool   IS_STATIC = false;

  static Class*       getFieldType ()
  {
    return ObjectTraits<R>::getType();
  }

  static Ref<Object>  get ( const Ref<Object>& obj,
                            R (*getter) ( T ) )
  {
    return ObjectTraits<R>::toObject (
      getter ( ObjectTraits<T>::toValue ( obj ) )
    );
  }

};


//-----------------------------------------------------------------------
//   class FieldGetterTraits<R (T::*) ()>
//-----------------------------------------------------------------------


template <class R, class T>

  class FieldGetterTraits<R (T::*) ()>

{
 public:

  static const bool   IS_STATIC = false;

  static Class*       getFieldType ()
  {
    return ObjectTraits<R>::getType ();
  }

  static Ref<Object>  get ( const Ref<Object>& obj,
                            R (T::*getter) () )
  {
    T& tmp = ObjectTraits<T&>::toValue ( obj );

    return ObjectTraits<R>::toObject ( (tmp.*getter)() );
  }
};


//-----------------------------------------------------------------------
//   class FieldGetterTraits<R (T::*) () const>
//-----------------------------------------------------------------------


template <class R, class T>

  class FieldGetterTraits<R (T::*) () const>

{
 public:

  static const bool   IS_STATIC = false;

  static Class*       getFieldType ()
  {
    return ObjectTraits<R>::getType ();
  }

  static Ref<Object>  get ( const Ref<Object>& obj,
                            R (T::*getter) () const )
  {
    const T& tmp = ObjectTraits<const T&>::toValue ( obj );

    return ObjectTraits<R>::toObject ( (tmp.*getter)() );
  }

};


//-----------------------------------------------------------------------
//   class FieldSetterTraits (primary template)
//-----------------------------------------------------------------------


template <class T>

  class FieldSetterTraits {};


//-----------------------------------------------------------------------
//   class FieldSetterTraits<R (*) ( A )>
//-----------------------------------------------------------------------


template <class R, class A>

  class FieldSetterTraits<R (*) ( A )>

{
 public:

  static const bool   IS_STATIC = true;

  static void         set ( const Ref<Object>&,
                            R (*setter) ( A ),
                            const Ref<Object>& value )
  {
    setter ( ObjectTraits<A>::toValue ( value ) );
  }

};


//-----------------------------------------------------------------------
//   class FieldSetterTraits<R (*) ( T, A )>
//-----------------------------------------------------------------------


template <class R, class T, class A>

  class FieldSetterTraits<R (*) ( T, A )>

{
 public:

  static const bool   IS_STATIC = false;

  static void         set ( const Ref<Object>& obj,
                            R (*setter) ( T, A ),
                            const Ref<Object>& value )
  {
    setter ( ObjectTraits<T>::toValue ( obj ),
             ObjectTraits<A>::toValue ( value ) );
  }

};


//-----------------------------------------------------------------------
//   class FieldSetterTraits<R (T::*) ( A )>
//-----------------------------------------------------------------------


template <class R, class T, class A>

  class FieldSetterTraits<R (T::*) ( A )>

{
 public:

  static const bool   IS_STATIC = false;

  static void         set ( const Ref<Object>& obj,
                            R (T::*setter) ( A ),
                            const Ref<Object>& value )
  {
    T& tmp = ObjectTraits<T&>::toValue ( obj );

    (tmp.*setter)( ObjectTraits<A>::toValue( value ) );
  }

};


JEM_END_PACKAGE_BASE

#endif
