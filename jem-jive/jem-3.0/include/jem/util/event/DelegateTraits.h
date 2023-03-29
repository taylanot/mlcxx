
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

#ifndef JEM_UTIL_EVENT_DELEGATETRAITS_H
#define JEM_UTIL_EVENT_DELEGATETRAITS_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class DelegateTraits (primary template)
//-----------------------------------------------------------------------


template <class T>

  class DelegateTraits {};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (*) ()>
//-----------------------------------------------------------------------


template <class R>

  class DelegateTraits<R (*) ()>

{
 public:

  template <class Args>
    static void call ( R (*f) (),
                       const Args& )
  {
    f ();
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (T::*) ()>
//-----------------------------------------------------------------------


template <class T, class R>

  class DelegateTraits<R (T::*) ()>

{
 public:

  template <class Args>
    static void call ( R (T::*m) (),
                       T& obj,
                       const Args& )
  {
    (obj.*m) ();
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (T::*) () const>
//-----------------------------------------------------------------------


template <class T, class R>

  class DelegateTraits<R (T::*) () const>

{
 public:

  template <class Args>
    static void call ( R (T::*m) () const,
                       const T& obj,
                       const Args& )
  {
    (obj.*m) ();
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (*) ( A )>
//-----------------------------------------------------------------------


template <class R, class A>

  class DelegateTraits<R (*) ( A )>

{
 public:

  template <class Args>
    static void call ( R (*f) ( A ),
                       const Args& args )
  {
    f ( args.first );
  }


  template <class Args>
    static void call ( R (*f) ( A ),
                       A obj,
                       const Args& )
  {
    f ( obj );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (T::*) ( A )>
//-----------------------------------------------------------------------


template <class T, class R, class A>

  class DelegateTraits<R (T::*) ( A )>

{
 public:

  template <class Args>
    static void call ( R (T::*m) ( A ),
                       T&          obj,
                       const Args& args )
  {
    (obj.*m) ( args.first );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (T::*) ( A ) const>
//-----------------------------------------------------------------------


template <class T, class R, class A>

  class DelegateTraits<R (T::*) ( A ) const>

{
 public:

  template <class Args>
    static void call ( R (T::*m) ( A ) const,
                       const T&    obj,
                       const Args& args )
  {
    (obj.*m) ( args.first );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (*) ( A, B )>
//-----------------------------------------------------------------------


template <class R, class A, class B>

  class DelegateTraits<R (*) ( A, B )>

{
 public:

  template <class Args>
    static void call ( R (*f) ( A, B ),
                       const Args& args )
  {
    f ( args.first, args.second );
  }


  template <class Args>
    static void call ( R (*f) ( A, B ),
                       A           obj,
                       const Args& args )
  {
    f ( obj, args.first );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (T::*) ( A, B )>
//-----------------------------------------------------------------------


template <class T, class R, class A, class B>

  class DelegateTraits<R (T::*) ( A, B )>

{
 public:

  template <class Args>
    static void call ( R (T::*m) ( A, B ),
                       T&          obj,
                       const Args& args )
  {
    (obj.*m) ( args.first, args.second );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (T::*) ( A, B ) const>
//-----------------------------------------------------------------------


template <class T, class R, class A, class B>

  class DelegateTraits<R (T::*) ( A, B ) const>

{
 public:

  template <class Args>
    static void call ( R (T::*m) ( A, B ) const,
                       const T&    obj,
                       const Args& args )
  {
    (obj.*m) ( args.first, args.second );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (*) ( A, B, C )>
//-----------------------------------------------------------------------


template <class R, class A, class B, class C>

  class DelegateTraits<R (*) ( A, B, C )>

{
 public:

  template <class Args>
    static void call ( R (*f) ( A, B, C ),
                       const Args& args )
  {
    f ( args.first, args.second, args.third );
  }


  template <class Args>
    static void call ( R (*f) ( A, B, C ),
                       A           obj,
                       const Args& args )
  {
    f ( obj, args.first, args.second );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (T::*) ( A, B, C )>
//-----------------------------------------------------------------------


template <class T, class R, class A, class B, class C>

  class DelegateTraits<R (T::*) ( A, B, C )>

{
 public:

  template <class Args>
    static void call ( R (T::*m) ( A, B, C ),
                       T&          obj,
                       const Args& args )
  {
    (obj.*m) ( args.first, args.second, args.third );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (T::*) ( A, B, C ) const>
//-----------------------------------------------------------------------


template <class T, class R, class A, class B, class C>

  class DelegateTraits<R (T::*) ( A, B, C ) const>

{
 public:

  template <class Args>
    static void call ( R (T::*m) ( A, B, C ) const,
                       const T&    obj,
                       const Args& args )
  {
    (obj.*m) ( args.first, args.second, args.third );
  }

};


//-----------------------------------------------------------------------
//   class DelegateTraits<R (*) ( A, B, C, D )>
//-----------------------------------------------------------------------


template <class R, class A, class B, class C, class D>

  class DelegateTraits<R (*) ( A, B, C, D )>

{
 public:

  template <class Args>
    static void call ( R (*f) ( A, B, C, D ),
                       A           obj,
                       const Args& args )
  {
    f ( obj, args.first, args.second, args.third );
  }

};


JEM_END_PACKAGE( util )

#endif
