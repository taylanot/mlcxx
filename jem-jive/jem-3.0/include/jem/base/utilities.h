
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

#ifndef JEM_BASE_UTILITIES_H
#define JEM_BASE_UTILITIES_H

#include <utility>
#include <jem/defines.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Nil
//-----------------------------------------------------------------------


class                 Nil {};
extern const Nil      NIL;


//-----------------------------------------------------------------------
//   class IntConstant
//-----------------------------------------------------------------------


template <int N>

  class IntConstant

{
 public:

  typedef int         Type;

  static const int    VALUE = N;

};


//-----------------------------------------------------------------------
//   class BoolConstant
//-----------------------------------------------------------------------


template <bool B>

  class BoolConstant

{
 public:

  typedef bool         Type;

  static const bool    VALUE = B;

};


typedef BoolConstant<true>   True;
typedef BoolConstant<false>  False;


//-----------------------------------------------------------------------
//   expression classes
//-----------------------------------------------------------------------


template <class Op, class A>            class UnaryExpr    {};
template <class Op, class A, class B>   class BinaryExpr   {};
template <class A,  class B, class C>   class WhereExpr    {};


//-----------------------------------------------------------------------
//   class Lock
//-----------------------------------------------------------------------


template <class M>

  class Lock

{
 public:

  explicit inline     Lock ( const M& mutex );
  inline             ~Lock ();


 private:

  M&                  mutex_;

};


//-----------------------------------------------------------------------
//   enumerations/constants
//-----------------------------------------------------------------------


enum                  ThrowMode
{
                        NO_THROW,
                        CAN_THROW
};


//-----------------------------------------------------------------------
//   utility functions
//-----------------------------------------------------------------------


void                  abort

  ( const char*         where,
    const char*         what,
    const char*         trace = nullptr ) noexcept;

template <class T, class... Args>

  constexpr inline
  const T&            min

    ( const T&          a,
      const Args& ...   args );

template <class T, class... Args>

  constexpr inline
  const T&            max

    ( const T&          a,
      const Args& ...   args );

template <class T>

  inline void         swap

  ( T&                  lhs,
    T&                  rhs );

template <class Iterator>

  inline idx_t        distance

  ( Iterator            first,
    Iterator            last );

template <class T>

  inline idx_t        distance

  ( T*                  first,
    T*                  last );

template <class T>

  inline idx_t        distance

  ( const T*            first,
    const T*            last );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   class BoolConstant
//-----------------------------------------------------------------------


template <bool B1, bool B2>

  inline BoolConstant<(B1 || B2)>  operator ||

  ( const BoolConstant<B1>&  lhs,
    const BoolConstant<B2>&  rhs )

{
  return BoolConstant<(B1 || B2)> ();
}


template <bool B1, bool B2>

  inline BoolConstant<(B1 && B2)>  operator &&

  ( const BoolConstant<B1>&  lhs,
    const BoolConstant<B2>&  rhs )

{
  return BoolConstant<(B1 && B2)> ();
}


//-----------------------------------------------------------------------
//   class Lock
//-----------------------------------------------------------------------


template <class M>

  inline Lock<M>::Lock ( const M& mutex ) :

    mutex_ ( const_cast<M&>( mutex ) )

{
  mutex_.lock();
}


template <class M>

  inline Lock<M>::~Lock ()

{
  mutex_.unlock();
}


//-----------------------------------------------------------------------
//   min
//-----------------------------------------------------------------------


template <class T>

  constexpr inline const T& min ( const T&          a,
                                  const T&          b )

{
  return (a < b) ? a : b;
}


template <class T>

  constexpr inline const T& min ( const IdxType<T>& a,
                                  const T&          b )

{
  return (a.value < b) ? a.value : b;
}


template <class T>

  constexpr inline const T& min ( const T&          a,
                                  const IdxType<T>& b )

{
  return (a < b.value) ? a : b.value;
}


template <class T, class... Args>

  constexpr inline const T& min ( const T&          a,
                                  const Args& ...   args )

{
  return jem::min ( a, jem::min( args... ) );
}


//-----------------------------------------------------------------------
//   max
//-----------------------------------------------------------------------


template <class T>

  constexpr inline const T& max ( const T&          a,
                                  const T&          b )

{
  return (a < b) ? b : a;
}


template <class T>

  constexpr inline const T& max ( const IdxType<T>& a,
                                  const T&          b )

{
  return (a.value < b) ? b : a.value;
}


template <class T>

  constexpr inline const T& max ( const T&          a,
                                  const IdxType<T>& b )

{
  return (a < b.value) ? b.value : a;
}


template <class T, class... Args>

  constexpr inline const T& max ( const T&          a,
                                  const Args& ...   args )

{
  return jem::max ( a, jem::max( args... ) );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void swap ( T& lhs, T& rhs )

{
  T tmp ( lhs );

  lhs = std::move ( rhs );
  rhs = std::move ( tmp );
}


//-----------------------------------------------------------------------
//   distance
//-----------------------------------------------------------------------


template <class Iterator>

  inline idx_t  distance

  ( Iterator      first,
    Iterator      last )

{
  idx_t  n = 0;

  for ( ; first != last; ++first )
  {
    n++;
  }

  return n;
}


template <class T>

  inline idx_t  distance

  ( T*            first,
    T*            last )

{
  return (idx_t) (last - first);
}


template <class T>

  inline idx_t  distance

  ( const T*      first,
    const T*      last )

{
  return (idx_t) (last - first);
}


JEM_END_PACKAGE_BASE

#endif
