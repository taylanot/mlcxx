
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

#ifndef JEM_BASE_TUPLE_TUPLE_H
#define JEM_BASE_TUPLE_TUPLE_H

#include <cstring>
#include <initializer_list>
#include <jem/base/TypeTraits.h>
#include <jem/base/tuple/base.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Tuple<T,M,N,Nil>
//-----------------------------------------------------------------------


template <class T, int M, int N>

  class Tuple<T,M,N,Nil>

{
 public:

  typedef T*                Iterator;
  typedef const T*          ConstIterator;

  static const int          SIZE = M * N;
  static const int          RANK = TupleRank<M,N>::RANK;


  inline                    Tuple     ();

  inline                    Tuple

    ( T                       a );

  inline                    Tuple

    ( T                       a,
      T                       b );

  inline                    Tuple

    ( T                       a,
      T                       b,
      T                       c );

  inline                    Tuple

    ( T                       a,
      T                       b,
      T                       c,
      T                       d );

  inline                    Tuple

    ( T                       a,
      T                       b,
      T                       c,
      T                       d,
      T                       e );

  inline                    Tuple

    ( T                       a,
      T                       b,
      T                       c,
      T                       d,
      T                       e,
      T                       f );

  template <class U, class E>
    inline                  Tuple

    ( const Tuple<U,M,N,E>&   rhs );

  inline                    Tuple

    ( const Tuple&            rhs );

  template <class U>
                            Tuple

    ( std::
      initializer_list<U>     list );

  inline Iterator           begin       ()       noexcept;
  inline ConstIterator      begin       () const noexcept;
  inline Iterator           end         ()       noexcept;
  inline ConstIterator      end         () const noexcept;

  template <class U, class E>
    inline Tuple&           operator =

    ( const Tuple<U,M,N,E>&   rhs );

  inline Tuple&             operator =

    ( const Tuple&            rhs );

  inline Tuple&             operator =

    ( T                       rhs );

  inline void               swap

    ( Tuple&                  rhs );

  template <class I>
    inline T&               operator    []

    ( I                       i );

  template <class I>
    inline const T&         operator    []

    ( I                       i )          const;

  template <class I>
    inline T&               operator    ()

    ( I                       i,
      I                       j );

  template <class I>
    inline const T&         operator    ()

    ( I                       i,
      I                       j )          const;

  Tuple<T,N,M>              transpose   () const;

  inline T*                 addr        ()       noexcept;
  inline const T*           addr        () const noexcept;


 private:

  T                         data_[SIZE];

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T, int M, int N>

  inline void               swap

  ( Tuple<T,M,N>&             lhs,
    Tuple<T,M,N>&             rhs );

template <class T, class U, int M, int N, class E1, class E2>

  inline bool               equal

  ( const Tuple<T,M,N,E1>&    lhs,
    const Tuple<U,M,N,E2>&    rhs );

template <class Input,
          class T, int M, int N>

  inline void               decode

  ( Input&                    in,
    Tuple<T,M,N>&             t );

template <class Output,
          class T, int M, int N>

  inline void               encode

  ( Output&                   out,
    const Tuple<T,M,N>&       t );

template <class Output,
          class T, int M, int N, class E>

  inline void               print

  ( Output&                   out,
    const Tuple<T,M,N,E>&     t );





//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE_BASE

#include <jem/base/tuple/Tuple.tcc>

#endif
