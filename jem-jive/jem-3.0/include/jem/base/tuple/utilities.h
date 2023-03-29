
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

#ifndef JEM_BASE_TUPLE_UTILITIES_H
#define JEM_BASE_TUPLE_UTILITIES_H

#include <jem/base/functional.h>
#include <jem/base/tuple/base.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   reduction functions
//-----------------------------------------------------------------------


template <class T, int M, int N, class E, class R>

  inline
  typename R::ResultType  reduce

  ( const Tuple<T,M,N,E>&   t,
    R                       r );

template <class T, int M, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          dot

  ( const Tuple<T,M,N,E>&   t );

template <class T, class U, int M, int N, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot

  ( const Tuple<T,M,N,E1>&  t,
    const Tuple<U,M,N,E2>&  u );

template <class T, int M, int N, class E>

  inline typename Maximum<T>
    ::ResultType          max

  ( const Tuple<T,M,N,E>&   t );

template <class T, int M, int N, class E>

  inline typename Minimum<T>
    ::ResultType          min

  ( const Tuple<T,M,N,E>&   t );

template <class T, int M, int N, class E>

  inline typename Plus<T>
    ::ResultType          sum

  ( const Tuple<T,M,N,E>&   t );

template <class T, int M, int N, class E>

  inline typename KahanSum<T>
    ::ResultType          ksum

  ( const Tuple<T,M,N,E>&   t );

template <class T, int M, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          product

  ( const Tuple<T,M,N,E>&   t );

template <class T, int M, int N, class E>

  inline bool             testall

  ( const Tuple<T,M,N,E>&   t );

template <class T, int M, int N, class E>

  inline bool             testany

  ( const Tuple<T,M,N,E>&   t );

template <class T, int M, int N, class E>

  inline int              count

  ( const Tuple<T,M,N,E>&   t );

template <class T, class U, int M, int N, class E>

  inline int              count

  ( const Tuple<T,M,N,E>&   t,
    const U&                v );


//-----------------------------------------------------------------------
//   minloc & maxloc
//-----------------------------------------------------------------------


template <class T, int M, class E>

  inline int              minloc

  ( const Tuple<T,M,1,E>&   t );

template <class T, int M, class E>

  inline int              maxloc

  ( const Tuple<T,M,1,E>&   t );


//-----------------------------------------------------------------------
//   makeTuple
//-----------------------------------------------------------------------


template <class T>

  inline Tuple<T,1>       makeTuple ( T a );

template <class T>

  inline Tuple<T,2>       makeTuple ( T a, T b );

template <class T>

  inline Tuple<T,3>       makeTuple ( T a, T b, T c );

template <class T>

  inline Tuple<T,4>       makeTuple ( T a, T b, T c,
                                      T d );

template <class T>

  inline Tuple<T,5>       makeTuple ( T a, T b, T c,
                                      T d, T e );

template <class T>

  inline Tuple<T,6>       makeTuple ( T a, T b, T c,
                                      T d, T e, T f );



//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE_BASE

#include <jem/base/tuple/utilities.tcc>

#endif
