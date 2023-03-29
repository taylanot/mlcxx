
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

#ifndef JEM_BASE_ARRAY_UTILITIES_H
#define JEM_BASE_ARRAY_UTILITIES_H

#include <jem/base/array/expression.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   array reduction operations
//-----------------------------------------------------------------------


template <class T, int N, class E, class R>

  inline
  typename R::ResultType  reduce

  ( const Array<T,N,E>&     e,
    R                       r );

template <class T, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          dot

  ( const Array<T,N,E>&     e );

template <class T, class U, int N, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot

  ( const Array<T,N,E1>&    e,
    const Array<U,N,E2>&    f );

template <class T, int N, class E>

  inline typename Maximum<T>
    ::ResultType          max

  ( const Array<T,N,E>&     e );

template <class T, int N, class E>

  inline typename Minimum<T>
    ::ResultType          min

  ( const Array<T,N,E>&     e );

template <class T, int N, class E>

  inline typename Plus<T>
    ::ResultType          sum

  ( const Array<T,N,E>&     e );

template <class T, int N, class E>

  inline typename KahanSum<T>
    ::ResultType          ksum

  ( const Array<T,N,E>&     e );

template <class T, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          product

  ( const Array<T,N,E>&     e );

template <class T, int N, class E>

  inline bool             testall

  ( const Array<T,N,E>&     e );

template <class T, int N, class E>

  inline bool             testany

  ( const Array<T,N,E>&     e );

template <class T, int N, class E>

  inline idx_t            count

  ( const Array<T,N,E>&     e );

template <class T, class U, int N, class E>

  inline idx_t            count

  ( const Array<T,N,E>&     e,
    const U&                v );


//-----------------------------------------------------------------------
//   minloc & maxloc
//-----------------------------------------------------------------------


template <class T, class E>

  inline idx_t            minloc

  ( const Array<T,1,E>&     e );

template <class T, class E>

  inline idx_t            maxloc

  ( const Array<T,1,E>&     e );


//-----------------------------------------------------------------------
//   shift & cshift
//-----------------------------------------------------------------------


template <class T, class E>

  void                    shift

  ( const Array<T,1,E>&     e,
    idx_t                   k );

template <class T, class E>

  void                    cshift

  ( const Array<T,1,E>&     e,
    idx_t                   k );


//-----------------------------------------------------------------------
//   reverse
//-----------------------------------------------------------------------


template <class T, class E>

  void                    reverse

  ( const Array<T,1,E>&     e );


//-----------------------------------------------------------------------
//   reshape
//-----------------------------------------------------------------------


template <class T, int M, int N>

  Array<T,M>              reshape

  ( const Array<T,N>&       a,
    const Tuple<idx_t,M>&   sh );

template <class T, int N>

  Array<T,1>              reshape

  ( const Array<T,N>&       a,
    idx_t                   n );

template <class T, int N>

  inline Array<T,2>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n );

template <class T, int N>

  inline Array<T,3>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p );

template <class T, int N>

  inline Array<T,4>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q );

template <class T, int N>

  inline Array<T,5>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r );

template <class T, int N>

  inline Array<T,6>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r,
    idx_t                   s );


//-----------------------------------------------------------------------
//   flatten
//-----------------------------------------------------------------------


template <class T, int M, int N>

  Array<T,M>              flatten

  ( const Array<T,N>&       a,
    const Tuple<idx_t,M>&   map );

template <class T, int N>

  Array<T,1>              flatten

  ( const Array<T,N>&       a );

template <class T, int N>

  inline Array<T,2>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n );

template <class T, int N>

  inline Array<T,3>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p );

template <class T, int N>

  inline Array<T,4>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q );

template <class T, int N>

  inline Array<T,5>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r );

template <class T, int N>

  inline Array<T,6>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r,
    idx_t                   s );


//-----------------------------------------------------------------------
//   equal
//-----------------------------------------------------------------------


template <class T, class U, int N, class E1, class E2>

  bool                    equal

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs );


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


template <class I, int N, class E>

  idx_t                   find

  ( Array<I,2>&             index,
    const Array<bool,N,E>&  expr );

template <class I, class E>

  idx_t                   find

  ( Array<I>&               index,
    const Array<bool,1,E>&  expr );


//-----------------------------------------------------------------------
//   iarray
//-----------------------------------------------------------------------


inline Array<idx_t,1,SliceTo>

                          iarray

  ( idx_t                   n );

inline Array<idx_t,1,SliceTo>

                          iarray

  ( const SliceTo&          s );

inline Array<idx_t,1,Slice>

                          iarray

  ( const Slice&            s );

inline Array<idx_t,1,SliceFromTo>

                          iarray

  ( const SliceFromTo&      s );


//-----------------------------------------------------------------------
//   makeContiguous
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N>       makeContiguous

  ( const Array<T,N>&       a );


//-----------------------------------------------------------------------
//   sort, permute & search
//-----------------------------------------------------------------------


template <class T>

  void                    sort

  ( const Array<T>&         a );

template <class I, class T>

  void                    sort

  ( const Array<I>&         iperm,
    const Array<T>&         a );

template <class T, class I>

  void                    permute

  ( const Array<T>&         a,
    const Array<I>&         iperm );

template <class T, class I>

  Array<T>                permuted

  ( const Array<T>&         a,
    const Array<I>&         iperm );

template <class T>

  idx_t                   binarySearch

  ( const T&                value,
    const Array<T>&         a );

template <class T>

  idx_t                   lowerBound

  ( const T&                value,
    const Array<T>&         a );

template <class T>

  idx_t                   upperBound

  ( const T&                value,
    const Array<T>&         a );

template <class T>

  SliceFromTo             equalRange

  ( const T&                value,
    const Array<T>&         a );





//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE_BASE

#include <jem/base/array/utilities.tcc>

#endif
