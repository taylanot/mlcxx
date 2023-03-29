
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

#ifndef JEM_BASE_ARRAY_TENSOR_H
#define JEM_BASE_ARRAY_TENSOR_H

#include <jem/base/functional.h>
#include <jem/base/array/Array.h>
#include <jem/base/array/tensor_index.h>


JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( tensor )


//-----------------------------------------------------------------------
//   class ReduceExpr
//-----------------------------------------------------------------------


template <int N, class Op, class A>

  class ReduceExpr {};


//-----------------------------------------------------------------------
//   class ReduceTraits
//-----------------------------------------------------------------------


template <int N, class Op, class E>

  class ReduceTraits

{
 public:

  typedef typename Op::FirstArgType   ArgType;
  typedef typename Op::ResultType     ValueType;
  typedef Array<ArgType,0,E>          ArrayType;

  typedef Array<ValueType, 0,
    ReduceExpr<N,Op,ArrayType> >      ReturnType;

};


//-----------------------------------------------------------------------
//   class DotTraits
//-----------------------------------------------------------------------


template <class T, class U, int N, class E1, class E2>

  class DotTraits

{
 public:

  typedef typename
    Multiplies<T,U>::ResultType       ValueType;

  typedef BinaryExpr
    <Multiplies<T,U>,
     Array<T,0,E1>,
     Array<U,0,E2> >                  DotExpr;

  typedef Array
    <ValueType,0,DotExpr>             ArrayType;

  typedef typename
    ReduceTraits
      <N,Plus<ValueType>,
       DotExpr>::ReturnType           ReturnType;

};


JEM_END_NAMESPACE( tensor )


//-----------------------------------------------------------------------
//   class Array<T,0,TensorExpr<N> >
//-----------------------------------------------------------------------


template <class T, int N>

  class Array<T,0,TensorExpr<N> >

{
 public:

  typedef T&              Reference;


                          Array

    ( const Array<T,1>&     a,
      const TensorIndex&    i );

  template <class I,
            class J>      Array

    ( const Array<T,2>&     a,
      const I&              i,
      const J&              j );

  template <class I,
            class J,
            class K>      Array

    ( const Array<T,3>&     a,
      const I&              i,
      const J&              j,
      const K&              k );

  template <class I,
            class J,
            class K,
            class L>      Array

    ( const Array<T,4>&     a,
      const I&              i,
      const J&              j,
      const K&              k,
      const L&              l );

  template <class I,
            class J,
            class K,
            class L,
            class P>      Array

    ( const Array<T,5>&     a,
      const I&              i,
      const J&              j,
      const K&              k,
      const L&              l,
      const P&              p );

  template <class I,
            class J,
            class K,
            class L,
            class P,
            class Q>      Array

    ( const Array<T,6>&     a,
      const I&              i,
      const J&              j,
      const K&              k,
      const L&              l,
      const P&              p,
      const Q&              q );

  inline                  Array

    ( const Array&          rhs );

  inline const Array&     operator =

    ( const Array&          rhs )          const;

  template <class X>
    inline const Array&   operator =

    ( const X&              rhs )          const;

  template <class X,
            class Op>
    inline void           assign

    ( const X&              rhs,
      const Op&             op )           const;

  inline T*               getPointer    () const;

  inline idx_t            getUbound

    ( idx_t*                ix )           const;

  inline void             checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const;

  inline idx_t            getStride

    ( idx_t*                ix )           const;


 private:

  template <class U,
            class Op>
    void                  assign_

    ( const True&           is1d,
      const U&              rhs,
      const Op&             op )           const;

  template <class U,
            class E,
            class Op>
    void                  assign_

    ( const True&           is1d,
      const Array<U,0,E>&   rhs,
      const Op&             op )           const;

  template <class U,
            class E,
            class Op>
    void                  assign_

    ( const False&          is1d,
      const Array<U,0,E>&   rhs,
      const Op&             op )           const;

  template <class U,
            class Op>
    void                  assign_

    ( const False&          is1d,
      const U&              rhs,
      const Op&             op )           const;

  template <class U,
            class A,
            class Op1,
            class Op2>
    void                  assign_

    ( const False&          is1d,
      const Array<U,0,
      tensor::ReduceExpr
      <1,Op1,A> >&          rhs,
      const Op2&            op )           const;

  template <class I>
    inline void           initIndex_

    ( I                     i,
      int                   idim,
      idx_t                 size,
      idx_t                 stride );

  inline void             initIndex_

    ( const TensorIndex&    i,
      int                   idim,
      idx_t                 size,
      idx_t                 stride );


 private:

  T*                      data_;
  Tuple<idx_t*,N>         index_;
  Tuple<idx_t,N>          ubound_;
  Tuple<idx_t,N>          stride_;

};


//-----------------------------------------------------------------------
//   class Array<T,0,tensor::ReduceExpr<Op,E,N> >
//-----------------------------------------------------------------------


template <class T, int N, class Op, class A>

  class Array<T,0,tensor::ReduceExpr<N,Op,A> >

{
 public:

  typedef A               ArgType;


  inline                  Array

    ( const ArgType&        t,
      TensorIndex&          i );

  inline                  Array

    ( const ArgType&        t,
      TensorIndex&          i,
      TensorIndex&          j );

  inline                  Array

    ( const ArgType&        t,
      TensorIndex&          i,
      TensorIndex&          j,
      TensorIndex&          k );

  inline                  Array

    ( const ArgType&        t,
      TensorIndex&          i,
      TensorIndex&          j,
      TensorIndex&          k,
      TensorIndex&          l );

  inline                  Array

    ( const ArgType&        t,
      TensorIndex&          i,
      TensorIndex&          j,
      TensorIndex&          k,
      TensorIndex&          l,
      TensorIndex&          p );

  inline                  Array

    ( const ArgType&        t,
      TensorIndex&          i,
      TensorIndex&          j,
      TensorIndex&          k,
      TensorIndex&          l,
      TensorIndex&          p,
      TensorIndex&          q );

  inline                  Array

    ( const Array&          rhs );


 private:

  Array&                  operator =

    ( const Array&          rhs );


 public:

  const Op                op;
  typename ArrayTraits
    <ArgType>::Reference  arg;
  const Tuple<idx_t*,N>   index;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T, class E>

  inline typename tensor::DotTraits<T,T,1,E,E>::ReturnType

                          dot

  ( const Array<T,0,E>&     t,
    TensorIndex&            i );

template <class T, class U, class E1, class E2>

  inline typename tensor::DotTraits<T,U,1,E1,E2>::ReturnType

                          dot

  ( const Array<T,0,E1>&    t,
    const Array<U,0,E2>&    u,
    TensorIndex&            i );

template <class T, class E>

  inline typename tensor::ReduceTraits<1,Maximum<T>,E>::ReturnType

                          max

  ( const Array<T,0,E>&     t,
    TensorIndex&            i );

template <class T, class E>

  inline typename tensor::ReduceTraits<1,Minimum<T>,E>::ReturnType

                          min

  ( const Array<T,0,E>&     t,
    TensorIndex&            i );

template <class T, class E>

  inline typename tensor::ReduceTraits<1,Plus<T>,E>::ReturnType

                          sum

  ( const Array<T,0,E>&     t,
    TensorIndex&            i );

template <class T, class E>

  inline typename tensor::ReduceTraits<1,Multiplies<T>,E>::ReturnType

                          product

  ( const Array<T,0,E>&     t,
    TensorIndex&            i );

template <class T, int N, class E>

  T                       trace

  ( const Array<T,N,E>&     e );


JEM_END_PACKAGE_BASE

#include <jem/base/array/tensor.tcc>

#endif
