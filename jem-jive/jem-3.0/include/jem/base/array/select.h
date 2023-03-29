
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

#ifndef JEM_BASE_ARRAY_SELECT_H
#define JEM_BASE_ARRAY_SELECT_H

#include <jem/base/array/Array.h>
#include <jem/base/array/assign.h>
#include <jem/base/array/expression.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Array<T,N,SelectExpr<V> >
//-----------------------------------------------------------------------


template <class T, int N, class V>

  class Array<T,N,SelectExpr<V> >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T&              Reference;
  typedef V               IndexType;
  typedef Array<V>        IndexArray;


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

  template <class U,
            class Op>
    void                  assign

    ( const U&              rhs,
      const Op&             op )           const;

  template <class U,
            class E,
            class Op>
    void                  assign

    ( const Array<U,N,E>&   rhs,
      const Op&             op )           const;

  template <class I>
    inline Reference      operator []

    ( const Tuple<I,N>&     i )            const;

  Reference               getFast

    ( idx_t                 i )            const;

  inline const Shape&     shape         () const;
  inline bool             isContiguous  () const;


 private:

  template <class I>
    inline void           initIndex_

    ( I                     idx,
      int                   idim,
      idx_t                 size,
      idx_t                 stride );

  inline void             initIndex_

    ( const IndexArray&     idx,
      int                   idim,
      idx_t                 size,
      idx_t                 stride );


 private:

  T*                      data_;
  Tuple<IndexType*,N>     index_;
  Tuple<idx_t,N>          shape_;
  Tuple<idx_t,N>          dstride_;
  Tuple<idx_t,N>          istride_;

};


//-----------------------------------------------------------------------
//   class Array<T,1,SelectExpr<V> >
//-----------------------------------------------------------------------


template <class T, class V>

  class Array<T,1,SelectExpr<V> >

{
 public:

  typedef Tuple<idx_t,1>  Shape;
  typedef T&              Reference;
  typedef V               IndexType;
  typedef Array<V>        IndexArray;


  inline                  Array

    ( const Array<T,1>&     a,
      const IndexArray&     i );

  template <class I,
            class J>

    inline                Array

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

  template <class U>
    const Array&          operator =

    ( const U&              rhs )          const;

  inline const Array&     operator =

    ( const Array&          rhs )          const;

  template <class U,
            class E>
    inline const Array&   operator =

    ( const Array<U,1,E>&   rhs )          const;

  template <class U,
            class Op>
    void                  assign

    ( const U&              rhs,
      const Op&             op )           const;

  template <class Op>
    void                  assign

    ( const Array&          rhs,
      const Op&             op )           const;

  template <class U,
            class E,
            class Op>
    void                  assign

    ( const Array<U,1,E>&   rhs,
      const Op&             op )           const;

  template <class I>
    inline Reference      operator []

    ( I                     i )            const;

  template <class I>
    inline Reference      operator []

    ( const Tuple<I,1>&     i )            const;

  inline Reference        getFast

    ( idx_t                 i )            const;

  inline idx_t            size          () const;
  inline Shape            shape         () const;
  inline bool             isContiguous  () const;


 private:

  template <class I>
    inline void           initIndex_

    ( I                     i,
      idx_t                 size,
      idx_t                 stride );

  inline void             initIndex_

    ( const IndexArray&     i,
      idx_t                 size,
      idx_t                 stride );


 private:

  T*                      data_;
  const IndexType*        index_;
  idx_t                   size_;
  idx_t                   dstride_;
  idx_t                   istride_;

};


//-----------------------------------------------------------------------
//   class Array<T,N,SelectLastExpr<V> >
//-----------------------------------------------------------------------


template <class T, int N, class V>

  class Array<T,N,SelectLastExpr<V> >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T&              Reference;
  typedef V               IndexType;
  typedef Array<V>        IndexArray;


                          Array

    ( const Array<T,N>&     a,
      const IndexArray&     i );

  inline                  Array

    ( const Array&          rhs );

  inline const Array&     operator =

    ( const Array&          rhs )          const;

  template <class X>
    inline const Array&   operator =

    ( const X&              rhs )          const;

  template <class U,
            class Op>
    void                  assign

    ( const U&              rhs,
      const Op&             op )           const;

  template <class U,
            class E,
            class Op>
    void                  assign

    ( const Array<U,N,E>&   rhs,
      const Op&             op )           const;

  template <class I>
    inline Reference      operator []

    ( const Tuple<I,N>&     i )            const;

  Reference               getFast

    ( idx_t                 i )            const;

  inline const Shape&     shape         () const;
  inline bool             isContiguous  () const;


 private:

  T*                      data_;
  IndexType*              index_;
  Tuple<idx_t,N>          shape_;
  Tuple<idx_t,N>          dstride_;
  idx_t                   istride_;

};


//-----------------------------------------------------------------------
//   class SelectTraits
//-----------------------------------------------------------------------


template <class E>

  class SelectTraits

{};


template <>

 class SelectTraits<Slice>

{
 public:

  typedef Array
    <idx_t,1,Slice>        ArrayType;

};


template <>

  class SelectTraits<SliceFromTo>

{
 public:

  typedef Array
    <idx_t,1,SliceFromTo>  ArrayType;

};


template <>

  class SelectTraits<SliceTo>

{
 public:

  typedef Array
    <idx_t,1,SliceTo>      ArrayType;

};


template <>

  class SelectTraits<SliceFrom>

{
 public:

  typedef Array
    <idx_t,1,SliceFromTo>  ArrayType;

};


template <>

  class SelectTraits<SliceAll>

{
 public:

  typedef Array
    <idx_t,1,SliceTo>      ArrayType;

};


template <class I, class E>

  class SelectTraits< Array<I,1,E> >

{
 public:

  typedef Array<I,1,E>    ArrayType;

};


//-----------------------------------------------------------------------
//   class Select<N>Expr
//-----------------------------------------------------------------------


template <class E, class I>

  class Select1Expr

{};


template <class E, class I, class J>

  class Select2Expr

{};


template <class E, class I,
          class J, class K>

  class Select3Expr

{};


template <class E, class I,
          class J, class K, class L>

  class Select4Expr

{};


template <class E, class I, class J,
          class K, class L, class P>

  class Select5Expr

{};


template <class E, class I, class J,
          class K, class L, class P, class Q>

  class Select6Expr

{};


//-----------------------------------------------------------------------
//   select
//-----------------------------------------------------------------------


template <class T, class E, class I>

  inline Array<T,1,Select1Expr<E,I> >

                        select

  ( const Array<T,1,E>&   arg,
    const I&              i );

template <class T, class E,
          class I, class J>

  inline Array<T,2,Select2Expr<E,I,J> >

                        select

  ( const Array<T,2,E>&   arg,
    const I&              i,
    const J&              j );

template <class T, class E,
          class I, class J, class K>

  inline Array<T,3,Select3Expr<E,I,J,K> >

                        select

  ( const Array<T,3,E>&   arg,
    const I&              i,
    const J&              j,
    const K&              k );

template <class T, class E,
          class I, class J, class K, class L>

  inline Array<T,4,Select4Expr<E,I,J,K,L> >

                        select

  ( const Array<T,4,E>&   arg,
    const I&              i,
    const J&              j,
    const K&              k,
    const L&              l );

template <class T, class E,
          class I, class J,
          class K, class L, class P>

  inline Array<T,5,Select5Expr<E,I,J,K,L,P> >

                        select

  ( const Array<T,5,E>&   arg,
    const I&              i,
    const J&              j,
    const K&              k,
    const L&              l,
    const P&              p );

template <class T, class E,
          class I, class J, class K,
          class L, class P, class Q>

  inline Array<T,6,Select6Expr<E,I,J,K,L,P,Q> >

                        select

  ( const Array<T,6,E>&   arg,
    const I&              i,
    const J&              j,
    const K&              k,
    const L&              l,
    const P&              p,
    const Q&              q );


JEM_END_PACKAGE_BASE

#include <jem/base/array/select.tcc>

#endif
