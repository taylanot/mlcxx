
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

#ifndef JEM_BASE_ARRAY_ARRAY_H
#define JEM_BASE_ARRAY_ARRAY_H

#include <initializer_list>
#include <jem/base/array/base.h>
#include <jem/base/array/Iterator.h>
#include <jem/base/array/DataBlock.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Array<T,N,Nil>
//-----------------------------------------------------------------------


template <class T, int N>

  class Array<T,N,Nil>

{
 public:

  typedef Tuple<idx_t,N>      Shape;
  typedef T                   ValueType;
  typedef T&                  Reference;
  typedef ArrayDataBlock<T>   DataBlock;
  typedef ArrayIterator<T,N>  Iterator;
  typedef Iterator            ConstIterator;

  enum CloneTag             { CLONE_TAG };

  static const int            RANK = N;


  inline                      Array         ()       noexcept;

  inline                      Array

    ( idx_t                     m,
      idx_t                     n );

  inline                      Array

    ( idx_t                     m,
      idx_t                     n ,
      idx_t                     p );

  inline                      Array

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q );

  inline                      Array

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q,
      idx_t                     r );

  inline                      Array

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q,
      idx_t                     r,
      idx_t                     s );

  explicit inline             Array

    ( const Shape&              sh );

  inline                      Array

    ( const Array&              rhs )                noexcept;

  inline                      Array

    ( Array&&                   rhs )                noexcept;

                              Array

    ( const Array&              rhs,
      CloneTag                  tag );

  template <class U, class E>
    explicit                  Array

    ( const Array<U,N,E>&       rhs );

                              Array

    ( T*                        data,
      const Shape&              sh );

  template <int   K,
            class I>
    inline                    Array

    ( const Array<T,K>&         a,
      const I&                  i );

  template <class I,
            class J>
    inline                    Array

    ( const Array<T,2>&         a,
      const I&                  i,
      const J&                  j );

  template <class I,
            class J,
            class K>          Array

    ( const Array<T,3>&         a,
      const I&                  i,
      const J&                  j,
      const K&                  k );

  template <class I,
            class J,
            class K,
            class L>          Array

    ( const Array<T,4>&         a,
      const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l );

  template <class I,
            class J,
            class K,
            class L,
            class P>          Array

    ( const Array<T,5>&         a,
      const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l,
      const P&                  p );

  template <class I,
            class J,
            class K,
            class L,
            class P,
            class Q>          Array

    ( const Array<T,6>&         a,
      const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l,
      const P&                  p,
      const Q&                  q );

  inline                      Array

    ( DataBlock*                block,
      T*                        data,
      const Shape&              sh,
      const Shape&              st );

  inline                     ~Array         ();

  const Array&                operator =

    ( const Array&              rhs )          const;

  template <class X>
    inline const Array&       operator =

    ( const X&                  rhs )          const;

  template <class U,
            class Op>
    inline void               assign

      ( const Array<U,N,Nil>&   rhs,
        const Op&               op )           const;

  template <class U,
            class E,
            class Op>
    inline void               assign

      ( const Array<U,N,E>&     rhs,
        const Op&               op )           const;

  template <class U,
            class Op>
    inline void               assign

      ( const U&                rhs,
        const Op&               op )           const;

  inline Iterator             begin         () const noexcept;
  inline Iterator             end           () const noexcept;
  inline Array                clone         () const;

  void                        ref

    ( const Array&              rhs );

  inline void                 ref

    ( Array&&                   rhs )                noexcept;

  void                        swap

    ( Array&                    rhs )                noexcept;

  template <class I>
    inline T&                 operator      []

    ( const Tuple<I,N>&         i )            const;

  template <class I>
    inline typename
    array::SliceLastTraits<T,N,I>::ResultType

                              operator      []

    ( const I&                  i )            const;

  template <class I,
            class J>
    inline typename
    array::SliceTraits<T,I,J>::ResultType

                              operator      ()

    ( const I&                  i,
      const J&                  j )            const;

  template <class I,
            class J,
            class K>
    inline typename
    array::SliceTraits<T,I,J,K>::ResultType

                              operator      ()

    ( const I&                  i,
      const J&                  j,
      const K&                  k )            const;

  template <class I,
            class J,
            class K,
            class L>
    inline typename
    array::SliceTraits<T,I,J,K,L>::ResultType

                              operator      ()

    ( const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l )            const;

  template <class I,
            class J,
            class K,
            class L,
            class P>
    inline typename
    array::SliceTraits<T,I,J,K,L,P>::ResultType

                              operator      ()

    ( const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l,
      const P&                  p )            const;

  template <class I,
            class J,
            class K,
            class L,
            class P,
            class Q>
    inline typename
    array::SliceTraits<T,I,J,K,L,P,Q>::ResultType

                              operator      ()

    ( const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l,
      const P&                  p,
      const Q&                  q )            const;

  inline T&                   getFast

    ( idx_t                      i )            const noexcept;

  inline Array                transpose     () const;

  Array                       transpose

    ( const Tuple<int,N>&       perm )         const;

  inline bool                 isAllocated   () const noexcept;
  inline bool                 isContiguous  () const noexcept;

  void                        resize

    ( idx_t                     m );

  inline void                 resize

    ( idx_t                     m,
      idx_t                     n );

  inline void                 resize

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p );

  inline void                 resize

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q );

  inline void                 resize

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q,
      idx_t                     r );

  inline void                 resize

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q,
      idx_t                     r,
      idx_t                     s );

  void                        resize

    ( const Shape&              sh );

  void                        reshape

    ( idx_t                     m );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q,
      idx_t                     r );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q,
      idx_t                     r,
      idx_t                     s );

  void                        reshape

    ( const Shape&              sh );

  inline idx_t                 size         () const;
  inline idx_t                 size

    ( int                       idim )         const;

  inline const Shape&         shape         () const noexcept;

  inline idx_t                stride        () const;
  inline idx_t                stride

    ( int                       idim )         const;

  inline const Shape&         strides       () const noexcept;

  inline T*                   addr          () const noexcept;
  inline T*                   addr

    ( idx_t                     offset )       const;

  inline idx_t                itemCount     () const;
  inline DataBlock*           getDataBlock  () const noexcept;


 private:

  friend class                array::Slicer<T&,0>;

  enum                        TransposeTag_
  {
                                TRANSPOSE_TAG_
  };


                              Array

    ( const Array&              rhs,
      TransposeTag_             tag );

  void                        init_

    ( const Shape&              sh );

  inline void                 init_

    ( idx_t                     len,
      const True&               fast );

  inline void                 init_

    ( idx_t                     len,
      const False&              fast );

  inline void                 grab_

    ( Array&                    rhs );

  template <class I>
    inline void               slice_

    ( I                         i,
      int                       idim,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const Slice&              s,
      int                       idim,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const SliceFromTo&        s,
      int                       idim,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const SliceTo&            s,
      int                       idim,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const SliceFrom&          s,
      int                       idim,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const SliceAll&           s,
      int                       idim,
      idx_t                     size,
      idx_t                     stride );


 private:

  DataBlock*                  block_;
  T*                          data_;
  Shape                       shape_;
  Shape                       stride_;

};


//-----------------------------------------------------------------------
//   class Array<T,1,Nil>
//-----------------------------------------------------------------------


template <class T>

  class Array<T,1,Nil>

{
 public:

  typedef Tuple<idx_t,1>      Shape;
  typedef T                   ValueType;
  typedef T&                  Reference;
  typedef ArrayDataBlock<T>   DataBlock;
  typedef ArrayIterator<T,1>  Iterator;
  typedef Iterator            ConstIterator;

  enum CloneTag             { CLONE_TAG };

  static const int            RANK = 1;


  inline                      Array         ()       noexcept;

  explicit inline             Array

    ( idx_t                     n );

  explicit inline             Array

    ( const Shape&              sh );

  inline                      Array

    ( const Array&              rhs )                noexcept;

  inline                      Array

    ( Array&&                   rhs )                noexcept;

                              Array

    ( const Array&              rhs,
      CloneTag                  tag );

  template <class U,
            class E>
    explicit                  Array

    ( const Array<U,1,E>&       rhs );

  inline                      Array

    ( T*                        data,
      idx_t                     size );

  template <class InputIterator>
                              Array

    ( InputIterator             first,
      InputIterator             last );

  template <class U>
                              Array

    ( std::initializer_list<U>  list );

  template <class I>
    inline                    Array

    ( const Array<T,1>&         a,
      const I&                  i );

  template <class I>
    inline                    Array

    ( const Array<T,2>&         a,
      I                         i );

  template <class I,
            class J>
    inline                    Array

    ( const Array<T,2>&         a,
      const I&                  i,
      const J&                  j );

  template <class I,
            class J,
            class K>          Array

    ( const Array<T,3>&         a,
      const I&                  i,
      const J&                  j,
      const K&                  k );

  template <class I,
            class J,
            class K,
            class L>          Array

    ( const Array<T,4>&         a,
      const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l );

  template <class I,
            class J,
            class K,
            class L,
            class P>          Array

    ( const Array<T,5>&         a,
      const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l,
      const P&                  p );

  template <class I,
            class J,
            class K,
            class L,
            class P,
            class Q>          Array

    ( const Array<T,6>&         a,
      const I&                  i,
      const J&                  j,
      const K&                  k,
      const L&                  l,
      const P&                  p,
      const Q&                  q );

  inline                      Array

    ( DataBlock*                block,
      T*                        data,
      idx_t                     size,
      idx_t                     stride );

  inline                     ~Array         ();

  const Array&                operator =

    ( const Array&              rhs )          const;

  template <class X>
    inline const Array&       operator =

    ( const X&                  rhs )          const;

  template <class U,
            class Op>
    void                      assign

      ( const Array<U,1,Nil>&   rhs,
        const Op&               op )           const;

  template <class U,
            class E,
            class Op>
    inline void               assign

      ( const Array<U,1,E>&     rhs,
        const Op&               op )           const;

  template <class U,
            class Op>
    void                      assign

      ( const U&                rhs,
        const Op&               op )           const;

  inline Iterator             begin         () const noexcept;
  inline Iterator             end           () const noexcept;
  inline T&                   front         () const;
  inline T&                   back          () const;
  inline Array                clone         () const;

  void                        ref

    ( const Array&              rhs );

  inline void                 ref

    ( Array&&                   rhs )                noexcept;

  void                        swap

    ( Array&                    rhs )                noexcept;

  template <class I>
    inline T&                 operator      []

    ( const Tuple<I,1>&         i )            const;

  template <class I>
    inline typename
    array::SliceTraits<T,I>::ResultType

                              operator      []

    ( const I&                  i )            const;

  inline T&                   getFast

    ( idx_t                      i )           const noexcept;

  inline bool                 isAllocated   () const noexcept;
  inline bool                 isContiguous  () const noexcept;

  void                        resize

    ( idx_t                     n );

  inline void                 resize

    ( const Shape&              sh );

  void                        reshape

    ( idx_t                     n );

  inline void                 reshape

    ( const Shape&              sh );

  inline idx_t                 size          () const noexcept;
  inline idx_t                 size

    ( int                       idim )         const;

  inline Shape                shape         () const;
  inline idx_t                stride        () const noexcept;
  inline idx_t                stride

    ( int                       idim )         const;

  inline Shape                strides       () const;

  inline T*                   addr          () const noexcept;
  inline T*                   addr

    ( idx_t                     offset )       const;

  inline idx_t                itemCount     () const;
  inline DataBlock*           getDataBlock  () const;


 private:

  friend class                array::Slicer<T&,0>;

  void                        init_

    ( idx_t                     len );

  inline void                 init_

    ( const True&               fast );

  inline void                 init_

    ( const False&              fast );

  inline void                 grab_

    ( Array&                    rhs );

  template <class I>
    inline void               slice_

    ( I                         i,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const Slice&              s,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const SliceFromTo&        s,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const SliceTo&            s,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const SliceFrom&          s,
      idx_t                     size,
      idx_t                     stride );

  inline void                 slice_

    ( const SliceAll&           s,
      idx_t                     size,
      idx_t                     stride );


 private:

  DataBlock*                  block_;
  T*                          data_;
  idx_t                       size_;
  idx_t                       stride_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input,
          class T, int N>

  void                        decode

  ( Input&                      in,
    Array<T,N>&                 a );

template <class Output,
          class T, int N>

  void                        encode

  ( Output&                     out,
    const Array<T,N>&           a );

template <class Output,
          class T, int N, class E>

  inline void                 print

  ( Output&                     out,
    const Array<T,N,E>&         a );

template <class T, int N>

  inline void                 swap

  ( Array<T,N>&                 lhs,
    Array<T,N>&                 rhs ) noexcept;


JEM_END_PACKAGE_BASE

#include <jem/base/array/Array.tcc>

#endif
