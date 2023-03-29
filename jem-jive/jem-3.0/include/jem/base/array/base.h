
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

#ifndef JEM_BASE_ARRAY_BASE_H
#define JEM_BASE_ARRAY_BASE_H

#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/base/TypeTraits.h>
#include <jem/base/tuple/Tuple.h>
#include <jem/base/array/forward.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ArrayTraits & specializations
//-----------------------------------------------------------------------


template <class T>

  class ArrayTraits

{};

template <class T, int N, class E>

  class ArrayTraits< Array<T,N,E> >

{
 public:

  typedef const Array<T,N>        Evaluate;
  typedef const Array<T,N,E>      Reference;

};

template <class T, int N>

  class ArrayTraits< Array<T,N,Nil> >

{
 public:

  typedef const Array<T,N>&       Evaluate;
  typedef const Array<T,N>&       Reference;

};

template <class T, int N, class I>

  class ArrayTraits< Array<T,N,SelectExpr<I> > >

{
 public:

  typedef const Array<T,N>        Evaluate;
  typedef const Array
    <T,N,SelectExpr<I> >&         Reference;

};

template <class T, int N, class I>

  class ArrayTraits<Array<T,N,SelectLastExpr<I> > >

{
 public:

  typedef const Array<T,N>        Evaluate;
  typedef const Array
    <T,N,SelectLastExpr<I> >&     Reference;

};

template <class T, int N>

  class ArrayTraits< Array<T,0,TensorExpr<N> > >

{
 public:

  typedef const Array
    <T,0,TensorExpr<N> >&         Reference;

};


//-----------------------------------------------------------------------
//   public utility functions
//-----------------------------------------------------------------------


inline Tuple<idx_t,1>     shape

  ( idx_t                   m );

template <class T, int N>

  inline void             swap

  ( Array<T,N>&             lhs,
    Array<T,N>&             rhs ) noexcept;


JEM_BEGIN_NAMESPACE( array )


//-----------------------------------------------------------------------
//   class Errors
//-----------------------------------------------------------------------


class Errors
{
 public:

  static const char*              INVALID_DIM;
  static const char*              INVALID_SIZE;
  static const char*              INVALID_SHAPE;
  static const char*              INVALID_SLICE;
  static const char*              INVALID_INDEX;
  static const char*              INVALID_INDEX0;
  static const char*              INVALID_INDEX1;
  static const char*              INVALID_INDEX2;
  static const char*              INVALID_INDEX3;
  static const char*              INVALID_INDEX4;
  static const char*              INVALID_INDEX5;
  static const char*              INVALID_SELECT;
  static const char*              SIZE_MISMATCH;
  static const char*              SHAPE_MISMATCH;
  static const char*              EMPTY_ARRAY;
  static const char*              RESHAPE_ERROR;

};

//-----------------------------------------------------------------------
//   class IndexTraits & specializations
//-----------------------------------------------------------------------


template <class I>

  class IndexTraits

{
 public:

  static const int                RANK = 0;

  typedef int                     IndexKind;

};

template <>

  class IndexTraits<TensorIndex>

{
 public:

  static const int                RANK = 1;

  typedef TensorIndex             IndexKind;

};

template <class I>

  class IndexTraits< Array<I> >

{
 public:

  static const int                RANK = 1;

  typedef Array<I>                IndexKind;

};

template <>

  class IndexTraits<Slice>

{
 public:

  static const int                RANK = 1;

  typedef Slice                   IndexKind;

};

template <>

  class IndexTraits<SliceTo>     : public IndexTraits<Slice>

{};

template <>

  class IndexTraits<SliceFrom>   : public IndexTraits<Slice>

{};

template <>

  class IndexTraits<SliceFromTo> : public IndexTraits<Slice>

{};

template <>

  class IndexTraits<SliceAll>    : public IndexTraits<Slice>

{};


//-----------------------------------------------------------------------
//   class Index2Kind & specializations
//-----------------------------------------------------------------------


template <class I, class J>

  class Index2Kind

{};

template <class I>

  class Index2Kind<I,I>

{
 public:

  typedef I                       IndexKind;
};

template <class I>

  class Index2Kind<int,I>

{
 public:

  typedef I                       IndexKind;

};

template <class I>

  class Index2Kind<I,int>

{
 public:

  typedef I                       IndexKind;

};

template <>

  class Index2Kind<int,int>

{
 public:

  typedef int                     IndexKind;

};


//-----------------------------------------------------------------------
//   class SliceKind & specializations
//-----------------------------------------------------------------------


template <class T, int N, class I>

  class SliceKind

{};

template <class T>

  class SliceKind<T,0,int>

{
 public:

  typedef T&                      ResultType;

};

template <class T, int N>

  class SliceKind<T,N,Slice>

{
 public:

  typedef Array<T,N,Nil>          ResultType;

};

template <class T, int N>

  class SliceKind<T,N,TensorIndex>

{
 public:

  typedef
    Array<T,0,TensorExpr<N> >     ResultType;

};

template <class T, int N, class I>

  class SliceKind<T,N,Array<I> >

{
 public:

  typedef Array
    <T,N,SelectExpr<I> >          ResultType;

};


//-----------------------------------------------------------------------
//   class SliceLastKind & specializations
//-----------------------------------------------------------------------


template <class T, int N, class I>

  class SliceLastKind

{};

template <class T, int N>

  class SliceLastKind<T,N,int>

{
 public:

  typedef Array<T,N,Nil>          ResultType;

};

template <class T>

  class SliceLastKind<T,0,int>

{
 public:

  typedef T&                      ResultType;

};

template <class T, int N>

  class SliceLastKind<T,N,Slice>

{
 public:

  typedef Array<T,N,Nil>          ResultType;

};

template <class T, int N, class I>

  class SliceLastKind<T,N,Array<I> >

{
 public:

  typedef Array
    <T,N,SelectLastExpr<I> >      ResultType;

};


//-----------------------------------------------------------------------
//   class SliceTraits
//-----------------------------------------------------------------------


template <class T,
          class I,
          class J = idx_t,
          class K = idx_t,
          class L = idx_t,
          class P = idx_t,
          class Q = idx_t>

  class SliceTraits

{
 public:

  static const int    DIM0       =  0;
  static const int    DIM1       =  DIM0 + IndexTraits<I>::RANK;
  static const int    DIM2       =  DIM1 + IndexTraits<J>::RANK;
  static const int    DIM3       =  DIM2 + IndexTraits<K>::RANK;
  static const int    DIM4       =  DIM3 + IndexTraits<L>::RANK;
  static const int    DIM5       =  DIM4 + IndexTraits<P>::RANK;
  static const int    DIM6       =  DIM5 + IndexTraits<Q>::RANK;

  static const int    RANK       =  DIM6;

  typedef typename    IndexTraits<I>::IndexKind              Kind0;
  typedef typename    Index2Kind
    <Kind0, typename  IndexTraits<J>::IndexKind>::IndexKind  Kind1;
  typedef typename    Index2Kind
    <Kind1, typename  IndexTraits<K>::IndexKind>::IndexKind  Kind2;
  typedef typename    Index2Kind
    <Kind2, typename  IndexTraits<L>::IndexKind>::IndexKind  Kind3;
  typedef typename    Index2Kind
    <Kind3, typename  IndexTraits<P>::IndexKind>::IndexKind  Kind4;
  typedef typename    Index2Kind
    <Kind4, typename  IndexTraits<Q>::IndexKind>::IndexKind  Kind5;

  typedef typename    SliceKind
    <T,RANK,Kind5> :: ResultType  ResultType;

};


//-----------------------------------------------------------------------
//   class SliceLastTraits
//-----------------------------------------------------------------------


template <class T, int N, class I>

  class SliceLastTraits

{
 public:

  static const int    RANK = N - 1 + IndexTraits<I>::RANK;

  typedef typename    IndexTraits<I>::IndexKind  IndexKind;

  typedef typename    SliceLastKind
    <T,RANK,IndexKind>::ResultType               ResultType;

};


//-----------------------------------------------------------------------
//   class Slicer
//-----------------------------------------------------------------------


template <class T, int N>  class Slicer;
template <class T>         class Slicer<T,0>;


//-----------------------------------------------------------------------
//   class Pass2nd
//-----------------------------------------------------------------------


class Pass2nd
{
 public:

  template <class T, class U>

    const U&            operator ()

    ( const T&            lhs,
      const U&            rhs )      const

  {
    return rhs;
  }

};


//-----------------------------------------------------------------------
//   private utility functions
//-----------------------------------------------------------------------


void                      getFastError  ();

template <int N>

  inline idx_t            size

  ( const Tuple<idx_t,N>&   shape );

template <class I, int N>

  inline idx_t            ioffset

  ( const Tuple<I,N>&       index,
    const Tuple<idx_t,N>&   stride );

template <int N>

  inline bool             incrIndex

  ( Tuple<idx_t,N>&         index,
    const Tuple<idx_t,N>&   shape );

template <int N>

  inline bool             incrIndex2

  ( Tuple<idx_t,N>&         index,
    const Tuple<idx_t,N>&   shape );

template <int N>

  inline bool             isContiguous

  ( const Tuple<idx_t,N>&   shape,
    const Tuple<idx_t,N>&   stride );

template <int N>

  inline bool             checkShape

  ( const Tuple<idx_t,N>&   shape );

using jem::checkIndex;

template <class I, int N>

  inline bool             checkIndex

  ( const Tuple<I,N>&       index,
    const Tuple<idx_t,N>&   shape );

template <class T, class U, class Op>

  void                    assignArray

  ( T*                      lhs,
    const U*                rhs,
    idx_t                   len,
    const Op&               op );

template <class T, class U, int N, class Op>

  inline void             assignArray

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const U*                rhs,
    const Tuple<idx_t,N>&   rst,
    const Tuple<idx_t,N>&   sh,
    const Op&               op );

template <class T, class U, class Op>

  void                    assignScalar

  ( T*                      lhs,
    const U&                rhs,
    idx_t                   len,
    const Op&               op );

template <class T, class U, int N, class Op>

  inline void             assignScalar

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const U&                rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op );

template <class Output,
          class T, int N>

  void                    print

  ( Output&                 out,
    const Array<T,N>&       a );


JEM_END_NAMESPACE( array )
JEM_END_PACKAGE_BASE

#include <jem/base/array/base.tcc>

#endif
