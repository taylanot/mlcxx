
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

#ifndef JEM_BASE_ARRAY_EXPRESSION_H
#define JEM_BASE_ARRAY_EXPRESSION_H

#include <jem/base/functional.h>
#include <jem/base/array/assign.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Array<T,N,UnaryExpr<Op,Array> >
//-----------------------------------------------------------------------


template <class T, int N, class Op, class U, class E>

  class Array<T,N,UnaryExpr<Op,Array<U,N,E> > >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               ValueType;
  typedef Array<U,N,E>    ArgType;
  typedef typename
    Op::ResultType        Reference;


  explicit                Array

    ( const ArgType&        a ) :

      op  (),
      arg ( a )

  {}

                          Array

    ( const Array&          rhs ) :

      op  ( rhs.op  ),
      arg ( rhs.arg )

  {}

  bool                    isContiguous  () const
  {
    return arg.isContiguous ();
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return op ( arg[i] );
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return op ( arg.getFast(i) );
  }

  idx_t                   size          () const
  {
    return arg.size ();
  }

  Shape                   shape         () const
  {
    return arg.shape ();
  }


 private:

  Array&                  operator = ( const Array& rhs );


 public:

  const Op                op;
  typename ArrayTraits
    <ArgType>::Reference  arg;

};


//-----------------------------------------------------------------------
//   class Array<T,N,BinaryExpr<Op,Array,Array> >
//-----------------------------------------------------------------------


template <class T, int N, class Op,
          class U, class V, class E1, class E2>

  class Array<T,N,BinaryExpr<Op,Array<U,N,E1>,Array<V,N,E2> > >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               ValueType;
  typedef Array<U,N,E1>   Arg1Type;
  typedef Array<V,N,E2>   Arg2Type;
  typedef typename
    Op::ResultType        Reference;


                          Array

    ( const Arg1Type&       a,
      const Arg2Type&       b ) :

      op   (),
      arg1 ( a ),
      arg2 ( b )

  {}

                          Array

    ( const Array&          rhs ) :

      op   ( rhs.op   ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  bool                    isContiguous  () const
  {
    return ( arg1.isContiguous() &&
             arg2.isContiguous() );
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return op ( arg1[i], arg2[i] );
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return op ( arg1.getFast(i), arg2.getFast(i) );
  }

  idx_t                   size          () const
  {
    JEM_ASSERT2 ( arg1.size() == arg2.size(),
                  array::Errors::SIZE_MISMATCH );

    return arg1.size ();
  }

  Shape                   shape         () const
  {
    JEM_ASSERT2 ( equal( arg1.shape(), arg2.shape() ),
                  array::Errors::SHAPE_MISMATCH );

    return arg1.shape ();
  }


 private:

  Array&                  operator =  ( const Array& rhs );


 public:

  const Op                op;
  typename ArrayTraits
    <Arg1Type>::Reference arg1;
  typename ArrayTraits
    <Arg2Type>::Reference arg2;

};


//-----------------------------------------------------------------------
//   class Array<T,N,BinaryExpr<Op,Array,Scalar> >
//-----------------------------------------------------------------------


template <class T, int N, class Op, class U, class V, class E>

  class Array<T,N,BinaryExpr<Op,Array<U,N,E>,ScalarExpr<V> > >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               ValueType;
  typedef Array<U,N,E>    Arg1Type;
  typedef V               Arg2Type;
  typedef typename
    Op::ResultType        Reference;


                          Array

    ( const Arg1Type&       a,
      const Arg2Type&       b ) :

      op   (),
      arg1 ( a ),
      arg2 ( b )

  {}

                          Array

    ( const Array&          rhs ) :

      op   ( rhs.op   ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  bool                    isContiguous  () const
  {
    return arg1.isContiguous ();
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return op ( arg1[i], arg2 );
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return op ( arg1.getFast(i), arg2 );
  }

  idx_t                   size          () const
  {
    return arg1.size ();
  }

  Shape                   shape         () const
  {
    return arg1.shape ();
  }


 private:

  Array&                  operator = ( const Array& rhs );


 public:

  const Op                op;
  typename ArrayTraits
    <Arg1Type>::Reference arg1;
  const Arg2Type          arg2;

};


//-----------------------------------------------------------------------
//   class Array<T,N,BinaryExpr<Op,Scalar,Array> >
//-----------------------------------------------------------------------


template <class T, int N, class Op, class U, class V, class E>

  class Array<T,N,BinaryExpr<Op,ScalarExpr<U>,Array<V,N,E> > >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               ValueType;
  typedef U               Arg1Type;
  typedef Array<V,N,E>    Arg2Type;
  typedef typename
    Op::ResultType        Reference;


                          Array

    ( const Arg1Type&       a,
      const Arg2Type&       b ) :

      op   (),
      arg1 ( a ),
      arg2 ( b )

  {}

                          Array

    ( const Array&          rhs ) :

      op   ( rhs.op   ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  bool                    isContiguous  () const
  {
    return arg2.isContiguous ();
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return op ( arg1, arg2[i] );
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return op ( arg1, arg2.getFast(i) );
  }

  idx_t                   size          () const
  {
    return arg2.size ();
  }

  Shape                   shape         () const
  {
    return arg2.shape ();
  }


 private:

  Array&                  operator = ( const Array& rhs );


 public:

  const Op                op;
  const Arg1Type          arg1;
  typename ArrayTraits
    <Arg2Type>::Reference arg2;

};


//-----------------------------------------------------------------------
//   class Array<T,N,WhereExpr<Array,Array,Array> >
//-----------------------------------------------------------------------


template <class T, int N, class U, class V,
          class E1, class E2, class E3>

  class Array<T,N,WhereExpr<
    Array<bool,N,E1>, Array<U,N,E2>, Array<V,N,E3> > >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef Array
    <bool,N,E1>           MaskType;
  typedef Array<U,N,E2>   Arg1Type;
  typedef Array<V,N,E3>   Arg2Type;
  typedef T               Reference;


                          Array

    ( const MaskType&       a,
      const Arg1Type&       b,
      const Arg2Type&       c ) :

      mask ( a ),
      arg1 ( b ),
      arg2 ( c )

  {}

                          Array

    ( const Array&          rhs ) :

      mask ( rhs.mask ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  bool                    isContiguous  () const
  {
    return ( mask.isContiguous() &&
             arg1.isContiguous() &&
             arg2.isContiguous() );
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return ( mask[i] ? arg1[i] : arg2[i] );
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return ( mask.getFast(i) ?
             arg1.getFast(i) :
             arg2.getFast(i) );
  }

  idx_t                   size          () const
  {
    JEM_ASSERT2 ( mask.size() == arg1.size() &&
                  mask.size() == arg2.size(),
                  array::Errors::SIZE_MISMATCH );

    return mask.size ();
  }

  Shape                   shape         () const
  {
    JEM_ASSERT2 ( equal( mask.shape(), arg1.shape() ) &&
                  equal( mask.shape(), arg2.shape() ),
                  array::Errors::SHAPE_MISMATCH );

    return mask.shape ();
  }


 private:

  Array&                  operator = ( const Array& rhs );


 public:

  typename ArrayTraits
    <MaskType>::Reference mask;
  typename ArrayTraits
    <Arg1Type>::Reference arg1;
  typename ArrayTraits
    <Arg2Type>::Reference arg2;

};


//-----------------------------------------------------------------------
//   class Array<T,N,WhereExpr<Array,Array,Scalar> >
//-----------------------------------------------------------------------


template <class T, int N, class U, class V, class E1, class E2>

  class Array<T,N,WhereExpr<Array<bool,N,E1>,
    Array<U,N,E2>, ScalarExpr<V> > >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               Reference;
  typedef Array
    <bool,N,E1>           MaskType;
  typedef Array<U,N,E2>   Arg1Type;
  typedef V               Arg2Type;


                          Array

    ( const MaskType&       a,
      const Arg1Type&       b,
      const Arg2Type&       c ) :

      mask ( a ),
      arg1 ( b ),
      arg2 ( c )

  {}

                          Array

    ( const Array&          rhs ) :

      mask ( rhs.mask ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  bool                    isContiguous  () const
  {
    return ( mask.isContiguous() &&
             arg1.isContiguous() );
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return ( mask[i] ? arg1[i] : arg2 );
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return ( mask.getFast(i) ? arg1.getFast(i) : arg2 );
  }

  idx_t                   size          () const
  {
    JEM_ASSERT2 ( mask.size() == arg1.size(),
                  array::Errors::SIZE_MISMATCH );

    return mask.size ();
  }

  Shape                   shape         () const
  {
    JEM_ASSERT2 ( equal( mask.shape(), arg1.shape() ),
                  array::Errors::SHAPE_MISMATCH );

    return mask.shape ();
  }


 private:

  Array&                  operator = ( const Array& rhs );


 public:

  typename ArrayTraits
    <MaskType>::Reference mask;
  typename ArrayTraits
    <Arg1Type>::Reference arg1;
  const Arg2Type          arg2;

};


//-----------------------------------------------------------------------
//   class Array<T,N,WhereExpr<Array,Scalar,Array> >
//-----------------------------------------------------------------------


template <class T, int N, class U, class V, class E1, class E2>

  class Array<T,N,WhereExpr<Array<bool,N,E1>,
    ScalarExpr<U>, Array<V,N,E2> > >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               Reference;
  typedef Array
    <bool,N,E1>           MaskType;
  typedef U               Arg1Type;
  typedef Array<V,N,E2>   Arg2Type;

                          Array

    ( const MaskType&       a,
      const Arg1Type&       b,
      const Arg2Type&       c ) :

      mask ( a ),
      arg1 ( b ),
      arg2 ( c )

  {}

                          Array

    ( const Array&          rhs ) :

      mask ( rhs.mask ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  bool                    isContiguous  () const
  {
    return ( mask.isContiguous() &&
             arg2.isContiguous() );
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return ( mask[i] ? arg1 : arg2[i] );
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return ( mask.getFast(i) ? arg1 : arg2.getFast(i) );
  }

  idx_t                   size          () const
  {
    JEM_ASSERT2 ( mask.size() == arg2.size(),
                  array::Errors::SIZE_MISMATCH );

    return mask.size ();
  }

  Shape                   shape         () const
  {
    JEM_ASSERT2 ( equal( mask.shape(), arg2.shape() ),
                  array::Errors::SHAPE_MISMATCH );

    return mask.shape ();
  }


 private:

  Array&                  operator = ( const Array& rhs );


 public:

  typename ArrayTraits
    <MaskType>::Reference mask;
  const Arg1Type          arg1;
  typename ArrayTraits
    <Arg2Type>::Reference arg2;

};


//-----------------------------------------------------------------------
//   class Array<T,N,WhereExpr<Array,Scalar,Scalar> >
//-----------------------------------------------------------------------


template <class T, int N, class U, class V, class E>

  class Array<T,N,WhereExpr<Array<bool,N,E>,
    ScalarExpr<U>, ScalarExpr<V> > >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               Reference;
  typedef Array
    <bool,N,E>            MaskType;
  typedef U               Arg1Type;
  typedef V               Arg2Type;

                          Array

    ( const MaskType&       a,
      const Arg1Type&       b,
      const Arg2Type&       c ) :

      mask ( a ),
      arg1 ( b ),
      arg2 ( c )

  {}

                          Array

    ( const Array&          rhs ) :

      mask ( rhs.mask ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  bool                    isContiguous  () const
  {
    return mask.isContiguous ();
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return ( mask[i] ? arg1 : arg2 );
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return ( mask.getFast(i) ? arg1 : arg2 );
  }

  idx_t                   size          () const
  {
    return mask.size ();
  }

  Shape                   shape         () const
  {
    return mask.shape ();
  }


 private:

  Array&                  operator = ( const Array& rhs );


 public:

  typename ArrayTraits
    <MaskType>::Reference mask;
  const Arg1Type          arg1;
  const Arg2Type          arg2;

};


//-----------------------------------------------------------------------
//   class Array<idx_t,1,Slice>
//-----------------------------------------------------------------------


template <>

  class Array<idx_t,1,Slice>

{
 public:

  typedef Tuple<idx_t,1>  Shape;
  typedef idx_t           Reference;


                          Array

    ( idx_t                 first,
      idx_t                 last,
      idx_t                 stride ) :

      first_  (  first ),
      last_   (   last ),
      stride_ ( stride )

  {}

                          Array

    ( const Array&          rhs ) :

      first_  ( rhs.first_  ),
      last_   ( rhs.last_   ),
      stride_ ( rhs.stride_ )

  {}

  bool                    isContiguous  () const
  {
    return true;
  }

  Reference               operator []

    ( idx_t                 i )            const

  {
    return (first_ + i * stride_);
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return (first_ + i * stride_);
  }

  idx_t                   size          () const
  {
    return ((last_ - first_ + stride_ - 1_idx) / stride_);
  }

  Shape                   shape         () const
  {
    return Shape ( size() );
  }


 private:

  Array&                  operator = ( const Array& rhs );


 private:

  const idx_t             first_;
  const idx_t             last_;
  const idx_t             stride_;

};


//-----------------------------------------------------------------------
//   class Array<idx_t,1,SliceFromTo>
//-----------------------------------------------------------------------


template <>

  class Array<idx_t,1,SliceFromTo>

{
 public:

  typedef Tuple<idx_t,1>  Shape;
  typedef idx_t           Reference;


                          Array

    ( idx_t                 first,
      idx_t                 last ) :

      first_  (  first ),
      last_   (   last )

  {}

                          Array

    ( const Array&          rhs ) :

      first_  ( rhs.first_ ),
      last_   ( rhs.last_  )

  {}

  bool                    isContiguous  () const
  {
    return true;
  }

  Reference               operator []

    ( idx_t                 i )            const

  {
    return (first_ + i);
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return (first_ + i);
  }

  idx_t                   size          () const
  {
    return (last_ - first_);
  }

  Shape                   shape         () const
  {
    return Shape ( last_ - first_ );
  }


 private:

  Array&                  operator = ( const Array& rhs );


 private:

  const idx_t             first_;
  const idx_t             last_;

};


//-----------------------------------------------------------------------
//   class Array<idx_t,1,SliceTo>
//-----------------------------------------------------------------------


template <>

  class Array<idx_t,1,SliceTo>

{
 public:

  typedef Tuple<idx_t,1>  Shape;
  typedef idx_t           Reference;


                          Array

    ( idx_t                 last ) :

      last_ ( last )

  {}

                          Array

    ( const Array&          rhs ) :

      last_ ( rhs.last_ )

  {}

  bool                    isContiguous  () const
  {
    return true;
  }

  Reference               operator []

    ( idx_t                 i )            const

  {
    return i;
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return i;
  }

  idx_t                   size          () const
  {
    return last_;
  }

  Shape                   shape         () const
  {
    return Shape ( last_ );
  }


 private:

  Array&                  operator = ( const Array& rhs );


 private:

  const idx_t             last_;

};


JEM_END_PACKAGE_BASE

#endif
