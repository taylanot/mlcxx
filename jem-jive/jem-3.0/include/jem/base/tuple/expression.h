
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

#ifndef JEM_BASE_TUPLE_EXPRESSION_H
#define JEM_BASE_TUPLE_EXPRESSION_H

#include <jem/base/tuple/base.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Tuple<T,0,0,Nil>
//-----------------------------------------------------------------------


template <class T>

  class Tuple<T,0,0,Nil>

{
 public:

  typedef T             ValueType;
  typedef const T&      Reference;


  explicit              Tuple

    ( const T&            value ) :

      value_ ( value )

  {}

                        Tuple

    ( const Tuple&        rhs ) :

      value_ ( rhs.value_ )

  {}

  Reference             operator [] ( int i ) const
  {
    return value_;
  }


 private:

  Tuple&                operator = ( const Tuple& rhs );


 private:

  const T               value_;

};


//-----------------------------------------------------------------------
//   class Tuple<T,M,N,UnaryExpr<Op,A> >
//-----------------------------------------------------------------------


template <class T, int M, int N, class Op, class A>

  class Tuple<T,M,N,UnaryExpr<Op,A> >

{
 public:

  typedef T             ValueType;
  typedef typename
    Op::ResultType      Reference;


  explicit              Tuple

    ( const A&            arg ) :

      op_  (),
      arg_ ( arg )

  {}

                        Tuple

    ( const Tuple&        rhs ) :

      op_  ( rhs.op_  ),
      arg_ ( rhs.arg_ )

  {}

  Reference             operator [] ( int i ) const
  {
    return op_ ( arg_[i] );
  }


 private:

  Tuple&                operator = ( const Tuple& rhs );


 private:

  const Op              op_;
  typename TupleTraits
    <A>::Reference      arg_;

};


//-----------------------------------------------------------------------
//   class Tuple<T,M,N,BinaryExpr<Op,A,B> >
//-----------------------------------------------------------------------


template <class T, int M, int N, class Op, class A, class B>

  class Tuple<T,M,N,BinaryExpr<Op,A,B> >

{
 public:

  typedef T             ValueType;
  typedef typename
    Op::ResultType      Reference;


                        Tuple

    ( const A&            lhs,
      const B&            rhs ) :

      op_  (),
      lhs_ ( lhs ),
      rhs_ ( rhs )

  {}

                        Tuple

    ( const Tuple&        rhs ) :

      op_  ( rhs.op_  ),
      lhs_ ( rhs.lhs_ ),
      rhs_ ( rhs.rhs_ )

  {}

  Reference             operator [] ( int i ) const
  {
    return op_ ( lhs_[i], rhs_[i] );
  }


 private:

  Tuple&                operator = ( const Tuple& rhs );


 private:

  const Op              op_;
  typename TupleTraits
    <A>::Reference      lhs_;
  typename TupleTraits
    <B>::Reference      rhs_;

};


//-----------------------------------------------------------------------
//   class Tuple<T,M,N,WhereExpr<A,B,C> >
//-----------------------------------------------------------------------


template <class T, int M, int N, class A, class B, class C>

  class Tuple<T,M,N,WhereExpr<A,B,C> >

{
 public:

  typedef T             ValueType;
  typedef T             Reference;


                        Tuple

    ( const A&            mask,
      const B&            lhs,
      const C&            rhs ) :

      mask_ ( mask ),
      lhs_  (  lhs ),
      rhs_  (  rhs )

  {}

                        Tuple

    ( const Tuple&        rhs ) :

      mask_ ( rhs.mask_ ),
      lhs_  ( rhs.lhs_  ),
      rhs_  ( rhs.rhs_  )

  {}

  Reference             operator [] ( int i ) const
  {
    return ( mask_[i] ? lhs_[i] : rhs_[i] );
  }


 private:

  Tuple&                operator = ( const Tuple& rhs );


 private:

  typename TupleTraits
    <A>::Reference      mask_;
  typename TupleTraits
    <B>::Reference      lhs_;
  typename TupleTraits
    <C>::Reference      rhs_;

};


JEM_END_PACKAGE_BASE

#endif
