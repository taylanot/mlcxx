
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

#ifndef JEM_BASE_TUPLE_OPERATORS_H
#define JEM_BASE_TUPLE_OPERATORS_H

#include <jem/base/functional.h>
#include <jem/base/tuple/expression.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   unary operators
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_UNARY_OPERATOR
#define JEM_DEFINE_UNARY_OPERATOR( OP, name )      \
                                                   \
template <class T, int M, int N, class E>          \
  inline Tuple< typename OP<T>::ResultType, M, N,  \
    UnaryExpr< OP<T>, Tuple<T,M,N,E> > >           \
  name ( const Tuple<T,M,N,E>& e )                 \
{                                                  \
  return Tuple< typename OP<T>::ResultType, M, N,  \
    UnaryExpr< OP<T>, Tuple<T,M,N,E> > > ( e );    \
}

JEM_DEFINE_UNARY_OPERATOR( Negate,      operator - )
JEM_DEFINE_UNARY_OPERATOR( LogicalNot,  operator ! )
JEM_DEFINE_UNARY_OPERATOR( BitwiseNot,  operator ~ )

#undef JEM_DEFINE_UNARY_OPERATOR


//-----------------------------------------------------------------------
//   binary operators
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_BINARY_OPERATOR
#define JEM_DEFINE_BINARY_OPERATOR( OP, name )                   \
                                                                 \
template <class T, class U, int M, int N, class E>               \
  inline Tuple< typename OP<T,U>::ResultType, M, N,              \
    BinaryExpr< OP<T,U>, Tuple<T,M,N,E>,                         \
      Tuple<U,0,0> > >                                           \
  name ( const Tuple<T,M,N,E>& lhs, const U& rhs )               \
{                                                                \
  return Tuple< typename OP<T,U>::ResultType, M, N,              \
    BinaryExpr< OP<T,U>, Tuple<T,M,N,E>,                         \
      Tuple<U,0,0> > > ( lhs, Tuple<U,0,0>( rhs ) );             \
}                                                                \
                                                                 \
template <class T, class U, int M, int N, class E>               \
  inline Tuple< typename OP<T,U>::ResultType, M, N,              \
    BinaryExpr< OP<T,U>, Tuple<T,0,0>,                           \
      Tuple<U,M,N,E> > >                                         \
  name ( const T& lhs, const Tuple<U,M,N,E>& rhs )               \
{                                                                \
  return Tuple< typename OP<T,U>::ResultType, M, N,              \
    BinaryExpr< OP<T,U>, Tuple<T,0,0>,                           \
      Tuple<U,M,N,E> > > ( Tuple<T,0,0>( lhs ), rhs );           \
}                                                                \
                                                                 \
template <class T, class U, int M, int N, class E1, class E2>    \
  inline Tuple< typename OP<T,U>::ResultType, M, N,              \
    BinaryExpr< OP<T,U>, Tuple<T,M,N,E1>,                        \
      Tuple<U,M,N,E2> > >                                        \
  name ( const Tuple<T,M,N,E1>& lhs,                             \
         const Tuple<U,M,N,E2>& rhs )                            \
{                                                                \
  return Tuple< typename OP<T,U>::ResultType, M, N,              \
    BinaryExpr< OP<T,U>, Tuple<T,M,N,E1>,                        \
      Tuple<U,M,N,E2> > >  ( lhs, rhs );                         \
}

JEM_DEFINE_BINARY_OPERATOR( Multiplies,         operator *  )
JEM_DEFINE_BINARY_OPERATOR( Divides,            operator /  )
JEM_DEFINE_BINARY_OPERATOR( Modulus,            operator %  )
JEM_DEFINE_BINARY_OPERATOR( Plus,               operator +  )
JEM_DEFINE_BINARY_OPERATOR( Minus,              operator -  )
JEM_DEFINE_BINARY_OPERATOR( BitwiseAnd,         operator &  )
JEM_DEFINE_BINARY_OPERATOR( BitwiseOr,          operator |  )
JEM_DEFINE_BINARY_OPERATOR( BitwiseXor,         operator ^  )
JEM_DEFINE_BINARY_OPERATOR( BitwiseLeftShift,   blshift     )
JEM_DEFINE_BINARY_OPERATOR( BitwiseRightShift,  brshift     )
JEM_DEFINE_BINARY_OPERATOR( LogicalAnd,         operator && )
JEM_DEFINE_BINARY_OPERATOR( LogicalOr,          operator || )
JEM_DEFINE_BINARY_OPERATOR( LessThan,           operator <  )
JEM_DEFINE_BINARY_OPERATOR( LessEqual,          operator <= )
JEM_DEFINE_BINARY_OPERATOR( GreaterThan,        operator >  )
JEM_DEFINE_BINARY_OPERATOR( GreaterEqual,       operator >= )
JEM_DEFINE_BINARY_OPERATOR( EqualTo,            operator == )
JEM_DEFINE_BINARY_OPERATOR( NotEqual,           operator != )

#undef JEM_DEFINE_BINARY_OPERATOR

JEM_BEGIN_NAMESPACE( tuple )

//-----------------------------------------------------------------------
//   assign ( tuple op= skalar )
//-----------------------------------------------------------------------


template <class T, class U, class Op, int M, int N, int P>

  void                    assign

  ( Tuple<T,M,N>&           lhs,
    const U&                rhs,
    const Op&               op,
    const IntConstant<P>&   sz )

{
  for ( int i = 0; i < P; i++ )
  {
    lhs[i] = op ( lhs[i], rhs );
  }
}


template <class T, class U, class Op, int M, int N>

  inline void             assign

  ( Tuple<T,M,N>&           lhs,
    const U&                rhs,
    const Op&               op,
    const IntConstant<0>&   sz )

{}


template <class T, class U, class Op, int M, int N>

  inline void             assign

  ( Tuple<T,M,N>&           lhs,
    const U&                rhs,
    const Op&               op,
    const IntConstant<1>&   sz )

{
  lhs[0] = op ( lhs[0], rhs );
}


template <class T, class U, class Op, int M, int N>

  inline void             assign

  ( Tuple<T,M,N>&           lhs,
    const U&                rhs,
    const Op&               op,
    const IntConstant<2>&   sz )

{
  lhs[0] = op ( lhs[0], rhs );
  lhs[1] = op ( lhs[1], rhs );
}


template <class T, class U, class Op, int M, int N>

  inline void             assign

  ( Tuple<T,M,N>&           lhs,
    const U&                rhs,
    const Op&               op,
    const IntConstant<3>&   sz )

{
  lhs[0] = op ( lhs[0], rhs );
  lhs[1] = op ( lhs[1], rhs );
  lhs[2] = op ( lhs[2], rhs );
}


//-----------------------------------------------------------------------
//   assign ( tuple op= tuple )
//-----------------------------------------------------------------------


template <class T, class U, class E, class Op, int M, int N, int P>

  void                    assign

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const Op&               op,
    const IntConstant<P>&   sz )

{
  for ( int i = 0; i < P; i++ )
  {
    lhs[i] = op ( lhs[i], rhs[i] );
  }
}


template <class T, class U, class E, class Op, int M, int N>

  inline void             assign

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const Op&               op,
    const IntConstant<0>&   sz )

{}


template <class T, class U, class E, class Op, int M, int N>

  inline void             assign

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const Op&               op,
    const IntConstant<1>&   sz )

{
  lhs[0] = op ( lhs[0], rhs[0] );
}


template <class T, class U, class E, class Op, int M, int N>

  inline void             assign

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const Op&               op,
    const IntConstant<2>&   sz )

{
  lhs[0] = op ( lhs[0], rhs[0] );
  lhs[1] = op ( lhs[1], rhs[1] );
}


template <class T, class U, class E, class Op, int M, int N>

  inline void             assign

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const Op&               op,
    const IntConstant<3>&   sz )

{
  lhs[0] = op ( lhs[0], rhs[0] );
  lhs[1] = op ( lhs[1], rhs[1] );
  lhs[2] = op ( lhs[2], rhs[2] );
}

JEM_END_NAMESPACE( tuple )

//-----------------------------------------------------------------------
//   computed assignment operators
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_ASSIGN
#define JEM_DEFINE_ASSIGN( OP, name )                    \
                                                         \
template <class T, class U, int M, int N>                \
  inline Tuple<T,M,N>& name                              \
  ( Tuple<T,M,N>& lhs, const U& rhs )                    \
{                                                        \
  tuple::assign ( lhs, rhs, OP<T,U>(),                   \
                  IntConstant<(M * N)>() );              \
  return lhs;                                            \
}                                                        \
                                                         \
template <class T, class U, int M, int N, class E>       \
  inline Tuple<T,M,N>& name                              \
  ( Tuple<T,M,N>& lhs, const Tuple<U,M,N,E>& rhs )       \
{                                                        \
  tuple::assign ( lhs, rhs, OP<T,U>(),                   \
                  IntConstant<(M * N)>() );              \
  return lhs;                                            \
}

JEM_DEFINE_ASSIGN( Multiplies,         operator *= )
JEM_DEFINE_ASSIGN( Divides,            operator /= )
JEM_DEFINE_ASSIGN( Modulus,            operator %= )
JEM_DEFINE_ASSIGN( Plus,               operator += )
JEM_DEFINE_ASSIGN( Minus,              operator -= )
JEM_DEFINE_ASSIGN( BitwiseAnd,         operator &= )
JEM_DEFINE_ASSIGN( BitwiseOr,          operator |= )
JEM_DEFINE_ASSIGN( BitwiseXor,         operator ^= )
JEM_DEFINE_ASSIGN( BitwiseLeftShift,   operator <<= )
JEM_DEFINE_ASSIGN( BitwiseRightShift,  operator >>= )

#undef JEM_DEFINE_ASSIGN


JEM_END_PACKAGE_BASE

#endif
