
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

#ifndef JEM_BASE_ARRAY_OPERATORS_H
#define JEM_BASE_ARRAY_OPERATORS_H

#include <jem/base/array/expression.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   unary operators
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_UNARY_OPERATOR
#define JEM_DEFINE_UNARY_OPERATOR( OP, name )   \
                                                \
template <class T, int N, class E>              \
  inline Array< typename OP<T>::ResultType, N,  \
    UnaryExpr< OP<T>, Array<T,N,E> > >          \
  name ( const Array<T,N,E>& e )                \
{                                               \
  return Array< typename OP<T>::ResultType, N,  \
    UnaryExpr< OP<T>, Array<T,N,E> > > ( e );   \
}

JEM_DEFINE_UNARY_OPERATOR( Negate,      operator - )
JEM_DEFINE_UNARY_OPERATOR( LogicalNot,  operator ! )
JEM_DEFINE_UNARY_OPERATOR( BitwiseNot,  operator ~ )

#undef JEM_DEFINE_UNARY_OPERATOR


//-----------------------------------------------------------------------
//   binary operators
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_BINARY_OPERATOR
#define JEM_DEFINE_BINARY_OPERATOR( OP, name )           \
                                                         \
template <class T, class U, int N, class E>              \
  inline Array< typename OP<T,U>::ResultType, N,         \
    BinaryExpr< OP<T,U>, Array<T,N,E>,                   \
      ScalarExpr<U> > >                                  \
  name ( const Array<T,N,E>& lhs, const U& rhs )         \
{                                                        \
  return Array< typename OP<T,U>::ResultType, N,         \
    BinaryExpr< OP<T,U>, Array<T,N,E>,                   \
      ScalarExpr<U> > > ( lhs, rhs );                    \
}                                                        \
                                                         \
template <class T, class U, int N, class E>              \
  inline Array< typename OP<T,U>::ResultType, N,         \
    BinaryExpr< OP<T,U>, ScalarExpr<T>,                  \
      Array<U,N,E> > >                                   \
  name ( const T& lhs, const Array<U,N,E>& rhs )         \
{                                                        \
  return Array< typename OP<T,U>::ResultType, N,         \
    BinaryExpr< OP<T,U>, ScalarExpr<T>,                  \
      Array<U,N,E> > > ( lhs, rhs );                     \
}                                                        \
                                                         \
template <class T, class U, int N, class E1, class E2>   \
  inline Array< typename OP<T,U>::ResultType, N,         \
    BinaryExpr< OP<T,U>, Array<T,N,E1>,                  \
      Array<U,N,E2> > >                                  \
  name ( const Array<T,N,E1>& lhs,                       \
         const Array<U,N,E2>& rhs )                      \
{                                                        \
  return Array< typename OP<T,U>::ResultType, N,         \
    BinaryExpr< OP<T,U>, Array<T,N,E1>,                  \
      Array<U,N,E2> > > ( lhs, rhs );                    \
}

JEM_DEFINE_BINARY_OPERATOR( Multiplies,          operator *  )
JEM_DEFINE_BINARY_OPERATOR( Divides,             operator /  )
JEM_DEFINE_BINARY_OPERATOR( Modulus,             operator %  )
JEM_DEFINE_BINARY_OPERATOR( Plus,                operator +  )
JEM_DEFINE_BINARY_OPERATOR( Minus,               operator -  )
JEM_DEFINE_BINARY_OPERATOR( BitwiseAnd,          operator &  )
JEM_DEFINE_BINARY_OPERATOR( BitwiseOr,           operator |  )
JEM_DEFINE_BINARY_OPERATOR( BitwiseXor,          operator ^  )
JEM_DEFINE_BINARY_OPERATOR( BitwiseLeftShift,    blshift     )
JEM_DEFINE_BINARY_OPERATOR( BitwiseRightShift,   brshift     )
JEM_DEFINE_BINARY_OPERATOR( LogicalAnd,          operator && )
JEM_DEFINE_BINARY_OPERATOR( LogicalOr,           operator || )
JEM_DEFINE_BINARY_OPERATOR( LessThan,            operator <  )
JEM_DEFINE_BINARY_OPERATOR( LessEqual,           operator <= )
JEM_DEFINE_BINARY_OPERATOR( GreaterThan,         operator >  )
JEM_DEFINE_BINARY_OPERATOR( GreaterEqual,        operator >= )
JEM_DEFINE_BINARY_OPERATOR( EqualTo,             operator == )
JEM_DEFINE_BINARY_OPERATOR( NotEqual,            operator != )

#undef JEM_DEFINE_BINARY_OPERATOR


//-----------------------------------------------------------------------
//   computed assignment operators
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_ASSIGN
#define JEM_DEFINE_ASSIGN( OP, name )                            \
                                                                 \
template <class T, class U, int N, class E>                      \
  inline const Array<T,N,E>& name                                \
  ( const Array<T,N,E>& lhs, const U& rhs )                      \
{                                                                \
  lhs.assign ( rhs, OP<T,U>() );                                 \
  return lhs;                                                    \
}                                                                \
                                                                 \
template <class T, class U, int N, class E1, class E2>           \
  inline const Array<T,N,E1>& name                               \
  ( const Array<T,N,E1>& lhs, const Array<U,N,E2>& rhs )         \
{                                                                \
  lhs.assign ( rhs, OP<T,U>() );                                 \
  return lhs;                                                    \
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
