
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

#ifndef JEM_BASE_ARRAY_INTRINSICS_H
#define JEM_BASE_ARRAY_INTRINSICS_H

#include <jem/base/intrinsics.h>
#include <jem/base/array/expression.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   unary functions
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_UNARY_FUNC
#define JEM_DEFINE_UNARY_FUNC( OP, NAME )       \
                                                \
template <class T, int N, class E>              \
  inline Array< typename OP<T>::ResultType, N,  \
    UnaryExpr< OP<T>, Array<T,N,E> > >          \
  NAME ( const Array<T,N,E>& e )                \
{                                               \
  return Array< typename OP<T>::ResultType, N,  \
    UnaryExpr< OP<T>, Array<T,N,E> > > ( e );   \
}

JEM_DEFINE_UNARY_FUNC( intrinsics::Abs,    abs   )
JEM_DEFINE_UNARY_FUNC( intrinsics::ArcCos, acos  )
JEM_DEFINE_UNARY_FUNC( intrinsics::ArcSin, asin  )
JEM_DEFINE_UNARY_FUNC( intrinsics::ArcTan, atan  )
JEM_DEFINE_UNARY_FUNC( intrinsics::Ceil,   ceil  )
JEM_DEFINE_UNARY_FUNC( intrinsics::Cos,    cos   )
JEM_DEFINE_UNARY_FUNC( intrinsics::Cosh,   cosh  )
JEM_DEFINE_UNARY_FUNC( intrinsics::Exp,    exp   )
JEM_DEFINE_UNARY_FUNC( intrinsics::Floor,  floor )
JEM_DEFINE_UNARY_FUNC( intrinsics::Log,    log   )
JEM_DEFINE_UNARY_FUNC( intrinsics::Log10,  log10 )
JEM_DEFINE_UNARY_FUNC( intrinsics::Sin,    sin   )
JEM_DEFINE_UNARY_FUNC( intrinsics::Sinh,   sinh  )
JEM_DEFINE_UNARY_FUNC( intrinsics::Sqrt,   sqrt  )
JEM_DEFINE_UNARY_FUNC( intrinsics::Tan,    tan   )
JEM_DEFINE_UNARY_FUNC( intrinsics::Tanh,   tanh  )

#undef JEM_DEFINE_UNARY_FUNC


//-----------------------------------------------------------------------
//   castTo
//-----------------------------------------------------------------------


template <class U, class T, int N, class E>

  inline Array< U, N,
    UnaryExpr< StaticCast<U,T>, Array<T,N,E> > >

  castTo ( const Array<T,N,E>& e )

{
  return Array< U, N,
    UnaryExpr<StaticCast<U,T>, Array<T,N,E> > > ( e );
}


//-----------------------------------------------------------------------
//   binary functions
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_BINARY_FUNC
#define JEM_DEFINE_BINARY_FUNC( OP, NAME )              \
                                                        \
template <class T, class U, int N, class E>             \
  inline Array< typename OP<T,U>::ResultType, N,        \
    BinaryExpr< OP<T,U>, Array<T,N,E>,                  \
      ScalarExpr<U> > >                                 \
  NAME ( const Array<T,N,E>& lhs, const U& rhs )        \
{                                                       \
  return Array< typename OP<T,U>::ResultType, N,        \
    BinaryExpr< OP<T,U>, Array<T,N,E>,                  \
      ScalarExpr<U> > > ( lhs, rhs );                   \
}                                                       \
                                                        \
template <class T, class U, int N, class E>             \
  inline Array< typename OP<T,U>::ResultType, N,        \
    BinaryExpr< OP<T,U>, ScalarExpr<T>,                 \
      Array<U,N,E> > >                                  \
  NAME ( const T& lhs, const Array<U,N,E>& rhs )        \
{                                                       \
  return Array< typename OP<T,U>::ResultType, N,        \
    BinaryExpr< OP<T,U>, ScalarExpr<T>,                 \
      Array<U,N,E> > > ( lhs, rhs );                    \
}                                                       \
                                                        \
template <class T, class U, int N, class E1, class E2>  \
  inline Array< typename OP<T,U>::ResultType, N,        \
    BinaryExpr< OP<T,U>, Array<T,N,E1>,                 \
      Array<U,N,E2> > >                                 \
  NAME ( const Array<T,N,E1>& lhs,                      \
         const Array<U,N,E2>& rhs )                     \
{                                                       \
  return Array< typename OP<T,U>::ResultType, N,        \
    BinaryExpr< OP<T,U>, Array<T,N,E1>,                 \
      Array<U,N,E2> > > ( lhs, rhs );                   \
}

JEM_DEFINE_BINARY_FUNC( intrinsics::Pow, pow )

#undef JEM_DEFINE_BINARY_FUNC


//-----------------------------------------------------------------------
//   where
//-----------------------------------------------------------------------


template <class T, class U, int N, class E1, class E2, class E3>

  inline Array< typename PromoteTypes<T,U>::ResultType, N,
    WhereExpr< Array<bool,N,E1>, Array<T,N,E2>, Array<U,N,E3> > >

  where ( const Array<bool,N,E1>& mask,
          const Array<T,N,E2>&    lhs,
          const Array<U,N,E3>&    rhs )

{
  return Array< typename PromoteTypes<T,U>::ResultType, N,
    WhereExpr< Array<bool,N,E1>, Array<T,N,E2>, Array<U,N,E3> > >
      ( mask, lhs, rhs );
}


template <class T, class U, int N, class E1, class E2>

  inline Array< typename PromoteTypes<T,U>::ResultType, N,
    WhereExpr< Array<bool,N,E1>, Array<T,N,E2>, ScalarExpr<U> > >

  where ( const Array<bool,N,E1>& mask,
          const Array<T,N,E2>&    lhs,
          const U&                rhs )

{
  return Array< typename PromoteTypes<T,U>::ResultType, N,
    WhereExpr< Array<bool,N,E1>, Array<T,N,E2>, ScalarExpr<U> > >
      ( mask, lhs, rhs );
}


template <class T, class U, int N, class E1, class E2>

  inline Array< typename PromoteTypes<T,U>::ResultType, N,
    WhereExpr< Array<bool,N,E1>, ScalarExpr<T>, Array<U,N,E2> > >

  where ( const Array<bool,N,E1>& mask,
          const T&                lhs,
          const Array<U,N,E2>&    rhs )

{
  return Array< typename PromoteTypes<T,U>::ResultType, N,
    WhereExpr< Array<bool,N,E1>, ScalarExpr<T>, Array<U,N,E2> > >
      ( mask, lhs, rhs );
}


template <class T, class U, int N, class E>

  inline Array< typename PromoteTypes<T,U>::ResultType, N,
    WhereExpr< Array<bool,N,E>, ScalarExpr<T>, ScalarExpr<U> > >

  where ( const Array<bool,N,E>&  mask,
          const T&                lhs,
          const U&                rhs )

{
  return Array< typename PromoteTypes<T,U>::ResultType, N,
    WhereExpr< Array<bool,N,E>, ScalarExpr<T>, ScalarExpr<U> > >
      ( mask, lhs, rhs );
}


JEM_END_PACKAGE_BASE

#ifdef JEM_STD_COMPLEX
#  include <jem/base/array/zintrinsics.h>
#endif

#endif
