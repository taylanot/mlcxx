
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

#ifndef JEM_BASE_TUPLE_INTRINSICS_H
#define JEM_BASE_TUPLE_INTRINSICS_H

#include <jem/base/intrinsics.h>
#include <jem/base/tuple/expression.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   unary functions
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_UNARY_FUNC
#define JEM_DEFINE_UNARY_FUNC( OP, NAME )          \
                                                   \
template <class T, int M, int N, class E>          \
  inline Tuple< typename OP<T>::ResultType, M, N,  \
    UnaryExpr< OP<T>, Tuple<T,M,N,E> > >           \
  NAME ( const Tuple<T,M,N,E>& e )                 \
{                                                  \
  return Tuple< typename OP<T>::ResultType, M, N,  \
    UnaryExpr< OP<T>, Tuple<T,M,N,E> > > ( e );    \
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


template <class U, class T, int M, int N, class E>

  inline Tuple< U, M, N,
    UnaryExpr< StaticCast<U,T>, Tuple<T,M,N,E> > >

  castTo ( const Tuple<T,M,N,E>& e )

{
  return Tuple< U, M, N,
    UnaryExpr< StaticCast<U,T>, Tuple<T,M,N,E> > > ( e );
}


//-----------------------------------------------------------------------
//   binary functions
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_BINARY_FUNC
#define JEM_DEFINE_BINARY_FUNC( OP, NAME )                    \
                                                              \
template <class T, class U, int M, int N, class E>            \
  inline Tuple< typename OP<T,U>::ResultType, M, N,           \
    BinaryExpr< OP<T,U>, Tuple<T,M,N,E>,                      \
      Tuple<U,0,0> > >                                        \
  NAME ( const Tuple<T,M,N,E>& lhs, const U& rhs )            \
{                                                             \
  return Tuple< typename OP<T,U>::ResultType, M, N,           \
    BinaryExpr< OP<T,U>, Tuple<T,M,N,E>,                      \
      Tuple<U,0,0> > > ( lhs, Tuple<U,0,0>( rhs ) );          \
}                                                             \
                                                              \
template <class T, class U, int M, int N, class E>            \
  inline Tuple< typename OP<T,U>::ResultType, M, N,           \
    BinaryExpr< OP<T,U>, Tuple<T,0,0>,                        \
      Tuple<U,M,N,E> > >                                      \
  NAME ( const T& lhs, const Tuple<U,M,N,E>& rhs )            \
{                                                             \
  return Tuple< typename OP<T,U>::ResultType, M, N,           \
    BinaryExpr< OP<T,U>, Tuple<T,0,0>,                        \
      Tuple<U,M,N,E> > > ( Tuple<T,0,0>( lhs ), rhs );        \
}                                                             \
                                                              \
template <class T, class U, int M, int N, class E1, class E2> \
  inline Tuple< typename OP<T,U>::ResultType, M, N,           \
    BinaryExpr< OP<T,U>, Tuple<T,M,N,E1>,                     \
      Tuple<U,M,N,E2> > >                                     \
  NAME ( const Tuple<T,M,N,E1>& lhs,                          \
         const Tuple<U,M,N,E2>& rhs )                         \
{                                                             \
  return Tuple< typename OP<T,U>::ResultType, M, N,           \
    BinaryExpr< OP<T,U>, Tuple<T,M,N,E1>,                     \
      Tuple<U,M,N,E2> > > ( lhs, rhs );                       \
}

JEM_DEFINE_BINARY_FUNC( intrinsics::Pow, pow )

#undef JEM_DEFINE_BINARY_FUNC


//-----------------------------------------------------------------------
//   where
//-----------------------------------------------------------------------


template <class T, class U, int M, int N,
          class E1, class E2, class E3>

  inline Tuple< typename PromoteTypes<T,U>::ResultType, M, N,
    WhereExpr< Tuple<bool,M,N,E1>, Tuple<T,M,N,E2>,
      Tuple<U,M,N,E3> > >

  where ( const Tuple<bool,M,N,E1>&  mask,
          const Tuple<T,M,N,E2>&     lhs,
          const Tuple<U,M,N,E3>&     rhs )

{
  return Tuple< typename PromoteTypes<T,U>::ResultType, M, N,
    WhereExpr< Tuple<bool,M,N,E1>, Tuple<T,M,N,E2>,
      Tuple<U,M,N,E3> > > ( mask, lhs, rhs );
}


template <class T, class U, int M, int N, class E1, class E2>

  inline Tuple< typename PromoteTypes<T,U>::ResultType, M, N,
    WhereExpr< Tuple<bool,M,N,E1>, Tuple<T,M,N,E2>, Tuple<U,0,0> > >

  where ( const Tuple<bool,M,N,E1>&  mask,
          const Tuple<T,M,N,E2>&     lhs,
          const U&                   rhs )

{
  return Tuple< typename PromoteTypes<T,U>::ResultType, M, N,
    WhereExpr< Tuple<bool,M,N,E1>, Tuple<T,M,N,E2>, Tuple<U,0,0> > >
      ( mask, lhs, Tuple<U,0,0>( rhs ) );
}


template <class T, class U, int M, int N, class E1, class E2>

  inline Tuple< typename PromoteTypes<T,U>::ResultType, M, N,
    WhereExpr< Tuple<bool,M,N,E1>, Tuple<T,0,0>, Tuple<U,M,N,E2> > >

  where ( const Tuple<bool,M,N,E1>&  mask,
          const T&                   lhs,
          const Tuple<U,M,N,E2>&     rhs )

{
  return Tuple< typename PromoteTypes<T,U>::ResultType, M, N,
    WhereExpr< Tuple<bool,M,N,E1>, Tuple<T,0,0>, Tuple<U,M,N,E2> > >
      ( mask, Tuple<T,0,0>( lhs ), rhs );
}


template <class T, class U, int M, int N, class E>

  inline Tuple< typename PromoteTypes<T,U>::ResultType, M, N,
    WhereExpr< Tuple<bool,M,N,E>, Tuple<T,0,0>, Tuple<U,0,0> > >

  where ( const Tuple<bool,M,N,E>&  mask,
          const T&                  lhs,
          const U&                  rhs )

{
  return Tuple< typename PromoteTypes<T,U>::ResultType, M, N,
    WhereExpr< Tuple<bool,M,N,E>, Tuple<T,0,0>, Tuple<U,0,0> > >
      ( mask, Tuple<T,0,0>( lhs ), Tuple<U,0,0>( rhs ) );
}


JEM_END_PACKAGE_BASE

#ifdef JEM_STD_COMPLEX
#  include <jem/base/tuple/zintrinsics.h>
#endif

#endif
