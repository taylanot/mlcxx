
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

#ifndef JEM_BASE_INTRINSICS_H
#define JEM_BASE_INTRINSICS_H

#include <cmath>
#include <cstdlib>
#include <jem/base/functional.h>


JEM_BEGIN_PACKAGE_BASE

JEM_BEGIN_NAMESPACE( intrinsics )


//-----------------------------------------------------------------------
//   unary function classes
//-----------------------------------------------------------------------


template <class T>

  inline typename
  UnaryFunc<T>::ResultType abs ( const T& arg )

{
  return (typename UnaryFunc<T>::ResultType) std::abs ( arg );
}

inline int     abs ( int arg )
{
  return std::abs ( arg );
}

inline long    abs ( long arg )
{
  return std::labs ( arg );
}

inline lint    abs ( lint arg )
{
  return std::llabs ( arg );
}

inline idx_t   abs ( idx_t arg )
{
  return idx_t ( abs( arg.value ) );
}

inline float   abs ( float arg )
{
  return (float) std::fabs ( arg );
}

inline double  abs ( double arg )
{
  return std::fabs ( arg );
}

template <class T>

  class Abs : public UnaryFunc<T>

{
 public:

  typename UnaryFunc<T>::ResultType

    operator () ( const T& arg ) const

  {
    return abs ( arg );
  }

};


#define JEM_DEFINE_UNARY_FUNC( NAME, name )              \
                                                         \
template <class T>                                       \
  inline typename                                        \
  UnaryFunc<T>::ResultType name ( const T& arg )         \
                                                         \
{                                                        \
  return (typename UnaryFunc<T>::ResultType)             \
    ::name ( arg );                                      \
}                                                        \
                                                         \
template <class T>                                       \
  class NAME : public UnaryFunc<T>                       \
{                                                        \
 public:                                                 \
                                                         \
  typename UnaryFunc<T>::ResultType                      \
    operator () ( const T& arg ) const                   \
  {                                                      \
    return name ( arg );                                 \
  }                                                      \
                                                         \
};


JEM_DEFINE_UNARY_FUNC( ArcCos, acos  )
JEM_DEFINE_UNARY_FUNC( ArcSin, asin  )
JEM_DEFINE_UNARY_FUNC( ArcTan, atan  )
JEM_DEFINE_UNARY_FUNC( Ceil,   ceil  )
JEM_DEFINE_UNARY_FUNC( Cos,    cos   )
JEM_DEFINE_UNARY_FUNC( Cosh,   cosh  )
JEM_DEFINE_UNARY_FUNC( Exp,    exp   )
JEM_DEFINE_UNARY_FUNC( Floor,  floor )
JEM_DEFINE_UNARY_FUNC( Log,    log   )
JEM_DEFINE_UNARY_FUNC( Log10,  log10 )
JEM_DEFINE_UNARY_FUNC( Sin,    sin   )
JEM_DEFINE_UNARY_FUNC( Sinh,   sinh  )
JEM_DEFINE_UNARY_FUNC( Sqrt,   sqrt  )
JEM_DEFINE_UNARY_FUNC( Tan,    tan   )
JEM_DEFINE_UNARY_FUNC( Tanh,   tanh  )

#undef JEM_DEFINE_UNARY_FUNC


//-----------------------------------------------------------------------
//   binary function classes
//-----------------------------------------------------------------------


template <class T, class U>

  inline typename BinaryFunc<T,U>::ResultType

    pow ( const T& x, const U& y )

{
  return (typename BinaryFunc<T,U>::ResultType) std::pow ( x, y );
}

template <class T, class U = T>

  class Pow : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return pow ( x, y );
  }

};


JEM_END_NAMESPACE( intrinsics )

JEM_END_PACKAGE_BASE

#endif
