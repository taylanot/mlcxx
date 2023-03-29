
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

#ifndef JEM_BASE_TUPLE_UTILITIES_TCC
#define JEM_BASE_TUPLE_UTILITIES_TCC


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   internal functions
//=======================================================================


JEM_BEGIN_NAMESPACE( tuple )


//-----------------------------------------------------------------------
//   reduce
//-----------------------------------------------------------------------


template <class T, int M, int N, int P, class E, class R>

  typename R::ResultType  reduce

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<P>&   sz,
    R                       r )

{
  r.init ( t[0] );

  for ( int i = 1; i < P; i++ )
  {
    r.feed ( t[i] );
  }

  return r.result ();
}


template <class T, int M, int N, class E, class R>

  inline
  typename R::ResultType  reduce

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<0>&   sz,
    R                       r )

{
  return typename R::ResultType ();
}


template <class T, int M, int N, class E, class R>

  inline
  typename R::ResultType  reduce

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<1>&   sz,
    R                       r )

{
  r.init ( t[0] );

  return r.result ();
}


template <class T, int M, int N, class E, class R>

  inline
  typename R::ResultType  reduce

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<2>&   sz,
    R                       r )

{
  r.init ( t[0] );
  r.feed ( t[1] );

  return r.result ();
}


template <class T, int M, int N, class E, class R>

  inline
  typename R::ResultType  reduce

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<3>&   sz,
    R                       r )

{
  r.init ( t[0] );
  r.feed ( t[1] );
  r.feed ( t[2] );

  return r.result ();
}


//-----------------------------------------------------------------------
//   reduce2
//-----------------------------------------------------------------------


template <class T, int M, int N, int P, class E, class Op>

  inline
  typename Op::ResultType reduce2

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<P>&   sz,
    const Op&               op )

{
  typename Op::ResultType  x = t[0];

  for ( int i = 1; i < P; i++ )
  {
    x = op( x, t[i] );
  }

  return x;
}


template <class T, int M, int N, class E, class Op>

  inline
  typename Op::ResultType reduce2

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<0>&   sz,
    const Op&               op )

{
  return typename Op::ResultType ();
}


template <class T, int M, int N, class E, class Op>

  inline
  typename Op::ResultType reduce2

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<1>&   sz,
    const Op&               op )

{
  return t[0];
}


template <class T, int M, int N, class E, class Op>

  inline
  typename Op::ResultType reduce2

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<2>&   sz,
    const Op&               op )

{
  return op ( t[0], t[1] );
}


template <class T, int M, int N, class E, class Op>

  inline
  typename Op::ResultType reduce2

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<3>&   sz,
    const Op&               op )

{
  typename Op::ResultType  x = op ( t[0], t[1] );

  return op ( x, t[2] );
}


//-----------------------------------------------------------------------
//   dot1
//-----------------------------------------------------------------------


template <class T, int M, int N, int P, class E>

  inline typename Multiplies<T>
    ::ResultType          dot1

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<P>&   sz )

{
  typename Multiplies<T>::ResultType  x = t[0] * t[0];

  for ( int i = 1; i < P; i++ )
  {
    x += t[i] * t[i];
  }

  return x;
}


template <class T, int M, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          dot1

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<0>&   sz )

{
  return typename Multiplies<T>::ResultType ();
}


template <class T, int M, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          dot1

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<1>&   sz )

{
  return (t[0] * t[0]);
}


template <class T, int M, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          dot1

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<2>&   sz )

{
  return (t[0] * t[0] + t[1] * t[1]);
}


template <class T, int M, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          dot1

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<3>&   sz )

{
  return (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
}


//-----------------------------------------------------------------------
//   dot2
//-----------------------------------------------------------------------


template <class T, class U,
          int M, int N, int P, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot2

  ( const Tuple<T,M,N,E1>&  t,
    const Tuple<U,M,N,E2>&  u,
    const IntConstant<P>&   sz )

{
  typename Multiplies<T,U>::ResultType  x = t[0] * u[0];

  for ( int i = 1; i < P; i++ )
  {
    x += t[i] * u[i];
  }

  return x;
}


template <class T, class U, int M, int N, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot2

  ( const Tuple<T,M,N,E1>&  t,
    const Tuple<U,M,N,E2>&  u,
    const IntConstant<0>&   sz )

{
  return typename Multiplies<T,U>::ResultType ();
}


template <class T, class U, int M, int N, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot2

  ( const Tuple<T,M,N,E1>&  t,
    const Tuple<U,M,N,E2>&  u,
    const IntConstant<1>&   sz )

{
  return (t[0] * u[0]);
}


template <class T, class U, int M, int N, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot2

  ( const Tuple<T,M,N,E1>&  t,
    const Tuple<U,M,N,E2>&  u,
    const IntConstant<2>&   sz )

{
  return (t[0] * u[0] + t[1] * u[1]);
}


template <class T, class U, int M, int N, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot2

  ( const Tuple<T,M,N,E1>&  t,
    const Tuple<U,M,N,E2>&  u,
    const IntConstant<3>&   sz )

{
  return (t[0] * u[0] + t[1] * u[1] + t[2] * u[2]);
}


//-----------------------------------------------------------------------
//   testany
//-----------------------------------------------------------------------


template <class T, int M, int N, int P, class E>

  inline bool             testany

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<P>&   sz,
    bool                    v )

{
  for ( int i = 0; i < P; i++ )
  {
    if ( ((bool) t[i]) == v )
    {
      return true;
    }
  }

  return false;
}


template <class T, int M, int N, class E>

  inline bool             testany

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<0>&   sz,
    bool                    v )

{
  return false;
}


template <class T, int M, int N, class E>

  inline bool             testany

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<1>&   sz,
    bool                    v )

{
  return ( ((bool) t[0]) == v );
}


template <class T, int M, int N, class E>

  inline bool             testany

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<2>&   sz,
    bool                    v )

{
  return ( (((bool) t[0]) == v) ||
           (((bool) t[1]) == v) );
}


template <class T, int M, int N, class E>

  inline bool             testany

  ( const Tuple<T,M,N,E>&   t,
    const IntConstant<3>&   sz,
    bool                    v )

{
  return ( (((bool) t[0]) == v) ||
           (((bool) t[1]) == v) ||
           (((bool) t[2]) == v) );
}


//-----------------------------------------------------------------------
//   minmaxloc
//-----------------------------------------------------------------------


template <class T, int M, class E, class Comp>

  static int              minmaxloc

  ( const Tuple<T,M,1,E>&   t,
    Comp                    comp )

{
  int  j = 0;

  for ( int i = 1; i < M; i++ )
  {
    if ( comp( t[i], t[j] ) )
    {
      j = i;
    }
  }

  return j;
}


template <class T, class E, class Comp>

  inline int              minmaxloc

  ( const Tuple<T,0,1,E>&   t,
    Comp                    comp )

{
  return 0;
}


template <class T, class E, class Comp>

  inline int              minmaxloc

  ( const Tuple<T,1,1,E>&   t,
    Comp                    comp )

{
  return 0;
}


template <class T, class E, class Comp>

  inline int              minmaxloc

  ( const Tuple<T,2,1,E>&   t,
    Comp                    comp )

{
  if ( comp( t[0], t[1] ) )
  {
    return 0;
  }
  else
  {
    return 1;
  }
}


template <class T, class E, class Comp>

  inline int              minmaxloc

  ( const Tuple<T,3,1,E>&   t,
    Comp                    comp )

{
  int  i;

  if ( comp( t[0], t[1] ) )
  {
    i = 0;
  }
  else
  {
    i = 1;
  }

  if ( comp( t[2], t[i] ) )
  {
    i = 2;
  }

  return i;
}


JEM_END_NAMESPACE( tuple )


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   reduce
//-----------------------------------------------------------------------


template <class T, int M, int N, class E, class R>

  inline
  typename R::ResultType  reduce

  ( const Tuple<T,M,N,E>&   t,
    R                       r )

{
  return tuple::reduce ( t, sizeOf(t), r );
}


//-----------------------------------------------------------------------
//   dot
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          dot

  ( const Tuple<T,M,N,E>&   t )

{
  return tuple::dot1 ( t, sizeOf(t) );
}


template <class T, class U, int M, int N, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot

  ( const Tuple<T,M,N,E1>&  t,
    const Tuple<U,M,N,E2>&  u )

{
  return tuple::dot2 ( t, u, sizeOf(t) );
}


//-----------------------------------------------------------------------
//   max
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline typename Maximum<T>
    ::ResultType          max

  ( const Tuple<T,M,N,E>&   t )

{
  return tuple::reduce2 ( t, sizeOf(t), Maximum<T>() );
}


//-----------------------------------------------------------------------
//   min
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline typename Minimum<T>
    ::ResultType          min

  ( const Tuple<T,M,N,E>&   t )

{
  return tuple::reduce2 ( t, sizeOf(t), Minimum<T>() );
}


//-----------------------------------------------------------------------
//   sum
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline typename Plus<T>
    ::ResultType          sum

  ( const Tuple<T,M,N,E>&   t )

{
  return tuple::reduce2 ( t, sizeOf(t), Plus<T>() );
}


//-----------------------------------------------------------------------
//   ksum
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline typename KahanSum<T>
    ::ResultType          ksum

  ( const Tuple<T,M,N,E>&   t )

{
  return tuple::reduce ( t, sizeOf(t), KahanSum<T>() );
}


//-----------------------------------------------------------------------
//   product
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          product

  ( const Tuple<T,M,N,E>&   t )

{
  return tuple::reduce2 ( t, sizeOf(t), Multiplies<T>() );
}


//-----------------------------------------------------------------------
//   testall
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline bool testall ( const Tuple<T,M,N,E>& t )

{
  return ! tuple::testany ( t, sizeOf(t), false );
}


//-----------------------------------------------------------------------
//   testany
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline bool testany ( const Tuple<T,M,N,E>& t )

{
  return tuple::testany ( t, sizeOf(t), true );
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T, int M, int N, class E>

  inline int count ( const Tuple<T,M,N,E>& t )

{
  return (int) tuple::reduce ( t, sizeOf(t), TrueCount<T>() );
}


template <class T, class U, int M, int N, class E>

  inline int count ( const Tuple<T,M,N,E>& t, const U& v )

{
  return (int) tuple::reduce ( t, sizeOf(t), EqualCount<T,U>( v ) );
}


//-----------------------------------------------------------------------
//   minloc & maxloc
//-----------------------------------------------------------------------


template <class T, int M, class E>

  inline int minloc ( const Tuple<T,M,1,E>& t )

{
  return tuple::minmaxloc ( t, LessThan<T>() );
}


template <class T, int M, class E>

  inline int maxloc ( const Tuple<T,M,1,E>& t )

{
  return tuple::minmaxloc ( t, GreaterThan<T>() );
}


//-----------------------------------------------------------------------
//   makeTuple
//-----------------------------------------------------------------------


template <class T>

  inline Tuple<T,1> makeTuple ( T a )

{
  return Tuple<T,1> ( a );
}


template <class T>

  inline Tuple<T,2> makeTuple ( T a, T b )

{
  return Tuple<T,2> ( a, b );
}


template <class T>

  inline Tuple<T,3> makeTuple ( T a, T b, T c )

{
  return Tuple<T,3> ( a, b, c );
}


template <class T>

  inline Tuple<T,4> makeTuple ( T a, T b, T c, T d )

{
  return Tuple<T,4> ( a, b, c, d );
}


template <class T>

  inline Tuple<T,5> makeTuple ( T a, T b, T c, T d, T e )

{
  return Tuple<T,5> ( a, b, c, d, e );
}


template <class T>

  inline Tuple<T,6> makeTuple ( T a, T b, T c, T d, T e, T f )

{
  return Tuple<T,6> ( a, b, c, d, e, f );
}


JEM_END_PACKAGE_BASE

#endif
