
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

#ifndef JEM_BASE_ARRAY_UTILITIES_TCC
#define JEM_BASE_ARRAY_UTILITIES_TCC

#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   private functions
//=======================================================================

JEM_BEGIN_NAMESPACE( array )

//-----------------------------------------------------------------------
//   reshapable
//-----------------------------------------------------------------------


template <class T, int N>

  inline bool           reshapable

  ( const Array<T,N>&     a )

{
  idx_t  st = a.stride (0);

  for ( int i = 1; i < N; i++ )
  {
    st *= a.size ( i - 1 );

    if ( st != a.stride(i) )
    {
      return false;
    }
  }

  return true;
}


template <class T>

  inline bool           reshapable

  ( const Array<T,1>&     a )

{
  return true;
}


template <class T>

  inline bool           reshapable

  ( const Array<T,2>&     a )

{
  return ( a.stride(1) == (a.stride(0) * a.size(0)) );
}


template <class T>

  inline bool           reshapable

  ( const Array<T,3>&     a )

{
  return ( a.stride(1) == (a.stride(0) * a.size(0)) &&
           a.stride(2) == (a.stride(1) * a.size(1)) );
}


template <class T>

  inline bool           reshapable

  ( const Array<T,4>&     a )

{
  return ( a.stride(1) == (a.stride(0) * a.size(0)) &&
           a.stride(2) == (a.stride(1) * a.size(1)) &&
           a.stride(3) == (a.stride(2) * a.size(2)) );
}


//-----------------------------------------------------------------------
//   reduceFast
//-----------------------------------------------------------------------


template <class T, int N, class E, class R>

  typename R::ResultType  reduceFast

  ( const Array<T,N,E>&     e,
    R                       r )

{
  const idx_t  n = size ( e.shape() );

  if ( n )
  {
    r.init ( e.getFast(0_idx) );

    for ( idx_t i = 1; i < n; i++ )
    {
      r.feed ( e.getFast(i) );
    }
  }

  return r.result ();
}


//-----------------------------------------------------------------------
//   reduceFast2
//-----------------------------------------------------------------------


template <class T, int N, class E, class Op>

  typename
  Op::ResultType          reduceFast2

  ( const Array<T,N,E>&     e,
    const Op&               op )

{
  const idx_t  n = size ( e.shape() );

  if ( ! n )
  {
    return typename Op::ResultType ();
  }
  else
  {
    typename Op::ResultType  t = e.getFast(0);

    for ( idx_t i = 1_idx; i < n; i++ )
    {
      t = op ( t, e.getFast(i) );
    }

    return t;
  }
}


template <class T, class Op>

  typename
  Op::ResultType          reduceFast2

  ( const T*                e,
    idx_t                   n,
    const Op&               op )

{
  if ( ! n )
  {
    return typename Op::ResultType ();
  }
  else
  {
    typename Op::ResultType  t = e[0];

    for ( idx_t i = 1; i < n; i++ )
    {
      t = op ( t, e[i] );
    }

    return t;
  }
}


template <class T, int N, class Op>

  typename
  Op::ResultType          reduceFast2

  ( const Array<T,N>&       e,
    const Op&               op )

{
  return reduceFast2 ( e.addr(), size( e.shape() ), op );
}


//-----------------------------------------------------------------------
//   reduceSlow
//-----------------------------------------------------------------------


template <class T, int N, class E, class R>

  typename R::ResultType  reduceSlow

  ( const Array<T,N,E>&     e,
    R                       r )

{
  const Tuple<idx_t,N>  sh = e.shape ();

  if ( size( sh ) > 0_idx )
  {
    const idx_t     n = sh[0];
    Tuple<idx_t,N>  i;

    i = 0_idx;

    r.init ( e[i] );

    i[0] = 1_idx;

    do
    {
      for ( ; i[0] < n; i[0]++ )
      {
        r.feed ( e[i] );
      }
    }
    while ( incrIndex( i, sh ) );
  }

  return r.result ();
}


template <class T, class E, class R>

  typename R::ResultType  reduceSlow

  ( const Array<T,1,E>&     e,
    R                       r )

{
  const idx_t  n = e.size ();

  if ( n > 0_idx )
  {
    r.init ( e[0] );

    for ( idx_t i = 1; i < n; i++ )
    {
      r.feed ( e[i] );
    }
  }

  return r.result ();
}

// Some special cases for better performance and smaller executables
// (no dependency on array rank).

template <class T, class R>

  void                    reduceSlow

  ( const T*                e,
    const idx_t*            st,
    const idx_t*            sh,
    idx_t*                  idx,
    int                     rank,
    R&                      r )

{
  JEM_ASSERT ( rank > 2 );

  const idx_t  st0 = st[0];

  const int    k   = rank - 1;
  const idx_t  m   = sh[0];
  const idx_t  n   = sh[1];

  idx_t        i   = 1;

  r.init ( e[0_idx] );

  do
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( ; i < m; i++ )
      {
        r.feed ( e[i * st0] );
      }

      e += st[1];
      i  = 0_idx;
    }

    for ( int j = 1; j < k; j++ )
    {
      e -= (sh[j] * st[j]);
      e += st[j + 1];

      idx[j] = 0_idx;
      idx[j + 1]++;

      if ( idx[j + 1] < sh[j + 1] )
      {
        break;
      }
    }
  }
  while ( idx[k] < sh[k] );
}


template <class T, int N, class R>

  inline
  typename R::ResultType  reduceSlow

  ( const Array<T,N,Nil>&   e,
    R                       r )

{
  if ( size( e.shape() ) > 0_idx )
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    reduceSlow ( e.addr(), e.strides().addr(),
                 e.shape().addr(), i.addr(), N, r );
  }

  return r.result ();
}


template <class T, class R>

  typename R::ResultType  reduceSlow

  ( const Array<T,1,Nil>&   e,
    R                       r )

{
  const idx_t  n = e.size ();

  if ( n > 0_idx )
  {
    const T*     ptr = e.addr   ();
    const idx_t  st  = e.stride ();

    r.init ( ptr[0] );

    for ( idx_t i = 1; i < n; i++ )
    {
      r.feed ( ptr[i * st] );
    }
  }

  return r.result ();
}


template <class T, class R>

  typename R::ResultType  reduceSlow

  ( const Array<T,2>&       e,
    R                       r )

{
  const idx_t  m = e.size (0);
  const idx_t  n = e.size (1);

  if ( (m * n) > 0_idx )
  {
    const T*     ptr = e.addr   ();
    const idx_t  st0 = e.stride (0);

    idx_t        i   = 1;

    r.init ( ptr[0] );

    for ( idx_t j = 0; j < n; j++ )
    {
      for ( ; i < m; i++ )
      {
        r.feed ( ptr[i * st0] );
      }

      ptr += e.stride (1);
      i    = 0_idx;
    }
  }

  return r.result ();
}


//-----------------------------------------------------------------------
//   reduce2
//-----------------------------------------------------------------------


template <class T, int N, class E, class Op>

  inline typename
  Op::ResultType          reduce2

  ( const Array<T,N,E>&     e,
    const Op&               op )

{
  if ( e.isContiguous() )
  {
    return reduceFast2 ( e, op );
  }
  else
  {
    return reduceSlow  ( e, BinaryReduction<Op>( op ) );
  }
}


//-----------------------------------------------------------------------
//   dot1Fast
//-----------------------------------------------------------------------


template <class T, int N, class E>

  typename Multiplies<T>
    ::ResultType          dot1Fast

  ( const Array<T,N,E>&     e )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  const idx_t  n = size ( e.shape() );

  if ( ! n )
  {
    return ResultType ();
  }
  else
  {
    ResultType  x = e.getFast (0);
    ResultType  t = x * x;

    for ( idx_t i = 1; i < n; i++ )
    {
      x  = e.getFast (i);
      t += x * x;
    }

    return t;
  }
}


template <class T>

  typename Multiplies<T>
    ::ResultType          dot1Fast

  ( const T*                e,
    idx_t                   n )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  if ( ! n )
  {
    return ResultType ();
  }
  else
  {
    ResultType  t = e[0] * e[0];

    for ( idx_t i = 1; i < n; i++ )
    {
      t += e[i] * e[i];
    }

    return t;
  }
}


double                    dot1Fast

  ( const double*           e,
    idx_t                    n );


template <class T, int N>

  inline typename Multiplies<T>
    ::ResultType          dot1Fast

  ( const Array<T,N>&       e )

{
  return dot1Fast ( e.addr(), size( e.shape() ) );
}


//-----------------------------------------------------------------------
//   dot2Fast
//-----------------------------------------------------------------------


template <class T, class U, int N, class E1, class E2>

  typename Multiplies<T,U>
    ::ResultType          dot2Fast

  ( const Array<T,N,E1>&    e,
    const Array<U,N,E2>&    f )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  const idx_t  n = size ( e.shape() );

  if ( ! n )
  {
    return ResultType ();
  }
  else
  {
    ResultType  t = e.getFast(0) * f.getFast(0);

    for ( idx_t i = 1; i < n; i++ )
    {
      t += e.getFast(i) * f.getFast(i);
    }

    return t;
  }
}


template <class T, class U>

  typename Multiplies<T,U>
    ::ResultType          dot2Fast

  ( const T*                e,
    const U*                f,
    idx_t                   n )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  if ( ! n )
  {
    return ResultType ();
  }
  else
  {
    ResultType  t = e[0] * f[0];

    for ( idx_t i = 1; i < n; i++ )
    {
      t += e[i] * f[i];
    }

    return t;
  }
}


double                    dot2Fast

  ( const double*           e,
    const double*           f,
    idx_t                   n );


template <class T, class U, int N>

  inline typename Multiplies<T,U>
    ::ResultType          dot2Fast

  ( const Array<T,N>&       e,
    const Array<U,N>&       f )

{
  return dot2Fast ( e.addr(), f.addr(), size( e.shape() ) );
}


//-----------------------------------------------------------------------
//   dot1Slow
//-----------------------------------------------------------------------


template <class T, int N, class E>

  typename Multiplies<T>
    ::ResultType          dot1Slow

  ( const Array<T,N,E>&     e )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  const Tuple<idx_t,N>  sh = e.shape ();

  if ( ! size( sh ) )
  {
    return ResultType ();
  }
  else
  {
    const idx_t     n = sh[0];
    Tuple<idx_t,N>  i;

    i = 0_idx;

    ResultType     x = e[i];
    ResultType     t = x * x;

    i[0] = 1_idx;

    do
    {
      for ( ; i[0] < n; i[0]++ )
      {
        x  = e[i];
        t += x * x;
      }
    }
    while ( incrIndex( i, sh ) );

    return t;
  }
}


template <class T, class E>

  typename Multiplies<T>
    ::ResultType          dot1Slow

  ( const Array<T,1,E>&     e )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  const idx_t  n = e.size ();

  if ( ! n )
  {
    return ResultType ();
  }
  else
  {
    ResultType  x = e[0];
    ResultType  t = x * x;

    for ( idx_t i = 1; i < n; i++ )
    {
      x  = e[i];
      t += x * x;
    }

    return t;
  }
}

// Some special cases for better performance and smaller executables
// (no dependency on array rank).

template <class T>

  typename Multiplies<T>
    ::ResultType          dot1Slow

  ( const T*                e,
    const idx_t*            st,
    const idx_t*            sh,
    idx_t*                  idx,
    int                     rank )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  JEM_ASSERT ( rank > 2 );

  const idx_t  st0 = st[0];

  const int    k   = rank - 1;
  const idx_t  m   = sh[0];
  const idx_t  n   = sh[1];

  idx_t        i   = 1_idx;

  ResultType   x   = e[0];
  ResultType   t   = x * x;

  do
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( ; i < m; i++ )
      {
        x  = e[i * st0];
        t += x * x;
      }

      e += st[1];
      i  = 0_idx;
    }

    for ( int j = 1; j < k; j++ )
    {
      e -= (sh[j] * st[j]);
      e += st[j + 1];

      idx[j] = 0_idx;
      idx[j + 1]++;

      if ( idx[j + 1] < sh[j + 1] )
      {
        break;
      }
    }
  }
  while ( idx[k] < sh[k] );

  return t;
}


template <class T, int N>

  inline typename Multiplies<T>
    ::ResultType          dot1Slow

  ( const Array<T,N,Nil>&   e )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  if ( ! size( e.shape() ) )
  {
    return ResultType ();
  }
  else
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    return dot1Slow ( e.addr(), e.strides().addr(),
                      e.shape().addr(), i.addr(), N );
  }
}


template <class T>

  typename Multiplies<T>
    ::ResultType          dot1Slow

  ( const Array<T,1,Nil>&   e )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  const idx_t  n = e.size ();

  if ( ! n )
  {
    return ResultType ();
  }
  else
  {
    const T*     ptr = e.addr   ();
    const idx_t  st  = e.stride ();

    ResultType   x   = ptr[0];
    ResultType   t   = x * x;

    for ( idx_t i = 1; i < n; i++ )
    {
      x  = ptr[i * st];
      t += x * x;
    }

    return t;
  }
}


template <class T>

  typename Multiplies<T>
    ::ResultType          dot1Slow

  ( const Array<T,2>&       e )

{
  typedef typename Multiplies<T>::ResultType  ResultType;

  const idx_t  m = e.size (0);
  const idx_t  n = e.size (1);

  if ( (m * n) == 0_idx )
  {
    return ResultType ();
  }
  else
  {
    const T*     ptr = e.addr   ();
    const idx_t  st0 = e.stride (0);

    idx_t        i   = 1_idx;

    ResultType   x   = ptr[0];
    ResultType   t   = x * x;

    for ( idx_t j = 0; j < n; j++ )
    {
      for ( ; i < m; i++ )
      {
        x  = ptr[i * st0];
        t += x * x;
      }

      ptr += e.stride (1);
      i    = 0_idx;
    }

    return t;
  }
}


//-----------------------------------------------------------------------
//   dot2Slow
//-----------------------------------------------------------------------


template <class T, class U, int N, class E1, class E2>

  typename Multiplies<T,U>
    ::ResultType        dot2Slow

  ( const Array<T,N,E1>&  e,
    const Array<U,N,E2>&  f )

{
  typedef typename Multiplies<T,U>::ResultType  ResultType;

  const Tuple<idx_t,N>  sh = e.shape ();

  if ( ! size( sh ) )
  {
    return ResultType ();
  }
  else
  {
    const idx_t     n = sh[0];
    Tuple<idx_t,N>  i;

    i = 0_idx;

    ResultType      t = e[i] * f[i];

    i[0] = 1_idx;

    do
    {
      for ( ; i[0] < n; i[0]++ )
      {
        t += e[i] * f[i];
      }
    }
    while ( incrIndex( i, sh ) );

    return t;
  }
}


template <class T, class U, class E1, class E2>

  typename Multiplies<T,U>
    ::ResultType          dot2Slow

  ( const Array<T,1,E1>&    e,
    const Array<U,1,E2>&    f )

{
  typedef typename Multiplies<T,U>::ResultType  ResultType;

  const idx_t  n = e.size ();

  if ( ! n )
  {
    return ResultType ();
  }
  else
  {
    ResultType  t = e[0] * f[0];

    for ( idx_t i = 1; i < n; i++ )
    {
      t += e[i] * f[i];
    }

    return t;
  }
}

// Some special cases for better performance and smaller executables
// (no dependency on array rank).

template <class T, class U>

  typename Multiplies<T,U>
    ::ResultType          dot2Slow

  ( const T*                e,
    const idx_t*            est,
    const U*                f,
    const idx_t*            fst,
    const idx_t*            sh,
    idx_t*                  idx,
    idx_t                   rank )

{
  typedef typename Multiplies<T,U>::ResultType  ResultType;

  JEM_ASSERT ( rank > 2 );

  const idx_t  est0 = est[0];
  const idx_t  fst0 = fst[0];

  const idx_t  k    = rank - 1;
  const idx_t  m    = sh[0];
  const idx_t  n    = sh[1];

  idx_t        i    = 1_idx;

  ResultType   t    = e[0] * f[0];

  do
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( ; i < m; i++ )
      {
        t += e[i * est0] * f[i * fst0];
      }

      e += est[1];
      f += fst[1];
      i  = 0_idx;
    }

    for ( idx_t j = 1; j < k; j++ )
    {
      e -= (sh[j] * est[j]);
      f -= (sh[j] * fst[j]);

      e += est[j + 1];
      f += fst[j + 1];

      idx[j] = 0_idx;
      idx[j + 1]++;

      if ( idx[j + 1] < sh[j + 1] )
      {
        break;
      }
    }
  }
  while ( idx[k] < sh[k] );

  return t;
}


template <class T, class U, int N>

  inline typename Multiplies<T,U>
    ::ResultType          dot2Slow

  ( const Array<T,N,Nil>&   e,
    const Array<U,N,Nil>&   f )

{
  typedef typename Multiplies<T,U>::ResultType  ResultType;

  if ( ! size( e.shape() ) )
  {
    return ResultType ();
  }
  else
  {
    Tuple<idx_t,N>  i;

    i = 0_idx;

    return dot2Slow ( e.addr(), e.strides().addr(),
                      f.addr(), f.strides().addr(),
                      e.shape().addr(), i.addr(), N );
  }
}


template <class T, class U>

  typename Multiplies<T,U>
    ::ResultType          dot2Slow

  ( const Array<T,1,Nil>&   e,
    const Array<U,1,Nil>&   f )

{
  typedef typename Multiplies<T,U>::ResultType  ResultType;

  const idx_t  n = e.size ();

  if ( ! n )
  {
    return ResultType ();
  }
  else
  {
    const T*     ept = e.addr   ();
    const U*     fpt = f.addr   ();

    const idx_t  est = e.stride ();
    const idx_t  fst = f.stride ();

    ResultType   t   = ept[0] * fpt[0];

    for ( idx_t i = 1; i < n; i++ )
    {
      t += ept[i * est] * fpt[i * fst];
    }

    return t;
  }
}


template <class T, class U>

  typename Multiplies<T,U>
    ::ResultType          dot2Slow

  ( const Array<T,2>&       e,
    const Array<U,2>&       f )

{
  typedef typename Multiplies<T,U>::ResultType  ResultType;

  const idx_t  m = e.size (0);
  const idx_t  n = e.size (1);

  if ( (m * n) == 0_idx )
  {
    return ResultType ();
  }
  else
  {
    const T*     ept  = e.addr   ();
    const U*     fpt  = f.addr   ();

    const idx_t  est0 = e.stride (0);
    const idx_t  fst0 = f.stride (0);

    idx_t        i    = 1;

    ResultType   t    = ept[0] * fpt[0];

    for ( idx_t j = 0; j < n; j++ )
    {
      for ( ; i < m; i++ )
      {
        t += ept[i * est0] * fpt[i * fst0];
      }

      ept += e.stride (1);
      fpt += f.stride (1);
      i    = 0_idx;
    }

    return t;
  }
}


//-----------------------------------------------------------------------
//   testanyFast
//-----------------------------------------------------------------------


template <class T, int N, class E>

  bool                  testanyFast

  ( const Array<T,N,E>&   e,
    bool                  v )

{
  const idx_t  n = size ( e.shape() );

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( ((bool) e.getFast(i)) == v )
    {
      return true;
    }
  }

  return false;
}


template <class T>

  bool                  testanyFast

  ( const T*              e,
    idx_t                 n,
    bool                  v )

{
  for ( idx_t i = 0; i < n; i++ )
  {
    if ( ((bool) e[i]) == v )
    {
      return true;
    }
  }

  return false;
}


template <class T, int N>

  inline bool           testanyFast

  ( const Array<T,N>&     e,
    bool                  v )

{
  return testanyFast ( e.addr(), size( e.shape() ), v );
}


//-----------------------------------------------------------------------
//   testanySlow
//-----------------------------------------------------------------------


template <class T, int N, class E>

  bool                  testanySlow

  ( const Array<T,N,E>&   e,
    bool                  v )

{
  const Tuple<idx_t,N>  sh = e.shape ();

  if ( size( sh ) > 0_idx )
  {
    const idx_t     n = sh[0];
    Tuple<idx_t,N>  i;

    i = 0_idx;

    do
    {
      for ( ; i[0] < n; i[0]++ )
      {
        if ( ((bool) e[i]) == v )
        {
          return true;
        }
      }
    }
    while ( incrIndex( i, sh ) );
  }

  return false;
}


template <class T, class E>

  bool                  testanySlow

  ( const Array<T,1,E>&   e,
    bool                  v )

{
  const idx_t  n = e.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( ((bool) e[i]) == v )
    {
      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   testany
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline bool           testany

  ( const Array<T,N,E>&   e,
    bool                  v )

{
  if ( e.isContiguous() )
  {
    return testanyFast ( e, v );
  }
  else
  {
    return testanySlow ( e, v );
  }
}


//-----------------------------------------------------------------------
//   minmaxloc
//-----------------------------------------------------------------------


template <class T, class E, class Comp>

  idx_t                 minmaxloc

  ( const Array<T,1,E>&   e,
    Comp                  comp )

{
  const idx_t  n = e.size ();

  idx_t        i, j;


  if      ( ! n )
  {
    j = -1_idx;
  }
  else if ( e.isContiguous() )
  {
    T  t = e.getFast(0);

    for ( i = 1, j = 0; i < n; i++ )
    {
      if ( comp( e.getFast(i), t ) )
      {
        t = e.getFast(i);
        j = i;
      }
    }
  }
  else
  {
    T  t = e[0];

    for ( i = 1, j = 0; i < n; i++ )
    {
      if ( comp( e[i], t ) )
      {
        t = e[i];
        j = i;
      }
    }
  }

  return j;
}


//-----------------------------------------------------------------------
//   crshift
//-----------------------------------------------------------------------


template <class T, class E>

  void                  crshift

  ( const Array<T,1,E>&   e,
    idx_t                 k )

{
  const idx_t  n = e.size ();


  k = k % n;

  if ( k == 1_idx )
  {
    T  tmp ( e[n - 1] );

    shift ( e, k );

    e[0] = tmp;
  }
  else
  {
    Array<T>  tmp ( k );

    idx_t      i, j;

    for ( i = 0, j = n - k; i < k; i++, j++ )
    {
      tmp[i] = e[j];
    }

    shift ( e, k );

    for ( i = 0; i < k; i++ )
    {
      e[i] = tmp[i];
    }
  }
}


//-----------------------------------------------------------------------
//   clshift
//-----------------------------------------------------------------------


template <class T, class E>

  void                  clshift

  ( const Array<T,1,E>&   e,
    idx_t                 k )

{
  const idx_t  n = e.size ();


  k = k % n;

  if ( k == 1_idx )
  {
    T  tmp ( e[0] );

    shift ( e, -k );

    e[n - 1] = tmp;
  }
  else
  {
    Array<T>  tmp ( k );

    idx_t      i, j;

    for ( i = 0; i < k; i++ )
    {
      tmp[i] = e[i];
    }

    shift ( e, -k );

    for ( i = 0, j = n - k; i < k; i++, j++ )
    {
      e[j] = tmp[i];
    }
  }
}

JEM_END_NAMESPACE( array )

//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   reduce
//-----------------------------------------------------------------------


template <class T, int N, class E, class R>

  inline
  typename R::ResultType  reduce

  ( const Array<T,N,E>&     e,
    R                       r )

{
  if ( e.isContiguous() )
  {
    return array::reduceFast ( e, r );
  }
  else
  {
    return array::reduceSlow ( e, r );
  }
}


template <class T, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          dot

  ( const Array<T,N,E>&     e )

{
  if ( e.isContiguous() )
  {
    return array::dot1Fast ( e );
  }
  else
  {
    return array::dot1Slow ( e );
  }
}


template <class T, class U, int N, class E1, class E2>

  inline typename Multiplies<T,U>
    ::ResultType          dot

  ( const Array<T,N,E1>&    e,
    const Array<U,N,E2>&    f )

{
  JEM_ASSERT2 ( equal( e.shape(), f.shape() ),
                array::Errors::SHAPE_MISMATCH );

  if ( e.isContiguous() && f.isContiguous() )
  {
    return array::dot2Fast ( e, f );
  }
  else
  {
    return array::dot2Slow ( e, f );
  }
}


//-----------------------------------------------------------------------
//   max
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline typename Maximum<T>
    ::ResultType          max

  ( const Array<T,N,E>&     e )

{
  return array::reduce2 ( e, Maximum<T>() );
}


//-----------------------------------------------------------------------
//   min
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline typename Minimum<T>
    ::ResultType          min

  ( const Array<T,N,E>&     e )

{
  return array::reduce2 ( e, Minimum<T>() );
}


//-----------------------------------------------------------------------
//   sum
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline typename Plus<T>
    ::ResultType          sum

  ( const Array<T,N,E>&     e )

{
  return array::reduce2 ( e, Plus<T>() );
}


//-----------------------------------------------------------------------
//   ksum
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline typename KahanSum<T>
    ::ResultType          ksum

  ( const Array<T,N,E>&     e )

{
  return reduce ( e, KahanSum<T>() );
}


//-----------------------------------------------------------------------
//   product
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline typename Multiplies<T>
    ::ResultType          product

  ( const Array<T,N,E>&     e )

{
  return array::reduce2 ( e, Multiplies<T>() );
}


//-----------------------------------------------------------------------
//   testall
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline bool             testall

  ( const Array<T,N,E>&     e )

{
  return ! array::testany ( e, false );
}


//-----------------------------------------------------------------------
//   testany
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline bool             testany

  ( const Array<T,N,E>&     e )

{
  return array::testany ( e, true );
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline idx_t            count

  ( const Array<T,N,E>&     e )

{
  return reduce ( e, TrueCount<T>() );
}


template <class T, class U, int N, class E>

  inline idx_t            count

  ( const Array<T,N,E>&     e,
    const U&                v )

{
  return reduce ( e, EqualCount<T,U>( v ) );
}


//-----------------------------------------------------------------------
//   minloc & maxloc
//-----------------------------------------------------------------------


template <class T, class E>

  inline idx_t            minloc

  ( const Array<T,1,E>&     e )

{
  return array::minmaxloc ( e, LessThan<T>() );
}


template <class T, class E>

  inline idx_t            maxloc

  ( const Array<T,1,E>&     e )

{
  return array::minmaxloc ( e, GreaterThan<T>() );
}


//-----------------------------------------------------------------------
//   shift
//-----------------------------------------------------------------------


template <class T, class E>

  void                    shift

  ( const Array<T,1,E>&     e,
    idx_t                   k )

{
  const idx_t  n = e.size ();

  idx_t        i;

  if ( k > 0_idx )
  {
    for ( i = n - 1; i >= k; i-- )
    {
      e[i] = e[i - k];
    }
  }
  else
  {
    for ( i = 0; i < n + k; i++ )
    {
      e[i] = e[i - k];
    }
  }
}


//-----------------------------------------------------------------------
//   cshift
//-----------------------------------------------------------------------


template <class T, class E>

  void                    cshift

  ( const Array<T,1,E>&     e,
    idx_t                   k )

{
  const idx_t  n = e.size();

  if ( ! n )
  {
    return;
  }

  if      ( k > 0_idx )
  {
    array::crshift ( e,  k );
  }
  else if ( k < 0_idx )
  {
    array::clshift ( e, -k );
  }
}


//-----------------------------------------------------------------------
//   reverse
//-----------------------------------------------------------------------


template <class T, class E>

  void                    reverse

  ( const Array<T,1,E>&     e )

{
  idx_t  i, j;


  j = e.size() - 1;

  if ( e.isContiguous() )
  {
    for ( i = 0; i < j; i++, j-- )
    {
      swap ( e.getFast(i), e.getFast(j) );
    }
  }
  else
  {
    for ( i = 0; i < j; i++, j-- )
    {
      swap ( e[i], e[j] );
    }
  }
}


//-----------------------------------------------------------------------
//   reshape
//-----------------------------------------------------------------------


template <class T, int M, int N>

  Array<T,M>              reshape

  ( const Array<T,N>&       a,
    const Tuple<idx_t,M>&   sh )

{
  JEM_PRECHECK2 ( array::reshapable( a ) &&
                  array::size( sh ) <= a.itemCount(),
                  array::Errors::RESHAPE_ERROR );

  Tuple<idx_t,M>  st;

  st[0] = a.stride (0);

  for ( int i = 1; i < M; i++ )
  {
    st[i] = st[i - 1] * sh[i - 1];
  }

  return Array<T,M> ( a.getDataBlock(), a.addr(), sh, st );
}


template <class T, int N>

  Array<T,1>              reshape

  ( const Array<T,N>&       a,
    idx_t                   m )

{
  JEM_PRECHECK2 ( array::reshapable( a ) && m <= a.itemCount(),
                  array::Errors::RESHAPE_ERROR );

  return Array<T,1> (
    a.getDataBlock (),
    a.addr         (),
    Tuple<idx_t,1>  ( m ),
    Tuple<idx_t,1>  ( a.stride(0) )
  );
}


template <class T, int N>

  inline Array<T,2>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n )

{
  return reshape ( a, Tuple<idx_t,2>( m, n ) );
}


template <class T, int N>

  inline Array<T,3>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p )

{
  return reshape ( a, Tuple<idx_t,3>( m, n, p ) );
}


template <class T, int N>

  inline Array<T,4>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q )

{
  return reshape ( a, Tuple<idx_t,4>( m, n, p, q ) );
}


template <class T, int N>

  inline Array<T,5>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r )

{
  return reshape ( a, Tuple<idx_t,5>( m, n, p, q, r ) );
}


template <class T, int N>

  inline Array<T,6>       reshape

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r,
    idx_t                   s )

{
  return reshape ( a, Tuple<idx_t,6>( m, n, p, q, r, s ) );
}


//-----------------------------------------------------------------------
//   flatten
//-----------------------------------------------------------------------


template <class T, int M, int N>

  Array<T,M>              flatten

  ( const Array<T,N>&       a,
    const Tuple<idx_t,M>&   map )

{
  static_assert ( M <= N, "M larger than N" );

  JEM_PRECHECK2 ( array::reshapable( a ),
                  array::Errors::RESHAPE_ERROR );

  Tuple<idx_t,M>  sh;
  Tuple<idx_t,M>  st;

  int             k = 0;

  for ( int i = 0; i < M; i++ )
  {
    sh[i] = 1_idx;

    for ( int j = 0; j < map[i]; j++, k++ )
    {
      sh[i] *= a.size ( k );
    }
  }

  st[0] = a.stride ( 0 );

  for ( int i = 1; i < M; i++ )
  {
    st[i] = st[i - 1] * sh[i - i];
  }

  return Array<T,M> ( a.getDataBlock(), a.addr(), sh, st );
}


template <class T, int N>

  Array<T,1>              flatten

  ( const Array<T,N>&       a )

{
  JEM_PRECHECK2 ( array::reshapable( a ),
                  array::Errors::RESHAPE_ERROR );

  return Array<T,1> (
    a.getDataBlock (),
    a.addr         (),
    a.itemCount    (),
    a.stride       (0)
  );
}


template <class T, int N>

  inline Array<T,2>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n )

{
  return flatten ( a, Tuple<idx_t,2>( m, n ) );
}


template <class T, int N>

  inline Array<T,3>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p )

{
  return flatten ( a, Tuple<idx_t,3>( m, n, p ) );
}


template <class T, int N>

  inline Array<T,4>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q )

{
  return flatten ( a, Tuple<idx_t,4>( m, n, p, q ) );
}


template <class T, int N>

  inline Array<T,5>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r )

{
  return flatten ( a, Tuple<idx_t,5>( m, n, p, q, r ) );
}


template <class T, int N>

  inline Array<T,6>       flatten

  ( const Array<T,N>&       a,
    idx_t                   m,
    idx_t                   n,
    idx_t                   p,
    idx_t                   q,
    idx_t                   r,
    idx_t                   s )

{
  return flatten ( a, Tuple<idx_t,6>( m, n, p, q, r, s ) );
}


//-----------------------------------------------------------------------
//   equal
//-----------------------------------------------------------------------


template <class T, class U, int N, class E1, class E2>

  bool                    equal

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs )

{
  if ( equal( lhs.shape(), rhs.shape() ) )
  {
    return ! testany
      ( Array< bool, N, BinaryExpr< NotEqual<T,U>,
          Array<T,N,E1>, Array<U,N,E2> > > ( lhs, rhs ) );
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


template <class I, int N, class E>

  idx_t                   find

  ( Array<I,2>&             index,
    const Array<bool,N,E>&  expr )

{
  const Tuple<idx_t,N>  sh = expr.shape ();
  const idx_t           cn = count ( expr );

  index.resize ( cn, N );

  if ( cn )
  {
    I*              ixp = index.addr ();
    Tuple<idx_t,N>  i;

    i = 0;

    if ( expr.isContiguous() )
    {
      idx_t  k = 0;

      do
      {
        for ( i[0] = 0; i[0] < sh[0]; i[0]++ )
        {
          if ( expr.getFast(k++) )
          {
            for ( int j = 0; j < N; j++ )
            {
              ixp[j * cn] = (I) i[j];
            }

            ixp++;
          }
        }
      }
      while ( array::incrIndex( i, sh ) );
    }
    else
    {
      do
      {
        for ( i[0] = 0; i[0] < sh[0]; i[0]++ )
        {
          if ( expr[i] )
          {
            for ( int j = 0; j < N; j++ )
            {
              ixp[j * cn] = (I) i[j];
            }

            ixp++;
          }
        }
      }
      while ( array::incrIndex( i, sh ) );
    }
  }

  return cn;
}


template <class I, class E>

  idx_t                   find

  ( Array<I>&               index,
    const Array<bool,1,E>&  expr )

{
  const idx_t  n = expr.size ();
  idx_t        k = count ( expr );

  index.resize ( k );

  if ( k )
  {
    I*  ixp = index.addr ();

    k = 0;

    if ( expr.isContiguous() )
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        if ( expr.getFast(i) )
        {
          ixp[k++] = (I) i;
        }
      }
    }
    else
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        if ( expr[i] )
        {
          ixp[k++] = (I) i;
        }
      }
    }
  }

  return k;
}


template <class I, class E>

  idx_t                   find

  ( Array<I,2>&             index,
    const Array<bool,1,E>&  expr )

{
  const idx_t  n = expr.size ();
  idx_t        k = count ( expr );

  index.resize ( k, 1 );

  k = 0;

  if ( k )
  {
    I*  ixp = index.addr ();

    k = 0;

    if ( expr.isContiguous() )
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        if ( expr.getFast(i) )
        {
          ixp[k++] = (I) i;
        }
      }
    }
    else
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        if ( expr[i] )
        {
          ixp[k++] = (I) i;
        }
      }
    }
  }

  return k;
}


template <class I, class E>

  idx_t                   find

  ( Array<I,2>&             index,
    const Array<bool,2,E>&  expr )

{
  const Tuple<idx_t,2>  sh = expr.shape ();
  idx_t                 cn  = count ( expr );

  index.resize ( cn, 2 );

  if ( cn )
  {
    I*  ixp = index.addr ();

    if ( expr.isContiguous() )
    {
      const idx_t  n = sh[0];
      idx_t        k = 0;

      for ( idx_t j = 0; j < sh[1]; j++ )
      {
        for ( idx_t i = 0; i < n; i++, k++ )
        {
          if ( expr.getFast(k) )
          {
            ixp[0]  = (I) i;
            ixp[cn] = (I) j;

            ixp++;
          }
        }
      }
    }
    else
    {
      Tuple<idx_t,2>  i;

      for ( i[1] = 0; i[1] < sh[1]; i[1]++ )
      {
        for ( i[0] = 0; i[0] < sh[0]; i[0]++ )
        {
          if ( expr[i] )
          {
            ixp[0]  = (I) i[0];
            ixp[cn] = (I) i[1];

            ixp++;
          }
        }
      }
    }
  }

  return cn;
}


//-----------------------------------------------------------------------
//   iarray
//-----------------------------------------------------------------------


inline Array<idx_t,1,SliceTo>

                          iarray

  ( idx_t                   n )

{
  JEM_ASSERT2 ( n >= 0, array::Errors::INVALID_SIZE );

  return Array<idx_t,1,SliceTo> ( n );
}


inline Array<idx_t,1,SliceTo>

                          iarray

  ( const SliceTo&          s )

{
  return Array<idx_t,1,SliceTo> ( s.last() );
}


inline Array<idx_t,1,Slice>

                          iarray

  ( const Slice&            s )

{
  return Array<idx_t,1,Slice> ( s.first(), s.last(), s.stride() );
}


inline Array<idx_t,1,SliceFromTo>

                          iarray

  ( const SliceFromTo&      s )

{
  return Array<idx_t,1,SliceFromTo> ( s.first(), s.last() );
}


//-----------------------------------------------------------------------
//   makeContiguous
//-----------------------------------------------------------------------


template <class T, int N>

  inline Array<T,N>       makeContiguous

  ( const Array<T,N>&       a )

{
  if ( a.isContiguous() )
  {
    return a;
  }
  else
  {
    return Array<T,N> ( a, Array<T,N>::CLONE_TAG );
  }
}


//-----------------------------------------------------------------------
//   sort
//-----------------------------------------------------------------------


template <class T>

  void                    sort

  ( const Array<T>&         a )

{
  const idx_t  st = a.stride ();
  T*           ap = a.addr   ();

  T            t;
  idx_t        i, j, k, m;


  m = st * (a.size() / 2);
  k = st * (a.size() - 1);

  while ( k > 0_idx )
  {
    if ( m > 0_idx )
    {
      m    -= st;
      t     = ap[m];
    }
    else
    {
      t     = ap[k];
      ap[k] = ap[0];
      k    -= st;
    }

    i = m;
    j = 2 * m + st;

    while ( j <= k )
    {
      if ( j < k && ap[j] < ap[j + st] )
      {
        j += st;
      }

      if ( t < ap[j] )
      {
        ap[i] = ap[j];
        i     = j;
        j     = j * 2 + st;
      }
      else
      {
        break;
      }
    }

    ap[i] = t;
  }
}


template <class I, class T>

  void                    sort

  ( const Array<I>&         iperm,
    const Array<T>&         a )

{
  const idx_t  ast = a    .stride ();
  const idx_t  ist = iperm.stride ();

  const T*     ap  = a    .addr   ();
  I*           ip  = iperm.addr   ();

  T            t;
  idx_t        i, j, k, m, p;


  m = ist * (iperm.size() / 2);
  k = ist * (iperm.size() - 1);

  while ( k > 0 )
  {
    if ( m > 0 )
    {
      m -= ist;
      p  = ip[m];
      t  = ap[p * ast];
    }
    else
    {
      p     = ip[k];
      t     = ap[p * ast];
      ip[k] = ip[0];
      k    -= ist;
    }

    i = m;
    j = 2 * m + ist;

    while ( j <= k )
    {
      if ( j < k && ap[ip[j] * ast] < ap[ip[j + ist] * ast] )
      {
        j += ist;
      }

      if ( t < ap[ip[j] * ast] )
      {
        ip[i] = ip[j];
        i     = j;
        j     = j * 2 + ist;
      }
      else
      {
        break;
      }
    }

    ip[i] = p;
  }
}


//-----------------------------------------------------------------------
//   permute
//-----------------------------------------------------------------------


template <class T, class I>

  void                    permute

  ( const Array<T>&         a,
    const Array<I>&         iperm )

{
  JEM_PRECHECK2 ( a.size() == iperm.size(), "array size mismatch" );

  const idx_t  n  = a.size ();

  Array<T>     t  ( n );

  const I*     ip = iperm.addr ();
  const T*     tp = t    .addr ();
  T*           ap = a    .addr ();


  t = a;

  if ( a.isContiguous() && iperm.isContiguous() )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      ap[i] = tp[ip[i]];
    }
  }
  else
  {
    const idx_t  ast = a    .stride ();
    const idx_t  ist = iperm.stride ();

    for ( idx_t i = 0; i < n; i++ )
    {
      ap[i * ast] = tp[ip[i * ist]];
    }
  }
}


//-----------------------------------------------------------------------
//   permuted
//-----------------------------------------------------------------------


template <class T, class I>

  Array<T>                permuted

  ( const Array<T>&         a,
    const Array<I>&         iperm )

{
  JEM_PRECHECK2 ( a.size() == iperm.size(), "array size mismatch" );

  const idx_t  n  = a.size ();

  Array<T>     t  ( n );

  const I*     ip = iperm.addr ();
  const T*     ap = a    .addr ();
  T*           tp = t    .addr ();


  if ( a.isContiguous() && iperm.isContiguous() )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      tp[i] = ap[ip[i]];
    }
  }
  else
  {
    const idx_t  ast = a    .stride ();
    const idx_t  ist = iperm.stride ();

    for ( idx_t i = 0; i < n; i++ )
    {
      tp[i] = ap[ip[i * ist] * ast];
    }
  }

  return t;
}


//-----------------------------------------------------------------------
//   binarySearch
//-----------------------------------------------------------------------


template <class T>

  idx_t                   binarySearch

  ( const T&                value,
    const Array<T>&         a )

{
  const T*     ap = a.addr   ();
  const idx_t  st = a.stride ();

  idx_t        lo = 0_idx;
  idx_t        hi = a.size ();
  idx_t        i;

  while ( hi > lo )
  {
    i = lo + (hi - lo) / 2;

    if      ( ap[i * st] < value )
    {
      lo = i + 1;
    }
    else if ( value < ap[i * st] )
    {
      hi = i;
    }
    else
    {
      return i;
    }
  }

  return -1_idx;
}


//-----------------------------------------------------------------------
//   lowerBound
//-----------------------------------------------------------------------


template <class T>

  idx_t                   lowerBound

  ( const T&                value,
    const Array<T>&         a )

{
  const T*     ap = a.addr   ();
  const idx_t  st = a.stride ();

  idx_t        lo = 0_idx;
  idx_t        hi = a.size ();
  idx_t        i;

  while ( hi > lo )
  {
    i = lo + (hi - lo) / 2;

    if ( ap[i * st] < value )
    {
      lo = i + 1;
    }
    else
    {
      hi = i;
    }
  }

  return lo;
}


//-----------------------------------------------------------------------
//   upperBound
//-----------------------------------------------------------------------


template <class T>

  idx_t                   upperBound

  ( const T&                value,
    const Array<T>&         a )

{
  const T*     ap = a.addr   ();
  const idx_t  st = a.stride ();

  idx_t        lo = 0_idx;
  idx_t        hi = a.size ();
  idx_t        i;

  while ( hi > lo )
  {
    i = lo + (hi - lo) / 2;

    if ( value < ap[i * st] )
    {
      hi = i;
    }
    else
    {
      lo = i + 1;
    }
  }

  return hi;
}


//-----------------------------------------------------------------------
//   equalRange
//-----------------------------------------------------------------------


template <class T>

  SliceFromTo             equalRange

  ( const T&                value,
    const Array<T>&         a )

{
  const T*     ap = a.addr   ();
  const idx_t  st = a.stride ();

  idx_t        lo = 0_idx;
  idx_t        hi = a.size ();
  idx_t        i, j, k;

  while ( hi > lo )
  {
    i = lo + (hi - lo) / 2;

    if      ( ap[i * st] < value )
    {
      lo = i + 1;
    }
    else if ( value < ap[i * st] )
    {
      hi = i;
    }
    else
    {
      j = lo + lowerBound ( value, a[slice(lo,i)] );
      k = i  + upperBound ( value, a[slice(i,hi)] );

      return SliceFromTo ( j, k );
    }
  }

  return SliceFromTo ( 0_idx, 0_idx );
}


JEM_END_PACKAGE_BASE

#endif
