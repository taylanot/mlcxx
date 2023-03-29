
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

#ifndef JEM_NUMERIC_ALGEBRA_UTILITIES_TCC
#define JEM_NUMERIC_ALGEBRA_UTILITIES_TCC

#include <jem/base/assert.h>
#include <jem/numeric/utilities.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   private functions
//=======================================================================

JEM_BEGIN_NAMESPACE( algebra )

//-----------------------------------------------------------------------
//   axpyFast
//-----------------------------------------------------------------------


template <class T>

  void                        axpyFast

  ( T*       JEM_RESTRICT       r,
    T                           t,
    const T* JEM_RESTRICT       b,
    idx_t                       n )

{
  for ( idx_t i = 0; i < n; i++ )
  {
    r[i] += t * b[i];
  }
}


void                          axpyFast

  ( double*       JEM_RESTRICT  r,
    double                      t,
    const double* JEM_RESTRICT  b,
    idx_t                       n );


template <class T, int N>

  inline void                 axpyFast

  ( const Array<T,N>&           r,
    T                           t,
    const Array<T,N>&           b )

{
  axpyFast ( r.addr(), t, b.addr(), r.itemCount() );
}


template <class T>

  void                        axpyFast

  ( T*       JEM_RESTRICT       r,
    const T* JEM_RESTRICT       a,
    T                           t,
    const T* JEM_RESTRICT       b,
    idx_t                       n )

{
  if ( r == b )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      r[i] = a[i] + t * r[i];
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      r[i] = a[i] + t * b[i];
    }
  }
}


void                          axpyFast

  ( double*       JEM_RESTRICT  r,
    const double* JEM_RESTRICT  a,
    double                      t,
    const double* JEM_RESTRICT  b,
    idx_t                       n );


template <class T, int N>

  inline void                 axpyFast

  ( const Array<T,N>&           r,
    const Array<T,N>&           a,
    T                           t,
    const Array<T,N>&           b )

{
  axpyFast ( r.addr(), a.addr(), t, b.addr(), r.itemCount() );
}


//-----------------------------------------------------------------------
//   axpySlow
//-----------------------------------------------------------------------


template <class T, int N>

  void                        axpySlow

  ( const Array<T,N>&           r,
    T                           t,
    const Array<T,N>&           b )

{
  const Tuple<idx_t,N>  sh = r.shape ();

  if ( array::size( sh ) > 0 )
  {
    T*       JEM_RESTRICT  rp = r.addr ();
    const T* JEM_RESTRICT  bp = b.addr ();

    const idx_t     n   = sh[0];
    const idx_t     rst = r.stride (0);
    const idx_t     bst = b.stride (0);

    Tuple<idx_t,N>  i;
    idx_t           j;

    i = 0;

    while ( true )
    {
      for ( j = 0; j < n; j++ )
      {
        rp[j * rst] += t * bp[j * bst];
      }

      i[0] = n;

      if ( ! array::incrIndex( i, sh ) )
      {
        break;
      }

      rp = r.addr() + array::ioffset ( i, r.stride() );
      bp = b.addr() + array::ioffset ( i. b.stride() );
    }
  }
}


template <class T>

  void                        axpySlow

  ( const Array<T,1>&           r,
    T                           t,
    const Array<T,1>&           b )

{
  T*       JEM_RESTRICT  rp = r.addr ();
  const T* JEM_RESTRICT  bp = b.addr ();

  const idx_t  n   = r.size   ();
  const idx_t  rst = r.stride ();
  const idx_t  bst = b.stride ();

  for ( idx_t i = 0; i < n; i++ )
  {
    rp[i * rst] += t * bp[i * bst];
  }
}


template <class T, int N>

  void                        axpySlow

  ( const Array<T,N>&           r,
    const Array<T,N>&           a,
    T                           t,
    const Array<T,N>&           b )

{
  const Tuple<idx_t,N>  sh = r.shape ();

  if ( array::size( sh ) > 0 )
  {
    T*       JEM_RESTRICT  rp = r.addr ();
    const T* JEM_RESTRICT  ap = a.addr ();
    const T* JEM_RESTRICT  bp = b.addr ();

    const idx_t     n   = sh[0];
    const idx_t     rst = r.stride (0);
    const idx_t     ast = a.stride (0);
    const idx_t     bst = b.stride (0);

    Tuple<idx_t,N>  i;
    idx_t           j;


    i = 0;

    while ( true )
    {
      for ( j = 0; j < n; j++ )
      {
        rp[j * rst] = ap[j * ast] + t * bp[j * bst];
      }

      i[0] = n;

      if ( ! array::incrIndex( i, sh ) )
      {
        break;
      }

      rp = r.addr() + array::ioffset ( i, r.stride() );
      ap = a.addr() + array::ioffset ( i, a.stride() );
      bp = b.addr() + array::ioffset ( i, b.stride() );
    }
  }
}


template <class T>

  void                        axpySlow

  ( const Array<T,1>&           r,
    const Array<T,1>&           a,
    T                           t,
    const Array<T,1>&           b )

{
  T*       JEM_RESTRICT  rp = r.addr ();
  const T* JEM_RESTRICT  ap = a.addr ();
  const T* JEM_RESTRICT  bp = b.addr ();

  const idx_t  n   = r.size   ();
  const idx_t  rst = r.stride ();
  const idx_t  ast = a.stride ();
  const idx_t  bst = b.stride ();

  for ( idx_t i = 0; i < n; i++ )
  {
    rp[i * rst] = ap[i *  ast] + t * bp[i * bst];
  }
}

JEM_END_NAMESPACE( algebra )

//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   crossProduct
//-----------------------------------------------------------------------


template <class T, class E1, class E2>

  inline Tuple<T,3>       crossProduct

  ( const Tuple<T,3,1,E1>&  a,
    const Tuple<T,3,1,E2>&  b )

{
  return Tuple<T,3> ( a[1] * b[2] - a[2] * b[1],
                      a[2] * b[0] - a[0] * b[2],
                      a[0] * b[1] - a[1] * b[0] );
}


template <class T, class E1, class E2>

  inline T                crossProduct

  ( const Tuple<T,2,1,E1>&  a,
    const Tuple<T,2,1,E2>&  b )

{
  return (a[0] * b[1] - a[1] * b[0]);
}


//-----------------------------------------------------------------------
//   dotProduct (Tuple)
//-----------------------------------------------------------------------


template <class T, int N, class E1, class E2>

  inline T                dotProduct

  ( const Tuple<T,N,1,E1>&  a,
    const Tuple<T,N,1,E2>&  b )

{
  T  t = a[0] * b[0];

  for ( int i = 1; i < N; i++ )
  {
    t += a[i] * b[i];
  }

  return t;
}


template <class T, class E1, class E2>

  inline T                dotProduct

  ( const Tuple<T,0,1,E1>&  a,
    const Tuple<T,0,1,E2>&  b )

{
  return T ();
}


template <class T, class E1, class E2>

  inline T                dotProduct

  ( const Tuple<T,1,1,E1>&  a,
    const Tuple<T,1,1,E2>&  b )

{
  return (a[0] * b[0]);
}


template <class T, class E1, class E2>

  inline T                dotProduct

  ( const Tuple<T,2,1,E1>&  a,
    const Tuple<T,2,1,E2>&  b )

{
  return (a[0] * b[0] + a[1] * b[1]);
}


template <class T, class E1, class E2>

  inline T                dotProduct

  ( const Tuple<T,3,1,E1>&  a,
    const Tuple<T,3,1,E2>&  b )

{
  return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}


//-----------------------------------------------------------------------
//   dotProduct (Array)
//-----------------------------------------------------------------------


template <class T, class E1, class E2>

  inline T              dotProduct

  ( const Array<T,1,E1>&  a,
    const Array<T,1,E2>&  b )

{
  return jem::dot ( a, b );
}


//-----------------------------------------------------------------------
//   norm2
//-----------------------------------------------------------------------


template <class T, int N, class E>

  inline T                norm2

  ( const Tuple<T,N,1,E>&   v )

{
  return sqrt ( dotProduct( v, v ) );
}


template <class T, class E>

  inline T                norm2

  ( const Array<T,1,E>&     v )

{
  return sqrt ( jem::dot( v ) );
}


//-----------------------------------------------------------------------
//   axpy
//-----------------------------------------------------------------------


template <class T, int N>

  inline void             axpy

  ( const Array<T,N>&       r,
    T                       t,
    const Array<T,N>&       b )

{
  JEM_ASSERT2 ( equal( r.shape(), b.shape() ),
                "Array shape mismatch" );

  if ( r.isContiguous() && b.isContiguous() )
  {
    algebra::axpyFast ( r, t, b );
  }
  else
  {
    algebra::axpySlow ( r, t, b );
  }
}


template <class T, int N>

  inline void             axpy

  ( const Array<T,N>&       r,
    const Array<T,N>&       a,
    T                       t,
    const Array<T,N>&       b )

{
  JEM_ASSERT2 ( equal( r.shape(), a.shape() ) &&
                equal( r.shape(), b.shape() ),
                "Array shape mismatch" );

  if ( r.isContiguous() && a.isContiguous() && b.isContiguous() )
  {
    algebra::axpyFast ( r, a, t, b );
  }
  else
  {
    algebra::axpySlow ( r, a, t, b );
  }
}


JEM_END_PACKAGE( numeric )

#endif
