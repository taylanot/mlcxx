
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

#ifndef JEM_NUMERIC_ALGEBRA_MATMUL_TCC
#define JEM_NUMERIC_ALGEBRA_MATMUL_TCC

#include <jem/base/assert.h>
#include <jem/base/PrecheckException.h>
#include <jem/numeric/NumberTraits.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   matmul in place
//=======================================================================

//-----------------------------------------------------------------------
//   vector = matrix * vector
//-----------------------------------------------------------------------


template <class T>

  void               matmul

  ( const Array<T,1>&  r,
    const Array<T,2>&  a,
    const Array<T,1>&  b )

{
  JEM_PRECHECK2 ( r.size() == a.size(0) &&
                  b.size() == a.size(1),
                  "matrix/vector size mismatch" );

  const idx_t  m = a.size (0);
  const idx_t  n = a.size (1);

  T            x;

  for ( idx_t i = 0; i < m; i++ )
  {
    x = NumberTraits<T>::zero ();

    for ( idx_t j = 0; j < n; j++ )
    {
      NumberTraits<T>::madd ( x, a(i,j), b[j] );
    }

    r[i] = x;
  }
}


//-----------------------------------------------------------------------
//   vector = vector * matrix
//-----------------------------------------------------------------------


template <class T>

  void               matmul

  ( const Array<T,1>&  r,
    const Array<T,1>&  a,
    const Array<T,2>&  b )

{
  JEM_PRECHECK2 ( r.size() == b.size(1) &&
                  a.size() == b.size(0),
                  "matrix/vector size mismatch" );

  const idx_t  m = b.size (0);
  const idx_t  n = b.size (1);

  T            x;

  for ( idx_t j = 0; j < n; j++ )
  {
    x = NumberTraits<T>::zero ();

    for ( idx_t i = 0; i < m; i++ )
    {
      NumberTraits<T>::madd ( x, a[i], b(i,j) );
    }

    r[j] = x;
  }
}


//-----------------------------------------------------------------------
//   matrix = vector * vector
//-----------------------------------------------------------------------


template <class T>

  void               matmul

  ( const Array<T,2>&  r,
    const Array<T,1>&  a,
    const Array<T,1>&  b )

{
  JEM_PRECHECK2 ( r.size(0) == a.size() &&
                  r.size(1) == b.size(),
                  "matrix/vector size mismatch" );

  const idx_t  m = a.size ();
  const idx_t  n = b.size ();

  T            x;

  r = NumberTraits<T>::zero ();

  for ( idx_t i = 0; i < m; i++ )
  {
    x = a[i];

    for ( idx_t j = 0; j < n; j++ )
    {
      NumberTraits<T>::madd ( r(i,j), b[j], x );
    }
  }
}


//-----------------------------------------------------------------------
//   matrix = matrix * matrix
//-----------------------------------------------------------------------


template <class T>

  void               matmul

  ( const Array<T,2>&  r,
    const Array<T,2>&  a,
    const Array<T,2>&  b )

{
  JEM_PRECHECK2 ( r.size(0) == a.size(0) &&
                  r.size(1) == b.size(1) &&
                  a.size(1) == b.size(0),
                  "matrix shape mismatch" );

  const idx_t  m = r.size (0);
  const idx_t  n = r.size (1);
  const idx_t  p = a.size (1);

  T            x;

  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      x = NumberTraits<T>::zero ();

      for ( idx_t k = 0; k < p; k++ )
      {
        NumberTraits<T>::madd ( x, a(i,k), b(k,j) );
      }

      r(i,j) = x;
    }
  }
}


//-----------------------------------------------------------------------
//   tuple = tuple * tuple
//-----------------------------------------------------------------------


template <class T, int M, int N, int P>

  inline void           matmul

  ( Tuple<T,M,P>&         r,
    const Tuple<T,M,N>&   a,
    const Tuple<T,N,P>&   b )

{
  T  x;

  for ( int j = 0; j < P; j++ )
  {
    for ( int i = 0; i < M; i++ )
    {
      x = NumberTraits<T>::zero ();

      for ( int k = 0; k < N; k++ )
      {
        NumberTraits<T>::madd ( x, a(i,k), b(k,j) );
      }

      r(i,j) = x;
    }
  }
}


template <class T, int M, int N>

  inline void             matmul

  ( Tuple<T,M,1>&           r,
    const Tuple<T,M,N>&     a,
    const Tuple<T,N,1>&     b )

{
  T  x;

  for ( int i = 0; i < M; i++ )
  {
    x = NumberTraits<T>::zero ();

    for ( int j = 0; j < N; j++ )
    {
      NumberTraits<T>::madd ( x, a(i,j), b[j] );
    }

    r[i] = x;
  }
}


template <class T, int M, int N>

  inline void             matmul

  ( Tuple<T,N,1>&           r,
    const Tuple<T,M,1>&     a,
    const Tuple<T,M,N>&     b )

{
  T  x;

  for ( int j = 0; j < N; j++ )
  {
    x = NumberTraits<T>::zero ();

    for ( int i = 0; i < M; i++ )
    {
      NumberTraits<T>::madd ( x, a[i], b(i,j) );
    }

    r[j] = x;
  }
}


template <class T, int M, int N>

  inline void             matmul

  ( Tuple<T,M,N>&           r,
    const Tuple<T,M,1>&     a,
    const Tuple<T,N,1>&     b )

{
  T  x;

  r = NumberTraits<T>::zero ();

  for ( int i = 0; i < M; i++ )
  {
    x = a[i];

    for ( int j = 0; j < N; j++ )
    {
      NumberTraits<T>::madd ( r(i,j), b[j], x );
    }
  }
}


//-----------------------------------------------------------------------
//   tuple = matrix * matrix
//-----------------------------------------------------------------------


template <class T, int M, int P>

  void                  matmul

  ( Tuple<T,M,P>&         r,
    const Array<T,2>&     a,
    const Array<T,2>&     b )

{
  JEM_ASSERT2 ( a.size(0) == M         &&
                a.size(1) == b.size(0) &&
                b.size(1) == P,
                "matrix shape mismatch" );

  T* JEM_RESTRICT  rp = r.addr ();

  const idx_t  n = a.size (1);

  T            t;


  t = NumberTraits<T>::zero ();

  for ( int i = 0; i < M * P; i++ )
  {
    rp[i] = t;
  }

  if ( ((a.stride(0) - 1) | (a.stride(1) - M) |
        (b.stride(0) - 1) | (b.stride(1) - n)) == 0 )
  {
    const T* JEM_RESTRICT  ap = a.addr ();
    const T* JEM_RESTRICT  bp = b.addr ();

    for ( idx_t k = 0; k < n; k++ )
    {
      for ( int j = 0; j < P; j++ )
      {
        t = bp[k + j * n];

        for ( int i = 0; i < M; i++ )
        {
          NumberTraits<T>::madd ( rp[i + j * M], ap[i + k * M], t );
        }
      }
    }
  }
  else
  {
    T  aa[M];

    for ( idx_t k = 0; k < n; k++ )
    {
      for ( int i = 0; i < M; i++ )
      {
        aa[i] = a(i,k);
      }

      for ( int j = 0; j < P; j++ )
      {
        t = b(k,j);

        for ( int i = 0; i < M; i++ )
        {
          NumberTraits<T>::madd ( rp[i + j * M], aa[i], t );
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   matrix = tuple * matrix
//-----------------------------------------------------------------------


template <class T, int M, int N>

  void                  matmul

  ( const Array<T,2>&     r,
    const Tuple<T,M,N>&   a,
    const Array<T,2>&     b )

{
  JEM_ASSERT2 ( r.size(0) == M         &&
                r.size(1) == b.size(1) &&
                b.size(0) == N,
                "matrix shape mismatch" );

  const T* JEM_RESTRICT  ap = a.addr ();

  const idx_t  p = b.size (1);

  T            t;


  if ( ((r.stride(0) - 1) | (r.stride(1) - M) |
        (b.stride(0) - 1) | (b.stride(1) - N)) == 0 )
  {
    T*       JEM_RESTRICT  rp = r.addr ();
    const T* JEM_RESTRICT  bp = b.addr ();

    for ( idx_t j = 0; j < p; j++ )
    {
      for ( int i = 0; i < M; i++ )
      {
        t = NumberTraits<T>::zero ();

        for ( int k = 0; k < N; k++ )
        {
          NumberTraits<T>::madd ( t, ap[i + k * M], bp[k + j * N] );
        }

        rp[i + j * M] = t;
      }
    }
  }
  else
  {
    T  bb[N];

    for ( idx_t j = 0; j < p; j++ )
    {
      for ( int k = 0; k < N; k++ )
      {
        bb[k] = b(k,j);
      }

      for ( int i = 0; i < M; i++ )
      {
        t = NumberTraits<T>::zero ();

        for ( int k = 0; k < N; k++ )
        {
          NumberTraits<T>::madd ( t, ap[i + k * M], bb[k] );
        }

        r(i,j) = t;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   matrix = matrix * tuple
//-----------------------------------------------------------------------


template <class T, int N, int P>

  void                  matmul

  ( const Array<T,2>&     r,
    const Array<T,2>&     a,
    const Tuple<T,N,P>&   b )

{
  JEM_ASSERT2 ( r.size(0) == a.size(0) &&
                r.size(1) == P         &&
                a.size(1) == N,
                "matrix shape mismatch" );

  const T* JEM_RESTRICT  bp = b.addr ();

  const idx_t  m = a.size (0);

  T            t;


  if ( ((r.stride(0) - 1) | (r.stride(1) - m) |
        (a.stride(0) - 1) | (a.stride(1) - m)) == 0 )
  {
    T*       JEM_RESTRICT  rp = r.addr ();
    const T* JEM_RESTRICT  ap = a.addr ();

    for ( idx_t i = 0; i < m; i++ )
    {
      for ( int j = 0; j < P; j++ )
      {
        t = NumberTraits<T>::zero ();

        for ( int k = 0; k < N; k++ )
        {
          NumberTraits<T>::madd ( t, ap[i + k * m], bp[k + j * N] );
        }

        rp[i + j * m] = t;
      }
    }
  }
  else
  {
    T  aa[N];

    for ( idx_t i = 0; i < m; i++ )
    {
      for ( int k = 0; k < N; k++ )
      {
        aa[k] = a(i,k);
      }

      for ( int j = 0; j < P; j++ )
      {
        t = NumberTraits<T>::zero ();

        for ( int k = 0; k < N; k++ )
        {
          NumberTraits<T>::madd ( t, aa[k], bp[k + j * N] );
        }

        r(i,j) = t;
      }
    }
  }
}


//=======================================================================
//   matmul returning a new Array or Tuple
//=======================================================================


template <class T>

  inline Array<T,1>     matmul

  ( const Array<T,2>&     a,
    const Array<T,1>&     b )

{
  Array<T>  r ( a.size(0) );

  matmul ( r, a, b );

  return r;
}


template <class T>

  inline Array<T,1>     matmul

  ( const Array<T,1>&     a,
    const Array<T,2>&     b )

{
  Array<T>  r ( b.size(1) );

  matmul ( r, a, b );

  return r;
}


template <class T>

  inline Array<T,2>     matmul

  ( const Array<T,1>&     a,
    const Array<T,1>&     b )

{
  Array<T,2>  r ( a.size(), b.size() );

  matmul ( r, a, b );

  return r;
}


template <class T>

  inline Array<T,2>     matmul

  ( const Array<T,2>&     a,
    const Array<T,2>&     b )

{
  Array<T,2>  r ( a.size(0), b.size(1) );

  matmul ( r, a, b );

  return r;
}


template <class T, int M, int N, int P>

  inline Tuple<T,M,P>   matmul

  ( const Tuple<T,M,N>&   a,
    const Tuple<T,N,P>&   b )

{
  Tuple<T,M,P>  r;

  matmul ( r, a, b );

  return r;
}


template <class T, int M, int N>

  inline Tuple<T,M,1>     matmul

  ( const Tuple<T,M,N>&     a,
    const Tuple<T,N,1>&     b )

{
  Tuple<T,M,1>  r;

  matmul ( r, a, b );

  return r;
}


template <class T, int M, int N>

  inline Tuple<T,N,1>     matmul

  ( const Tuple<T,M,1>&     a,
    const Tuple<T,M,N>&     b )

{
  Tuple<T,N,1>  r;

  matmul ( r, a, b );

  return r;
}


template <class T, int M, int N>

  inline Tuple<T,M,N>     matmul

  ( const Tuple<T,M,1>&     a,
    const Tuple<T,N,1>&     b )

{
  Tuple<T,M,N>  r;

  matmul ( r, a, b );

  return r;
}


template <class T, int M, int N>

  inline Array<T,2>     matmul

  ( const Tuple<T,M,N>&   a,
    const Array<T,2>&     b )

{
  Array<T,2>  r ( M, b.size(1) );

  matmul ( r, a, b );

  return r;
}


template <class T, int N, int P>

  inline Array<T,2>     matmul

  ( const Array<T,2>&     a,
    const Tuple<T,N,P>&   b )

{
  Array<T,2>  r ( a.size(0), P );

  matmul ( r, a, b );

  return r;
}


JEM_END_PACKAGE( numeric )

#endif
