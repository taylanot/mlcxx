
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


#include <jem/pragmas.h>
#include <jem/base/PrecheckException.h>
#include <jem/numeric/NumberTraits.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   vecmul_
//-----------------------------------------------------------------------


template <class T>

  Array<T,2>&         vecmul_

  ( Array<T,2>&         mout,
    const Array<T>&     lhs,
    const Array<T>&     rhs )

{
  const T* JEM_RESTRICT  lp = lhs.addr ();

  T*       JEM_RESTRICT  mp;

  const idx_t  lst = lhs.stride ();

  const idx_t  m   = lhs.size ();
  const idx_t  n   = rhs.size ();

  T            t;
  idx_t        i, j;


  if ( mout.size(0) != m || mout.size(1) != n )
  {
    mout.resize ( m, n );
  }

  mp = mout.addr ();

  if ( lst == 1_idx )
  {
    for ( j = 0; j < n; j++ )
    {
      t = rhs[j];

JEM_IVDEP

      for ( i = 0; i < m; i++ )
      {
        mp[i] = t * lp[i];
      }

      mp += mout.stride(1);
    }
  }
  else
  {
    for ( j = 0; j < n; j++ )
    {
      t = rhs[j];

JEM_IVDEP

      for ( i = 0; i < m; i++ )
      {
        mp[i] = t * lp[i * lst];
      }

      mp += mout.stride (1);
    }
  }

  return mout;
}


//=======================================================================
//   class MatmulChain<T,0>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline MatmulChain<T,0>::MatmulChain () noexcept

{}


//-----------------------------------------------------------------------
//   matmul (vector * vector)
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T,2>& MatmulChain<T,0>::matmul

  ( const Array<T>&  v,
    const Array<T>&  w )

{
  return vecmul_ ( mout_, v, w );
}


//=======================================================================
//   class MatmulChain<T,1>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline MatmulChain<T,1>::MatmulChain () noexcept

{}


//-----------------------------------------------------------------------
//   matmul (matrix * vector)
//-----------------------------------------------------------------------


template <class T>

  const Array<T>& MatmulChain<T,1>::matmul

  ( const Array<T,2>&  a,
    const Array<T>&    v )

{
  JEM_PRECHECK2 ( a.size(1) == v.size(),
                  "matrix/vector size mismatch" );

  const T* JEM_RESTRICT  arow = a.addr ();

  const T* JEM_RESTRICT  arow1;
  T*       JEM_RESTRICT  vcol;

  const idx_t  ast = a.stride (1);

  const idx_t  m   = a.size   (0);
  const idx_t  n   = a.size   (1);

  T            x, y, z;
  idx_t        i, j;


  vresize_ ( m, n );

  if ( v.isContiguous() )
  {
    vcol = v.addr ();
  }
  else
  {
    vcol = args_.addr ();

    for ( i = 0; i < n; i++ )
    {
      vcol[i] = v[i];
    }
  }

  // Compute vout_ = a * v

  for ( i = 0; i < m - 1; i += 2_idx )
  {
    x = y = NumberTraits<T>::zero ();
    arow1 = arow + a.stride (0);

JEM_IVDEP

    for ( j = 0; j < n; j++ )
    {
      z  = vcol [j];
      x += arow [j * ast] * z;
      y += arow1[j * ast] * z;
    }

    vout_[i + 0] = x;
    vout_[i + 1] = y;
    arow         = arow1 + a.stride (0);
  }

  if ( i < m )
  {
    x = NumberTraits<T>::zero ();

    for ( j = 0; j < n; j++ )
    {
      x += arow[j * ast] * vcol[j];
    }

    vout_[i] = x;
  }

  return vout_;
}


//-----------------------------------------------------------------------
//   matmul (vector * matrix)
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T>& MatmulChain<T,1>::matmul

  ( const Array<T>&    v,
    const Array<T,2>&  a )

{
  return matmul ( a.transpose(), v );
}



//-----------------------------------------------------------------------
//   matmul (vector * matrix * vector)
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T,2>& MatmulChain<T,1>::matmul

  ( const Array<T>&    v,
    const Array<T,2>&  a,
    const Array<T>&    w )

{
  return vecmul_ ( mout_, v, matmul( a, w ) );
}


//-----------------------------------------------------------------------
//   vresize_
//-----------------------------------------------------------------------


template <class T>

  inline void MatmulChain<T,1>::vresize_

  ( idx_t m, idx_t n )

{
  if ( m != vout_.size() )
  {
    vout_.resize ( m );
  }

  if ( n > args_.size() )
  {
    args_.resize ( n );
  }
}


//=======================================================================
//   class MatmulChain<T,2>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline MatmulChain<T,2>::MatmulChain () noexcept

{}


//-----------------------------------------------------------------------
//   matmul (matrix * matrix)
//-----------------------------------------------------------------------


template <class T>

  const Array<T,2>& MatmulChain<T,2>::matmul

  ( const Array<T,2>&  a,
    const Array<T,2>&  b )

{
  JEM_PRECHECK2 ( a.size(1) == b.size(0), "matrix shape mismatch" );

  T* JEM_RESTRICT  mout;
  T* JEM_RESTRICT  arow;
  T* JEM_RESTRICT  arow1;
  T* JEM_RESTRICT  bcol;

  const idx_t  m = a.size (0);
  const idx_t  n = a.size (1);
  const idx_t  p = b.size (1);

  T            x, y, z;
  idx_t        i, j, k;


  mresize_ ( m, n, p );

  mout = mout_.addr ();
  arow = args_.addr ();

  for ( i = k = 0; i < m; i++ )
  {
    for ( j = 0; j < n; j++, k++ )
    {
      arow[k] = a(i,j);
    }
  }

  if ( b.stride(0) == 1_idx && b.stride(1) == n )
  {
    bcol = b.addr ();
  }
  else
  {
    bcol = arow + (m * n);

    for ( j = k = 0; j < p; j++ )
    {
      for ( i = 0; i < n; i++, k++ )
      {
        bcol[k] = b(i,j);
      }
    }
  }

  // Compute mout_ = a * b

  for ( j = 0; j < p; j++ )
  {
    arow = args_.addr ();

    for ( i = 0; i < m - 1; i += 2_idx )
    {
      x = y = NumberTraits<T>::zero ();
      arow1 = arow + n;

      for ( k = 0; k < n; k++ )
      {
        z  = bcol [k];
        x += arow [k] * z;
        y += arow1[k] * z;
      }

      mout[i + 0] = x;
      mout[i + 1] = y;
      arow        = arow1 + n;
    }

    if ( i < m )
    {
      x = NumberTraits<T>::zero ();

      for ( k = 0; k < n; k++ )
      {
        x += arow[k] * bcol[k];
      }

      mout[i] = x;
    }

    mout += m;
    bcol += n;
  }

  return mout_;
}


//-----------------------------------------------------------------------
//   matmul (matrix * matrix * vector)
//-----------------------------------------------------------------------


template <class T>

  const Array<T>& MatmulChain<T,2>::matmul

  ( const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T>&    v )

{
  JEM_PRECHECK2 ( a.size(1) == b.size(0) &&
                  b.size(1) == v.size(),
                  "matrix/vector size mismatch" );

  const T* JEM_RESTRICT  acol = a.addr ();
  const T* JEM_RESTRICT  brow = b.addr ();

  const T* JEM_RESTRICT  acol1;
  const T* JEM_RESTRICT  brow1;
  T*       JEM_RESTRICT  vout;
  T*       JEM_RESTRICT  vcol;

  const idx_t  ast = a.stride (0);
  const idx_t  bst = b.stride (1);

  const idx_t  m   = a.size   (0);
  const idx_t  n   = a.size   (1);
  const idx_t  p   = b.size   (1);

  T            x, y, z;
  idx_t        i, j;


  vresize_ ( m, p );

  vout = vout_.addr ();

  if ( v.isContiguous() )
  {
    vcol = v.addr ();
  }
  else
  {
    vcol = args_.addr ();

    for ( i = 0; i < p; i++ )
    {
      vcol[i] = v[i];
    }
  }

  x = NumberTraits<T>::zero ();

  for ( i = 0; i < m; i++ )
  {
    vout[i] = x;
  }

  // Compute vout_ = a * b * v

  for ( i = 0; i < n - 1; i += 2_idx )
  {
    x = y = NumberTraits<T>::zero ();
    acol1 = acol + a.stride (1);
    brow1 = brow + b.stride (0);

    for ( j = 0; j < p; j++ )
    {
      z  = vcol [j];
      x += brow [j * bst] * z;
      y += brow1[j * bst] * z;
    }

JEM_IVDEP

    for ( j = 0; j < m; j++ )
    {
      vout[j] += acol[j * ast] * x + acol1[j * ast] * y;
    }

    acol = acol1 + a.stride (1);
    brow = brow1 + b.stride (0);
  }

  if ( i < n )
  {
    x = NumberTraits<T>::zero ();

    for ( j = 0; j < p; j++ )
    {
      x += brow[j * bst] * vcol[j];
    }

JEM_IVDEP

    for ( j = 0; j < m; j++ )
    {
      vout[j] += acol[j * ast] * x;
    }
  }

  return vout_;
}


//-----------------------------------------------------------------------
//   matmul (vector * matrix * matrix)
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T>& MatmulChain<T,2>::matmul

  ( const Array<T>&    v,
    const Array<T,2>&  a,
    const Array<T,2>&  b )

{
  return matmul ( b.transpose(), a.transpose(), v );
}


//-----------------------------------------------------------------------
//   matmul (vector * matrix * matrix * vector)
//-----------------------------------------------------------------------


template <class T>
  inline const Array<T,2>& MatmulChain<T,2>::matmul

  ( const Array<T>&    v,
    const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T>&    w )

{
  return vecmul_ ( mout_, v, matmul( a, b, w ) );
}


//-----------------------------------------------------------------------
//   mresize_
//-----------------------------------------------------------------------


template <class T>

  inline void MatmulChain<T,2>::mresize_

  ( idx_t m, idx_t n, idx_t p )

{
  if ( m != mout_.size(0) || p != mout_.size(1) )
  {
    mout_.resize ( m, p );
  }

  if ( n * (m + p) > args_.size() )
  {
    args_.resize ( n * (m + p) );
  }
}


//-----------------------------------------------------------------------
//   vresize_
//-----------------------------------------------------------------------


template <class T>

  inline void MatmulChain<T,2>::vresize_

  ( idx_t m, idx_t p )

{
  if ( m != vout_.size() )
  {
    vout_.resize ( m );
  }

  if ( p > args_.size() )
  {
    args_.resize ( p );
  }
}


//=======================================================================
//   class MatmulChain<T,3>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline MatmulChain<T,3>::MatmulChain () noexcept

{}


//-----------------------------------------------------------------------
//   matmul (matrix * matrix * matrix)
//-----------------------------------------------------------------------


template <class T>

  const Array<T,2>& MatmulChain<T,3>::matmul

  ( const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T,2>&  c )

{
  JEM_PRECHECK2 ( a.size(1) == b.size(0) &&
                  b.size(1) == c.size(0),
                  "matrix shape mismatch" );

  T* JEM_RESTRICT  mout;
  T* JEM_RESTRICT  acol;
  T* JEM_RESTRICT  acol1;
  T* JEM_RESTRICT  brow;
  T* JEM_RESTRICT  brow1;
  T* JEM_RESTRICT  ccol;

  const idx_t  m = a.size (0);
  const idx_t  n = a.size (1);
  const idx_t  p = b.size (1);
  const idx_t  q = c.size (1);

  T            x, y, z;
  idx_t        i, j, k;


  mresize_ ( m, n, p, q );

  mout = mout_.addr ();
  acol = args_.addr ();
  brow = acol + (m * n);

  for ( j = k = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++, k++ )
    {
      acol[k] = a(i,j);
    }
  }

  for ( i = k = 0; i < n; i++ )
  {
    for ( j = 0; j < p; j++, k++ )
    {
      brow[k] = b(i,j);
    }
  }

  if ( c.stride(0) == 1_idx && c.stride(1) == p )
  {
    ccol = c.addr ();
  }
  else
  {
    ccol = brow + (n * p);

    for ( j = k = 0; j < q; j++ )
    {
      for ( i = 0; i < p; i++, k++ )
      {
        ccol[k] = c(i,j);
      }
    }
  }

  // Compute mout_ = a * b * c

  for ( j = 0; j < q; j++ )
  {
    acol = args_.addr ();
    brow = acol + (m * n);
    x    = NumberTraits<T>::zero ();

    for ( i = 0; i < m; i++ )
    {
      mout[i] = x;
    }

    for ( i = 0; i < n - 1; i += 2_idx )
    {
      x = y = NumberTraits<T>::zero ();
      acol1 = acol + m;
      brow1 = brow + p;

      for ( k = 0; k < p; k++ )
      {
        z  = ccol [k];
        x += brow [k] * z;
        y += brow1[k] * z;
      }

JEM_IVDEP

      for ( k = 0; k < m; k++ )
      {
        mout[k] += acol[k] * x + acol1[k] * y;
      }

      acol = acol1 + m;
      brow = brow1 + p;
    }

    if ( i < n )
    {
      x = NumberTraits<T>::zero ();

      for ( k = 0; k < p; k++ )
      {
        x += brow[k] * ccol[k];
      }

JEM_IVDEP

      for ( k = 0; k < m; k++ )
      {
        mout[k] += acol[k] * x;
      }
    }

    mout += m;
    ccol += p;
  }

  return mout_;
}


//-----------------------------------------------------------------------
//   matmul (matrix * matrix * matrix * vector)
//-----------------------------------------------------------------------


template <class T>

  const Array<T>& MatmulChain<T,3>::matmul

  ( const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T,2>&  c,
    const Array<T>&    v )

{
  JEM_PRECHECK2 ( a.size(1) == b.size(0) &&
                  b.size(1) == c.size(0) &&
                  c.size(1) == v.size(),
                  "matrix/vector size mismatch" );


  const T* JEM_RESTRICT  arow = a.addr ();
  const T* JEM_RESTRICT  bcol = b.addr ();
  const T* JEM_RESTRICT  crow = c.addr ();

  const T* JEM_RESTRICT  arow1;
  const T* JEM_RESTRICT  bcol1;
  const T* JEM_RESTRICT  crow1;
  T*       JEM_RESTRICT  tcol;
  T*       JEM_RESTRICT  vcol;

  const idx_t  ast = a.stride (1);
  const idx_t  bst = b.stride (0);
  const idx_t  cst = c.stride (1);

  const idx_t  m   = a.size   (0);
  const idx_t  n   = a.size   (1);
  const idx_t  p   = b.size   (1);
  const idx_t  q   = c.size   (1);

  T            x, y, z;
  idx_t        i, j;


  vresize_ ( m, n, q );

  tcol = tmp_ .addr ();

  if ( v.isContiguous() )
  {
    vcol = v.addr ();
  }
  else
  {
    vcol = args_.addr ();

    for ( i = 0; i < q; i++ )
    {
      vcol[i] = v[i];
    }
  }

  x = NumberTraits<T>::zero ();

  for ( i = 0; i < n; i++ )
  {
    tcol[i] = x;
  }

  // Compute vout_ = a * b * c * v

  for ( i = 0; i < p - 1; i += 2_idx )
  {
    x = y = NumberTraits<T>::zero ();
    bcol1 = bcol + b.stride (1);
    crow1 = crow + c.stride (0);

    for ( j = 0; j < q; j++ )
    {
      z  = vcol [j];
      x += crow [j * cst] * z;
      y += crow1[j * cst] * z;
    }

JEM_IVDEP

    for ( j = 0; j < n; j++ )
    {
      tcol[j] += bcol[j * bst] * x + bcol1[j * bst] * y;
    }

    crow = crow1 + c.stride (0);
    bcol = bcol1 + b.stride (1);
  }

  if ( i < p )
  {
    x = NumberTraits<T>::zero ();

    for ( j = 0; j < q; j++ )
    {
      x += crow[j * cst] * vcol[j];
    }

JEM_IVDEP

    for ( j = 0; j < n; j++ )
    {
      tcol[j] += bcol[j * bst] * x;
    }
  }

  for ( i = 0; i < m - 1; i += 2_idx )
  {
    x = y = NumberTraits<T>::zero ();
    arow1 = arow + a.stride (0);

    for ( j = 0; j < n; j++ )
    {
      z  = tcol [j];
      x += arow [j * ast] * z;
      y += arow1[j * ast] * z;
    }

    vout_[i + 0] = x;
    vout_[i + 1] = y;
    arow         = arow1 + a.stride (0);
  }

  if ( i < m )
  {
    x = NumberTraits<T>::zero ();

    for ( j = 0; j < n; j++ )
    {
      x += arow[j * ast] * tcol[j];
    }

    vout_[i] = x;
  }

  return vout_;
}


//-----------------------------------------------------------------------
//   matmul (vector * matrix * matrix * matrix)
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T>& MatmulChain<T,3>::matmul

  ( const Array<T>&    v,
    const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T,2>&  c )

{
  return matmul ( c.transpose(), b.transpose(), a.transpose(), v );
}


//-----------------------------------------------------------------------
//   matmul (vector * matrix * matrix * matrix * vector)
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T,2>& MatmulChain<T,3>::matmul

  ( const Array<T>&    v,
    const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T,2>&  c,
    const Array<T>&    w )

{
  return vecmul_ ( mout_, v, matmul( a, b, c, w ) );
}


//-----------------------------------------------------------------------
//   mresize_
//-----------------------------------------------------------------------


template <class T>

  inline void MatmulChain<T,3>::mresize_

  ( idx_t m, idx_t n, idx_t p, idx_t q )

{
  if ( m != mout_.size(0) || q != mout_.size(1) )
  {
    mout_.resize ( m, q );
  }

  if ( (n * (m + p) + p * q) > args_.size() )
  {
    args_.resize ( n * (m + p) + p * q );
  }
}


//-----------------------------------------------------------------------
//   vresize_
//-----------------------------------------------------------------------


template <class T>

  inline void MatmulChain<T,3>::vresize_

  ( idx_t m, idx_t n, idx_t q )

{
  if ( m != vout_.size() )
  {
    vout_.resize ( m );
  }

  if ( n > tmp_.size() )
  {
    tmp_.resize ( n );
  }

  if ( q > args_.size() )
  {
    args_.resize ( q );
  }
}


//=======================================================================
//   class MatmulChain<T,4>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline MatmulChain<T,4>::MatmulChain () noexcept

{}


//-----------------------------------------------------------------------
//   matmul (matrix * matrix * matrix * matrix)
//-----------------------------------------------------------------------


template <class T>

  const Array<T,2>& MatmulChain<T,4>::matmul

  ( const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T,2>&  c,
    const Array<T,2>&  d )

{
  JEM_PRECHECK2 ( a.size(1) == b.size(0) &&
                  b.size(1) == c.size(0) &&
                  c.size(1) == d.size(0),
                  "matrix shape mismatch" );

  T* JEM_RESTRICT  tcol;
  T* JEM_RESTRICT  arow;
  T* JEM_RESTRICT  arow1;
  T* JEM_RESTRICT  bcol;
  T* JEM_RESTRICT  bcol1;
  T* JEM_RESTRICT  crow;
  T* JEM_RESTRICT  crow1;
  T* JEM_RESTRICT  dcol;

  const idx_t  m = a.size (0);
  const idx_t  n = a.size (1);
  const idx_t  p = b.size (1);
  const idx_t  q = c.size (1);
  const idx_t  r = d.size (1);

  T            x, y, z;
  idx_t        i, j, k;


  mresize_ ( m, n, p, q, r );

  tcol = tmp_ .addr ();
  arow = args_.addr ();
  bcol = arow + (m * n);
  crow = bcol + (n * p);

  for ( i = k = 0; i < m; i++ )
  {
    for ( j = 0; j < n; j++, k++ )
    {
      arow[k] = a(i,j);
    }
  }

  for ( j = k = 0; j < p; j++ )
  {
    for ( i = 0; i < n; i++, k++ )
    {
      bcol[k] = b(i,j);
    }
  }

  for ( i = k = 0; i < p; i++ )
  {
    for ( j = 0; j < q; j++, k++ )
    {
      crow[k] = c(i,j);
    }
  }


  if ( d.stride(0) == 1_idx && d.stride(1) == q )
  {
    dcol = d.addr ();
  }
  else
  {
    dcol = crow + (p * q);

    for ( j = k = 0; j < r; j++ )
    {
      for ( i = 0; i < q; i++, k++ )
      {
        dcol[k] = d(i,j);
      }
    }
  }

  // Compute mout_ = a * b * c * d

  for ( j = 0; j < r; j++ )
  {
    arow = args_.addr ();
    bcol = arow + (m * n);
    crow = bcol + (n * p);
    x    = NumberTraits<T>::zero ();

    for ( i = 0; i < n; i++ )
    {
      tcol[i] = x;
    }

    for ( i = 0; i < p - 1; i += 2_idx )
    {
      x = y = NumberTraits<T>::zero ();
      bcol1 = bcol + n;
      crow1 = crow + q;

      for ( k = 0; k < q; k++ )
      {
        z  = dcol [k];
        x += crow [k] * z;
        y += crow1[k] * z;
      }

JEM_IVDEP

      for ( k = 0; k < n; k++ )
      {
        tcol[k] += bcol[k] * x + bcol1[k] * y;
      }

      bcol = bcol1 + n;
      crow = crow1 + q;
    }

    if ( i < p )
    {
      x = NumberTraits<T>::zero ();

      for ( k = 0; k < q; k++ )
      {
        x += crow[k] * dcol[k];
      }

JEM_IVDEP

      for ( k = 0; k < n; k++ )
      {
        tcol[k] += bcol[k] * x;
      }
    }

    for ( i = 0; i < m - 1; i += 2_idx )
    {
      x = y = NumberTraits<T>::zero ();
      arow1 = arow + n;

      for ( k = 0; k < n; k++ )
      {
        z  = tcol [k];
        x += arow [k] * z;
        y += arow1[k] * z;
      }

      mout_(i + 0,j) = x;
      mout_(i + 1,j) = y;
      arow           = arow1 + n;
    }

    if ( i < m )
    {
      x = NumberTraits<T>::zero ();

      for ( k = 0; k < n; k++ )
      {
        x += arow[k] * tcol[k];
      }

      mout_(i,j) = x;
    }

    dcol += q;
  }

  return mout_;
}


//-----------------------------------------------------------------------
//   matmul (matrix * matrix * matrix * matrix * vector)
//-----------------------------------------------------------------------


template <class T>

  const Array<T>& MatmulChain<T,4>::matmul

  ( const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T,2>&  c,
    const Array<T,2>&  d,
    const Array<T>&    v )

{
  JEM_PRECHECK2 ( a.size(1) == b.size(0) &&
                  b.size(1) == c.size(0) &&
                  c.size(1) == d.size(0) &&
                  d.size(1) == v.size(),
                  "matrix/vector size mismatch" );

  const T* JEM_RESTRICT  acol = a.addr ();
  const T* JEM_RESTRICT  brow = b.addr ();
  const T* JEM_RESTRICT  ccol = c.addr ();
  const T* JEM_RESTRICT  drow = d.addr ();

  const T* JEM_RESTRICT  acol1;
  const T* JEM_RESTRICT  brow1;
  const T* JEM_RESTRICT  ccol1;
  const T* JEM_RESTRICT  drow1;

  T*       JEM_RESTRICT  vout;
  T*       JEM_RESTRICT  tcol;
  T*       JEM_RESTRICT  vcol;

  const idx_t  ast = a.stride (0);
  const idx_t  bst = b.stride (1);
  const idx_t  cst = c.stride (0);
  const idx_t  dst = d.stride (1);

  const idx_t  m   = a.size   (0);
  const idx_t  n   = a.size   (1);
  const idx_t  p   = b.size   (1);
  const idx_t  q   = c.size   (1);
  const idx_t  r   = d.size   (1);

  T            x, y, z;
  idx_t        i, j;


  vresize_ ( m, p, r );

  vout = vout_.addr ();
  tcol = tmp_ .addr ();

  if ( v.stride() == 1 )
  {
    vcol = v.addr ();
  }
  else
  {
    vcol = args_.addr ();

    for ( i = 0; i < r; i++ )
    {
      vcol[i] = v[i];
    }
  }

  x = NumberTraits<T>::zero ();

  for ( i = 0; i < p; i++ )
  {
    tcol[i] = x;
  }

  for ( i = 0; i < m; i++ )
  {
    vout[i] = x;
  }

  // Compute vout_ = a * b * c * d * v

  for ( i = 0; i < q - 1; i += 2_idx )
  {
    x = y = NumberTraits<T>::zero ();
    drow1 = drow + d.stride (0);
    ccol1 = ccol + c.stride (1);

    for ( j = 0; j < r; j++ )
    {
      z  = vcol [j];
      x += drow [j * dst] * z;
      y += drow1[j * dst] * z;
    }

JEM_IVDEP

    for ( j = 0; j < p; j++ )
    {
      tcol[j] += ccol[j * cst] * x + ccol1[j * cst] * y;
    }

    drow = drow1 + d.stride (0);
    ccol = ccol1 + c.stride (1);
  }

  if ( i < q )
  {
    x = NumberTraits<T>::zero ();

    for ( j = 0; j < r; j++ )
    {
      x += drow[j * dst] * vcol[j];
    }

JEM_IVDEP

    for ( j = 0; j < p; j++ )
    {
      tcol[j] += ccol[j * cst] * x;
    }
  }

  for ( i = 0; i < n - 1; i += 2_idx )
  {
    x = y = NumberTraits<T>::zero ();
    brow1 = brow + b.stride (0);
    acol1 = acol + a.stride (1);

    for ( j = 0; j < p; j++ )
    {
      z  = tcol [j];
      x += brow [j * bst] * z;
      y += brow1[j * bst] * z;
    }

JEM_IVDEP

    for ( j = 0; j < m; j++ )
    {
      vout[j] += acol[j * ast] * x + acol1[j * ast] * y;
    }

    brow = brow1 + b.stride (0);
    acol = acol1 + a.stride (1);
  }

  if ( i < n )
  {
    x = NumberTraits<T>::zero ();

    for ( j = 0; j < p; j++ )
    {
      x += brow[j * bst] * tcol[j];
    }

JEM_IVDEP

    for ( j = 0; j < m; j++ )
    {
      vout[j] += acol[j * ast] * x;
    }
  }

  return vout_;
}


//-----------------------------------------------------------------------
//   matmul (vector * matrix * matrix * matrix * matrix)
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T>& MatmulChain<T,4>::matmul

  ( const Array<T>&    v,
    const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T,2>&  c,
    const Array<T,2>&  d )

{
  return matmul ( d.transpose(), c.transpose(),
                  b.transpose(), a.transpose(), v );
}


//-----------------------------------------------------------------------
//   matmul (vector * matrix * matrix * matrix * matrix * vector)
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T,2>& MatmulChain<T,4>::matmul

  ( const Array<T>&    v,
    const Array<T,2>&  a,
    const Array<T,2>&  b,
    const Array<T,2>&  c,
    const Array<T,2>&  d,
    const Array<T>&    w )

{
  return vecmul_ ( mout_, v, matmul( a, b, c, d, w ) );
}


//-----------------------------------------------------------------------
//   mresize_
//-----------------------------------------------------------------------


template <class T>

  inline void MatmulChain<T,4>::mresize_

  ( idx_t m, idx_t n, idx_t p, idx_t q, idx_t r )

{
  if ( m != mout_.size(0) || r != mout_.size(1) )
  {
    mout_.resize ( m, r );
  }

  if ( n > tmp_.size() )
  {
    tmp_.resize ( n );
  }

  if ( (n * (m + p) + q * (p + r)) > args_.size() )
  {
    args_.resize ( n * (m + p) + q * (p + r) );
  }
}


//-----------------------------------------------------------------------
//   vresize_
//-----------------------------------------------------------------------


template <class T>

  inline void MatmulChain<T,4>::vresize_

  ( idx_t m, idx_t p, idx_t r )

{
  if ( m != vout_.size() )
  {
    vout_.resize ( m );
  }

  if ( p > tmp_.size() )
  {
    tmp_.resize ( p );
  }

  if ( r > args_.size() )
  {
    args_.resize ( r );
  }
}


JEM_END_PACKAGE( numeric )
