
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


#include <cstring>
#include <jem/pragmas.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/algebra/LUSolver.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class LUSolver
//=======================================================================

//-----------------------------------------------------------------------
//   factor (Array)
//-----------------------------------------------------------------------


bool LUSolver::factor

  ( const Matrix&     a,
    const Vector&     scale,
    const IdxVector&  jperm,
    int&              sign )

{
  JEM_PRECHECK2 ( a    .size(0) == a.size(1) &&
                  scale.size()  >= a.size(1) &&
                  jperm.size()  >= a.size(1),
                  "Array size mismatch" );

  const idx_t  n = a.size (0);


  if ( a.isContiguous() &&
       scale.stride() + jperm.stride() == 2_lint )
  {
    return factor_ ( n, sign, a.addr(),  scale.addr(), jperm.addr() );
  }
  else
  {
    Matrix     b  ( makeContiguous( a     ) );
    Vector     sc ( makeContiguous( scale ) );
    IdxVector  jp ( makeContiguous( jperm ) );

    bool       result;

    result = factor_ ( n, sign, b.addr(), sc.addr(), jp.addr() );

    if ( result )
    {
      if ( a.addr() != b.addr() )
      {
        a = b;
      }
      if ( jperm.addr() != jp.addr() )
      {
        jperm = jp;
      }
    }

    return result;
  }
}


//-----------------------------------------------------------------------
//   solve (Array)
//-----------------------------------------------------------------------


void LUSolver::solve

  ( const Vector&     lhs,
    const Matrix&     lu,
    const Vector&     rhs,
    const IdxVector&  jperm )

{
  JEM_PRECHECK2  ( lhs.size()  == lu   .size(0) &&
                   lhs.size()  == lu   .size(1) &&
                   lhs.size()  == rhs  .size()  &&
                   lhs.size()  <= jperm.size(),
                   "Array size mismatch" );

  const idx_t  n = lhs.size ();


  lhs = rhs;

  if ( lu.isContiguous() &&
       lhs.stride() + jperm.stride() == 2_lint )
  {
    solve1_ ( n, 0, lhs.addr(), lu.addr(), jperm.addr() );
  }
  else
  {
    Vector     x  ( makeContiguous( lhs   ) );
    Matrix     a  ( makeContiguous( lu    ) );
    IdxVector  jp ( makeContiguous( jperm ) );

    solve1_ ( n, 0, x.addr(), a.addr(), jp.addr() );

    if ( x.addr() != lhs.addr() )
    {
      lhs = x;
    }
  }
}


void LUSolver::solve

  ( const Matrix&     lhs,
    const Matrix&     lu,
    const Matrix&     rhs,
    const IdxVector&  jperm )

{
  JEM_PRECHECK2 ( lhs.size(0) == lu   .size(0) &&
                  lhs.size(0) == lu   .size(1) &&
                  lhs.size(0) == rhs  .size(0) &&
                  lhs.size(1) == rhs  .size(1) &&
                  lhs.size(0) <= jperm.size(),
                  "Array size mismatch" );

  const idx_t  m = lhs.size (0);
  const idx_t  n = lhs.size (1);


  lhs = rhs;

  if ( lu.isContiguous() &&
       lhs.stride(0) + jperm.stride() == 2_lint )
  {
    msolve_ ( m, n, lhs.addr(), lu.addr(), jperm.addr() );
  }
  else
  {
    Matrix     x  ( makeContiguous( lhs   ) );
    Matrix     a  ( makeContiguous( lu    ) );
    IdxVector  jp ( makeContiguous( jperm ) );

    msolve_ ( m, n, x.addr(), a.addr(), jp.addr() );

    if ( lhs.addr() != x.addr() )
    {
      lhs = x;
    }
  }
}


//-----------------------------------------------------------------------
//   invert (Array)
//-----------------------------------------------------------------------


bool LUSolver::invert

  ( const Matrix&  a,
    double&        d )

{
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  const idx_t  n = a.size(0);

  bool         result;


  if ( ! a.isContiguous() )
  {
    Matrix  tmp ( a.clone() );

    if ( invert( tmp, d ) )
    {
      a = tmp;

      return true;
    }
    else
    {
      return false;
    }
  }

  switch ( n )
  {
  case 0:

    result = true;

    break;

  case 1:

    d      = a(0,0);
    result = ! jem::isTiny( d );

    if ( result )
    {
      a(0,0) = 1.0 / d;
    }

    break;

  case 2:

    result = invert2x2_ ( a.addr(), d );

    break;

  case 3:

    result = invert3x3_ ( a.addr(), d );

    break;

  case 4:
  case 5:
  case 6:

    {
      double  b[36];

      std::memcpy ( b, a.addr(), (size_t) (n * n) * sizeof(double) );

      result = gaussi_ ( n, a.addr(), b, d );
    }

    break;

  default:

    {
      Matrix     b     ( n, n );
      Vector     scale ( n );
      IdxVector  jperm ( n );

      b = a;

      result = invert_ ( n,  a.addr(), b.addr(),
                         scale.addr(), jperm.addr(), d );
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   det (Array)
//-----------------------------------------------------------------------


double LUSolver::det ( const Array<double,2>& lu )
{
  JEM_PRECHECK2 ( lu.size(0) == lu.size(1), "non-square matrix" );

  const idx_t  n = lu.size(0);

  double       d = 1.0;

  for ( idx_t i = 0; i < n; i++ )
  {
    d *= lu(i,i);
  }

  return (1.0 / d);
}


//-----------------------------------------------------------------------
//   invert3x3_
//-----------------------------------------------------------------------


bool LUSolver::invert3x3_

  ( double* JEM_RESTRICT  a,
    double&               d )

{
  // Matrix layout:
  //
  //       | a00 a01 a02 |   | a[0] a[3] a[6] |
  //   a = | a10 a11 a12 | = | a[1] a[4] a[7] |
  //       | a20 a21 a22 |   | a[2] a[5] a[8] |

  double  c00, c01, c02;
  double  c10, c11, c12;
  double  c20, c21, c22;

  double  s;

  // Compute the co-factors:

  c00 = a[4] * a[8] - a[5] * a[7];
  c10 = a[5] * a[6] - a[3] * a[8];
  c20 = a[3] * a[7] - a[4] * a[6];

  c01 = a[2] * a[7] - a[1] * a[8];
  c11 = a[0] * a[8] - a[2] * a[6];
  c21 = a[1] * a[6] - a[0] * a[7];

  c02 = a[1] * a[5] - a[2] * a[4];
  c12 = a[2] * a[3] - a[0] * a[5];
  c22 = a[0] * a[4] - a[1] * a[3];

  // Compute the determinant:

  d = a[0] * c00 + a[1] * c10 + a[2] * c20;

  if ( jem::isTiny( d ) )
  {
    return false;
  }

  s = 1.0 / d;

  a[0] = s * c00;
  a[1] = s * c01;
  a[2] = s * c02;

  a[3] = s * c10;
  a[4] = s * c11;
  a[5] = s * c12;

  a[6] = s * c20;
  a[7] = s * c21;
  a[8] = s * c22;

  return true;
}


//-----------------------------------------------------------------------
//   factor_
//-----------------------------------------------------------------------


bool LUSolver::factor_

  ( idx_t                 n,
    int&                  sign,
    double* JEM_RESTRICT  aa,
    double* JEM_RESTRICT  scale,
    idx_t*  JEM_RESTRICT  jperm )

{
  double*  JEM_RESTRICT  acol;
  double*  JEM_RESTRICT  acol1;
  double*  JEM_RESTRICT  acol2;
  double*  JEM_RESTRICT  acol3;
  double*  JEM_RESTRICT  pcol;

  double   s, t;
  double   u, v, w, x, z;

  lint     flopCount;
  idx_t    i, j, k, p;


  sign      = 1;
  flopCount = 0;
  acol      = aa;

  for ( j = 0; j < n; j++ )
  {
    s = 0.0;

    for ( i = 0; i < n; i++ )
    {
      t = (double) abs ( acol[i] );

      if ( t > s )
      {
        s = t;
      }
    }

    if ( isTiny( s ) )
    {
      return false;
    }

    scale[j] = 1.0 / s;
    acol    += n;
  }

  // Main loop:

  for ( k = 0; k < n; k++ )
  {
    pcol = aa + (k * n);

    // Find the pivot

    p = k;
    s = scale[k] * abs( pcol[k] );

    for ( j = k + 1; j < n; j++ )
    {
      t = scale[j] * abs( aa[k + j * n] );

      if ( t > s )
      {
        s = t;
        p = j;
      }
    }

    // Check for singular matrix

    if ( s <= Limits<double>::EPSILON )
    {
      return false;
    }

    // Store the column index

    jperm[k] = p;

    // Swap columns if necessary

    if ( p != k )
    {
      acol = aa + (p * n);

      for ( i = 0; i < n; i++ )
      {
        x       = acol[i];
        acol[i] = pcol[i];
        pcol[i] = x;
      }

      sign = -sign;
    }

    // Compute the k-th column of L

    x = pcol[k] = 1.0 / pcol[k];

    for ( i = k + 1; i < n; i++ )
    {
      pcol[i] *= x;
    }

    // Factor the remaining (n-k) x (n-k) submatrix

    acol = pcol + n;

    for ( j = n - k - 1; j > 3; j -= 4 )
    {
      acol1 =  acol  + n;
      acol2 =  acol1 + n;
      acol3 =  acol2 + n;

      u     = -acol [k];
      v     = -acol1[k];
      w     = -acol2[k];
      x     = -acol3[k];

JEM_IVDEP

      for ( i = k + 1; i < n; i++ )
      {
        z         = pcol[i];
        acol [i] += z * u;
        acol1[i] += z * v;
        acol2[i] += z * w;
        acol3[i] += z * x;
      }

      acol = acol3 + n;
    }

    switch ( j )
    {
    case 0:

      break;

    case 1:

      u = -acol[k];

JEM_IVDEP

      for ( i = k + 1; i < n; i++ )
      {
        acol[i] += u * pcol[i];
      }

      break;

    case 2:

      acol1 =  acol  + n;

      u     = -acol [k];
      v     = -acol1[k];

JEM_IVDEP

      for ( i = k + 1; i < n; i++ )
      {
        z         = pcol[i];
        acol [i] += u * z;
        acol1[i] += v * z;
      }

      break;

    case 3:

      acol1 =  acol  + n;
      acol2 =  acol1 + n;

      u     = -acol [k];
      v     = -acol1[k];
      w     = -acol2[k];

JEM_IVDEP

      for ( i = k + 1; i < n; i++ )
      {
        z         = pcol[i];
        acol [i] += u * z;
        acol1[i] += v * z;
        acol2[i] += w * z;
      }

      break;
    }

    flopCount = flopCount + (n - k) * (n - k);

    if ( flopCount > 1000000_lint )
    {
      testCancelled ( flopCount, JEM_FUNC,
                      "LU factorization cancelled" );
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   solve1_
//-----------------------------------------------------------------------


void LUSolver::solve1_

  ( idx_t                       n,
    idx_t                       k,
    double*       JEM_RESTRICT  lhs,
    const double* JEM_RESTRICT  lu,
    const idx_t*  JEM_RESTRICT  jperm ) noexcept

{
  const double* JEM_RESTRICT  lcol;
  const double* JEM_RESTRICT  ucol;

  double  u;
  idx_t   i, j;


  lcol = lu + (k * n);

  for ( j = k; j < n; j++ )
  {
    u = -lhs[j];

    for ( i = j + 1; i < n; i++ )
    {
      lhs[i] += u * lcol[i];
    }

    lcol += n;
  }

  ucol = lu + (n * n);

  for ( j = n - 1; j >= 0; j-- )
  {
    ucol  -= n;

    lhs[j] =  lhs[j] * ucol[j];
    u      = -lhs[j];

    for ( i = j - 1; i >= 0; i-- )
    {
      lhs[i] += u * ucol[i];
    }
  }

  for ( i = n - 1; i >= 0; i-- )
  {
    j = jperm[i];

    if ( j != i )
    {
      u      = lhs[i];
      lhs[i] = lhs[j];
      lhs[j] = u;
    }
  }
}


//-----------------------------------------------------------------------
//   solve2_
//-----------------------------------------------------------------------


void LUSolver::solve2_

  ( idx_t                       n,
    idx_t                       k,
    double*       JEM_RESTRICT  lhs,
    const double* JEM_RESTRICT  lu,
    const idx_t*  JEM_RESTRICT  jperm ) noexcept

{
  double*       JEM_RESTRICT  lhs1 = lhs + n;

  const double* JEM_RESTRICT  lcol;
  const double* JEM_RESTRICT  ucol;

  double  u, v, z;
  idx_t   i, j;


  lcol = lu + (k * n);

  for ( j = k; j < n; j++ )
  {
    u = -lhs [j];
    v = -lhs1[j];

JEM_IVDEP

    for ( i = j + 1; i < n; i++ )
    {
      z        = lcol[i];
      lhs [i] += u * z;
      lhs1[i] += v * z;
    }

    lcol += n;
  }

  ucol = lu + (n * n);

  for ( j = n - 1; j >= 0; j-- )
  {
    ucol   -= n;

    z       =  ucol[j];
    lhs [j] =  lhs [j] * z;
    lhs1[j] =  lhs1[j] * z;
    u       = -lhs [j];
    v       = -lhs1[j];

JEM_IVDEP

    for ( i = j - 1; i >= 0; i-- )
    {
      z        = ucol[i];
      lhs [i] += u * z;
      lhs1[i] += v * z;
    }
  }

  for ( i = n - 1; i >= 0; i-- )
  {
    j = jperm[i];

    if ( j != i )
    {
      u       = lhs [i];
      v       = lhs1[i];
      lhs [i] = lhs [j];
      lhs1[i] = lhs1[j];
      lhs [j] = u;
      lhs1[j] = v;
    }
  }
}


//-----------------------------------------------------------------------
//   msolve_
//-----------------------------------------------------------------------


void LUSolver::msolve_

  ( idx_t                       m,
    idx_t                       n,
    double*       JEM_RESTRICT  lhs,
    const double* JEM_RESTRICT  lu,
    const idx_t*  JEM_RESTRICT  jperm ) noexcept

{
  idx_t  j;

  for ( j = 0; j < n - 1; j += 2 )
  {
    solve2_ ( m, 0, lhs, lu, jperm );

    lhs += (m * 2);
  }

  if ( j < n )
  {
    solve1_ ( m, 0, lhs, lu, jperm );
  }
}


//-----------------------------------------------------------------------
//   invert_
//-----------------------------------------------------------------------


bool LUSolver::invert_

  ( idx_t                 n,
    double* JEM_RESTRICT  aa,
    double* JEM_RESTRICT  bb,
    double* JEM_RESTRICT  scale,
    idx_t*  JEM_RESTRICT  jperm,
    double&               dd )

{
  double*  JEM_RESTRICT  acol;

  int      sign;
  idx_t    j;


  if ( ! factor_( n, sign, bb, scale, jperm ) )
  {
    dd = 0.0; return false;
  }

  dd = (double) sign;

  for ( j = 0; j < n; j++ )
  {
    dd *= bb[j + j * n];
  }

  dd = 1.0 / dd;

  for ( j = 0; j < n * n; j++ )
  {
    aa[j] = 0.0;
  }

  acol = aa;

  for ( j = 0; j < n - 1; j += 2 )
  {
    acol[j]         = 1.0;
    acol[j + n + 1] = 1.0;

    solve2_ ( n, j, acol, bb, jperm );

    acol += (2 * n);
  }

  if ( j < n )
  {
    acol[j] = 1.0;

    solve1_ ( n, j, acol, bb, jperm );
  }

  return true;
}


//-----------------------------------------------------------------------
//   gaussi_
//-----------------------------------------------------------------------

bool LUSolver::gaussi_

  ( idx_t                 sz,
    double* JEM_RESTRICT  aa,
    double* JEM_RESTRICT  bb,
    double&               dd )

{
  const idx_t  n  = sz;
  const idx_t  nn = n * n;

  double       s, t;
  idx_t        ij, kj, kn;
  idx_t        i,  k,  p;


  for ( i = 0; i < nn; i++ )
  {
    aa[i] = 0.0;
  }

  dd = 1.0;

  // Scale the matrices so that largest absolute value on each
  // row equals 1.0.

  for ( i = 0; i < n; i++ )
  {
    s = std::fabs ( bb[i] );

    for ( ij = i + n; ij < nn; ij += n )
    {
      t = std::fabs ( bb[ij] );

      if ( t > s )
      {
        s = t;
      }
    }

    if ( isTiny( s ) )
    {
      dd = 0.0; return false;
    }

    dd *= s;
    s   = 1.0 / s;

    for ( ij = i; ij < nn; ij += n )
    {
      bb[ij] *= s;
    }

    aa[i + i * n] = s;
  }

  // Simple Gauss elimination with row pivoting.

  for ( k = 0; k < n; k++ )
  {
    kn = k * n;
    p  = k;
    s  = std::fabs ( bb[k + kn] );

    for ( i = k + 1; i < n; i++ )
    {
      t = std::fabs ( bb[i + kn] );

      if ( t > s )
      {
        p = i;
        s = t;
      }
    }

    if ( s < Limits<double>::EPSILON )
    {
      dd = 0.0; return false;
    }

    if ( p != k )
    {
      // Swap rows k and p.

      for ( ij = p, kj = k; ij < kn; ij += n, kj += n )
      {
        t      = aa[kj];
        aa[kj] = aa[ij];
        aa[ij] = t;
      }

      for ( ; ij < nn; ij += n, kj += n )
      {
        t      = aa[kj];
        aa[kj] = aa[ij];
        aa[ij] = t;

        t      = bb[kj];
        bb[kj] = bb[ij];
        bb[ij] = t;
      }

      dd = -dd;
    }

    dd *= bb[k + kn];
    s   = 1.0 / bb[k + kn];

    for ( kj = k; kj < (kn + n); kj += n )
    {
      aa[kj] *= s;
    }

    for ( ; kj < nn; kj += n )
    {
      aa[kj] *= s;
      bb[kj] *= s;
    }

    for ( i = k + 1; i < n; i++ )
    {
      t = -bb[i + kn];

      for ( ij = i, kj = k; ij < (kn + n); ij += n, kj += n )
      {
        aa[ij] += t * aa[kj];
      }

      for ( ; ij < nn; ij += n, kj += n )
      {
        aa[ij] += t * aa[kj];
        bb[ij] += t * bb[kj];
      }
    }
  }

  // Backward substitution

  for ( k = n - 1; k > 0; k-- )
  {
    kn = k * n;

    for ( i = k - 1; i >= 0; i-- )
    {
      t  = -bb[i + kn];

      for ( ij = i, kj = k; ij < nn; ij += n, kj += n )
      {
        aa[ij] += t * aa[kj];
      }
    }
  }

  return true;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   det (Array)
//-----------------------------------------------------------------------


double det ( const Array<double,2>& a )
{
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  const idx_t  n = a.size(0);

  double       d;

  switch ( n )
  {
  case 0:

    d = 0.0;

    break;

  case 1:

    d = a(0,0);

    break;

  case 2:

    d = (a(0,0) * a(1,1) - a(0,1) * a(1,0));

    break;

  case 3:

    d = (a(0,0) * ( a(1,1) * a(2,2) - a(1,2) * a(2,1) ) -
         a(1,0) * ( a(0,1) * a(2,2) - a(0,2) * a(2,1) ) +
         a(2,0) * ( a(0,1) * a(1,2) - a(0,2) * a(1,1) ));

    break;

  case 4:
  case 5:
  case 6:

    {
      double  lu    [36];
      double  scale [6];
      idx_t   jperm [6];

      int     sign;
      idx_t   i, j, k;


      if ( a.isContiguous() )
      {
        std::memcpy ( lu, a.addr(), (size_t) (n * n) * sizeof(double) );
      }
      else
      {
        k = 0;

        for ( j = 0; j < n; j++ )
        {
          for ( i = 0; i < n; i++ )
          {
            lu[k++] = a(i,j);
          }
        }
      }

      if ( ! LUSolver::factor_( n, sign, lu, scale, jperm ) )
      {
        d = 0.0;
      }
      else
      {
        d = (double) sign;

        for ( i = 0; i < n; i++ )
        {
          d *= lu[i * (n + 1)];
        }

        d = 1.0 / d;
      }
    }

    break;

  default:

    {
      Array<double,2>  lu    ( n, n );
      Array<double>    scale ( n );
      Array<idx_t>     jperm ( n );

      int              sign;

      lu = a;

      if ( LUSolver::factor( lu, scale, jperm, sign ) )
      {
        d = sign * LUSolver::det ( lu );
      }
      else
      {
        d = 0.0;
      }
    }
  }

  return d;
}


//-----------------------------------------------------------------------
//   inverse (Array)
//-----------------------------------------------------------------------


Array<double,2>  inverse ( const Array<double,2>& a )
{
  Array<double,2>  tmp ( a.clone() );

  if ( ! invert( tmp ) )
  {
    throw ArithmeticException ( JEM_FUNC, "singular matrix" );
  }

  return tmp;
}


JEM_END_PACKAGE( numeric )
