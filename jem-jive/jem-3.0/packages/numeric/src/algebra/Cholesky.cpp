
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
#include <jem/base/assert.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/algebra/Cholesky.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class Cholesky
//=======================================================================

//-----------------------------------------------------------------------
//   factor (Array)
//-----------------------------------------------------------------------


bool Cholesky::factor ( const Matrix& a )
{
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  if ( a.isContiguous() )
  {
    return factor_ ( a.size(0), a.addr() );
  }
  else
  {
    Matrix  b ( a.clone() );

    if ( factor_( b.size(0), b.addr() ) )
    {
      a = b;

      return true;
    }
    else
    {
      return false;
    }
  }
}


//-----------------------------------------------------------------------
//   fsub (Array)
//-----------------------------------------------------------------------


void Cholesky::fsub

  ( const Vector&  x,
    const Matrix&  a )

{
  JEM_PRECHECK2 ( x.size()  == a.size(0),
                  "matrix/vector size mismatch" );
  JEM_PRECHECK2 ( a.size(0) == a.size(1),
                  "non-square matrix" );

  if ( x.isContiguous() && a.isContiguous() )
  {
    fsub1_ ( x.size(), 0, x.addr(), a.addr() );
  }
  else
  {
    Vector  y ( makeContiguous( x ) );
    Matrix  b ( makeContiguous( a ) );

    fsub1_ ( y.size(), 0, y.addr(), b.addr() );

    if ( x.addr() != y.addr() )
    {
      x = y;
    }
  }
}


void Cholesky::fsub

  ( const Matrix&  x,
    const Matrix&  a )

{
  JEM_PRECHECK2 ( x.size(0) == a.size(0), "matrix shape mismatch" );
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  if ( a.isContiguous() && x.stride(0) == 1_idx )
  {
    mfsub_ ( x.size(0), x.size(1), x.addr(), a.addr() );
  }
  else
  {
    Matrix  y ( makeContiguous( x ) );
    Matrix  b ( makeContiguous( a ) );

    mfsub_ ( y.size(0), y.size(1), y.addr(), b.addr() );

    if ( x.addr() != y.addr() )
    {
      x = y;
    }
  }
}


//-----------------------------------------------------------------------
//   fsubt (Array)
//-----------------------------------------------------------------------


void Cholesky::fsubt

  ( const Matrix&  x,
    const Matrix&  a )

{
  JEM_PRECHECK2 ( x.size(1) == a.size(0), "matrix shape mismatch" );
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  if ( x.isContiguous() && a.isContiguous() )
  {
    fsubt_ ( x.size(0), x.size(1), x.addr(), a.addr() );
  }
  else
  {
    Matrix  y ( makeContiguous( x ) );
    Matrix  b ( makeContiguous( a ) );

    fsubt_ ( y.size(0), y.size(1), y.addr(), b.addr() );

    if ( x.addr() != y.addr() )
    {
      x = y;
    }
  }
}


//-----------------------------------------------------------------------
//   bsub (Array)
//-----------------------------------------------------------------------


void Cholesky::bsub

  ( const Vector&  x,
    const Matrix&  a )

{
  JEM_PRECHECK2 ( x.size()  == a.size(0), "matrix shape mismatch" );
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  if ( a.isContiguous() && x.isContiguous() )
  {
    bsub1_ ( x.size(), x.addr(), a.addr() );
  }
  else
  {
    Vector  y ( makeContiguous( x ) );
    Matrix  b ( makeContiguous( a ) );

    bsub1_ ( y.size(), y.addr(), b.addr() );

    if ( x.addr() != y.addr() )
    {
      x = y;
    }
  }
}


void Cholesky::bsub

  ( const Matrix&  x,
    const Matrix&  a )

{
  JEM_PRECHECK2 ( x.size(0) == a.size(0), "matrix shape mismatch" );
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  if ( a.isContiguous() && x.stride(0) == 1_idx )
  {
    mbsub_ ( x.size(0), x.size(1), x.addr(), a.addr() );
  }
  else
  {
    Matrix  y ( makeContiguous( x ) );
    Matrix  b ( makeContiguous( a ) );

    mbsub_ ( y.size(0), y.size(1), y.addr(), b.addr() );

    if ( x.addr() != y.addr() )
    {
      x = y;
    }
  }
}


//-----------------------------------------------------------------------
//   solve (Array)
//-----------------------------------------------------------------------


void Cholesky::solve

  ( const Vector&  x,
    const Matrix&  a )

{
  JEM_PRECHECK2 ( x.size()  == a.size(0),
                  "matrix/vector size mismatch" );
  JEM_PRECHECK2 ( a.size(0) == a.size(1),
                  "non-square matrix" );

  if ( a.isContiguous() && x.isContiguous() )
  {
    fsub1_ ( x.size(), 0, x.addr(), a.addr() );
    bsub1_ ( x.size(),    x.addr(), a.addr() );
  }
  else
  {
    Vector  y ( makeContiguous( x ) );
    Matrix  b ( makeContiguous( a ) );

    fsub1_ ( y.size(), 0, y.addr(), b.addr() );
    bsub1_ ( y.size(),    y.addr(), b.addr() );

    if ( x.addr() != y.addr() )
    {
      x = y;
    }
  }
}


void Cholesky::solve

  ( const Matrix&  x,
    const Matrix&  a )

{
  JEM_PRECHECK2 ( x.size(0) == a.size(0), "matrix shape mismatch" );
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  if ( a.isContiguous() && x.stride(0) == 1_idx )
  {
    mfsub_ ( x.size(0), x.size(1), x.addr(), a.addr() );
    mbsub_ ( x.size(0), x.size(1), x.addr(), a.addr() );
  }
  else
  {
    Matrix  y ( makeContiguous( x ) );
    Matrix  b ( makeContiguous( a ) );

    mfsub_ ( y.size(0), y.size(1), y.addr(), b.addr() );
    mbsub_ ( y.size(0), y.size(1), y.addr(), b.addr() );

    if ( x.addr() != y.addr() )
    {
      x = y;
    }
  }
}


//-----------------------------------------------------------------------
//   invert (Array)
//-----------------------------------------------------------------------


bool Cholesky::invert ( const Matrix& a )
{
  JEM_PRECHECK2 ( a.size(0) == a.size(1), "non-square matrix" );

  const idx_t  n = a.size (0);


  if ( ! a.isContiguous() )
  {
    Matrix  b ( a.clone() );

    if ( invert( b ) )
    {
      a = b;

      return true;
    }
    else
    {
      return false;
    }
  }

  if ( n <= 6 )
  {
    double  b[36];

    std::memcpy ( b, a.addr(), (size_t) (n * n) * sizeof(double) );

    return invert_ ( n, a.addr(), b );
  }
  else
  {
    Matrix  b ( n, n );

    b = a;

    return invert_ ( n, a.addr(), b.addr() );
  }
}


//-----------------------------------------------------------------------
//   factor_
//-----------------------------------------------------------------------


bool Cholesky::factor_

  ( idx_t                 n,
    double* JEM_RESTRICT  a )

{
  double*  JEM_RESTRICT  acol;
  double*  JEM_RESTRICT  lcol;

  double   t;

  lint     flopCount;
  idx_t    i, j, k;


  flopCount = 0;
  acol      = a;

  for ( k = 0; k < n; k++ )
  {
    lcol = a;

    for ( j = 0; j < k; j++ )
    {
      t = -acol[j];

      for ( i = 0; i < j; i++ )
      {
        t += lcol[i] * acol[i];
      }

      acol[j] = -t / lcol[j];
      lcol   +=  n;
    }

    t = 0.0;

    for ( i = 0; i < k; i++ )
    {
      t += acol[i] * acol[i];
    }

    t = acol[k] - t;

    if ( ! (t > 0.0) )
    {
      return false;
    }

    acol[k]    = std::sqrt ( t );
    acol      += n;
    flopCount += (lint) k * (lint) k / 2_lint;

    if ( flopCount > 1000000_lint )
    {
      testCancelled ( flopCount, JEM_FUNC,
                      "Cholesky factorization cancelled" );
    }
  }

  acol = a;

  for ( j = 0; j < n; j++ )
  {
    for ( i = j + 1; i < n; i++ )
    {
      acol[i] = a[i * n + j];
    }

    acol += n;
  }

  return true;
}


//-----------------------------------------------------------------------
//   fsub1_
//-----------------------------------------------------------------------


void Cholesky::fsub1_

  ( idx_t                       n,
    idx_t                       k,
    double*       JEM_RESTRICT  x,
    const double* JEM_RESTRICT  a ) noexcept

{
  double  t;
  idx_t   i, j;


  a += (k * n);

  for ( j = k; j < n; j++ )
  {
    t = -x[j];

    for ( i = k; i < j; i++ )
    {
      t += a[i] * x[i];
    }

    x[j] = -t / a[j];
    a   +=  n;
  }
}


//-----------------------------------------------------------------------
//   fsub2_
//-----------------------------------------------------------------------


void Cholesky::fsub2_

  ( idx_t                       n,
    idx_t                       k,
    double*       JEM_RESTRICT  x,
    const double* JEM_RESTRICT  a ) noexcept

{
  double*  JEM_RESTRICT  y = x + n;

  double   t, u;
  idx_t    i, j;


  a += (k * n);

  for ( j = k; j < n; j++ )
  {
    t = -x[j];
    u = -y[j];

    for ( i = k; i < j; i++ )
    {
      t += a[i] * x[i];
      u += a[i] * y[i];
    }

    x[j] = -t / a[j];
    y[j] = -u / a[j];
    a   +=  n;
  }
}


//-----------------------------------------------------------------------
//   mfsub_
//-----------------------------------------------------------------------


void Cholesky::mfsub_

  ( idx_t                       m,
    idx_t                       n,
    double*       JEM_RESTRICT  x,
    const double* JEM_RESTRICT  a ) noexcept

{
  idx_t  j;


  for ( j = 0; j < n - 1; j += 2 )
  {
    fsub2_ ( m, 0, x, a );

    x += (2 * m);
  }

  if ( j < n )
  {
    fsub1_ ( m, 0, x, a );
  }
}


//-----------------------------------------------------------------------
//   fsubt_
//-----------------------------------------------------------------------


void Cholesky::fsubt_

  ( idx_t                       m,
    idx_t                       n,
    double*       JEM_RESTRICT  x,
    const double* JEM_RESTRICT  a ) noexcept

{
  double*  JEM_RESTRICT  xrow;
  double*  JEM_RESTRICT  trow;
  double*  JEM_RESTRICT  trow1;

  double   t, u;
  idx_t    i, j, k;


  xrow = x;

  for ( k = 0; k < n; k++ )
  {
    trow  = x;

    for ( j = 0; j < k - 1; j += 2 )
    {
      trow1 = trow + m;

      t     = -a[j];
      u     = -a[j + 1];

JEM_IVDEP

      for ( i = 0; i < m; i++ )
      {
        xrow[i] += (t * trow[i] + u * trow1[i]);
      }

      trow = trow1 + m;
    }

    if ( j < k )
    {
      t = -a[j];

JEM_IVDEP

      for ( i = 0; i < m; i++ )
      {
        xrow[i] += t * trow[i];
      }
    }

    t = 1.0 / a[k];

    for ( i = 0; i < m; i++ )
    {
      xrow[i] *= t;
    }

    xrow += m;
    a    += n;
  }
}


//-----------------------------------------------------------------------
//   bsub1_
//-----------------------------------------------------------------------


void Cholesky::bsub1_

  ( idx_t                       n,
    double*       JEM_RESTRICT  x,
    const double* JEM_RESTRICT  a ) noexcept

{
  double  t;
  idx_t   i, j;


  a += (n * n);

  for ( j = n - 1; j >= 0; j-- )
  {
    a -=  n;
    t  = -x[j];

    for ( i = n - 1; i > j; i-- )
    {
      t += a[i] * x[i];
    }

    x[j] = -t / a[j];
  }
}


//-----------------------------------------------------------------------
//   bsub2_
//-----------------------------------------------------------------------


void Cholesky::bsub2_

  ( idx_t                       n,
    double*       JEM_RESTRICT  x,
    const double* JEM_RESTRICT  a ) noexcept

{
  double*  JEM_RESTRICT  y = x + n;

  double   t, u;
  idx_t    i, j;


  a += (n * n);

  for ( j = n - 1; j >= 0; j-- )
  {
    a -=  n;
    t  = -x[j];
    u  = -y[j];

    for ( i = n - 1; i > j; i-- )
    {
      t += a[i] * x[i];
      u += a[i] * y[i];
    }

    x[j] = -t / a[j];
    y[j] = -u / a[j];
  }
}


//-----------------------------------------------------------------------
//   mbsub_
//-----------------------------------------------------------------------


void Cholesky::mbsub_

  ( idx_t                       m,
    idx_t                       n,
    double*       JEM_RESTRICT  x,
    const double* JEM_RESTRICT  a ) noexcept

{
  idx_t  j;


  for ( j = 0; j < n - 1; j += 2 )
  {
    bsub2_ ( m, x, a );

    x += (2 * m);
  }

  if ( j < n )
  {
    bsub1_ ( m, x, a );
  }
}


//-----------------------------------------------------------------------
//   invert_
//-----------------------------------------------------------------------


bool Cholesky::invert_

  ( idx_t                  n,
    double*  JEM_RESTRICT  a,
    double*  JEM_RESTRICT  b )

{
  idx_t  j;

  if ( ! factor_( n, b ) )
  {
    return false;
  }

  for ( j = 0; j < n * n; j++ )
  {
    a[j] = 0.0;
  }

  for ( j = 0; j < n - 1; j += 2 )
  {
    a[j]         = 1.0;
    a[j + n + 1] = 1.0;

    fsub2_ ( n, j, a, b );
    bsub2_ ( n,    a, b );

    a += (2 * n);
  }

  if ( j < n )
  {
    a[j] = 1.0;

    fsub1_ ( n, j, a, b );
    bsub1_ ( n,    a, b );
  }

  return true;
}


JEM_END_PACKAGE( numeric )
