
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
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Float.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/NumberTraits.h>
#include <jem/numeric/algebra/EigenUtils.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class EigenUtils
//=======================================================================

//-----------------------------------------------------------------------
//   symSolve (without eigenvectors)
//-----------------------------------------------------------------------


void EigenUtils::symSolve

  ( const Vector&  evals,
    const Matrix&  symmat )

{
  JEM_PRECHECK2 ( evals.size() == symmat.size(0) &&
                  evals.size() == symmat.size(1),
                  "Array size mismatch" );

  const idx_t  n = evals.size ();

  Vector  sd ( n );


  if ( evals.isContiguous() && symmat.isContiguous() )
  {
    triReduce_ ( n, evals.addr(), sd.addr(), symmat.addr() );
    triSolve_  ( n, evals.addr(), sd.addr(), nullptr       );
  }
  else
  {
    Vector  dd ( makeContiguous( evals  ) );
    Matrix  aa ( makeContiguous( symmat ) );

    triReduce_ ( n, dd.addr(), sd.addr(), aa.addr() );
    triSolve_  ( n, dd.addr(), sd.addr(), nullptr   );

    if ( dd.addr() != evals.addr() )
    {
      evals = dd;
    }
  }
}


//-----------------------------------------------------------------------
//   symSolve (with eigenvectors)
//-----------------------------------------------------------------------


void EigenUtils::symSolve

  ( const Vector&  evals,
    const Matrix&  evecs,
    const Matrix&  symmat )

{
  JEM_PRECHECK2 ( evals.size() == evecs .size(0) &&
                  evals.size() == evecs .size(1) &&
                  evals.size() == symmat.size(0) &&
                  evals.size() == symmat.size(1),
                  "Array size mismatch" );

  const idx_t  n = evals.size ();

  Vector  sd ( n );


  evecs = symmat;

  if ( evals.isContiguous() && evecs.isContiguous() )
  {
    triReduce_ ( n, evals.addr(), sd.addr(), evecs.addr() );
    triSolve_  ( n, evals.addr(), sd.addr(), evecs.addr() );
  }
  else
  {
    Vector  dd ( makeContiguous( evals ) );
    Matrix  aa ( makeContiguous( evecs ) );

    triReduce_ ( n, dd.addr(), sd.addr(), aa.addr() );
    triSolve_  ( n, dd.addr(), sd.addr(), aa.addr() );

    if ( dd.addr() != evals.addr() )
    {
      evals = dd;
    }

    if ( aa.addr() != evecs.addr() )
    {
      evecs = aa;
    }
  }
}


//-----------------------------------------------------------------------
//   triSolve (without eigenvectors)
//-----------------------------------------------------------------------


void EigenUtils::triSolve

  ( const Vector&  diag,
    const Vector&  subdiag )

{
  JEM_PRECHECK2 ( diag.size() == subdiag.size(), "Array size mismatch" );

  const idx_t  n = diag.size ();


  if ( diag.isContiguous() && subdiag.isContiguous() )
  {
    triSolve_ ( n, diag.addr(), subdiag.addr(), nullptr );
  }
  else
  {
    Vector  dd ( makeContiguous( diag    ) );
    Vector  sd ( makeContiguous( subdiag ) );

    triSolve_ ( n, dd.addr(), sd.addr(), nullptr );

    if ( dd.addr() != diag.addr() )
    {
      diag = dd;
    }
  }
}


//-----------------------------------------------------------------------
//   triSolve (with eigenvectors)
//-----------------------------------------------------------------------


void EigenUtils::triSolve

  ( const Vector&  diag,
    const Vector&  subdiag,
    const Matrix&  evecs )

{
  JEM_PRECHECK2 ( diag.size() == subdiag.size()  &&
                  diag.size() == evecs  .size(0) &&
                  diag.size() == evecs  .size(1),
                  "Array size mismatch" );

  const idx_t  n = diag.size ();


  if ( diag .isContiguous() && subdiag.isContiguous() &&
       evecs.isContiguous() )
  {
    triSolve_ ( n, diag.addr(), subdiag.addr(), evecs.addr() );
  }
  else
  {
    Vector  dd ( makeContiguous( diag    ) );
    Vector  sd ( makeContiguous( subdiag ) );
    Matrix  ev ( makeContiguous( evecs   ) );

    triSolve_ ( n, dd.addr(), sd.addr(), ev.addr() );

    if ( dd.addr() != diag.addr() )
    {
      diag = dd;
    }

    if ( ev.addr() != evecs.addr() )
    {
      evecs = ev;
    }
  }
}


//-----------------------------------------------------------------------
//   triReduce
//-----------------------------------------------------------------------


void EigenUtils::triReduce

  ( const Vector&  diag,
    const Vector&  subdiag,
    const Matrix&  symmat )

{
  JEM_PRECHECK2 ( diag.size() == subdiag.size() &&
                  diag.size() == symmat.size(0) &&
                  diag.size() == symmat.size(1),
                  "Array size mismatch" );

  const idx_t  n = diag.size ();


  if ( diag  .isContiguous() && subdiag.isContiguous() &&
       symmat.isContiguous() )
  {
    triReduce_ ( n, diag.addr(), subdiag.addr(), symmat.addr() );
  }
  else
  {
    Vector  dd ( makeContiguous( diag    ) );
    Vector  sd ( makeContiguous( subdiag ) );
    Matrix  aa ( makeContiguous( symmat  ) );

    triReduce_ ( n, dd.addr(), sd.addr(), aa.addr() );

    if ( dd.addr() != diag.addr() )
    {
      diag = dd;
    }

    if ( sd.addr() != subdiag.addr() )
    {
      subdiag = sd;
    }

    if ( aa.addr() != symmat.addr() )
    {
      symmat = aa;
    }
  }
}


//-----------------------------------------------------------------------
//   triSolve_
//-----------------------------------------------------------------------


void EigenUtils::triSolve_

  ( idx_t                 n,
    double* JEM_RESTRICT  dd,
    double* JEM_RESTRICT  sd,
    double* JEM_RESTRICT  ev )

{
  const char*  CONTEXT  = "jem::numeric::EigenUtils::triSolve";

  const int    MAX_ITER = 40;

  double       a,  b, u, v, w;
  double       cs, sn;

  lint         flopCount;
  int          iiter;
  idx_t        ipiv;
  idx_t        i, j, k;
  idx_t        p, q;


  if ( n == 0 )
  {
    return;
  }

  for ( i = 1; i < n; i++ )
  {
    sd[i - 1] = sd[i];
  }

  sd[n - 1] = 0.0;
  flopCount = 0;

  for ( i = 0; i < n; i++ )
  {
    for ( iiter = 0; iiter < MAX_ITER; iiter++ )
    {
      for ( ipiv = i; ipiv < n - 1; ipiv++ )
      {
        a = abs( dd[ipiv] ) + abs( dd[ipiv + 1] );

        if ( abs( sd[ipiv] ) <= Limits<double>::EPSILON * a )
        {
          break;
        }
      }

      if ( ipiv == i )
      {
        break;
      }

      u = (dd[i + 1] - dd[i]) / (2.0 * sd[i]);
      v = modulus ( u, 1.0 );
      u = dd[ipiv] - dd[i] + sd[i] / (u + sign( v, u ));

      w  = 0.0;
      cs = sn = 1.0;

      for ( j = ipiv - 1; j >= i; j-- )
      {
        a = sn * sd[j];
        b = cs * sd[j];
        v = modulus ( a, u );

        sd[j + 1] = v;

        if ( isTiny( v ) )
        {
          dd[j + 1] -= w;
          sd[ipiv]   = 0.0;
          break;
        }

        sn = a / v;
        cs = u / v;
        u  = dd[j + 1] - w;
        v  = (dd[j] - u) * sn + 2.0 * cs * b;
        w  = sn * v;

        dd[j + 1] = u + w;

        u  = cs * v - b;

        if ( ev )
        {
          p = j * n;
          q = p + n;

JEM_IVDEP

          for ( k = 0; k < n; k++ )
          {
            a         =      ev[k + q];
            ev[k + q] = sn * ev[k + p] + cs * a;
            ev[k + p] = cs * ev[k + p] - sn * a;
          }

          flopCount += 4_lint * (lint) n;
        }
      }

      if ( ! isTiny( v ) || j < i )
      {
        dd[i]   -= w;
        sd[i]    = u;
        sd[ipiv] = 0.0;
      }

      flopCount += (10_lint * ipiv);
    }

    if ( iiter >= MAX_ITER )
    {
      throw ArithmeticException (
        CONTEXT,
        "failed to determine the eigenvalues of a tri-"
        "diagonal matrix"
      );
    }

    if ( n > 10 )
    {
      testCancelled ( flopCount, CONTEXT,
                      "tri-diagonal QL algorithm cancelled" );
    }
  }
}


//-----------------------------------------------------------------------
//   triReduce_
//-----------------------------------------------------------------------


void EigenUtils::triReduce_

  ( idx_t                 n,
    double* JEM_RESTRICT  dd,
    double* JEM_RESTRICT  sd,
    double* JEM_RESTRICT  aa )

{
  const char*  CONTEXT = "jem::numeric::EigenUtils::triReduce";

  double*      JEM_RESTRICT  ai;
  double*      JEM_RESTRICT  aj;

  double       scale;
  double       u, v, w, t;

  lint         flopCount;
  idx_t        i, j, k;


  if ( n == 0 )
  {
    return;
  }

  flopCount = 0;

  for ( i = n - 1; i > 1; i-- )
  {
    ai    = aa + (i * n);
    scale = 0.0;

    for ( j = 0; j < i; j++ )
    {
      scale += abs ( ai[j] );
    }

    if ( Float::isNaN( scale ) )
    {
      throw ArithmeticException (
        CONTEXT,
        "invalid floating point number: NaN"
      );
    }

    if ( isTiny( scale ) )
    {
      sd[i] = ai[i - 1];
      dd[i] = 0.0;

      continue;
    }

    scale = 1.0 / scale;
    u     = 0.0;

    for ( j = 0; j < i; j++ )
    {
      t     = ai[j] * scale;
      u    += t * t;
      ai[j] = t;
    }

    v     = ai[i - 1];
    w     = (v < 0.0 ? std::sqrt( u ) : -std:: sqrt( u ));
    sd[i] = w / scale;
    u    -= v * w;

    ai[i - 1] = v - w;

    v     = 0.0;
    t     = 1.0 / u;

    for ( j = 0; j < i; j++ )
    {
      aj    = aa + (j * n);
      aj[i] = ai[j] * t;
      w     = 0.0;

      for ( k = 0; k <= j; k++ )
      {
        w += aj[k] * ai[k];
      }

      for ( k = j + 1; k < i; k++ )
      {
        w += aa[j + k * n] * ai[k];
      }

      sd[j] = w * t;
      v    += sd[j] * ai[j];
    }

    t = v / (u + u);

    for ( j = 0; j < i; j++ )
    {
      aj     = aa + (j * n);
      v      = ai[j];
      sd[j] -= t * v;
      w      = sd[j];

JEM_IVDEP

      for ( k = 0; k <= j; k++ )
      {
        aj[k] -= (v * sd[k] + w * ai[k]);
      }
    }

    dd[i] = u;

    if ( n > 10 )
    {
      flopCount += (2_lint * i * i);

      testCancelled (
        flopCount, CONTEXT,
        "tri-diagonal reduction algorithm cancelled"
      );
    }
  }

  // Transpose the matrix aa.

  for ( i = 0; i < n; i++ )
  {
    for ( j = 0; j < i; j++ )
    {
      u             = aa[i + j * n];
      aa[i + j * n] = aa[j + i * n];
      aa[j + i * n] = u;
    }
  }

  sd[0] = 0.0;
  dd[0] = 0.0;

  if ( n > 1 )
  {
    sd[1] = aa[1];
    dd[1] = 0.0;
  }

  for ( i = 0; i < n; i++ )
  {
    ai = aa + (i * n);

    if ( isTiny( sd[i] ) )
    {
      // This is to avoid numerical problems in the case that a
      // matrix is completely filled with very small values.

      sd[i] = 0.0;
    }

    if ( ! isTiny( dd[i] ) )
    {
      for ( j = 0; j < i; j++ )
      {
        aj = aa + (j * n);
        u  = 0.0;

        for ( k = 0; k < i; k++ )
        {
          u += aa[i + k * n] * aj[k];
        }

JEM_IVDEP

        for ( k = 0; k < i; k++ )
        {
          aj[k] -= u * ai[k];
        }
      }
    }

    dd[i] = ai[i];
    ai[i] = 1.0;

    for ( j = 0; j < i; j++ )
    {
      ai[j] = aa[i + j * n] = 0.0;
    }
  }
}


JEM_END_PACKAGE( numeric )
