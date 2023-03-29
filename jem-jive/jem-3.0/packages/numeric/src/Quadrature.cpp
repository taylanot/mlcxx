
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/ArithmeticException.h>
#include <jem/numeric/Quadrature.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class Quadrature
//=======================================================================

//-----------------------------------------------------------------------
//   gaussLegendre
//-----------------------------------------------------------------------


void Quadrature::gaussLegendre

  ( const Array<double>&  x,
    const Array<double>&  w )

{
  JEM_PRECHECK2 ( x.size() == w.size(), "Array size mismatch" );
  JEM_PRECHECK2 ( x.size() > 0, "number of Gauss points must "
                  "be larger than zero" );

  const idx_t  MAX_ITER = 10;

  const idx_t  n = x.size ();
  const idx_t  m = (n + 1) / 2;

  double       z, z1;
  double       p, p1, p2, p3;
  idx_t        i, j, k;


  for ( i = 1; i <= m; i++ )
  {
    k = 0;
    z = std::cos ( 3.14159265358979 * ((double) i - 0.25) /
                                      ((double) n + 0.50) );
    do
    {
      p1 = 1.0;
      p2 = 0.0;

      for ( j = 1; j <= n; j++ )
      {
        p3 = p2;
        p2 = p1;
        p1 = ((double) (2 * j - 1) * z * p2 -
              (double) (j - 1) * p3) / (double) j;
      }

      p  = (double) n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z  = z1 - p1 / p;

      k++;
    }
    while ( std::fabs(z - z1) > 3.0e-14 && k <= MAX_ITER );

    if ( k > MAX_ITER )
    {
      throw ArithmeticException (
        JEM_FUNC,
        String::format (
          "failed to compute %d Gauss-Legendre abscissas and weights "
          "(too many iterations)", n
        )
      );
    }

    x[i - 1] = -z;
    x[n - i] =  z;
    w[i - 1] =  2.0 / ((1.0 - z * z) * p * p);
    w[n - i] =  w[i - 1];
  }
}


//-----------------------------------------------------------------------
//   gaussJacobi
//-----------------------------------------------------------------------


void Quadrature::gaussJacobi

  ( double                alfa,
    double                beta,
    const Array<double>&  x,
    const Array<double>&  w )

{
  JEM_PRECHECK2 ( x.size() == w.size(), "Array size mismatch" );
  JEM_PRECHECK2 ( x.size() > 0, "number of Gauss points must "
                  "be larger than zero" );

  const idx_t   MAX_ITER = 10;

  const idx_t   n  = x.size ();
  const double  nn = (double) n;

  double        an, bn, ab;
  double        z,  z1;
  double        a,  b, c;
  double        r1, r2, r3;
  double        p,  p1, p2, p3;
  double        t,  t1;
  idx_t         i, j, k;


  z = 0.0;

  for ( i = 1; i <= n; i++ )
  {
    if      ( i == 1 )
    {
      an  = alfa / nn;
      bn  = beta / nn;
      r1  = (1.0 + alfa) * (2.78 / (4.0 + nn * nn) + 0.768 * an / nn);
      r2  = 1.0  + 1.48  * an + 0.96 * bn + 0.452 * an * an;
      r2 += 0.83 * an * bn;
      z   = 1.0 - r1 / r2;
    }
    else if ( i == 2 )
    {
      r1 = (4.1 + alfa) / ((1.0 + alfa) * (1.0 + 0.156 * alfa));
      r2 = 1.0 + 0.06  * (nn - 8.0) * (1.0 + 0.12 * alfa) / nn;
      r3 = 1.0 + 0.012 * beta * (1.0 + 0.25 * std::fabs(alfa)) / nn;
      z -= (1.0 - z) * r1 * r2 * r3;
    }
    else if ( i == 3 )
    {
      r1 = (1.67 + 0.28 * alfa) / (1.0 + 0.37 * alfa);
      r2 = 1.0   + 0.22 * (nn - 8.0) / nn;
      r3 = 1.0   + 8.0  * beta / ((6.28 + beta) * nn * nn);
      z -= (x[0] - z) * r1 * r2 * r3;
    }
    else if ( i == n - 1 )
    {
      r1 = (1.0 + 0.235 * beta) / (0.766 + 0.199 * beta);
      r2 = 1.0 / (1.0 + 0.639 * (nn - 4.0) / (1.0 + 0.71 * (nn - 4.0)));
      r3 = 1.0 / (1.0 + 20.0  * alfa / ((7.5 + alfa) * nn * nn));
      z += (z - x[n - 4]) * r1 * r2 * r3;
    }
    else if ( i == n )
    {
      r1 = (1.0 + 0.37 * beta) / (1.67 + 0.28 * beta);
      r2 = 1.0 / (1.0 + 0.22 * (nn - 8.0) / nn);
      r3 = 1.0 / (1.0 + 8.0  * alfa / ((6.28 + alfa) * nn * nn));
      z += (z - x[n - 3]) * r1 * r2 * r3;
    }
    else
    {
      z = 3.0 * x[i - 2] - 3.0 * x[i - 3] + x[i - 4];
    }

    ab = alfa + beta;

    for ( k = 1; k <= MAX_ITER; k++ )
    {
      t  = 2.0 + ab;
      p1 = (alfa - beta + t * z) / 2.0;
      p2 = 1.0;

      for ( j = 2; j <= n; j++ )
      {
        double  jj = (double) j;

        p3 = p2;
        p2 = p1;
        t  = 2.0 * jj + ab;
        a  = 2.0 * jj * (jj + ab) * (t - 2.0);
        b  = alfa * alfa - beta * beta + t * (t - 2.0) * z;
        b *= t - 1.0;
        c  = 2.0 * (jj - 1.0 + alfa) * (jj - 1.0 + beta) * t;
        p1 = (b * p2 - c * p3) / a;
      }

      p  = nn  * (alfa - beta - t * z) * p1;
      p += 2.0 * (nn + alfa) * (nn + beta) * p2;
      p /= t   * (1.0 - z * z);
      z1 = z;
      z  = z1 - p1 / p;

      if ( std::fabs(z - z1) <= 3.0e-14 )
      {
        break;
      }
    }

    if ( k > MAX_ITER )
    {
      throw ArithmeticException (
        JEM_FUNC,
        String::format (
          "failed to compute %d Gauss-Jacobi abscissas and weights "
          "(too many iterations)", n
        )
      );
    }

    t1  = logGamma_( alfa + nn ) + logGamma_( beta      + nn );
    t1 -= logGamma_( 1.0  + nn ) + logGamma_( 1.0  + ab + nn );

    x[i - 1] = z;
    w[i - 1] = ::exp( t1 ) * t * ::pow( 2.0, ab ) / (p * p2);
  }
}


//-----------------------------------------------------------------------
//   logGamma_
//-----------------------------------------------------------------------


double Quadrature::logGamma_ ( double x )
{
  static const double coeff[6] =

    { 76.18009172947146,     -86.50532032941677,
      24.01409824083091,     -1.231739572450155,
      0.1208650973866179e-2, -0.5395239384953e-5 };

  double  y, s, tmp;
  int     j;

  y   = x;
  tmp = x + 5.5;
  tmp = tmp - (x + 0.5) * ::log( tmp );
  s   = 1.000000000190015;

  for ( j = 0; j <= 5; j++ )
  {
    y += 1.0;
    s += coeff[j] / y;
  }

  return -tmp + ::log( 2.5066282746310005 * s / x);
}


JEM_END_PACKAGE( numeric )
