
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
#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ArithmeticException.h>
#include <jem/numeric/Quaternion.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   internal macro's
//=======================================================================

//-----------------------------------------------------------------------
//   M_PI
//-----------------------------------------------------------------------


#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif


//-----------------------------------------------------------------------
//   JIVE_QUATERNION_RMULT
//-----------------------------------------------------------------------


#undef  JIVE_QUATERNION_RMULT
#define JIVE_QUATERNION_RMULT( r, a, b )                \
                                                        \
  r[0] = a(0,0) * b[0] + a(0,1) * b[1] + a(0,2) * b[2]; \
  r[1] = a(1,0) * b[0] + a(1,1) * b[1] + a(1,2) * b[2]; \
  r[2] = a(2,0) * b[0] + a(2,1) * b[1] + a(2,2) * b[2]


//-----------------------------------------------------------------------
//   JIVE_QUATERNION_NORMALIZE
//-----------------------------------------------------------------------


#undef  JIVE_QUATERNION_NORMALIZE
#define JIVE_QUATERNION_NORMALIZE( q, t )       \
                                                \
  t = q[0] * q[0] + q[1] * q[1] +               \
      q[2] * q[2] + q[3] * q[3];                \
                                                \
  if ( isTiny( t ) )                            \
  {                                             \
    normalizeError_ ( JEM_FUNC );               \
  }                                             \
                                                \
  t     = 1.0 / t;                              \
                                                \
  q[0] *= t;                                    \
  q[1] *= t;                                    \
  q[2] *= t;                                    \
  q[3] *= t


//=======================================================================
//   class Quaternion
//=======================================================================

//-----------------------------------------------------------------------
//   qmult (Vec4 * Matrix)
//-----------------------------------------------------------------------


void Quaternion::qmult

  ( const Matrix&  r,
    const Vec4&    a,
    const Matrix&  b )

{
  JEM_PRECHECK2 ( r.size(0) == 4 &&
                  b.size(0) == 4 &&
                  r.size(1) == b.size(1),
                  "invalid Array sizes" );

  const idx_t  n = r.size(1);

  idx_t        j;


  if ( r.stride(0) + b.stride(0) == 2L )
  {
          double*  JEM_RESTRICT rj = r.addr ();
    const double*  JEM_RESTRICT bj = b.addr ();

    const idx_t    rst = r.stride (1);
    const idx_t    bst = b.stride (1);

    for ( j = 0; j < n; j++ )
    {
      JIVE_QUATERNION_QMULT( rj, a, bj );

      rj += rst;
      bj += bst;
    }
  }
  else
  {
    Vec4  rj, bj;

    for ( j = 0; j < n; j++ )
    {
      bj[0]  = b(0,j);
      bj[1]  = b(1,j);
      bj[2]  = b(2,j);
      bj[3]  = b(3,j);

      JIVE_QUATERNION_QMULT( rj, a, bj );

      r(0,j) = rj[0];
      r(1,j) = rj[1];
      r(2,j) = rj[2];
      r(3,j) = rj[3];
    }
  }
}


//-----------------------------------------------------------------------
//   qmult (Vector * Matrix)
//-----------------------------------------------------------------------


void Quaternion::qmult

  ( const Matrix&  r,
    const Vector&  a,
    const Matrix&  b )

{
  JEM_PRECHECK2 ( a.size() == 4, "Array should have size 4" );

  Vec4  aa;

  aa[0] = a[0];
  aa[1] = a[1];
  aa[2] = a[2];
  aa[3] = a[3];

  qmult ( r, aa, b );
}


//-----------------------------------------------------------------------
//   qmult (Matrix * Matrix)
//-----------------------------------------------------------------------


void Quaternion::qmult

  ( const Matrix&  r,
    const Matrix&  a,
    const Matrix&  b )

{
  JEM_PRECHECK2 ( r.size(0) == 4         &&
                  a.size(0) == 4         &&
                  b.size(0) == 4         &&
                  r.size(1) == a.size(1) &&
                  a.size(1) == b.size(1),
                  "invalid Array sizes" );

  const idx_t  n = r.size(1);

  idx_t        j;


  if ( r.stride(0) + a.stride(0) + b.stride(0) == 3L )
  {
          double*  JEM_RESTRICT rj = r.addr ();
    const double*  JEM_RESTRICT aj = a.addr ();
    const double*  JEM_RESTRICT bj = b.addr ();

    const idx_t    rst = r.stride (1);
    const idx_t    ast = a.stride (1);
    const idx_t    bst = b.stride (1);

    for ( j = 0; j < n; j++ )
    {
      JIVE_QUATERNION_QMULT( rj, aj, bj );

      rj += rst;
      aj += ast;
      bj += bst;
    }
  }
   else
  {
    Vec4  rj, aj, bj;

    for ( j = 0; j < n; j++ )
    {
      aj[0]  = a(0,j);
      aj[1]  = a(1,j);
      aj[2]  = a(2,j);
      aj[3]  = a(3,j);

      bj[0]  = b(0,j);
      bj[1]  = b(1,j);
      bj[2]  = b(2,j);
      bj[3]  = b(3,j);

      JIVE_QUATERNION_QMULT( rj, aj, bj );

      r(0,j) = rj[0];
      r(1,j) = rj[1];
      r(2,j) = rj[2];
      r(3,j) = rj[3];
    }
  }
}


//-----------------------------------------------------------------------
//   vmult (Vec4 * Vec3)
//-----------------------------------------------------------------------


void Quaternion::vmult

  ( Vec3&        r,
    const Vec4&  a,
    const Vec3&  b ) noexcept

{
  Mat3x3  rm;

  toRotmat ( rm, a );

  JIVE_QUATERNION_RMULT( r, rm, b );
}


//-----------------------------------------------------------------------
//   vmult (Vector * Vector)
//-----------------------------------------------------------------------


void Quaternion::vmult

  ( const Vector&  r,
    const Vector&  a,
    const Vector&  b )

{
  JEM_PRECHECK2 ( r.size() == 3 && a.size() == 4 && b.size() == 3,
                  "invalid Array sizes" );

  Mat3x3  rm;
  Vec4    aa;


  aa[0] = a[0];
  aa[1] = a[1];
  aa[2] = a[2];
  aa[3] = a[3];

  toRotmat ( rm, aa );

  JIVE_QUATERNION_RMULT( r, rm, b );
}


//-----------------------------------------------------------------------
//   vmult (Vec4 * Matrix)
//-----------------------------------------------------------------------


void Quaternion::vmult

  ( const Matrix&  r,
    const Vec4&    a,
    const Matrix&  b )

{
  JEM_PRECHECK2 ( r.size(0) == 3 &&
                  b.size(0) == 3 &&
                  r.size(1) == b.size(1),
                  "invalid Array sizes" );

  const idx_t  n = r.size(1);

  Mat3x3       rm;
  idx_t        j;


  toRotmat ( rm, a );

  if ( r.stride(0) + b.stride(0) == 2L )
  {
    double*        JEM_RESTRICT  rj = r.addr ();
    const double*  JEM_RESTRICT  bj = b.addr ();

    const idx_t    rst = r.stride (1);
    const idx_t    bst = b.stride (1);

    for ( j = 0; j < n; j++ )
    {
      JIVE_QUATERNION_RMULT( rj, rm, bj );

      rj += rst;
      bj += bst;
    }
  }
  else
  {
    Vec3  rj, bj;

    for ( j = 0; j < n; j++ )
    {
      bj[0]  = b(0,j);
      bj[1]  = b(1,j);
      bj[2]  = b(2,j);

      JIVE_QUATERNION_RMULT( rj, rm, bj );

      r(0,j) = rj[0];
      r(1,j) = rj[1];
      r(2,j) = rj[2];
    }
  }
}


//-----------------------------------------------------------------------
//   vmult (Vector * Matrix)
//-----------------------------------------------------------------------


void Quaternion::vmult

  ( const Matrix&  r,
    const Vector&  a,
    const Matrix&  b )

{
  JEM_PRECHECK2 ( a.size() == 4, "Array should have size 4" );

  Vec4  aa;

  aa[0] = a[0];
  aa[1] = a[1];
  aa[2] = a[2];
  aa[3] = a[3];

  vmult ( r, aa, b );
}


//-----------------------------------------------------------------------
//   vmult (Matrix * Matrix)
//-----------------------------------------------------------------------


void Quaternion::vmult

  ( const Matrix&  r,
    const Matrix&  a,
    const Matrix&  b )

{
  JEM_PRECHECK2 ( r.size(0) == 3         &&
                  a.size(0) == 4         &&
                  b.size(0) == 3         &&
                  r.size(1) == a.size(1) &&
                  a.size(1) == b.size(1),
                  "invalid Array sizes" );

  const idx_t  n = r.size(1);

  Mat3x3       rm;
  Vec4         aj;
  idx_t        j;


  if ( r.stride(0) + b.stride(0) == 2L )
  {
    double*        JEM_RESTRICT  rj = r.addr ();
    const double*  JEM_RESTRICT  bj = b.addr ();

    const idx_t    rst = r.stride (1);
    const idx_t    bst = b.stride (1);

    for ( j = 0; j < n; j++ )
    {
      aj[0] = a(0,j);
      aj[1] = a(1,j);
      aj[2] = a(2,j);
      aj[3] = a(3,j);

      toRotmat ( rm, aj );

      JIVE_QUATERNION_RMULT( rj, rm, bj );

      rj += rst;
      bj += bst;
    }
  }
  else
  {
    Vec3  rj, bj;

    for ( j = 0; j < n; j++ )
    {
      aj[0]  = a(0,j);
      aj[1]  = a(1,j);
      aj[2]  = a(2,j);
      aj[3]  = a(3,j);

      bj[0]  = b(0,j);
      bj[1]  = b(1,j);
      bj[2]  = b(2,j);

      toRotmat ( rm, aj );

      JIVE_QUATERNION_RMULT( rj, rm, bj );

      r(0,j) = rj[0];
      r(1,j) = rj[1];
      r(2,j) = rj[2];
    }
  }
}


//-----------------------------------------------------------------------
//   invert (Matrix)
//-----------------------------------------------------------------------


void Quaternion::invert ( const Matrix& q )
{
  JEM_PRECHECK2 ( q.size(0) == 4, "matrix should have 4 rows" );

  const idx_t  n = q.size(1);

  idx_t        j;


  if ( q.stride(0) == 1L )
  {
    double*      qj  = q.addr   ();
    const idx_t  qst = q.stride (1);

    for ( j = 0; j < n; j++ )
    {
      qj[1] = -qj[1];
      qj[2] = -qj[2];
      qj[3] = -qj[3];

      qj   += qst;
    }
  }
  else
  {
    for ( j = 0; j < n; j++ )
    {
      q(1,j) = -q(1,j);
      q(2,j) = -q(2,j);
      q(3,j) = -q(3,j);
    }
  }
}


//-----------------------------------------------------------------------
//   normalize (Vec4)
//-----------------------------------------------------------------------


void Quaternion::normalize ( Vec4& q )
{
  double  t;

  JIVE_QUATERNION_NORMALIZE( q, t );
}


//-----------------------------------------------------------------------
//   normalize (Vector)
//-----------------------------------------------------------------------


void Quaternion::normalize ( const Vector& q )
{
  JEM_PRECHECK2 ( q.size() == 4, "Array should have size 4" );

  double  t;

  JIVE_QUATERNION_NORMALIZE( q, t );
}


//-----------------------------------------------------------------------
//   normalize (Matrix)
//-----------------------------------------------------------------------


void Quaternion::normalize ( const Matrix& q )
{
  JEM_PRECHECK2 ( q.size(0) == 4, "matrix should have 4 rows" );

  const idx_t  n = q.size(1);

  double       t;
  idx_t        j;


  if ( q.stride(0) == 1_idx )
  {
    double*      qj  = q.addr   ();
    const idx_t  qst = q.stride (1);

    for ( j = 0; j < n; j++ )
    {
      JIVE_QUATERNION_NORMALIZE( qj, t );

      qj += qst;
    }
  }
  else
  {
    for ( j = 0; j < n; j++ )
    {
      t = q(0,j) * q(0,j) + q(1,j) * q(1,j) +
          q(2,j) * q(2,j) + q(3,j) * q(3,j);

      if ( isTiny( t ) )
      {
        normalizeError_ ( JEM_FUNC );
      }

      t       = 1.0 / t;
      q(0,j) *= t;
      q(1,j) *= t;
      q(2,j) *= t;
      q(3,j) *= t;
    }
  }
}


//-----------------------------------------------------------------------
//   toRotmat
//-----------------------------------------------------------------------


void Quaternion::toRotmat

  ( Mat3x3&      rm,
    const Vec4&  q ) noexcept

{
  double  a, b, c;


  a        = q[1] * q[1];
  b        = q[2] * q[2];
  c        = q[3] * q[3];

  rm(0,0)  = 1.0 - 2.0 * (b + c);
  rm(1,1)  = 1.0 - 2.0 * (a + c);
  rm(2,2)  = 1.0 - 2.0 * (a + b);

  rm(1,0)  = 2.0 * q[1] * q[2];
  rm(0,1)  = rm(1,0);
  a        = 2.0 * q[0] * q[3];
  rm(1,0) += a;
  rm(0,1) -= a;

  rm(2,0)  = 2.0 * q[1] * q[3];
  rm(0,2)  = rm(2,0);
  a        = 2.0 * q[0] * q[2];
  rm(2,0) -= a;
  rm(0,2) += a;

  rm(2,1)  = 2.0 * q[2] * q[3];
  rm(1,2)  = rm(2,1);
  a        = 2.0 * q[0] * q[1];
  rm(2,1) += a;
  rm(1,2) -= a;
}


//-----------------------------------------------------------------------
//   toRotmat (Matrix)
//-----------------------------------------------------------------------


void Quaternion::toRotmat

  ( const Matrix&  rm,
    const Vector&  q )

{
  JEM_PRECHECK2 ( rm.size(0) == 3 &&
                  rm.size(1) == 3 && q.size() == 4,
                  "invalid Array sizes" );

  Mat3x3  tm;
  Vec4    tq;


  tq[0] = q[0];
  tq[1] = q[1];
  tq[2] = q[2];
  tq[3] = q[3];

  toRotmat ( tm, tq );

  if ( rm.isContiguous() )
  {
    std::memcpy ( rm.addr(), tm.addr(), 9 * sizeof(double) );
  }
  else
  {
    rm(0,0) = tm(0,0);
    rm(1,0) = tm(1,0);
    rm(2,0) = tm(2,0);
    rm(0,1) = tm(0,1);
    rm(1,1) = tm(1,1);
    rm(2,1) = tm(2,1);
    rm(0,2) = tm(0,2);
    rm(1,2) = tm(1,2);
    rm(2,2) = tm(2,2);
  }
}


//-----------------------------------------------------------------------
//   toRotvec (Vec4)
//-----------------------------------------------------------------------


void Quaternion::toRotvec

  ( Vec4&        rv,
    const Vec4&  q ) noexcept

{
  double  alpha;
  double  scale;


  if      ( q[0] <= -1.0 )
  {
    alpha = M_PI;
  }
  else if ( q[0] >= 1.0 )
  {
    alpha = 0.0;
  }
  else
  {
    alpha = std::acos ( q[0] );
  }

  scale = std::sin ( alpha );

  if ( isTiny( scale ) )
  {
    rv[0] = 0.0;
    rv[1] = 1.0;
    rv[2] = 0.0;
    rv[3] = 0.0;
  }
  else
  {
    scale = 1.0  / scale;
    rv[0] = 2.0  * alpha;
    rv[1] = q[1] * scale;
    rv[2] = q[2] * scale;
    rv[3] = q[3] * scale;
  }
}


//-----------------------------------------------------------------------
//   toRotvec (Vector)
//-----------------------------------------------------------------------


void Quaternion::toRotvec

  ( const Vector&  rv,
    const Vector&  q )

{
  JEM_PRECHECK2 ( rv.size() == 4 && q.size() == 4,
                  "Arrays should have size 4" );

  Vec4  tq;
  Vec4  tv;

  tq[0] = q[0];
  tq[1] = q[1];
  tq[2] = q[2];
  tq[3] = q[3];

  toRotvec ( tv, tq );

  rv[0] = tv[0];
  rv[1] = tv[1];
  rv[2] = tv[2];
  rv[3] = tv[3];
}


//-----------------------------------------------------------------------
//   fromRotvec (Vec4)
//-----------------------------------------------------------------------


void Quaternion::fromRotvec

  ( Vec4&        q,
    const Vec4&  rv )

{
  double  len = rv[1] * rv[1] + rv[2] * rv[2] + rv[3] * rv[3];

  double  scale;


  if ( isTiny( len ) )
  {
    throw ArithmeticException (
      JEM_FUNC,
      "rotation axis has zero length"
    );
  }

  scale = std::sin ( 0.5 * rv[0] ) / std::sqrt ( len );

  q[0]  = std::cos ( 0.5 * rv[0] );

  q[1]  = rv[1] * scale;
  q[2]  = rv[2] * scale;
  q[3]  = rv[3] * scale;
}


//-----------------------------------------------------------------------
//   fromRotvec (Vector)
//-----------------------------------------------------------------------


/* void Quaternion::fromRotvec

  ( const Vector&  q,
    const Vector&  rv )

{
  JEM_PRECHECK2 ( rv.size() == 4 && q.size() == 4,
                  "Arrays should have size 4" );


  Vec4  tq;
  Vec4  tv;

  tq[0] = q[0];
  tq[1] = q[1];
  tq[2] = q[2];
  tq[3] = q[3];

  fromRotvec ( tv, tq );

  rv[0] = tv[0];
  rv[1] = tv[1];
  rv[2] = tv[2];
  rv[3] = tv[3];
} */


//-----------------------------------------------------------------------
//   normalizeError_
//-----------------------------------------------------------------------


void Quaternion::normalizeError_ ( const char* where )
{
  throw ArithmeticException (
    where,
    "normalization of quaternion failed (zero length)"
  );
}


JEM_END_PACKAGE( numeric )
