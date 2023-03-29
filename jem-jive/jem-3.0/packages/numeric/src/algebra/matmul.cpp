
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
#include <jem/numeric/algebra/matmul.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   matmul in place
//=======================================================================

//-----------------------------------------------------------------------
//   vector = matrix * vector
//-----------------------------------------------------------------------


void                      matmul

  ( const Array<double,1>&  r,
    const Array<double,2>&  a,
    const Array<double,1>&  b )

{
  JEM_PRECHECK2 ( r.size() == a.size(0) &&
                  b.size() == a.size(1),
                  "matrix/vector size mismatch" );

  const idx_t  m = a.size (0);
  const idx_t  n = a.size (1);

  idx_t        i, j;


  if ( a.stride(0) > a.stride(1) )
  {
    // Assume row-major orientation.

    const double* JEM_RESTRICT  arow0 = a.addr ();
    const double* JEM_RESTRICT  bcol0 = b.addr ();

    double  b0, r0, r1, r2, r3;

    if ( a.stride(1) + b.stride() == 2_idx )
    {
      const double* JEM_RESTRICT  arow1 = nullptr;
      const double* JEM_RESTRICT  arow2 = nullptr;
      const double* JEM_RESTRICT  arow3 = nullptr;

      for ( i = 0; i < (m - 3); i += 4 )
      {
        r0 = r1 = r2 = r3 = 0.0;

        arow1 = arow0 + a.stride (0);
        arow2 = arow1 + a.stride (0);
        arow3 = arow2 + a.stride (0);

        for ( j = 0; j < n; j++ )
        {
          b0  =      bcol0[j];
          r0 += b0 * arow0[j];
          r1 += b0 * arow1[j];
          r2 += b0 * arow2[j];
          r3 += b0 * arow3[j];
        }

        r[i + 0] = r0;
        r[i + 1] = r1;
        r[i + 2] = r2;
        r[i + 3] = r3;

        arow0    = arow3 + a.stride (0);
      }

      switch ( m - i )
      {
      case 0:

        break;

      case 1:

        r0 = 0.0;

        for ( j = 0; j < n; j++ )
        {
          r0 += bcol0[j] * arow0[j];
        }

        r[i] = r0;

        break;

      case 2:

        r0 = r1 = 0.0;

        arow1 = arow0 + a.stride (0);

        for ( j = 0; j < n; j++ )
        {
          b0  =      bcol0[j];
          r0 += b0 * arow0[j];
          r1 += b0 * arow1[j];
        }

        r[i + 0] = r0;
        r[i + 1] = r1;

        break;

      case 3:

        r0 = r1 = r2 = 0.0;

        arow1 = arow0 + a.stride (0);
        arow2 = arow1 + a.stride (0);

        for ( j = 0; j < n; j++ )
        {
          b0  =      bcol0[j];
          r0 += b0 * arow0[j];
          r1 += b0 * arow1[j];
          r2 += b0 * arow2[j];
        }

        r[i + 0] = r0;
        r[i + 1] = r1;
        r[i + 2] = r2;

        break;
      }
    }
    else
    {
      const idx_t  ast1 = a.stride (1);
      const idx_t  bst  = b.stride ();

      for ( i = 0; i < m; i++ )
      {
        r0 = 0.0;

        for ( j = 0; j < n; j++ )
        {
          r0 += arow0[j * ast1] * bcol0[j * bst];
        }

        r[i]   = r0;
        arow0 += a.stride (0);
      }
    }
  }
  else
  {
    // Assume column-major orientation.

    const double* JEM_RESTRICT  acol0 = a.addr ();
    double*       JEM_RESTRICT  rcol0 = r.addr ();

    double  b0, b1, b2, b3;

    if ( a.stride(0) + r.stride() == 2_idx )
    {
      const double* JEM_RESTRICT  acol1 = nullptr;
      const double* JEM_RESTRICT  acol2 = nullptr;
      const double* JEM_RESTRICT  acol3 = nullptr;

      for ( i = 0; i < m; i++ )
      {
        rcol0[i] = 0.0;
      }

      for ( j = 0; j < (n - 3); j += 4 )
      {
        b0 = b[j + 0];
        b1 = b[j + 1];
        b2 = b[j + 2];
        b3 = b[j + 3];

        acol1 = acol0 + a.stride (1);
        acol2 = acol1 + a.stride (1);
        acol3 = acol2 + a.stride (1);

JEM_IVDEP

        for ( i = 0; i < m; i++ )
        {
          rcol0[i] += (acol0[i] * b0 + acol1[i] * b1 +
                       acol2[i] * b2 + acol3[i] * b3);
        }

        acol0 = acol3 + a.stride (1);
      }

      switch ( n - j )
      {
      case 0:

        break;

      case 1:

        b0 = b[j];

JEM_IVDEP

        for ( i = 0; i < m; i++ )
        {
          rcol0[i] += (acol0[i] * b0);
        }

        break;

      case 2:

        b0 = b[j + 0];
        b1 = b[j + 1];

        acol1 = acol0 + a.stride (1);

JEM_IVDEP

        for ( i = 0; i < m; i++ )
        {
          rcol0[i] += (acol0[i] * b0 + acol1[i] * b1);
        }

        break;

      case 3:

        b0 = b[j + 0];
        b1 = b[j + 1];
        b2 = b[j + 2];

        acol1 = acol0 + a.stride (1);
        acol2 = acol1 + a.stride (1);

JEM_IVDEP

        for ( i = 0; i < m; i++ )
        {
          rcol0[i] += (acol0[i] * b0 + acol1[i] * b1 +
                       acol2[i] * b2);
        }

        break;
      }
    }
    else
    {
      const idx_t  ast0 = a.stride (0);
      const idx_t  rst  = r.stride ();

      for ( i = 0; i < m; i++ )
      {
        rcol0[i * rst] = 0.0;
      }

      for ( j = 0; j < n; j++ )
      {
        b0 = b[j];

JEM_IVDEP

        for ( i = 0; i < m; i++ )
        {
          rcol0[i * rst] += acol0[i * ast0] * b0;
        }

        acol0 += a.stride (1);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   vector = vector * matrix
//-----------------------------------------------------------------------


void                      matmul

  ( const Array<double,1>&  r,
    const Array<double,1>&  a,
    const Array<double,2>&  b )

{
  JEM_PRECHECK2 ( r.size() == b.size(1) &&
                  a.size() == b.size(0),
                  "matrix/vector size mismatch" );

  const idx_t  m = b.size (0);
  const idx_t  n = b.size (1);

  idx_t        i, j;


  if ( b.stride(0) > b.stride(1) )
  {
    // Assume row-major orientation.

    const double* JEM_RESTRICT  brow0 = b.addr ();
    double*       JEM_RESTRICT  rrow0 = r.addr ();

    double  a0, a1, a2, a3;

    if ( b.stride(1) + r.stride() == 2_idx )
    {
      const double* JEM_RESTRICT  brow1;
      const double* JEM_RESTRICT  brow2;
      const double* JEM_RESTRICT  brow3;

      for ( j = 0; j < n; j++ )
      {
        rrow0[j] = 0.0;
      }

      for ( i = 0; i < (m - 3); i += 4 )
      {
        a0 = a[i + 0];
        a1 = a[i + 1];
        a2 = a[i + 2];
        a3 = a[i + 3];

        brow1 = brow0 + b.stride (0);
        brow2 = brow1 + b.stride (0);
        brow3 = brow2 + b.stride (0);

JEM_IVDEP

        for ( j = 0; j < n; j++ )
        {
          rrow0[j] += (a0 * brow0[j] + a1 * brow1[j] +
                       a2 * brow2[j] + a3 * brow3[j]);
        }

        brow0 = brow3 + b.stride (0);
      }

      switch ( m - i )
      {
      case 0:

        break;

      case 1:

        a0 = a[i];

JEM_IVDEP

        for ( j = 0; j < n; j++ )
        {
          rrow0[j] += (a0 * brow0[j]);
        }

        break;

      case 2:

        a0 = a[i + 0];
        a1 = a[i + 1];

        brow1 = brow0 + b.stride (0);

JEM_IVDEP

        for ( j = 0; j < n; j++ )
        {
          rrow0[j] += (a0 * brow0[j] + a1 * brow1[j]);
        }

        break;

      case 3:

        a0 = a[i + 0];
        a1 = a[i + 1];
        a2 = a[i + 2];

        brow1 = brow0 + b.stride (0);
        brow2 = brow1 + b.stride (0);

JEM_IVDEP

        for ( j = 0; j < n; j++ )
        {
          rrow0[j] += (a0 * brow0[j] + a1 * brow1[j] +
                       a2 * brow2[j]);
        }

        break;
      }
    }
    else
    {
      const idx_t  bst1 = b.stride (1);
      const idx_t  rst  = r.stride ();

      for ( j = 0; j < n; j++ )
      {
        rrow0[j * rst] = 0.0;
      }

      for ( i = 0; i < m; i++ )
      {
        a0 = a[i];

JEM_IVDEP

        for ( j = 0; j < n; j++ )
        {
          rrow0[j * rst] += a0 * brow0[j * bst1];
        }

        brow0 += b.stride (0);
      }
    }
  }
  else
  {
    // Assume column-major orientation.

    const double* JEM_RESTRICT  arow0 = a.addr ();
    const double* JEM_RESTRICT  bcol0 = b.addr ();

    double  a0, r0, r1, r2, r3;

    if ( a.stride() + b.stride(0) == 2_idx )
    {
      const double* JEM_RESTRICT  bcol1;
      const double* JEM_RESTRICT  bcol2;
      const double* JEM_RESTRICT  bcol3;

      for ( j = 0; j < (n - 3); j += 4 )
      {
        r0 = r1 = r2 = r3 = 0.0;

        bcol1 = bcol0 + b.stride (1);
        bcol2 = bcol1 + b.stride (1);
        bcol3 = bcol2 + b.stride (1);

        for ( i = 0; i < m; i++ )
        {
          a0  =      arow0[i];
          r0 += a0 * bcol0[i];
          r1 += a0 * bcol1[i];
          r2 += a0 * bcol2[i];
          r3 += a0 * bcol3[i];
        }

        r[j + 0] = r0;
        r[j + 1] = r1;
        r[j + 2] = r2;
        r[j + 3] = r3;

        bcol0    = bcol3 + b.stride (1);
      }

      switch ( n - j )
      {
      case 0:

        break;

      case 1:

        r0 = 0.0;

        for ( i = 0; i < m; i++ )
        {
          r0 += arow0[i] * bcol0[i];
        }

        r[j] = r0;

        break;

      case 2:

        r0 = r1 = 0.0;

        bcol1 = bcol0 + b.stride (1);

        for ( i = 0; i < m; i++ )
        {
          a0  =      arow0[i];
          r0 += a0 * bcol0[i];
          r1 += a0 * bcol1[i];
        }

        r[j + 0] = r0;
        r[j + 1] = r1;

        break;

      case 3:

        r0 = r1 = r2 = 0.0;

        bcol1 = bcol0 + b.stride (1);
        bcol2 = bcol1 + b.stride (1);

        for ( i = 0; i < m; i++ )
        {
          a0  =      arow0[i];
          r0 += a0 * bcol0[i];
          r1 += a0 * bcol1[i];
          r2 += a0 * bcol2[i];
        }

        r[j + 0] = r0;
        r[j + 1] = r1;
        r[j + 2] = r2;

        break;
      }
    }
    else
    {
      const idx_t  ast  = a.stride ();
      const idx_t  bst0 = b.stride (0);

      for ( j = 0; j < n; j++ )
      {
        r0 = 0.0;

        for ( i = 0; i < m; i++ )
        {
          r0 += arow0[i * ast] * bcol0[i * bst0];
        }

        r[j]   = r0;
        bcol0 += b.stride(1);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   matrix = vector * vector
//-----------------------------------------------------------------------


void                      matmul

  ( const Array<double,2>&  r,
    const Array<double,1>&  a,
    const Array<double,1>&  b )

{
  JEM_PRECHECK2 ( r.size(0) == a.size() &&
                  r.size(1) == b.size(),
                  "matrix/vector size mismatch" );

  const idx_t  m = a.size ();
  const idx_t  n = b.size ();

  idx_t        i, j;


  if ( r.stride(0) > r.stride(1) )
  {
    // Assume row-major orientation.

    const double* JEM_RESTRICT  brow0 = b.addr ();
    double*       JEM_RESTRICT  rrow0 = r.addr ();

    double  a0;

    if ( b.stride() + r.stride(1) == 2_idx )
    {
      for ( i = 0; i < m; i++ )
      {
        a0 = a[i];

        for ( j = 0; j < n; j++ )
        {
          rrow0[j] = a0 * brow0[j];
        }

        rrow0 += r.stride (0);
      }
    }
    else
    {
      const idx_t  bst  = b.stride ();
      const idx_t  rst1 = r.stride (1);

      for ( i = 0; i < m; i++ )
      {
        a0 = a[i];

        for ( j = 0; j < n; j++ )
        {
          rrow0[j * rst1] = a0 * brow0[j * bst];
        }

        rrow0 += r.stride (0);
      }
    }
  }
  else
  {
    // Assume column-major orientation.

    const double* JEM_RESTRICT  acol0 = a.addr ();
    double*       JEM_RESTRICT  rcol0 = r.addr ();

    double  b0;

    if ( a.stride() + r.stride(0) == 2_idx )
    {
      for ( j = 0; j < n; j++ )
      {
        b0 = b[j];

        for ( i = 0; i < m; i++ )
        {
          rcol0[i] = acol0[i] * b0;
        }

        rcol0 += r.stride (1);
      }
    }
    else
    {
      const idx_t  ast  = a.stride ();
      const idx_t  rst0 = r.stride (0);

      for ( j = 0; j < n; j++ )
      {
        b0 = b[j];

        for ( i = 0; i < m; i++ )
        {
          rcol0[i * rst0] = acol0[i * ast] * b0;
        }

        rcol0 += r.stride (1);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   matrix = matrix * matrix
//-----------------------------------------------------------------------


void                      matmul

  ( const Array<double,2>&  r,
    const Array<double,2>&  a,
    const Array<double,2>&  b )

{
  JEM_PRECHECK2 ( r.size(0) == a.size(0) &&
                  r.size(1) == b.size(1) &&
                  a.size(1) == b.size(0),
                  "matrix shape mismatch" );

  const idx_t  m = r.size (0);
  const idx_t  n = r.size (1);
  const idx_t  p = a.size (1);

  idx_t        i, j, k;


  if ( a.stride(0) > a.stride(1) )
  {
    // Assume row-major orientation.

    const double* JEM_RESTRICT  arow0 = a.addr ();
    const double* JEM_RESTRICT  bcol0 = b.addr ();

    double  a0,  a1,  a2,  a3;
    double  b0,  b1,  b2,  b3;
    double  r00, r01, r02, r03;
    double  r10, r11, r12, r13;
    double  r20, r21, r22, r23;
    double  r30, r31, r32, r33;

    if ( a.stride(1) + b.stride(0) == 2_idx )
    {
      // Contiguous, row-major storage; full loop unrolling.

      const double* JEM_RESTRICT  arow1 = nullptr;
      const double* JEM_RESTRICT  arow2 = nullptr;
      const double* JEM_RESTRICT  arow3 = nullptr;
      const double* JEM_RESTRICT  bcol1 = nullptr;
      const double* JEM_RESTRICT  bcol2 = nullptr;
      const double* JEM_RESTRICT  bcol3 = nullptr;

      const idx_t  ast0 = a.stride (0);

      for ( j = 0; j < (n - 3); j += 4 )
      {
        arow0 =         a.addr   ();
        bcol1 = bcol0 + b.stride (1);
        bcol2 = bcol1 + b.stride (1);
        bcol3 = bcol2 + b.stride (1);

        for ( i = 0; i < (m - 3); i += 4 )
        {
          arow1 = arow0 + ast0;
          arow2 = arow1 + ast0;
          arow3 = arow2 + ast0;

          r00 = r01 = r02 = r03 = 0.0;
          r10 = r11 = r12 = r13 = 0.0;
          r20 = r21 = r22 = r23 = 0.0;
          r30 = r31 = r32 = r33 = 0.0;

          for ( k = 0; k < p; k++ )
          {
            a0 = arow0[k];
            a1 = arow1[k];
            a2 = arow2[k];
            a3 = arow3[k];

            b0 = bcol0[k];
            b1 = bcol1[k];
            b2 = bcol2[k];
            b3 = bcol3[k];

            r00 += a0 * b0;
            r01 += a0 * b1;
            r02 += a0 * b2;
            r03 += a0 * b3;

            r10 += a1 * b0;
            r11 += a1 * b1;
            r12 += a1 * b2;
            r13 += a1 * b3;

            r20 += a2 * b0;
            r21 += a2 * b1;
            r22 += a2 * b2;
            r23 += a2 * b3;

            r30 += a3 * b0;
            r31 += a3 * b1;
            r32 += a3 * b2;
            r33 += a3 * b3;
          }

          r(i + 0,j + 0) = r00;
          r(i + 1,j + 0) = r10;
          r(i + 2,j + 0) = r20;
          r(i + 3,j + 0) = r30;

          r(i + 0,j + 1) = r01;
          r(i + 1,j + 1) = r11;
          r(i + 2,j + 1) = r21;
          r(i + 3,j + 1) = r31;

          r(i + 0,j + 2) = r02;
          r(i + 1,j + 2) = r12;
          r(i + 2,j + 2) = r22;
          r(i + 3,j + 2) = r32;

          r(i + 0,j + 3) = r03;
          r(i + 1,j + 3) = r13;
          r(i + 2,j + 3) = r23;
          r(i + 3,j + 3) = r33;

          arow0 = arow3 + ast0;
        }

        for ( ; i < m; i++ )
        {
          r00 = r01 = r02 = r03 = 0.0;

          for ( k = 0; k < p; k++ )
          {
            a0 = arow0[k];

            b0 = bcol0[k];
            b1 = bcol1[k];
            b2 = bcol2[k];
            b3 = bcol3[k];

            r00 += a0 * b0;
            r01 += a0 * b1;
            r02 += a0 * b2;
            r03 += a0 * b3;
          }

          r(i + 0,j + 0) = r00;
          r(i + 0,j + 1) = r01;
          r(i + 0,j + 2) = r02;
          r(i + 0,j + 3) = r03;

          arow0 += ast0;
        }

        bcol0 = bcol3 + b.stride (1);
      }

      switch ( n - j )
      {
      case 0:

        break;

      case 1:

        arow0 = a.addr ();

        for ( i = 0; i < m; i++ )
        {
          r00 = 0.0;

          for ( k = 0; k < p; k++ )
          {
            r00 += arow0[k] * bcol0[k];
          }

          r(i,j) = r00;

          arow0 += ast0;
        }

        break;

      case 2:

        arow0 =         a.addr   ();
        bcol1 = bcol0 + b.stride (1);

        for ( i = 0; i < m; i++ )
        {
          r00 = r01 = 0.0;

          for ( k = 0; k < p; k++ )
          {
            a0   =      arow0[k];
            r00 += a0 * bcol0[k];
            r01 += a0 * bcol1[k];
          }

          r(i,j + 0) = r00;
          r(i,j + 1) = r01;

          arow0     += ast0;
        }

        break;

      case 3:

        arow0 =         a.addr   ();
        bcol1 = bcol0 + b.stride (1);
        bcol2 = bcol1 + b.stride (1);

        for ( i = 0; i < m; i++ )
        {
          r00 = r01 = r02 = 0.0;

          for ( k = 0; k < p; k++ )
          {
            a0   =      arow0[k];
            r00 += a0 * bcol0[k];
            r01 += a0 * bcol1[k];
            r02 += a0 * bcol2[k];
          }

          r(i,j + 0) = r00;
          r(i,j + 1) = r01;
          r(i,j + 2) = r02;

          arow0     += ast0;
        }

        break;
      }
    }
    else
    {
      // Non-contiguous, row-major storage; no loop unrolling.

      const idx_t  ast1 = a.stride (1);
      const idx_t  bst0 = b.stride (0);

      for ( j = 0; j < n; j++ )
      {
        arow0 = a.addr();

        for ( i = 0; i < m; i++ )
        {
          r00 = 0.0;

          for ( k = 0; k < p; k++ )
          {
            r00 += arow0[k * ast1] * bcol0[k * bst0];
          }

          r(i,j) = r00;
          arow0 += a.stride (0);
        }

        bcol0 += b.stride (1);
      }
    }
  }
  else
  {
    // Assume column-major orientation.

    const double* JEM_RESTRICT  acol0 = a.addr ();
    double*       JEM_RESTRICT  rcol0 = r.addr ();

    double  a0,  a1,  a2,  a3;
    double  b00, b01, b02, b03;
    double  b10, b11, b12, b13;
    double  b20, b21, b22, b23;
    double  b30, b31, b32, b33;

    if ( r.stride(0) + a.stride(0) == 2_idx )
    {
      // Contiguous, column-major storage; full loop unrolling.

      const double* JEM_RESTRICT  acol1 = nullptr;
      const double* JEM_RESTRICT  acol2 = nullptr;
      const double* JEM_RESTRICT  acol3 = nullptr;
      double*       JEM_RESTRICT  rcol1 = nullptr;
      double*       JEM_RESTRICT  rcol2 = nullptr;
      double*       JEM_RESTRICT  rcol3 = nullptr;

      const idx_t  rst1 = r.stride (1);

      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          rcol0[i] = 0.0;
        }

        rcol0 += rst1;
      }

      for ( k = 0; k < (p - 3); k += 4 )
      {
        rcol0 =         r.addr   ();
        acol1 = acol0 + a.stride (1);
        acol2 = acol1 + a.stride (1);
        acol3 = acol2 + a.stride (1);

        for ( j = 0; j < (n - 3); j += 4 )
        {
          rcol1 = rcol0 + rst1;
          rcol2 = rcol1 + rst1;
          rcol3 = rcol2 + rst1;

          b00 = b(k + 0,j + 0);
          b10 = b(k + 1,j + 0);
          b20 = b(k + 2,j + 0);
          b30 = b(k + 3,j + 0);

          b01 = b(k + 0,j + 1);
          b11 = b(k + 1,j + 1);
          b21 = b(k + 2,j + 1);
          b31 = b(k + 3,j + 1);

          b02 = b(k + 0,j + 2);
          b12 = b(k + 1,j + 2);
          b22 = b(k + 2,j + 2);
          b32 = b(k + 3,j + 2);

          b03 = b(k + 0,j + 3);
          b13 = b(k + 1,j + 3);
          b23 = b(k + 2,j + 3);
          b33 = b(k + 3,j + 3);

JEM_IVDEP

          for ( i = 0; i < m; i++ )
          {
            a0 = acol0[i];
            a1 = acol1[i];
            a2 = acol2[i];
            a3 = acol3[i];

            rcol0[i] += (a0 * b00 + a1 * b10 + a2 * b20 + a3 * b30);
            rcol1[i] += (a0 * b01 + a1 * b11 + a2 * b21 + a3 * b31);
            rcol2[i] += (a0 * b02 + a1 * b12 + a2 * b22 + a3 * b32);
            rcol3[i] += (a0 * b03 + a1 * b13 + a2 * b23 + a3 * b33);
          }

          rcol0 = rcol3 + rst1;
        }

        for ( ; j < n; j++ )
        {
          b00 = b(k + 0,j);
          b10 = b(k + 1,j);
          b20 = b(k + 2,j);
          b30 = b(k + 3,j);

JEM_IVDEP

          for ( i = 0; i < m; i++ )
          {
            rcol0[i] += (acol0[i] * b00 + acol1[i] * b10 +
                         acol2[i] * b20 + acol3[i] * b30);
          }

          rcol0 += rst1;
        }

        acol0 = acol3 + a.stride (1);
      }

      switch ( p - k )
      {
      case 0:

        break;

      case 1:

        rcol0 = r.addr ();

        for ( j = 0; j < n; j++ )
        {
          b00 = b(k,j);

JEM_IVDEP

          for ( i = 0; i < m; i++ )
          {
            rcol0[i] += (b00 * acol0[i]);
          }

          rcol0 += rst1;
        }

        break;

      case 2:

        rcol0 =         r.addr   ();
        acol1 = acol0 + a.stride (1);

        for ( j = 0; j < n; j++ )
        {
          b00 = b(k + 0,j);
          b10 = b(k + 1,j);

JEM_IVDEP

          for ( i = 0; i < m; i++ )
          {
            rcol0[i] += (acol0[i] * b00 + acol1[i] * b10);
          }

          rcol0 += rst1;
        }

        break;

      case 3:

        rcol0 =         r.addr   ();
        acol1 = acol0 + a.stride (1);
        acol2 = acol1 + a.stride (1);

        for ( j = 0; j < n; j++ )
        {
          b00 = b(k + 0,j);
          b10 = b(k + 1,j);
          b20 = b(k + 2,j);

JEM_IVDEP

          for ( i = 0; i < m; i++ )
          {
            rcol0[i] += (acol0[i] * b00 + acol1[i] * b10 +
                         acol2[i] * b20);
          }

          rcol0 += rst1;
        }

        break;
      }
    }
    else
    {
      // Non-contiguous, column-major storage; no loop unrolling.

      const idx_t  ast0 = a.stride (0);
      const idx_t  rst0 = r.stride (0);

      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          rcol0[i * rst0] = 0.0;
        }

        rcol0 += r.stride (1);
      }

      for ( k = 0; k < p; k++ )
      {
        rcol0 = r.addr ();

        for ( j = 0; j < n; j++ )
        {
          b00 = b(k,j);

JEM_IVDEP

          for ( i = 0; i < m; i++ )
          {
            rcol0[i * rst0] += b00 * acol0[i * ast0];
          }

          rcol0 += r.stride (1);
        }

        acol0 += a.stride (1);
      }
    }
  }
}


JEM_END_PACKAGE( numeric )
