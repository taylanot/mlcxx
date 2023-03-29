
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_GEOM_UTILITIES_H
#define JIVE_GEOM_UTILITIES_H

#include <jive/Array.h>
#include <jive/geom/Tuple.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


void                        getGrad2Map

  ( const IdxVector&          imap,
    const IdxVector&          jmap,
    idx_t                     rank );

template <int M, int N>

  inline void               getJacobi

  ( Tuple<double,M,N>&        ja,
    const Matrix&             grads,
    const Matrix&             coords );

template <int M, int N>

  inline void               getJacobi

  ( Tuple<double,M,N>&        ja,
    const Cubix&              grads,
    idx_t                     ip,
    const Matrix&             coords );

template <int M, int N>

  inline void               applyJacobi

  ( const Matrix&             ggrads,
    const Tuple<double,M,N>&  ja,
    const Matrix&             sgrads );

template <int M, int N>

  inline void               applyJacobi

  ( const Cubix&              ggrads,
    const Tuple<double,M,N>&  ja,
    const Cubix&              sgrads,
    idx_t                     ip );

void                        getQujobi

  ( const Matrix&             qu,
    const Matrix&             ja );

template <int M, int N, int P, int Q>

  void                      getQujobi

  ( Tuple<double,M,N>&        qu,
    const Tuple<double,P,Q>&  ja );

template <int M>

  inline void               getGlobalPoint

  ( Tuple<double,M>&          x,
    const Vector&             funcs,
    const Matrix&             coords );

template <int M>

  inline void               getGlobalPoint

  ( Tuple<double,M>&          x,
    const Matrix&             funcs,
    idx_t                     ip,
    const Matrix&             coords );






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getJacobi
//-----------------------------------------------------------------------


template <int M, int N>

  inline void         getJacobi

  ( Tuple<double,M,N>&  ja,
    const Matrix&       grads,
    const Matrix&       coords )

{
  JEM_ASSERT2 ( grads .size(0) == M &&
                coords.size(0) == N &&
                coords.size(1) == grads.size(1),
                "Array shape mismatch" );

  double* JEM_RESTRICT  jp = ja.addr ();

  const idx_t  p = coords.size (1);


  for ( int i = 0; i < M * N; i++ )
  {
    jp[i] = 0.0;
  }

  if ( ((grads .stride(0) - 1) | (grads .stride(1) - M) |
        (coords.stride(0) - 1) | (coords.stride(1) - N)) == 0 )
  {
    const double* JEM_RESTRICT  gp = grads .addr ();
    const double* JEM_RESTRICT  cp = coords.addr ();

    for ( idx_t k = 0; k < p; k++ )
    {
      for ( int j = 0; j < N; j++ )
      {
        double  c = cp[j + k * N];

        for ( int i = 0; i < M; i++ )
        {
          jp[i + j * M] += c * gp[i + k * M];
        }
      }
    }
  }
  else
  {
    double  g[M];

    for ( idx_t k = 0; k < p; k++ )
    {
      for ( int i = 0; i < M; i++ )
      {
        g[i] = grads(i,k);
      }

      for ( int j = 0; j < N; j++ )
      {
        double c = coords(j,k);

        for ( int i = 0; i < M; i++ )
        {
          jp[i + j * M] += c * g[i];
        }
      }
    }
  }
}


template <int M, int N>

  inline void         getJacobi

  ( Tuple<double,M,N>&  ja,
    const Cubix&        grads,
    idx_t               ip,
    const Matrix&       coords )

{
  JEM_ASSERT2 ( grads .size(0) == M &&
                coords.size(0) == N &&
                coords.size(1) == grads.size(1),
                "Array shape mismatch" );

  double* JEM_RESTRICT  jp = ja.addr ();

  const idx_t  p = coords.size (1);


  for ( int i = 0; i < M * N; i++ )
  {
    jp[i] = 0.0;
  }

  if ( ((grads .stride(0) - 1) | (grads .stride(1) - M) |
        (coords.stride(0) - 1) | (coords.stride(1) - N)) == 0 )
  {
    const double* JEM_RESTRICT  gp = grads .addr ();
    const double* JEM_RESTRICT  cp = coords.addr ();

    gp += (ip * grads.stride(2));

    for ( idx_t k = 0; k < p; k++ )
    {
      for ( int j = 0; j < N; j++ )
      {
        double  c = cp[j + k * N];

        for ( int i = 0; i < M; i++ )
        {
          jp[i + j * M] += c * gp[i + k * M];
        }
      }
    }
  }
  else
  {
    double  g[M];

    for ( idx_t k = 0; k < p; k++ )
    {
      for ( int i = 0; i < M; i++ )
      {
        g[i] = grads(i,k,ip);
      }

      for ( int j = 0; j < N; j++ )
      {
        double  c = coords(j,k);

        for ( int i = 0; i < M; i++ )
        {
          jp[i + j * M] += c * g[i];
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   applyJacobi
//-----------------------------------------------------------------------


template <int M, int N>

  inline void               applyJacobi

  ( const Matrix&             ggrads,
    const Tuple<double,M,N>&  ja,
    const Matrix&             sgrads )

{
  JEM_ASSERT2 ( ggrads.size(0) == M &&
                sgrads.size(0) == N &&
                sgrads.size(1) == ggrads.size(1),
                "Array shape mismatch" );

  const double* JEM_RESTRICT  jp = ja.addr ();

  const idx_t  p = ggrads.size (1);


  if ( ((ggrads.stride(0) - 1) | (ggrads.stride(1) - M) |
        (sgrads.stride(0) - 1) | (sgrads.stride(1) - N)) == 0 )
  {
    double*       JEM_RESTRICT  gp = ggrads.addr ();
    const double* JEM_RESTRICT  sp = sgrads.addr ();

    for ( idx_t k = 0; k < p; k++ )
    {
      for ( int i = 0; i < M; i++ )
      {
        double  t = 0.0;

        for ( int j = 0; j < N; j++ )
        {
          t += jp[i + j * M] * sp[j + k * N];
        }

        gp[i + k * M] = t;
      }
    }
  }
  else
  {
    double  s[N];

    for ( idx_t k = 0; k < p; k++ )
    {
      for ( int j = 0; j < N; j++ )
      {
        s[j] = sgrads(j,k);
      }

      for ( int i = 0; i < M; i++ )
      {
        double  t = 0.0;

        for ( int j = 0; j < N; j++ )
        {
          t += jp[i + j * M] * s[j];
        }

        ggrads(i,k) = t;
      }
    }
  }
}


template <int M, int N>

  inline void               applyJacobi

  ( const Cubix&              ggrads,
    const Tuple<double,M,N>&  ja,
    const Cubix&              sgrads,
    idx_t                     ip )

{
  JEM_ASSERT2 ( ggrads.size(0) == M &&
                sgrads.size(0) == N &&
                sgrads.size(1) == ggrads.size(1),
                "Array shape mismatch" );

  const double* JEM_RESTRICT  jp = ja.addr ();

  const idx_t  p = ggrads.size (1);


  if ( ((ggrads.stride(0) - 1) | (ggrads.stride(1) - M) |
        (sgrads.stride(0) - 1) | (sgrads.stride(1) - N)) == 0 )
  {
    double*       JEM_RESTRICT  gp = ggrads.addr ();
    const double* JEM_RESTRICT  sp = sgrads.addr ();

    gp += (ip * ggrads.stride(2));
    sp += (ip * sgrads.stride(2));

    for ( idx_t k = 0; k < p; k++ )
    {
      for ( int i = 0; i < M; i++ )
      {
        double  t = 0.0;

        for ( int j = 0; j < N; j++ )
        {
          t += jp[i + j * M] * sp[j + k * N];
        }

        gp[i + k * M] = t;
      }
    }
  }
  else
  {
    double  s[N];

    for ( idx_t k = 0; k < p; k++ )
    {
      for ( int j = 0; j < N; j++ )
      {
        s[j] = sgrads(j,k,ip);
      }

      for ( int i = 0; i < M; i++ )
      {
        double  t = 0.0;

        for ( int j = 0; j < N; j++ )
        {
          t += jp[i + j * M] * s[j];
        }

        ggrads(i,k,ip) = t;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   macro JIVE_GET_QUJOBI
//-----------------------------------------------------------------------


#undef  JIVE_GET_QUJOBI
#define JIVE_GET_QUJOBI( qu, ja, m, n, itype )          \
                                                        \
  itype  i, ii, ik;                                     \
  itype  j, jj, jk;                                     \
                                                        \
                                                        \
  for ( i = 0; i < m; i++ )                             \
  {                                                     \
    for ( j = 0; j < n; j++ )                           \
    {                                                   \
      qu(i,j) = ja(i,j) * ja(i,j);                      \
    }                                                   \
                                                        \
    jk = n;                                             \
                                                        \
    for ( j = 0; j < n; j++ )                           \
    {                                                   \
      for ( jj = j + 1; jj < n; jj++, jk++ )            \
      {                                                 \
        qu(i,jk) = 2.0 * ja(i,j) * ja(i,jj);            \
      }                                                 \
    }                                                   \
  }                                                     \
                                                        \
  ik = m;                                               \
                                                        \
  for ( i = 0; i < m; i++ )                             \
  {                                                     \
    for ( ii = i + 1; ii < m; ii++, ik++ )              \
    {                                                   \
      for ( j = 0; j < n; j++ )                         \
      {                                                 \
        qu(ik,j) = ja(i,j) * ja(ii,j);                  \
      }                                                 \
                                                        \
      jk = n;                                           \
                                                        \
      for ( j = 0; j < n; j++ )                         \
      {                                                 \
        for ( jj = j + 1; jj < n; jj++, jk++ )          \
        {                                               \
          qu(ik,jk) = ja(i,j)  * ja(ii,jj) +            \
                      ja(i,jj) * ja(ii,j);              \
        }                                               \
      }                                                 \
    }                                                   \
  }


//-----------------------------------------------------------------------
//   getQujobi
//-----------------------------------------------------------------------


template <int M, int N, int P, int Q>

  void                        getQujobi

  ( Tuple<double,M,N>&          qu,
    const Tuple<double,P,Q>&    ja )

{
  JIVE_GET_QUJOBI( qu, ja, P, Q, int )
}


//-----------------------------------------------------------------------
//   getQujobi (overloaded versions)
//-----------------------------------------------------------------------


inline void                 getQujobi

  ( Tuple<double,1,1>&        qu,
    const Tuple<double,1,1>&  ja )

{
  qu(0,0) = ja(0,0) * ja(0,0);
}


inline void                 getQujobi

  ( Tuple<double,3,3>&        qu,
    const Tuple<double,2,2>&  ja )

{
  qu(0,0) = ja(0,0) * ja(0,0);
  qu(1,0) = ja(1,0) * ja(1,0);
  qu(2,0) = ja(0,0) * ja(1,0);

  qu(0,1) = ja(0,1) * ja(0,1);
  qu(1,1) = ja(1,1) * ja(1,1);
  qu(2,1) = ja(0,1) * ja(1,1);

  qu(0,2) = 2.0 * ja(0,0) * ja(0,1);
  qu(1,2) = 2.0 * ja(1,0) * ja(1,1);
  qu(2,2) = ja(0,0) * ja(1,1) + ja(0,1) * ja(1,0);
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


template <int M>

  inline void       getGlobalPoint

  ( Tuple<double,M>&  x,
    const Vector&     funcs,
    const Matrix&     coords )

{
  JEM_ASSERT2 ( coords.size(0) == M &&
                coords.size(1) == funcs.size(),
                "Array size mismatch" );

  double* JEM_RESTRICT  xp = x.addr ();

  const idx_t  n = funcs.size ();


  for ( int i = 0; i < M; i++ )
  {
    xp[i] = 0.0;
  }

  if ( ((funcs .stride()  - 1) |
        (coords.stride(0) - 1) | (coords.stride(1) - M)) == 0 )
  {
    const double* JEM_RESTRICT  fp = funcs .addr ();
    const double* JEM_RESTRICT  cp = coords.addr ();

    for ( idx_t j = 0; j < n; j++ )
    {
      double  f = fp[j];

      for ( int i = 0; i < M; i++ )
      {
        xp[i] += f * cp[i + j * M];
      }
    }
  }
  else
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      double  f = funcs[j];

      for ( int i = 0; i < M; i++ )
      {
        xp[i] += f * coords(i,j);
      }
    }
  }
}


template <int M>

  inline void       getGlobalPoint

  ( Tuple<double,M>&  x,
    const Matrix&     funcs,
    idx_t             ip,
    const Matrix&     coords )

{
  JEM_ASSERT2 ( coords.size(0) == M &&
                coords.size(1) == funcs.size(0),
                "Array size mismatch" );

  double* JEM_RESTRICT  xp = x.addr ();

  const idx_t  n = funcs.size (0);


  for ( int i = 0; i < M; i++ )
  {
    xp[i] = 0.0;
  }

  if ( ((funcs .stride(0) - 1) |
        (coords.stride(0) - 1) | (coords.stride(1) - M)) == 0 )
  {
    const double* JEM_RESTRICT  fp = funcs .addr ();
    const double* JEM_RESTRICT  cp = coords.addr ();

    fp += (ip * funcs.stride(1));

    for ( idx_t j = 0; j < n; j++ )
    {
      double  f = fp[j];

      for ( int i = 0; i < M; i++ )
      {
        xp[i] += f * cp[i + j * M];
      }
    }
  }
  else
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      double  f = funcs(j,ip);

      for ( int i = 0; i < M; i++ )
      {
        xp[i] += f * coords(i,j);
      }
    }
  }
}


JIVE_END_PACKAGE( geom )

#endif
