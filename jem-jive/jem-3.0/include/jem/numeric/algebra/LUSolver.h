
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

#ifndef JEM_NUMERIC_ALGEBRA_LUSOLVER_H
#define JEM_NUMERIC_ALGEBRA_LUSOLVER_H

#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class LUSolver
//-----------------------------------------------------------------------


class LUSolver
{
 public:

  typedef Array<double>         Vector;
  typedef Array<idx_t>          IdxVector;
  typedef Array<double,2>       Matrix;


  static bool                   factor

    ( const Matrix&               a,
      const Vector&               scale,
      const IdxVector&            jperm,
      int&                        sign );

  template <int N>

    static inline bool          factor

    ( Tuple<double,N,N>&          a,
      Tuple<idx_t,N>&             jperm,
      int&                        sign );

  static void                   solve

    ( const Vector&               lhs,
      const Matrix&               lu,
      const Vector&               rhs,
      const IdxVector&            jperm );

  static void                   solve

    ( const Matrix&               lhs,
      const Matrix&               lu,
      const Matrix&               rhs,
      const IdxVector&            jperm );

  template <int N>

    static inline void          solve

    ( Tuple<double,N>&            lhs,
      const Tuple<double,N,N>&    lu,
      const Tuple<double,N>&      rhs,
      const Tuple<idx_t,N>&       jperm )  noexcept;

  static bool                   invert

    ( const Matrix&               a,
      double&                     d );

  template <int N>

    static inline bool          invert

    ( Tuple<double,N,N>&          a,
      double&                     d );

  static double                 det

    ( const Matrix&               lu );

  template <int N>

    static inline double        det

    ( const Tuple<double,N,N>&    lu );


 public:

  // Should be private, but are used by some related functions.

  static inline bool            invert2x2_

    ( double*       JEM_RESTRICT  a,
      double&                     d );

  static bool                   invert3x3_

    ( double*       JEM_RESTRICT  a,
      double&                     d );

  static bool                   factor_

    ( idx_t                       n,
      int&                        sign,
      double*       JEM_RESTRICT  aa,
      double*       JEM_RESTRICT  scale,
      idx_t*        JEM_RESTRICT  jperm );


 private:

  static void                   solve1_

    ( idx_t                       n,
      idx_t                       k,
      double*       JEM_RESTRICT  lhs,
      const double* JEM_RESTRICT  lu,
      const idx_t*  JEM_RESTRICT  jperm )  noexcept;

  static void                   solve2_

    ( idx_t                       n,
      idx_t                       k,
      double*       JEM_RESTRICT  lhs,
      const double* JEM_RESTRICT  lu,
      const idx_t*  JEM_RESTRICT  jperm )  noexcept;

  static void                   msolve_

    ( idx_t                       m,
      idx_t                       n,
      double*       JEM_RESTRICT  lhs,
      const double* JEM_RESTRICT  lu,
      const idx_t*  JEM_RESTRICT  jperm )  noexcept;

  static bool                   invert_

    ( idx_t                       n,
      double*       JEM_RESTRICT  aa,
      double*       JEM_RESTRICT  bb,
      double*       JEM_RESTRICT  scale,
      idx_t*        JEM_RESTRICT  jperm,
      double&                     dd );

  static bool                   gaussi_

    ( idx_t                       n,
      double*       JEM_RESTRICT  aa,
      double*       JEM_RESTRICT  bb,
      double&                     dd );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


double                          det

  ( const Array<double,2>&        a );

template <int N>

  double                        det

  ( const Tuple<double,N,N>&      a );

Array<double,2>                 inverse

  ( const Array<double,2>&        a );

template <int N>

  Tuple<double,N,N>             inverse

  ( const Tuple<double,N,N>&      a );

inline bool                     invert

  ( const Array<double,2>&        a );

inline bool                     invert

  ( const Array<double,2>&        a,
    double&                       d );

template <int N>

  inline bool                   invert

  ( Tuple<double,N,N>&            a );

template <int N>

  inline bool                   invert

  ( Tuple<double,N,N>&            a,
    double&                       d );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class LUSolver
//=======================================================================

//-----------------------------------------------------------------------
//   factor (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline bool LUSolver::factor

  ( Tuple<double,N,N>&  a,
    Tuple<idx_t,N>&     jperm,
    int&                sign )

{
  double  scale[N];

  return factor_ ( N, sign, a.addr(), scale, jperm.addr() );
}


//-----------------------------------------------------------------------
//   solve (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline void LUSolver::solve

  ( Tuple<double,N>&          lhs,
    const Tuple<double,N,N>&  lu,
    const Tuple<double,N>&    rhs,
    const Tuple<idx_t,N>&     jperm ) noexcept

{
  lhs = rhs;

  solve1_ ( N, 0, lhs.addr(), lu.addr(), jperm.addr() );
}


//-----------------------------------------------------------------------
//   invert (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline bool LUSolver::invert

  ( Tuple<double,N,N>&  a,
    double&             d )

{
  Tuple<double,N,N>  b ( a );

  return gaussi_ ( N, a.addr(), b.addr(), d );
}


//-----------------------------------------------------------------------
//   invert2x2_
//-----------------------------------------------------------------------


inline bool LUSolver::invert2x2_

  ( double* JEM_RESTRICT  a,
    double&               d )

{
  // Matrix layout:
  //
  //       | a00 a01 |   | a[0] a[2] |
  //   a = |         | = |           |
  //       | a10 a11 |   | a[1] a[3] |

  d = a[0] * a[3] - a[2] * a[1];

  if ( jem::isTiny( d ) )
  {
    return false;
  }

  double  s = 1.0 / d;
  double  t = a[0];

  a[0] =  a[3] * s;
  a[1] = -a[1] * s;
  a[2] = -a[2] * s;
  a[3] =  t    * s;

  return true;
}


//-----------------------------------------------------------------------
//   det (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline double LUSolver::det ( const Tuple<double,N,N>& lu )

{
  double  d;
  int     i;

  d = 1.0;

  for ( i = 0; i < N; i++ )
  {
    d *= lu(i,i);
  }

  return (1.0 / d);
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   det (Tuple)
//-----------------------------------------------------------------------


template <int N>

  double det ( const Tuple<double,N,N>& a )

{
  double  lu[N * N];
  double  scale[N];
  idx_t   jperm[N];

  double  d;
  int     sign;
  int     i;


  for ( i = 0; i < N * N; i++ )
  {
    lu[i] = a[i];
  }

  if ( LUSolver::factor_( N, sign, lu, scale, jperm ) )
  {
    d = (double) sign;

    for ( i = 0; i < N; i++ )
    {
      d *= lu[i + i * N];
    }

    d = 1.0 / d;
  }
  else
  {
    d = 0.0;
  }

  return d;
}


//-----------------------------------------------------------------------
//   det (Tuple<double,1,1>)
//-----------------------------------------------------------------------


inline double det ( const Tuple<double,1,1>& a )
{
  return a(0,0);
}


//-----------------------------------------------------------------------
//   det (Tuple<double,2,2>)
//-----------------------------------------------------------------------


inline double det ( const Tuple<double,2,2>& a )
{
  return ( a(0,0) * a(1,1) - a(0,1) * a(1,0) );
}


//-----------------------------------------------------------------------
//   det (Tuple<double,3,3>)
//-----------------------------------------------------------------------


inline double det ( const Tuple<double,3,3>& a )
{
  return ( a(0,0) * ( a(1,1) * a(2,2) - a(1,2) * a(2,1) ) -
           a(1,0) * ( a(0,1) * a(2,2) - a(0,2) * a(2,1) ) +
           a(2,0) * ( a(0,1) * a(1,2) - a(0,2) * a(1,1) ) );
}


//-----------------------------------------------------------------------
//   inverse (Tuple)
//-----------------------------------------------------------------------


template <int N>

  Tuple<double,N,N>  inverse ( const Tuple<double,N,N>& a )

{
  Tuple<double,N,N>  tmp ( a );

  if ( ! invert( tmp ) )
  {
    throw ArithmeticException ( JEM_FUNC, "singular matrix" );
  }

  return tmp;
}


//-----------------------------------------------------------------------
//   invert (Array)
//-----------------------------------------------------------------------


inline bool invert ( const Array<double,2>& a )
{
  double  dummy;

  return invert ( a, dummy );
}


inline bool invert ( const Array<double,2>& a, double& d )
{
  return LUSolver::invert ( a, d );
}


//-----------------------------------------------------------------------
//   invert (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline bool invert ( Tuple<double,N,N>& a )

{
  double  dummy;

  return invert ( a, dummy );
}


template <int N>

  inline bool invert ( Tuple<double,N,N>& a, double& d )

{
  return LUSolver::invert ( a, d );
}


//-----------------------------------------------------------------------
//   invert (Tuple<double,1,1>)
//-----------------------------------------------------------------------


inline bool invert ( Tuple<double,1,1>& a, double& d )
{
  d = a(0,0);

  if ( jem::isTiny( d ) )
  {
    return false;
  }

  a(0,0) = 1.0 / d;

  return true;
}


//-----------------------------------------------------------------------
//   invert (Tuple<double,2,2>)
//-----------------------------------------------------------------------


inline bool invert ( Tuple<double,2,2>& a, double& d )
{
  return LUSolver::invert2x2_ ( a.addr(), d );
}


//-----------------------------------------------------------------------
//   invert (Tuple<double,3,3>)
//-----------------------------------------------------------------------


inline bool invert ( Tuple<double,3,3>& a, double& d )
{
  return LUSolver::invert3x3_ ( a.addr(), d );
}


JEM_END_PACKAGE( numeric )

#endif
