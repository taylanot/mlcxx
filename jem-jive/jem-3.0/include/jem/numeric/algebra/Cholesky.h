
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

#ifndef JEM_NUMERIC_ALGEBRA_CHOLESKY_H
#define JEM_NUMERIC_ALGEBRA_CHOLESKY_H

#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class Cholesky
//-----------------------------------------------------------------------


class Cholesky
{
 public:

  typedef Array<double>         Vector;
  typedef Array<double,2>       Matrix;


  static bool                   factor

    ( const Matrix&               a );

  template <int N>

    static inline bool          factor

    ( Tuple<double,N,N>&          a );

  static void                   fsub

    ( const Vector&               x,
      const Matrix&               a );

  static void                   fsub

    ( const Matrix&               x,
      const Matrix&               a );

  template <int N>

    static inline void          fsub

    ( Tuple<double,N>&            x,
      const Tuple<double,N,N>&    a )  noexcept;

  template <int M, int N>

    static inline void          fsub

    ( Tuple<double,M,N>&          x,
      const Tuple<double,M,M>&    a )  noexcept;

  static void                   fsubt

    ( const Matrix&               x,
      const Matrix&               a );

  template <int M, int N>

    static inline void          fsubt

    ( Tuple<double,M,N>&          x,
      const Tuple<double,N,N>&    a )  noexcept;

  static void                   bsub

    ( const Vector&               x,
      const Matrix&               a );

  static void                   bsub

    ( const Matrix&               x,
      const Matrix&               a );

  template <int N>

    static void                 bsub

    ( Tuple<double,N>&            x,
      const Tuple<double,N,N>&    a )  noexcept;

  template <int M, int N>

    static inline void          bsub

    ( Tuple<double,M,N>&         x,
      const Tuple<double,M,M>&   a )  noexcept;

  static void                   solve

    ( const Vector&               x,
      const Matrix&               a );

  static void                   solve

    ( const Matrix&               x,
      const Matrix&               a );

  template <int N>

    static inline void          solve

    ( Tuple<double,N>&            x,
      const Tuple<double,N,N>&    a )  noexcept;

  template <int M, int N>

    static inline void          solve

    ( Tuple<double,M,N>&          x,
      const Tuple<double,M,M>&    a )  noexcept;

  static bool                   invert

    ( const Matrix&               a );

  template <int N>

    static inline bool          invert

    ( Tuple<double,N,N>&          a );


 private:

  static bool                   factor_

    ( idx_t                       n,
      double*       JEM_RESTRICT  a );

  static void                   fsub1_

    ( idx_t                       n,
      idx_t                       k,
      double*       JEM_RESTRICT  x,
      const double* JEM_RESTRICT  a )  noexcept;

  static void                   fsub2_

    ( idx_t                       n,
      idx_t                       k,
      double*       JEM_RESTRICT  x,
      const double* JEM_RESTRICT  a )  noexcept;

  static void                   mfsub_

    ( idx_t                       m,
      idx_t                       n,
      double*       JEM_RESTRICT  x,
      const double* JEM_RESTRICT  a )  noexcept;

  static void                   fsubt_

    ( idx_t                       m,
      idx_t                       n,
      double*       JEM_RESTRICT  x,
      const double* JEM_RESTRICT  a )  noexcept;

  static void                   bsub1_

    ( idx_t                       n,
      double*       JEM_RESTRICT  x,
      const double* JEM_RESTRICT  a )  noexcept;

  static void                   bsub2_

    ( idx_t                       n,
      double*       JEM_RESTRICT  x,
      const double* JEM_RESTRICT  a )  noexcept;

  static void                   mbsub_

    ( idx_t                       m,
      idx_t                       n,
      double*       JEM_RESTRICT  x,
      const double* JEM_RESTRICT  a )  noexcept;

  static bool                   invert_

    ( idx_t                       n,
      double*       JEM_RESTRICT  a,
      double*       JEM_RESTRICT  b );

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Cholesky
//=======================================================================

//-----------------------------------------------------------------------
//   factor (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline bool Cholesky::factor ( Tuple<double,N,N>& a )

{
  return factor_ ( N, a.addr() );
}


//-----------------------------------------------------------------------
//   fsub (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline void Cholesky::fsub

  ( Tuple<double,N>&          x,
    const Tuple<double,N,N>&  a ) noexcept

{
  fsub1_ ( N, 0, x.addr(), a.addr() );
}


template <int M, int N>

  inline void Cholesky::fsub

  ( Tuple<double,M,N>&        x,
    const Tuple<double,M,M>&  a ) noexcept

{
  mfsub_ ( M, N, x.addr(), a.addr() );
}


//-----------------------------------------------------------------------
//   fsubt (Tuple)
//-----------------------------------------------------------------------


template <int M, int N>

  inline void Cholesky::fsubt

  ( Tuple<double,M,N>&        x,
    const Tuple<double,N,N>&  a ) noexcept

{
  fsubt_ ( M, N, x.addr(), a.addr() );
}


//-----------------------------------------------------------------------
//   bsub (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline void Cholesky::bsub

    ( Tuple<double,N>&          x,
      const Tuple<double,N,N>&  a ) noexcept

{
  bsub1_ ( N, x.addr(), a.addr() );
}


template <int M, int N>

  inline void Cholesky::bsub

  ( Tuple<double,M,N>&        x,
    const Tuple<double,M,M>&  a ) noexcept

{
  mbsub_ ( M, N, x.addr(), a.addr() );
}


//-----------------------------------------------------------------------
//   solve (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline void Cholesky::solve

    ( Tuple<double,N>&          x,
      const Tuple<double,N,N>&  a ) noexcept

{
  fsub1_ ( N, 0, x.addr(), a.addr() );
  bsub1_ ( N,    x.addr(), a.addr() );
}


template <int M, int N>

  inline void Cholesky::solve

  ( Tuple<double,M,N>&        x,
    const Tuple<double,M,M>&  a ) noexcept

{
  mfsub_ ( M, N, x.addr(), a.addr() );
  mbsub_ ( M, N, x.addr(), a.addr() );
}


//-----------------------------------------------------------------------
//   invert (Tuple)
//-----------------------------------------------------------------------


template <int N>

  inline bool Cholesky::invert ( Tuple<double,N,N>& a )

{
  Tuple<double,N,N>  b ( a );

  return invert_ ( N, a.addr(), b.addr() );
}


JEM_END_PACKAGE( numeric )

#endif
