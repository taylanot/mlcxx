
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

#ifndef JEM_NUMERIC_ALGEBRA_EIGENUTILS_H
#define JEM_NUMERIC_ALGEBRA_EIGENUTILS_H

#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class EigenUtils
//-----------------------------------------------------------------------


class EigenUtils
{
 public:

  typedef Array<double>       Vector;
  typedef Array<double,2>     Matrix;


  static void                 symSolve

    ( const Vector&             evals,
      const Matrix&             symmat );

  static void                 symSolve

    ( const Vector&             evals,
      const Matrix&             evecs,
      const Matrix&             symmat );

  static void                 triSolve

    ( const Vector&             diag,
      const Vector&             subdiag );

  static void                 triSolve

    ( const Vector&             diag,
      const Vector&             subdiag,
      const Matrix&             evecs );

  static void                 triReduce

    ( const Vector&             diag,
      const Vector&             subdiag,
      const Matrix&             symmat );


  template <int N>

    static inline void        symSolve

    ( Tuple<double,N>&          evals,
      Tuple<double,N,N>&        symmat );

  template <int N>

    static inline void        symSolve

    ( Tuple<double,N>&          evals,
      Tuple<double,N,N>&        evecs,
      const Tuple<double,N,N>&  symmat );

  template <int N>

    static inline void        triSolve

    ( Tuple<double,N>&          diag,
      Tuple<double,N>&          subdiag );

  template <int N>

    static inline void        triSolve

    ( Tuple<double,N>&          diag,
      Tuple<double,N>&          subdiag,
      Tuple<double,N,N>&        evecs );

  template <int N>

    static inline void        triReduce

    ( Tuple<double,N>&          diag,
      Tuple<double,N>&          subdiag,
      Tuple<double,N,N>&        symmat );


 private:

  static void                 triSolve_

    ( idx_t                     n,
      double* JEM_RESTRICT      dd,
      double* JEM_RESTRICT      sd,
      double* JEM_RESTRICT      ev );

  static void                 triReduce_

    ( idx_t                     n,
      double* JEM_RESTRICT      dd,
      double* JEM_RESTRICT      sd,
      double* JEM_RESTRICT      aa );

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   symSolve
//-----------------------------------------------------------------------


template <int N>

  inline void  EigenUtils::symSolve

  ( Tuple<double,N>&    evals,
    Tuple<double,N,N>&  symmat )

{
  Tuple<double,N>  sd;

  triReduce_ ( N, evals.addr(), sd.addr(), symmat.addr() );
  triSolve_  ( N, evals.addr(), sd.addr(), nullptr       );
}


template <int N>

  inline void  EigenUtils::symSolve

  (       Tuple<double,N>&    evals,
          Tuple<double,N,N>&  evecs,
    const Tuple<double,N,N>&  symmat )

{
  Tuple<double,N>  sd;

  evecs = symmat;

  triReduce_ ( N, evals.addr(), sd.addr(), evecs.addr() );
  triSolve_  ( N, evals.addr(), sd.addr(), evecs.addr() );
}


//-----------------------------------------------------------------------
//   triSolve
//-----------------------------------------------------------------------


template <int N>

  inline void  EigenUtils::triSolve

  ( Tuple<double,N>&    diag,
    Tuple<double,N>&    subdiag )

{
  triSolve_ ( N, diag.addr(), subdiag.addr(), nullptr );
}


template <int N>

  inline void  EigenUtils::triSolve

  ( Tuple<double,N>&    diag,
    Tuple<double,N>&    subdiag,
    Tuple<double,N,N>&  evecs )

{
  triSolve_ ( N, diag.addr(), subdiag.addr(), evecs.addr() );
}


//-----------------------------------------------------------------------
//   triReduce
//-----------------------------------------------------------------------


template <int N>

  inline void  EigenUtils::triReduce

  ( Tuple<double,N>&    diag,
    Tuple<double,N>&    subdiag,
    Tuple<double,N,N>&  symmat )

{
  triReduce_ ( N, diag.addr(), subdiag.addr(), symmat.addr() );
}


JEM_END_PACKAGE( numeric )

#endif
