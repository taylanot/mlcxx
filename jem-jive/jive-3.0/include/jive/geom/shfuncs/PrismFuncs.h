
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

#ifndef JIVE_GEOM_PRISMFUNCS_H
#define JIVE_GEOM_PRISMFUNCS_H

#include <jem/base/tuple/Tuple.h>
#include <jive/geom/shfuncs/LineFuncs.h>
#include <jive/geom/shfuncs/TriangFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class PrismFuncs
//-----------------------------------------------------------------------


template <class Base, class Line>

  class PrismFuncs

{
 public:

  static const int            RANK        = Base::RANK + 1;
  static const int            FUNC_COUNT  = Base::FUNC_COUNT *
                                            Line::FUNC_COUNT;
  static const int            GRAD_COUNT  = RANK;
  static const int            GRAD2_COUNT = Base::GRAD2_COUNT + RANK;


  typedef Tuple
    <double,RANK>             Point;

  typedef Tuple
    <double,FUNC_COUNT>       FuncVector;

  typedef Tuple
    <double,
     GRAD_COUNT,FUNC_COUNT>   GradMatrix;

  typedef Tuple
    <double,
     GRAD2_COUNT,FUNC_COUNT>  Grad2Matrix;


  static inline void          evalFuncs

    ( FuncVector&               f,
      const Point&              u );

  static inline void          evalGrads

    ( FuncVector&               f,
      GradMatrix&               g,
      const Point&              u );

  static inline void          evalGrads2

    ( FuncVector&               f,
      GradMatrix&               g,
      Grad2Matrix&              h,
      const Point&              u );

};


//-----------------------------------------------------------------------
//   helper functions
//-----------------------------------------------------------------------


template <int N, class VectorType>

  inline void             getBasePoint

  ( Tuple<double,N>&        p,
    const VectorType&       u );

template <class VectorType>

  inline void             getBasePoint

  ( Tuple<double,1>&        p,
    const VectorType&       u );

template <class VectorType>

  inline void             getBasePoint

  ( Tuple<double,2>&        p,
    const VectorType&       u );

template <class VectorType>

  inline void             getBasePoint

  ( Tuple<double,3>&        p,
    const VectorType&       u );


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_PRISM_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_PRISM_FUNCS
#define JIVE_EVAL_PRISM_FUNCS( f, u, Base, Line )       \
                                                        \
  do                                                    \
  {                                                     \
    jem::Tuple < double,                                \
                 Base::FUNC_COUNT >  f1;                \
    jem::Tuple < double,                                \
                 Line::FUNC_COUNT >  f0;                \
    jem::Tuple < double,                                \
                 Base::RANK >        p;                 \
                                                        \
    getBasePoint    ( p, u );                           \
                                                        \
    Base::evalFuncs ( f1, p );                          \
    Line::evalFuncs ( f0, u[Base::RANK] );              \
                                                        \
    for ( int i = 0; i < Line::FUNC_COUNT; i++ )        \
    {                                                   \
      for ( int j = 0; j < Base::FUNC_COUNT; j++ )      \
      {                                                 \
        f[i * Base::FUNC_COUNT + j] = f1[j] * f0[i];    \
      }                                                 \
    }                                                   \
  }                                                     \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_PRISM_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_PRISM_GRADS
#define JIVE_EVAL_PRISM_GRADS( f, g, u, Base, Line )    \
                                                        \
  do                                                    \
  {                                                     \
    jem::Tuple < double,                                \
                 Base::FUNC_COUNT >  f1;                \
    jem::Tuple < double,                                \
                 Line::FUNC_COUNT >  f0;                \
    jem::Tuple < double,                                \
                 Base::GRAD_COUNT,                      \
                 Base::FUNC_COUNT >  g0;                \
    jem::Tuple < double, 1,                             \
                 Line::FUNC_COUNT >  g1;                \
    jem::Tuple < double,                                \
                 Base::RANK >        p;                 \
                                                        \
    getBasePoint    ( p,  u );                          \
                                                        \
    Base::evalGrads ( f1, g0, p );                      \
    Line::evalGrads ( f0, g1, u[Base::RANK] );          \
                                                        \
    for ( int i = 0; i < Line::FUNC_COUNT; i++ )        \
    {                                                   \
      for ( int j = 0; j < Base::FUNC_COUNT; j++ )      \
      {                                                 \
        int  ifunc = i * Base::FUNC_COUNT + j;          \
                                                        \
        f[ifunc] = f1[j] * f0[i];                       \
                                                        \
        for ( int ii = 0; ii < Base::RANK; ii++ )       \
        {                                               \
          g(ii,ifunc) = g0(ii,j) * f0[i];               \
        }                                               \
                                                        \
        g(Base::RANK,ifunc) = f1[j] * g1(0,i);          \
      }                                                 \
    }                                                   \
  }                                                     \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_PRISM_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_PRISM_GRADS2
#define JIVE_EVAL_PRISM_GRADS2( f, g, h, u,             \
                                Base, Line )            \
                                                        \
  do                                                    \
  {                                                     \
    jem::Tuple < double,                                \
                 Base::FUNC_COUNT >  f1;                \
    jem::Tuple < double,                                \
                 Line::FUNC_COUNT >  f0;                \
    jem::Tuple < double,                                \
                 Base::GRAD_COUNT,                      \
                 Base::FUNC_COUNT >  g0;                \
    jem::Tuple < double, 1,                             \
                 Line::FUNC_COUNT >  g1;                \
    jem::Tuple < double,                                \
                 Base::GRAD2_COUNT,                     \
                 Base::FUNC_COUNT >  h0;                \
    jem::Tuple < double, 1,                             \
                 Line::FUNC_COUNT >  h1;                \
    jem::Tuple < double,                                \
                 Base::RANK >        p;                 \
                                                        \
    getBasePoint     ( p,  u );                         \
                                                        \
    Base::evalGrads2 ( f1, g0, h0, p );                 \
    Line::evalGrads2 ( f0, g1, h1, u[Base::RANK] );     \
                                                        \
    for ( int i = 0; i < Line::FUNC_COUNT; i++ )        \
    {                                                   \
      for ( int j = 0; j < Base::FUNC_COUNT; j++ )      \
      {                                                 \
        int  ifunc = i * Base::FUNC_COUNT + j;          \
                                                        \
        f[ifunc] = f1[j] * f0[i];                       \
                                                        \
        for ( int ii = 0; ii < Base::RANK; ii++ )       \
        {                                               \
          g(ii,ifunc) = g0(ii,j) * f0[i];               \
          h(ii,ifunc) = h0(ii,j) * f0[i];               \
        }                                               \
                                                        \
        g(Base::RANK,ifunc) = f1[j] * g1(0,i);          \
        h(Base::RANK,ifunc) = f1[j] * h1(0,i);          \
                                                        \
        int  ik = Base::RANK + 1;                       \
        int  jk = Base::RANK;                           \
                                                        \
        for ( int ii = 0; ii < Base::RANK; ii++, ik++ ) \
        {                                               \
          for ( int jj = ii + 1; jj < Base::RANK;       \
                                 jj++, ik++, jk++ )     \
          {                                             \
            h(ik,ifunc) = h0(jk,j) * f0[i];             \
          }                                             \
                                                        \
          h(ik,ifunc) = g0(ii,j) * g1(0,i);             \
        }                                               \
      }                                                 \
    }                                                   \
  }                                                     \
  while ( false )





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   helper functions
//=======================================================================


template <int N, class VectorType>

  inline void         getBasePoint

  ( Tuple<double,N>&    p,
    const VectorType&   u )

{
  for ( int i = 0; i < N; i++ )
  {
    p[i] = u[i];
  }
}


template <class VectorType>

  inline void         getBasePoint

  ( Tuple<double,1>&    p,
    const VectorType&   u )

{
  p[0] = u[0];
}


template <class VectorType>

  inline void         getBasePoint

  ( Tuple<double,2>&    p,
    const VectorType&   u )

{
  p[0] = u[0];
  p[1] = u[1];
}


template <class VectorType>

  inline void         getBasePoint

  ( Tuple<double,3>&    p,
    const VectorType&   u )

{
  p[0] = u[0];
  p[1] = u[1];
  p[2] = u[2];
}


//=======================================================================
//   class PrismFuncs
//=======================================================================


template <class Base, class Line>

  inline void PrismFuncs<Base,Line>::evalFuncs

  ( FuncVector&    f,
    const Point&   u )

{
  JIVE_EVAL_PRISM_FUNCS( f, u, Base, Line );
}


template <class Base, class Line>

  inline void PrismFuncs<Base,Line>::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS( f, g, u, Base, Line );
}


template <class Base, class Line>

  inline void PrismFuncs<Base,Line>::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u, Base, Line );
}


JIVE_END_PACKAGE( geom )

#endif
