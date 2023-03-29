
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

#ifndef JIVE_GEOM_TETFUNCS_H
#define JIVE_GEOM_TETFUNCS_H

#include <jive/geom/shfuncs/ShapeFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class Tet4Funcs
//-----------------------------------------------------------------------


class Tet4Funcs : public ShapeFuncs<3,4>
{
 public:

  static inline void    evalFuncs

    ( FuncVector&         f,
      const Point&        u );

  static inline void    evalGrads

    ( FuncVector&         f,
      GradMatrix&         g,
      const Point&        u );

  static inline void    evalGrads2

    ( FuncVector&         f,
      GradMatrix&         g,
      Grad2Matrix&        h,
      const Point&        u );

};


//-----------------------------------------------------------------------
//   class Tet10Funcs
//-----------------------------------------------------------------------


class Tet10Funcs : public ShapeFuncs<3,10>
{
 public:

  static const double   GRADS2[6][10];


  static inline void    evalFuncs

    ( FuncVector&         f,
      const Point&        u );

  static inline void    evalGrads

    ( FuncVector&         f,
      GradMatrix&         g,
      const Point&        u );

  static inline void    evalGrads2

    ( FuncVector&         f,
      GradMatrix&         g,
      Grad2Matrix&        h,
      const Point&        u );

};


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TET4_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TET4_FUNCS
#define JIVE_EVAL_TET4_FUNCS( f, u )                      \
                                                          \
  f[0] = 1.0 - u[0] - u[1] - u[2];                        \
  f[1] = u[0];                                            \
  f[2] = u[1];                                            \
  f[3] = u[2]


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TET4_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TET4_GRADS
#define JIVE_EVAL_TET4_GRADS( f, g, u )                   \
                                                          \
  f[0]   = 1.0 - u[0] - u[1] - u[2];                      \
  f[1]   = u[0];                                          \
  f[2]   = u[1];                                          \
  f[3]   = u[2];                                          \
                                                          \
  g(0,0) = -1.0;                                          \
  g(1,0) = -1.0;                                          \
  g(2,0) = -1.0;                                          \
                                                          \
  g(0,1) =  1.0;                                          \
  g(1,1) =  0.0;                                          \
  g(2,1) =  0.0;                                          \
                                                          \
  g(0,2) =  0.0;                                          \
  g(1,2) =  1.0;                                          \
  g(2,2) =  0.0;                                          \
                                                          \
  g(0,3) =  0.0;                                          \
  g(1,3) =  0.0;                                          \
  g(2,3) =  1.0


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TET4_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TET4_GRADS2
#define JIVE_EVAL_TET4_GRADS2( f, g, h, u )               \
                                                          \
  f[0]   = 1.0 - u[0] - u[1] - u[2];                      \
  f[1]   = u[0];                                          \
  f[2]   = u[1];                                          \
  f[3]   = u[2];                                          \
                                                          \
  g(0,0) = -1.0;                                          \
  g(1,0) = -1.0;                                          \
  g(2,0) = -1.0;                                          \
                                                          \
  g(0,1) =  1.0;                                          \
  g(1,1) =  0.0;                                          \
  g(2,1) =  0.0;                                          \
                                                          \
  g(0,2) =  0.0;                                          \
  g(1,2) =  1.0;                                          \
  g(2,2) =  0.0;                                          \
                                                          \
  g(0,3) =  0.0;                                          \
  g(1,3) =  0.0;                                          \
  g(2,3) =  1.0;                                          \
                                                          \
  h      = 0.0


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TET10_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TET10_FUNCS
#define JIVE_EVAL_TET10_FUNCS( f, u )                     \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Tet4Funcs::FuncVector  q;                 \
                                                          \
    JIVE_EVAL_TET4_FUNCS( q, u );                         \
                                                          \
    f[0] = q[0] * (2.0 * q[0] - 1.0);                     \
    f[1] =  4.0 * q[0] * q[1];                            \
    f[2] = q[1] * (2.0 * q[1] - 1.0);                     \
    f[3] =  4.0 * q[1] * q[2];                            \
    f[4] = q[2] * (2.0 * q[2] - 1.0);                     \
    f[5] =  4.0 * q[2] * q[0];                            \
    f[6] =  4.0 * q[0] * q[3];                            \
    f[7] =  4.0 * q[1] * q[3];                            \
    f[8] =  4.0 * q[2] * q[3];                            \
    f[9] = q[3] * (2.0 * q[3] - 1.0);                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TET10_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TET10_GRADS
#define JIVE_EVAL_TET10_GRADS( f, g, u )                  \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Tet4Funcs::GradMatrix  r;                 \
    jive::geom::Tet4Funcs::FuncVector  q;                 \
                                                          \
    JIVE_EVAL_TET4_GRADS( q, r, u );                      \
                                                          \
    f[0] = q[0] * (2.0 * q[0] - 1.0);                     \
    f[1] =  4.0 * q[0] * q[1];                            \
    f[2] = q[1] * (2.0 * q[1] - 1.0);                     \
    f[3] =  4.0 * q[1] * q[2];                            \
    f[4] = q[2] * (2.0 * q[2] - 1.0);                     \
    f[5] =  4.0 * q[2] * q[0];                            \
    f[6] =  4.0 * q[0] * q[3];                            \
    f[7] =  4.0 * q[1] * q[3];                            \
    f[8] =  4.0 * q[2] * q[3];                            \
    f[9] = q[3] * (2.0 * q[3] - 1.0);                     \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      g(i,0) = r(i,0) * (4.0    * q[0] - 1.0);            \
      g(i,1) =    4.0 * (r(i,0) * q[1] + r(i,1) * q[0]);  \
      g(i,2) = r(i,1) * (4.0    * q[1] - 1.0 );           \
      g(i,3) =    4.0 * (r(i,1) * q[2] + r(i,2) * q[1]);  \
      g(i,4) = r(i,2) * (4.0    * q[2] - 1.0 );           \
      g(i,5) =    4.0 * (r(i,2) * q[0] + r(i,0) * q[2]);  \
      g(i,6) =    4.0 * (r(i,0) * q[3] + r(i,3) * q[0]);  \
      g(i,7) =    4.0 * (r(i,1) * q[3] + r(i,3) * q[1]);  \
      g(i,8) =    4.0 * (r(i,2) * q[3] + r(i,3) * q[2]);  \
      g(i,9) = r(i,3) * (4.0    * q[3] - 1.0);            \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TET10_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TET10_GRADS2
#define JIVE_EVAL_TET10_GRADS2( f, g, h, u )              \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Tet4Funcs::GradMatrix  r;                 \
    jive::geom::Tet4Funcs::FuncVector  q;                 \
                                                          \
    JIVE_EVAL_TET4_GRADS( q, r, u );                      \
                                                          \
    f[0] = q[0] * (2.0 * q[0] - 1.0);                     \
    f[1] =  4.0 * q[0] * q[1];                            \
    f[2] = q[1] * (2.0 * q[1] - 1.0);                     \
    f[3] =  4.0 * q[1] * q[2];                            \
    f[4] = q[2] * (2.0 * q[2] - 1.0);                     \
    f[5] =  4.0 * q[2] * q[0];                            \
    f[6] =  4.0 * q[0] * q[3];                            \
    f[7] =  4.0 * q[1] * q[3];                            \
    f[8] =  4.0 * q[2] * q[3];                            \
    f[9] = q[3] * (2.0 * q[3] - 1.0);                     \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      g(i,0) = r(i,0) * (4.0    * q[0] - 1.0);            \
      g(i,1) =    4.0 * (r(i,0) * q[1] + r(i,1) * q[0]);  \
      g(i,2) = r(i,1) * (4.0    * q[1] - 1.0 );           \
      g(i,3) =    4.0 * (r(i,1) * q[2] + r(i,2) * q[1]);  \
      g(i,4) = r(i,2) * (4.0    * q[2] - 1.0 );           \
      g(i,5) =    4.0 * (r(i,2) * q[0] + r(i,0) * q[2]);  \
      g(i,6) =    4.0 * (r(i,0) * q[3] + r(i,3) * q[0]);  \
      g(i,7) =    4.0 * (r(i,1) * q[3] + r(i,3) * q[1]);  \
      g(i,8) =    4.0 * (r(i,2) * q[3] + r(i,3) * q[2]);  \
      g(i,9) = r(i,3) * (4.0    * q[3] - 1.0);            \
    }                                                     \
                                                          \
    for ( int i = 0; i < 6; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 10; j++ )                      \
      {                                                   \
        h(i,j) =                                          \
          jive::geom::Tet10Funcs::GRADS2[i][j];           \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Tet4Funcs
//=======================================================================


inline void Tet4Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_TET4_FUNCS( f, u );
}


inline void Tet4Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_TET4_GRADS( f, g, u );
}


inline void Tet4Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_TET4_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Tet10Funcs
//=======================================================================


inline void Tet10Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_TET10_FUNCS( f, u );
}


inline void Tet10Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_TET10_GRADS( f, g, u );
}


inline void Tet10Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_TET10_GRADS2( f, g, h, u );
}

JIVE_END_PACKAGE( geom )

#endif
