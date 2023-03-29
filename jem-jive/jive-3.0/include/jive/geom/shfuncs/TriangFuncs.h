
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

#ifndef JIVE_GEOM_TRIANGFUNCS_H
#define JIVE_GEOM_TRIANGFUNCS_H

#include <jive/geom/shfuncs/ShapeFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class Triang3Funcs
//-----------------------------------------------------------------------


class Triang3Funcs : public ShapeFuncs<2,3>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

};


//-----------------------------------------------------------------------
//   class Triang6Funcs
//-----------------------------------------------------------------------


class Triang6Funcs : public ShapeFuncs<2,6>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

};


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TRIANG3_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TRIANG3_FUNCS
#define JIVE_EVAL_TRIANG3_FUNCS( f, u )                   \
                                                          \
  f[0] = 1.0 - u[0] - u[1];                               \
  f[1] = u[0];                                            \
  f[2] = u[1]


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TRIANG3_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TRIANG3_GRADS
#define JIVE_EVAL_TRIANG3_GRADS( f, g, u )                \
                                                          \
  f[0]   = 1.0 - u[0] - u[1];                             \
  f[1]   = u[0];                                          \
  f[2]   = u[1];                                          \
                                                          \
  g(0,0) = -1.0;                                          \
  g(1,0) = -1.0;                                          \
  g(0,1) =  1.0;                                          \
  g(1,1) =  0.0;                                          \
  g(0,2) =  0.0;                                          \
  g(1,2) =  1.0


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TRIANG3_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TRIANG3_GRADS2
#define JIVE_EVAL_TRIANG3_GRADS2( f, g, h, u )            \
                                                          \
  f[0]   = 1.0 - u[0] - u[1];                             \
  f[1]   = u[0];                                          \
  f[2]   = u[1];                                          \
                                                          \
  g(0,0) = -1.0;                                          \
  g(1,0) = -1.0;                                          \
  g(0,1) =  1.0;                                          \
  g(1,1) =  0.0;                                          \
  g(0,2) =  0.0;                                          \
  g(1,2) =  1.0;                                          \
                                                          \
  h      =  0.0


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TRIANG6_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TRIANG6_FUNCS
#define JIVE_EVAL_TRIANG6_FUNCS( f, u )                   \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Triang3Funcs::FuncVector  q;              \
                                                          \
    JIVE_EVAL_TRIANG3_FUNCS( q, u );                      \
                                                          \
    f[0] = q[0] * (2.0 * q[0] - 1.0);                     \
    f[1] =  4.0 * q[0] * q[1];                            \
    f[2] = q[1] * (2.0 * q[1] - 1.0);                     \
    f[3] =  4.0 * q[1] * q[2];                            \
    f[4] = q[2] * (2.0 * q[2] - 1.0);                     \
    f[5] =  4.0 * q[2] * q[0];                            \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TRIANG6_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TRIANG6_GRADS
#define JIVE_EVAL_TRIANG6_GRADS( f, g, u )                \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Triang3Funcs::GradMatrix  r;              \
    jive::geom::Triang3Funcs::FuncVector  q;              \
                                                          \
    JIVE_EVAL_TRIANG3_GRADS( q, r, u );                   \
                                                          \
    f[0] = q[0] * (2.0 * q[0] - 1.0);                     \
    f[1] =  4.0 * q[0] * q[1];                            \
    f[2] = q[1] * (2.0 * q[1] - 1.0);                     \
    f[3] =  4.0 * q[1] * q[2];                            \
    f[4] = q[2] * (2.0 * q[2] - 1.0);                     \
    f[5] =  4.0 * q[2] * q[0];                            \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      g(i,0) = r(i,0) * (4.0    * q[0] - 1.0);            \
      g(i,1) =    4.0 * (r(i,0) * q[1] + r(i,1) * q[0]);  \
      g(i,2) = r(i,1) * (4.0    * q[1] - 1.0);            \
      g(i,3) =    4.0 * (r(i,1) * q[2] + r(i,2) * q[1]);  \
      g(i,4) = r(i,2) * (4.0    * q[2] - 1.0);            \
      g(i,5) =    4.0 * (r(i,2) * q[0] + r(i,0) * q[2]);  \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_TRIANG6_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_TRIANG6_GRADS2
#define JIVE_EVAL_TRIANG6_GRADS2( f, g, h, u )            \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Triang3Funcs::GradMatrix  r;              \
    jive::geom::Triang3Funcs::FuncVector  q;              \
                                                          \
    JIVE_EVAL_TRIANG3_GRADS( q, r, u );                   \
                                                          \
    f[0] = q[0] * (2.0 * q[0] - 1.0);                     \
    f[1] =  4.0 * q[0] * q[1];                            \
    f[2] = q[1] * (2.0 * q[1] - 1.0);                     \
    f[3] =  4.0 * q[1] * q[2];                            \
    f[4] = q[2] * (2.0 * q[2] - 1.0);                     \
    f[5] =  4.0 * q[2] * q[0];                            \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      g(i,0) = r(i,0) * (4.0    * q[0] - 1.0);            \
      g(i,1) =    4.0 * (r(i,0) * q[1] + r(i,1) * q[0]);  \
      g(i,2) = r(i,1) * (4.0    * q[1] - 1.0);            \
      g(i,3) =    4.0 * (r(i,1) * q[2] + r(i,2) * q[1]);  \
      g(i,4) = r(i,2) * (4.0    * q[2] - 1.0);            \
      g(i,5) =    4.0 * (r(i,2) * q[0] + r(i,0) * q[2]);  \
    }                                                     \
                                                          \
    h(0,0) =  4.0;                                        \
    h(1,0) =  4.0;                                        \
    h(2,0) =  4.0;                                        \
    h(0,1) = -8.0;                                        \
    h(1,1) =  0.0;                                        \
    h(2,1) = -4.0;                                        \
    h(0,2) =  4.0;                                        \
    h(1,2) =  0.0;                                        \
    h(2,2) =  0.0;                                        \
    h(0,3) =  0.0;                                        \
    h(1,3) =  0.0;                                        \
    h(2,3) =  4.0;                                        \
    h(0,4) =  0.0;                                        \
    h(1,4) =  4.0;                                        \
    h(2,4) =  0.0;                                        \
    h(0,5) =  0.0;                                        \
    h(1,5) = -8.0;                                        \
    h(2,5) = -4.0;                                        \
  }                                                       \
  while ( false )



//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Triang3Funcs
//=======================================================================


inline void Triang3Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_TRIANG3_FUNCS( f, u );
}


inline void Triang3Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_TRIANG3_GRADS( f, g, u );
}


inline void Triang3Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_TRIANG3_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Triang6Funcs
//=======================================================================


inline void Triang6Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_TRIANG6_FUNCS( f, u );
}


inline void Triang6Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_TRIANG6_GRADS( f, g, u );
}


inline void Triang6Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_TRIANG6_GRADS2( f, g, h, u );
}


JIVE_END_PACKAGE( geom )

#endif
