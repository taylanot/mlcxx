
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

#ifndef JIVE_GEOM_CUBEFUNCS_H
#define JIVE_GEOM_CUBEFUNCS_H

#include <jive/geom/shfuncs/LineFuncs.h>
#include <jive/geom/shfuncs/SquareFuncs.h>
#include <jive/geom/shfuncs/PrismFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class Cube8Funcs
//-----------------------------------------------------------------------


class Cube8Funcs : public ShapeFuncs<3,8>
{
 public:

  static const int    VERTEX_COORDS[3][8];


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
//   class Cube16Funcs
//-----------------------------------------------------------------------


class Cube16Funcs : public ShapeFuncs<3,16>
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
//   class Cube20Funcs
//-----------------------------------------------------------------------


class Cube20Funcs : public ShapeFuncs<3,20>
{
 public:

  static const int    VERTEX_COORDS[3][20];


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
//   class Cube27BFuncs
//-----------------------------------------------------------------------


class Cube27BFuncs : public ShapeFuncs<3,27>
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
//   class Cube3Bx2x2Funcs
//-----------------------------------------------------------------------


class Cube3Bx2x2Funcs : public ShapeFuncs<3,12>
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
//   macro JIVE_EVAL_CUBE8_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE8_FUNCS
#define JIVE_EVAL_CUBE8_FUNCS( f, u )             \
                                                  \
  do                                              \
  {                                               \
    const int*  p =                               \
      jive::geom::Cube8Funcs::VERTEX_COORDS[0];   \
    const int*  q =                               \
      jive::geom::Cube8Funcs::VERTEX_COORDS[1];   \
    const int*  r =                               \
      jive::geom::Cube8Funcs::VERTEX_COORDS[2];   \
                                                  \
    const double  x = u[0];                       \
    const double  y = u[1];                       \
    const double  z = u[2];                       \
                                                  \
    for ( int i = 0; i < 8; i++ )                 \
    {                                             \
      f[i] = 0.125 * (1.0 + p[i] * x) *           \
                     (1.0 + q[i] * y) *           \
                     (1.0 + r[i] * z);            \
    }                                             \
  }                                               \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE8_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE8_GRADS
#define JIVE_EVAL_CUBE8_GRADS( f, g, u )                 \
                                                         \
  do                                                     \
  {                                                      \
    const int*  p =                                      \
      jive::geom::Cube8Funcs::VERTEX_COORDS[0];          \
    const int*  q =                                      \
      jive::geom::Cube8Funcs::VERTEX_COORDS[1];          \
    const int*  r =                                      \
      jive::geom::Cube8Funcs::VERTEX_COORDS[2];          \
                                                         \
    const double  x = u[0];                              \
    const double  y = u[1];                              \
    const double  z = u[2];                              \
                                                         \
    for ( int i = 0; i < 8; i++ )                        \
    {                                                    \
      f[i]   = 0.125 *        (1.0 + p[i] * x) *         \
                              (1.0 + q[i] * y) *         \
                              (1.0 + r[i] * z);          \
                                                         \
      g(0,i) = 0.125 * p[i] * (1.0 + q[i] * y) *         \
                              (1.0 + r[i] * z);          \
      g(1,i) = 0.125 * q[i] * (1.0 + p[i] * x) *         \
                              (1.0 + r[i] * z);          \
      g(2,i) = 0.125 * r[i] * (1.0 + p[i] * x) *         \
                              (1.0 + q[i] * y);          \
    }                                                    \
  }                                                      \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE8_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE8_GRADS2
#define JIVE_EVAL_CUBE8_GRADS2( f, g, h, u )              \
                                                          \
  do                                                      \
  {                                                       \
    const int*  p =                                       \
      jive::geom::Cube8Funcs::VERTEX_COORDS[0];           \
    const int*  q =                                       \
      jive::geom::Cube8Funcs::VERTEX_COORDS[1];           \
    const int*  r =                                       \
      jive::geom::Cube8Funcs::VERTEX_COORDS[2];           \
                                                          \
    const double  x = u[0];                               \
    const double  y = u[1];                               \
    const double  z = u[2];                               \
                                                          \
    for ( int i = 0; i < 8; i++ )                         \
    {                                                     \
      f[i]   = 0.125 *        (1.0 + p[i] * x) *          \
                              (1.0 + q[i] * y) *          \
                              (1.0 + r[i] * z);           \
                                                          \
      g(0,i) = 0.125 * p[i] * (1.0 + q[i] * y) *          \
                              (1.0 + r[i] * z);           \
      g(1,i) = 0.125 * q[i] * (1.0 + p[i] * x) *          \
                              (1.0 + r[i] * z);           \
      g(2,i) = 0.125 * r[i] * (1.0 + p[i] * x) *          \
                              (1.0 + q[i] * y);           \
                                                          \
      h(0,i) = 0.0;                                       \
      h(1,i) = 0.0;                                       \
      h(2,i) = 0.0;                                       \
      h(3,i) = 0.125 * p[i] * q[i] *                      \
                (1.0 + r[i] * z);                         \
      h(4,i) = 0.125 * p[i] * r[i] *                      \
                (1.0 + q[i] * y);                         \
      h(5,i) = 0.125 * q[i] * r[i] *                      \
                (1.0 + p[i] * x);                         \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE16_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE16_FUNCS
#define JIVE_EVAL_CUBE16_FUNCS( f, u )                    \
                                                          \
  JIVE_EVAL_PRISM_FUNCS( f, u,                            \
                         jive::geom::Square8Funcs,        \
                         jive::geom::Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE16_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE16_GRADS
#define JIVE_EVAL_CUBE16_GRADS( f, g, u )                 \
                                                          \
  JIVE_EVAL_PRISM_GRADS( f, g, u,                         \
                         jive::geom::Square8Funcs,        \
                         jive::geom::Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE16_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE16_GRADS2
#define JIVE_EVAL_CUBE16_GRADS2( f, g, h, u )             \
                                                          \
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u,                     \
                          jive::geom::Square8Funcs,       \
                          jive::geom::Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE20_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE20_FUNCS
#define JIVE_EVAL_CUBE20_FUNCS( f, u )                  \
                                                        \
  do                                                    \
  {                                                     \
    const int*  p =                                     \
      jive::geom::Cube20Funcs::VERTEX_COORDS[0];        \
    const int*  q =                                     \
      jive::geom::Cube20Funcs::VERTEX_COORDS[1];        \
    const int*  r =                                     \
      jive::geom::Cube20Funcs::VERTEX_COORDS[2];        \
                                                        \
    const double      x = u[0];                         \
    const double      y = u[1];                         \
    const double      z = u[2];                         \
                                                        \
    jive::geom::Vec3  v;                                \
    jive::geom::Vec3  w;                                \
                                                        \
    v[0] = x * x;                                       \
    v[1] = y * y;                                       \
    v[2] = z * z;                                       \
                                                        \
    for ( int i = 0; i < 20; i++ )                      \
    {                                                   \
      w[0] = x * p[i];                                  \
      w[1] = y * q[i];                                  \
      w[2] = z * r[i];                                  \
                                                        \
      if ( p[i] == 0 )                                  \
      {                                                 \
        f[i] = 0.25 * (1.0 - v[0]) * (1.0 + w[1]) *     \
                      (1.0 + w[2]);                     \
      }                                                 \
      else if ( q[i] == 0 )                             \
      {                                                 \
        f[i] = 0.25 * (1.0 + w[0]) * ( 1.0 - v[1]) *    \
                      (1.0 + w[2]);                     \
      }                                                 \
      else if ( r[i] == 0 )                             \
      {                                                 \
        f[i] = 0.25 * (1.0 + w[0]) * ( 1.0 + w[1]) *    \
                      (1.0 - v[2]);                     \
      }                                                 \
      else                                              \
      {                                                 \
        f[i] = 0.125 * (1.0 + w[0]) * (1.0 + w[1]) *    \
                       (1.0 + w[2]) *                   \
                       (w[0] + w[1] + w[2] - 2.0);      \
      }                                                 \
    }                                                   \
  }                                                     \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE20_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE20_GRADS
#define JIVE_EVAL_CUBE20_GRADS( f, g, u )                        \
                                                                 \
  do                                                             \
  {                                                              \
    const int*  p =                                              \
      jive::geom::Cube20Funcs::VERTEX_COORDS[0];                 \
    const int*  q =                                              \
      jive::geom::Cube20Funcs::VERTEX_COORDS[1];                 \
    const int*  r =                                              \
      jive::geom::Cube20Funcs::VERTEX_COORDS[2];                 \
                                                                 \
    const double      x = u[0];                                  \
    const double      y = u[1];                                  \
    const double      z = u[2];                                  \
                                                                 \
    jive::geom::Vec3  v;                                         \
    jive::geom::Vec3  w;                                         \
                                                                 \
    v[0] = x * x;                                                \
    v[1] = y * y;                                                \
    v[2] = z * z;                                                \
                                                                 \
    for ( int i = 0; i < 20; i++ )                               \
    {                                                            \
      w[0] = x * p[i];                                           \
      w[1] = y * q[i];                                           \
      w[2] = z * r[i];                                           \
                                                                 \
      if ( p[i] == 0 )                                           \
      {                                                          \
        f[i]   =  0.25 * (1.0 - v[0]) * (1.0 + w[1]) *           \
                         (1.0 + w[2]);                           \
                                                                 \
        g(0,i) = -0.50 * x *    (1.0 + w[1]) * (1.0 + w[2]);     \
        g(1,i) =  0.25 * q[i] * (1.0 - v[0]) * (1.0 + w[2]);     \
        g(2,i) =  0.25 * r[i] * (1.0 - v[0]) * (1.0 + w[1]);     \
      }                                                          \
      else if ( q[i] == 0 )                                      \
      {                                                          \
        f[i]   =  0.25 * (1.0 + w[0]) * ( 1.0 - v[1]) *          \
                         (1.0 + w[2]);                           \
                                                                 \
        g(0,i) =  0.25 * p[i] * (1.0 - v[1]) * (1.0 + w[2]);     \
        g(1,i) = -0.50 * y    * (1.0 + w[0]) * (1.0 + w[2]);     \
        g(2,i) =  0.25 * r[i] * (1.0 + w[0]) * (1.0 - v[1]);     \
      }                                                          \
      else if ( r[i] == 0 )                                      \
      {                                                          \
        f[i]   =  0.25 * (1.0 + w[0]) * ( 1.0 + w[1]) *          \
                         (1.0 - v[2]);                           \
                                                                 \
        g(0,i) =  0.25 * p[i] * (1.0 + w[1]) * (1.0 - v[2]);     \
        g(1,i) =  0.25 * q[i] * (1.0 + w[0]) * (1.0 - v[2]);     \
        g(2,i) = -0.50 * z    * (1.0 + w[0]) * (1.0 + w[1]);     \
      }                                                          \
      else                                                       \
      {                                                          \
        f[i]   = 0.125 * (1.0 + w[0]) * (1.0 + w[1]) *           \
                         (1.0 + w[2]) *                          \
                         (w[0] + w[1] + w[2] - 2.0);             \
                                                                 \
        g(0,i) = 0.125 * p[i] * (1.0 + w[1]) * (1.0 + w[2]) *    \
                           (2.0 * w[0] + w[1] + w[2] - 1.0);     \
        g(1,i) = 0.125 * q[i] * (1.0 + w[0]) * (1.0 + w[2]) *    \
                           (2.0 * w[1] + w[0] + w[2] - 1.0);     \
        g(2,i) = 0.125 * r[i] * (1.0 + w[0]) * (1.0 + w[1]) *    \
                           (2.0 * w[2] + w[0] + w[1] - 1.0);     \
      }                                                          \
    }                                                            \
  }                                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE20_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE20_GRADS2
#define JIVE_EVAL_CUBE20_GRADS2( f, g, h, u )                    \
                                                                 \
  do                                                             \
  {                                                              \
    const int*  p =                                              \
      jive::geom::Cube20Funcs::VERTEX_COORDS[0];                 \
    const int*  q =                                              \
      jive::geom::Cube20Funcs::VERTEX_COORDS[1];                 \
    const int*  r =                                              \
      jive::geom::Cube20Funcs::VERTEX_COORDS[2];                 \
                                                                 \
    const double      x = u[0];                                  \
    const double      y = u[1];                                  \
    const double      z = u[2];                                  \
                                                                 \
    jive::geom::Vec3  v;                                         \
    jive::geom::Vec3  w;                                         \
                                                                 \
    v[0] = x * x;                                                \
    v[1] = y * y;                                                \
    v[2] = z * z;                                                \
                                                                 \
    for ( int i = 0; i < 20; i++ )                               \
    {                                                            \
      w[0] = x * p[i];                                           \
      w[1] = y * q[i];                                           \
      w[2] = z * r[i];                                           \
                                                                 \
      if ( p[i] == 0 )                                           \
      {                                                          \
        f[i]   =  0.25 * (1.0 - v[0]) * (1.0 + w[1]) *           \
                         (1.0 + w[2]);                           \
                                                                 \
        g(0,i) = -0.50 * x *    (1.0 + w[1]) * (1.0 + w[2]);     \
        g(1,i) =  0.25 * q[i] * (1.0 - v[0]) * (1.0 + w[2]);     \
        g(2,i) =  0.25 * r[i] * (1.0 - v[0]) * (1.0 + w[1]);     \
                                                                 \
        h(0,i) = -0.50 * (1.0 + w[1]) * (1.0 + w[2]);            \
        h(1,i) =  0.0;                                           \
        h(2,i) =  0.0;                                           \
        h(3,i) = -0.50 * x    * q[i]  * (1.0 + w[2]);            \
        h(4,i) = -0.50 * x    * r[i]  * (1.0 + w[1]);            \
        h(5,i) =  0.25 * q[i] * r[i]  * (1.0 - v[0]);            \
      }                                                          \
      else if ( q[i] == 0 )                                      \
      {                                                          \
        f[i]   =  0.25 * (1.0 + w[0]) * ( 1.0 - v[1]) *          \
                         (1.0 + w[2]);                           \
                                                                 \
        g(0,i) =  0.25 * p[i] * (1.0 - v[1]) * (1.0 + w[2]);     \
        g(1,i) = -0.50 * y    * (1.0 + w[0]) * (1.0 + w[2]);     \
        g(2,i) =  0.25 * r[i] * (1.0 + w[0]) * (1.0 - v[1]);     \
                                                                 \
        h(0,i) =  0.0;                                           \
        h(1,i) = -0.50 * (1.0 + w[0]) * (1.0 + w[2]);            \
        h(2,i) =  0.0;                                           \
        h(3,i) = -0.50 * y    * p[i]  * (1.0 + w[2]);            \
        h(4,i) =  0.25 * p[i] * r[i]  * (1.0 - v[1]);            \
        h(5,i) = -0.50 * y    * r[i]  * (1.0 + w[0]);            \
      }                                                          \
      else if ( r[i] == 0 )                                      \
      {                                                          \
        f[i]   =  0.25 * (1.0 + w[0]) * ( 1.0 + w[1]) *          \
                         (1.0 - v[2]);                           \
                                                                 \
        g(0,i) =  0.25 * p[i] * (1.0 + w[1]) * (1.0 - v[2]);     \
        g(1,i) =  0.25 * q[i] * (1.0 + w[0]) * (1.0 - v[2]);     \
        g(2,i) = -0.50 * z    * (1.0 + w[0]) * (1.0 + w[1]);     \
                                                                 \
        h(0,i) =  0.0;                                           \
        h(1,i) =  0.0;                                           \
        h(2,i) = -0.50 * (1.0 + w[0]) * (1.0 + w[1]);            \
        h(3,i) =  0.25 * p[i] * q[i]  * (1.0 - v[2]);            \
        h(4,i) = -0.50 * z    * p[i]  * (1.0 + w[1]);            \
        h(5,i) = -0.50 * z    * q[i]  * (1.0 + w[0]);            \
      }                                                          \
      else                                                       \
      {                                                          \
        f[i]   = 0.125 * (1.0 + w[0]) * (1.0 + w[1]) *           \
                         (1.0 + w[2]) *                          \
                         (w[0] + w[1] + w[2] - 2.0);             \
                                                                 \
        g(0,i) = 0.125 * p[i] * (1.0 + w[1]) * (1.0 + w[2]) *    \
                           (2.0 * w[0] + w[1] + w[2] - 1.0);     \
        g(1,i) = 0.125 * q[i] * (1.0 + w[0]) * (1.0 + w[2]) *    \
                           (2.0 * w[1] + w[0] + w[2] - 1.0);     \
        g(2,i) = 0.125 * r[i] * (1.0 + w[0]) * (1.0 + w[1]) *    \
                           (2.0 * w[2] + w[0] + w[1] - 1.0);     \
                                                                 \
        h(0,i) = 0.250 * (1.0 + w[1]) * (1.0 + w[2]);            \
        h(1,i) = 0.250 * (1.0 + w[0]) * (1.0 + w[2]);            \
        h(2,i) = 0.250 * (1.0 + w[0]) * (1.0 + w[1]);            \
        h(3,i) = 0.125 * p[i] * q[i]  * (1.0 + w[2]) *           \
                      (2.0 * w[0] + 2.0 * w[1] + w[2]);          \
        h(4,i) = 0.125 * p[i] * r[i]  * (1.0 + w[1]) *           \
                      (2.0 * w[0] + 2.0 * w[2] + w[1]);          \
        h(5,i) = 0.125 * q[i] * r[i]  * (1.0 + w[0]) *           \
                      (2.0 * w[1] + 2.0 * w[2] + w[0]);          \
      }                                                          \
    }                                                            \
  }                                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE27B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE27B_FUNCS
#define JIVE_EVAL_CUBE27B_FUNCS( f, u )                   \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector  f0, f1, f2;      \
                                                          \
    jive::geom::Line3BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line3BFuncs::evalFuncs ( f1, u[1] );      \
    jive::geom::Line3BFuncs::evalFuncs ( f2, u[2] );      \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        for ( int k = 0; k < 3; k++ )                     \
        {                                                 \
          f[9 * i + 3 * j + k] = f2[i] * f1[j] * f0[k];   \
        }                                                 \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE27B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE27B_GRADS
#define JIVE_EVAL_CUBE27B_GRADS( f, g, u )                \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector  f0, f1, f2;      \
    jive::geom::Line3BFuncs::GradMatrix  g0, g1, g2;      \
                                                          \
    jive::geom::Line3BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line3BFuncs::evalGrads ( f1, g1, u[1] );  \
    jive::geom::Line3BFuncs::evalGrads ( f2, g2, u[2] );  \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        for ( int k = 0; k < 3; k++ )                     \
        {                                                 \
          int  p = 9 * i + 3 * j + k;                     \
                                                          \
          f[p]   = f0[k]   * f1[j]   * f2[i];             \
                                                          \
          g(0,p) = g0(0,k) * f1[j]   * f2[i];             \
          g(1,p) = f0[k]   * g1(0,j) * f2[i];             \
          g(2,p) = f0[k]   * f1[j]   * g2(0,i);           \
        }                                                 \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE27B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE27B_GRADS2
#define JIVE_EVAL_CUBE27B_GRADS2( f, g, h, u )            \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector   f0, f1, f2;     \
    jive::geom::Line3BFuncs::GradMatrix   g0, g1, g2;     \
    jive::geom::Line3BFuncs::Grad2Matrix  h0, h1, h2;     \
                                                          \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f1, g1, h1, u[1] );       \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f2, g2, h2, u[2] );       \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        for ( int k = 0; k < 3; k++ )                     \
        {                                                 \
          double  f01 = f0[k] * f1[j];                    \
          double  f02 = f0[k] * f2[i];                    \
          double  f12 = f1[j] * f2[i];                    \
          int     p   = 9 * i + 3 * j + k;                \
                                                          \
          f[p]   = f01     * f2[i];                       \
                                                          \
          g(0,p) = f12     * g0(0,k);                     \
          g(1,p) = f02     * g1(0,j);                     \
          g(2,p) = f01     * g2(0,i);                     \
                                                          \
          h(0,p) = f12     * h0(0,k);                     \
          h(1,p) = f02     * h1(0,j);                     \
          h(2,p) = f01     * h2(0,i);                     \
          h(3,p) = g0(0,k) * g1(0,j) * f2[i];             \
          h(4,p) = g0(0,k) * f1[j]   * g2(0,i);           \
          h(5,p) = f0[k]   * g1(0,j) * g2(0,i);           \
        }                                                 \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE3BX2X2_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE3BX2X2_FUNCS
#define JIVE_EVAL_CUBE3BX2X2_FUNCS( f, u )                \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector  f0;              \
    jive::geom::Line2Funcs::FuncVector   f1, f2;          \
                                                          \
    jive::geom::Line3BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line2Funcs::evalFuncs  ( f1, u[1] );      \
    jive::geom::Line2Funcs::evalFuncs  ( f2, u[2] );      \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 2; j++ )                       \
      {                                                   \
        for ( int k = 0; k < 3; k++ )                     \
        {                                                 \
          f[6 * i + 3 * j + k] = f2[i] * f1[j] * f0[k];   \
        }                                                 \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE3BX2X2_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE3BX2X2_GRADS
#define JIVE_EVAL_CUBE3BX2X2_GRADS( f, g, u )             \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector  f0;              \
    jive::geom::Line2Funcs::FuncVector   f1, f2;          \
    jive::geom::Line3BFuncs::GradMatrix  g0;              \
    jive::geom::Line2Funcs::GradMatrix   g1, g2;          \
                                                          \
    jive::geom::Line3BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line2Funcs::evalGrads  ( f1, g1, u[1] );  \
    jive::geom::Line2Funcs::evalGrads  ( f2, g2, u[2] );  \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 2; j++ )                       \
      {                                                   \
        for ( int k = 0; k < 3; k++ )                     \
        {                                                 \
          int  p = 6 * i + 3 * j + k;                     \
                                                          \
          f[p]   = f0[k]   * f1[j]   * f2[i];             \
                                                          \
          g(0,p) = g0(0,k) * f1[j]   * f2[i];             \
          g(1,p) = f0[k]   * g1(0,j) * f2[i];             \
          g(2,p) = f0[k]   * f1[j]   * g2(0,i);           \
        }                                                 \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_CUBE3BX2X2_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_CUBE3BX2X2_GRADS2
#define JIVE_EVAL_CUBE3BX2X2_GRADS2( f, g, h, u )         \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector   f0;             \
    jive::geom::Line2Funcs::FuncVector    f1, f2;         \
    jive::geom::Line3BFuncs::GradMatrix   g0;             \
    jive::geom::Line2Funcs::GradMatrix    g1, g2;         \
    jive::geom::Line3BFuncs::Grad2Matrix  h0;             \
    jive::geom::Line2Funcs::Grad2Matrix   h1, h2;         \
                                                          \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line2Funcs::evalGrads2  ( f1, g1, h1, u[1] );       \
    jive::geom::                                          \
      Line2Funcs::evalGrads2  ( f2, g2, h2, u[2] );       \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 2; j++ )                       \
      {                                                   \
        for ( int k = 0; k < 3; k++ )                     \
        {                                                 \
          double  f01 = f0[k] * f1[j];                    \
          double  f02 = f0[k] * f2[i];                    \
          double  f12 = f1[j] * f2[i];                    \
          int     p   = 6 * i + 3 * j + k;                \
                                                          \
          f[p]   = f01     * f2[i];                       \
                                                          \
          g(0,p) = f12     * g0(0,k);                     \
          g(1,p) = f02     * g1(0,j);                     \
          g(2,p) = f01     * g2(0,i);                     \
                                                          \
          h(0,p) = f12     * h0(0,k);                     \
          h(1,p) = f02     * h1(0,j);                     \
          h(2,p) = f01     * h2(0,i);                     \
          h(3,p) = g0(0,k) * g1(0,j) * f2[i];             \
          h(4,p) = g0(0,k) * f1[j]   * g2(0,i);           \
          h(5,p) = f0[k]   * g1(0,j) * g2(0,i);           \
        }                                                 \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Cube8Funcs
//=======================================================================


inline void Cube8Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_CUBE8_FUNCS( f, u );
}


inline void Cube8Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_CUBE8_GRADS( f, g, u );
}


inline void Cube8Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_CUBE8_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Cube16Funcs
//=======================================================================


inline void Cube16Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_CUBE16_FUNCS( f, u );
}


inline void Cube16Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_CUBE16_GRADS( f, g, u );
}


inline void Cube16Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_CUBE16_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Cube20Funcs
//=======================================================================


inline void Cube20Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_CUBE20_FUNCS( f, u );
}


inline void Cube20Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_CUBE20_GRADS( f, g, u );
}


inline void Cube20Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_CUBE20_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Cube27BFuncs
//=======================================================================


inline void Cube27BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_CUBE27B_FUNCS( f, u );
}


inline void Cube27BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_CUBE27B_GRADS( f, g, u );
}


inline void Cube27BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_CUBE27B_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Cube3Bx2x2Funcs
//=======================================================================


inline void Cube3Bx2x2Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_CUBE3BX2X2_FUNCS( f, u );
}


inline void Cube3Bx2x2Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_CUBE3BX2X2_GRADS( f, g, u );
}


inline void Cube3Bx2x2Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_CUBE3BX2X2_GRADS2( f, g, h, u );
}


JIVE_END_PACKAGE( geom )

#endif
