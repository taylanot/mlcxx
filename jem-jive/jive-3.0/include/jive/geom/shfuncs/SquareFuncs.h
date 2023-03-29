
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

#ifndef JIVE_GEOM_SQUAREFUNCS_H
#define JIVE_GEOM_SQUAREFUNCS_H

#include <jive/geom/shfuncs/LineFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class Square4Funcs
//-----------------------------------------------------------------------


class Square4Funcs : public ShapeFuncs<2,4>
{
 public:

  static const int      VERTEX_COORDS[2][4];


  static inline void    evalFuncs

    ( FuncVector&         f,
      const Point&        u );

  static inline void    evalGrads

    ( FuncVector&         f,
      GradMatrix&         g,
      const Point&        u );

  static inline void    evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

};


//-----------------------------------------------------------------------
//   class Square6Funcs
//-----------------------------------------------------------------------


class Square6Funcs : public ShapeFuncs<2,6>
{
 public:

  static const int      IPERM[6];


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
//   class Square8Funcs
//-----------------------------------------------------------------------


class Square8Funcs : public ShapeFuncs<2,8>
{
 public:

  static const int      VERTEX_COORDS[2][8];


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
//   class Square9Funcs
//-----------------------------------------------------------------------


class Square9Funcs : public ShapeFuncs<2,9>
{
 public:

  static const int      IPERM[9];


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
//   class Square12Funcs
//-----------------------------------------------------------------------


class Square12Funcs : public ShapeFuncs<2,12>
{
 public:

  static const int      VERTEX_COLORS   [12];
  static const double   VERTEX_COORDS[2][12];


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
//   class Square9BFuncs
//-----------------------------------------------------------------------


class Square9BFuncs : public ShapeFuncs<2,9>
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
//   class Square16BFuncs
//-----------------------------------------------------------------------


class Square16BFuncs : public ShapeFuncs<2,16>
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
//   class Square25BFuncs
//-----------------------------------------------------------------------


class Square25BFuncs : public ShapeFuncs<2,25>
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
//   class Square3Bx2Funcs
//-----------------------------------------------------------------------


class Square3Bx2Funcs : public ShapeFuncs<2,6>
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
//   class Square4Bx2Funcs
//-----------------------------------------------------------------------


class Square4Bx2Funcs : public ShapeFuncs<2,8>
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
//   class Square4Bx3BFuncs
//-----------------------------------------------------------------------


class Square4Bx3BFuncs : public ShapeFuncs<2,12>
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
//   class Square5Bx2Funcs
//-----------------------------------------------------------------------


class Square5Bx2Funcs : public ShapeFuncs<2,10>
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
//   class Square5Bx3BFuncs
//-----------------------------------------------------------------------


class Square5Bx3BFuncs : public ShapeFuncs<2,15>
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
//   class Square6Bx3BFuncs
//-----------------------------------------------------------------------


class Square6Bx3BFuncs : public ShapeFuncs<2,18>
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
//   macro JIVE_EVAL_SQUARE4_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4_FUNCS
#define JIVE_EVAL_SQUARE4_FUNCS( f, u )                   \
                                                          \
  do                                                      \
  {                                                       \
    const int*    p =                                     \
      jive::geom::Square4Funcs::VERTEX_COORDS[0];         \
    const int*    q =                                     \
      jive::geom::Square4Funcs::VERTEX_COORDS[1];         \
                                                          \
    const double  x = u[0];                               \
    const double  y = u[1];                               \
                                                          \
    for ( int i = 0; i < 4; i++ )                         \
    {                                                     \
      f[i] = 0.25 * (1.0 + p[i] * x) * (1.0 + q[i] * y);  \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE4_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4_GRADS
#define JIVE_EVAL_SQUARE4_GRADS( f, g, u )                \
                                                          \
  do                                                      \
  {                                                       \
    const int*    p =                                     \
      jive::geom::Square4Funcs::VERTEX_COORDS[0];         \
    const int*    q =                                     \
      jive::geom::Square4Funcs::VERTEX_COORDS[1];         \
                                                          \
    const double  x = u[0];                               \
    const double  y = u[1];                               \
                                                          \
    for ( int i = 0; i < 4; i++ )                         \
    {                                                     \
      f[i]   = 0.25 * (1.0 + p[i] * x) *                  \
                      (1.0 + q[i] * y);                   \
                                                          \
      g(0,i) = 0.25 * p[i] * (1.0 + q[i] * y);            \
      g(1,i) = 0.25 * q[i] * (1.0 + p[i] * x);            \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE4_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4_GRADS2
#define JIVE_EVAL_SQUARE4_GRADS2( f, g, h, u )            \
                                                          \
  do                                                      \
  {                                                       \
    const int*    p =                                     \
      jive::geom::Square4Funcs::VERTEX_COORDS[0];         \
    const int*    q =                                     \
      jive::geom::Square4Funcs::VERTEX_COORDS[1];         \
                                                          \
    const double  x = u[0];                               \
    const double  y = u[1];                               \
                                                          \
    for ( int i = 0; i < 4; i++ )                         \
    {                                                     \
      f[i]   = 0.25 * (1.0 + p[i] * x) *                  \
                      (1.0 + q[i] * y);                   \
                                                          \
      g(0,i) = 0.25 * p[i] * (1.0 + q[i] * y);            \
      g(1,i) = 0.25 * q[i] * (1.0 + p[i] * x);            \
                                                          \
      h(0,i) = 0.0;                                       \
      h(1,i) = 0.0;                                       \
      h(2,i) = 0.25 * p[i] * q[i];                        \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE6_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE6_FUNCS
#define JIVE_EVAL_SQUARE6_FUNCS( f, u )                   \
                                                          \
  do                                                      \
  {                                                       \
    const int*  iperm =                                   \
      jive::geom::Square6Funcs::IPERM;                    \
                                                          \
    jive::geom::Line3Funcs::FuncVector  f0;               \
    jive::geom::Line2Funcs::FuncVector  f1;               \
                                                          \
    jive::geom::Line3Funcs::evalFuncs ( f0, u[0] );       \
    jive::geom::Line2Funcs::evalFuncs ( f1, u[1] );       \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        f[iperm[3 * i + j]] = f1[i] * f0[j];              \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE6_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE6_GRADS
#define JIVE_EVAL_SQUARE6_GRADS( f, g, u )                \
                                                          \
  do                                                      \
  {                                                       \
    const int*  iperm =                                   \
      jive::geom::Square6Funcs::IPERM;                    \
                                                          \
    jive::geom::Line3Funcs::FuncVector  f0;               \
    jive::geom::Line2Funcs::FuncVector  f1;               \
    jive::geom::Line3Funcs::GradMatrix  g0;               \
    jive::geom::Line2Funcs::GradMatrix  g1;               \
                                                          \
    jive::geom::Line3Funcs::evalGrads ( f0, g0, u[0] );   \
    jive::geom::Line2Funcs::evalGrads ( f1, g1, u[1] );   \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        int  k = iperm[3 * i + j];                        \
                                                          \
        f[k]   = f1[i] * f0[j];                           \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE6_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE6_GRADS2
#define JIVE_EVAL_SQUARE6_GRADS2( f, g, h, u )            \
                                                          \
  do                                                      \
  {                                                       \
    const int*  iperm =                                   \
      jive::geom::Square6Funcs::IPERM;                    \
                                                          \
    jive::geom::Line3Funcs::FuncVector   f0;              \
    jive::geom::Line2Funcs::FuncVector   f1;              \
    jive::geom::Line3Funcs::GradMatrix   g0;              \
    jive::geom::Line2Funcs::GradMatrix   g1;              \
    jive::geom::Line3Funcs::Grad2Matrix  h0;              \
    jive::geom::Line2Funcs::Grad2Matrix  h1;              \
                                                          \
    jive::geom::                                          \
      Line3Funcs::evalGrads2 ( f0, g0, h0, u[0] );        \
    jive::geom::                                          \
      Line2Funcs::evalGrads2 ( f1, g1, h1, u[1] );        \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        int  k = iperm[3 * i + j];                        \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE8_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE8_FUNCS
#define JIVE_EVAL_SQUARE8_FUNCS( f, u )                   \
                                                          \
  do                                                      \
  {                                                       \
    const int*        p =                                 \
      jive::geom::Square8Funcs::VERTEX_COORDS[0];         \
    const int*        q =                                 \
      jive::geom::Square8Funcs::VERTEX_COORDS[1];         \
                                                          \
    const double      x = u[0];                           \
    const double      y = u[1];                           \
                                                          \
    jive::geom::Vec2  v;                                  \
    jive::geom::Vec2  w;                                  \
                                                          \
    v[0] = x * x;                                         \
    v[1] = y * y;                                         \
                                                          \
    for ( int i = 0; i < 8; i++ )                         \
    {                                                     \
      w[0] = x * p[i];                                    \
      w[1] = y * q[i];                                    \
                                                          \
      if ( p[i] == 0 )                                    \
      {                                                   \
        f[i] = 0.5 * (1.0 - v[0]) * (1.0 + w[1]);         \
      }                                                   \
      else if ( q[i] == 0 )                               \
      {                                                   \
        f[i] = 0.5 * (1.0 - v[1]) * (1.0 + w[0]);         \
      }                                                   \
      else                                                \
      {                                                   \
        f[i] = 0.25 * (1.0 + w[0]) * (1.0 + w[1]) *       \
                      (w[0] + w[1] - 1.0);                \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE8_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE8_GRADS
#define JIVE_EVAL_SQUARE8_GRADS( f, g, u )                \
                                                          \
  do                                                      \
  {                                                       \
    const int*        p =                                 \
      jive::geom::Square8Funcs::VERTEX_COORDS[0];         \
    const int*        q =                                 \
      jive::geom::Square8Funcs::VERTEX_COORDS[1];         \
                                                          \
    const double      x = u[0];                           \
    const double      y = u[1];                           \
                                                          \
    jive::geom::Vec2  v;                                  \
    jive::geom::Vec2  w;                                  \
                                                          \
    v[0] = x * x;                                         \
    v[1] = y * y;                                         \
                                                          \
    for ( int i = 0; i < 8; i++ )                         \
    {                                                     \
      w[0] = x * p[i];                                    \
      w[1] = y * q[i];                                    \
                                                          \
      if ( p[i] == 0 )                                    \
      {                                                   \
        f[i]   = 0.5 * (1.0 - v[0]) * (1.0 + w[1]);       \
                                                          \
        g(0,i) =      -x    * (1.0 + w[1]);               \
        g(1,i) = 0.5 * q[i] * (1.0 - v[0]);               \
      }                                                   \
      else if ( q[i] == 0 )                               \
      {                                                   \
        f[i]   = 0.5 * (1.0 - v[1]) * (1.0 + w[0]);       \
                                                          \
        g(0,i) = 0.5 * p[i] * (1.0 - v[1]);               \
        g(1,i) =      -y    * (1.0 + w[0]);               \
    }                                                     \
      else                                                \
      {                                                   \
        f[i]   = 0.25 * (1.0 + w[0]) * (1.0 + w[1]) *     \
                        (w[0] + w[1] - 1.0);              \
                                                          \
        g(0,i) = 0.25 * p[i] * (1.0 + w[1]) *             \
                        (2.0 * w[0] + w[1]);              \
        g(1,i) = 0.25 * q[i] * (1.0 + w[0]) *             \
                        (2.0 * w[1] + w[0]);              \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE8_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE8_GRADS2
#define JIVE_EVAL_SQUARE8_GRADS2( f, g, h, u )            \
                                                          \
  do                                                      \
  {                                                       \
    const int*        p =                                 \
      jive::geom::Square8Funcs::VERTEX_COORDS[0];         \
    const int*        q =                                 \
      jive::geom::Square8Funcs::VERTEX_COORDS[1];         \
                                                          \
    const double      x = u[0];                           \
    const double      y = u[1];                           \
                                                          \
    jive::geom::Vec2  v;                                  \
    jive::geom::Vec2  w;                                  \
                                                          \
    v[0] = x * x;                                         \
    v[1] = y * y;                                         \
                                                          \
    for ( int i = 0; i < 8; i++ )                         \
    {                                                     \
      w[0] = x * p[i];                                    \
      w[1] = y * q[i];                                    \
                                                          \
      if ( p[i] == 0 )                                    \
      {                                                   \
        f[i]   = 0.5 * (1.0 - v[0]) * (1.0 + w[1]);       \
                                                          \
        g(0,i) =      -x    * (1.0 + w[1]);               \
        g(1,i) = 0.5 * q[i] * (1.0 - v[0]);               \
                                                          \
        h(0,i) = -1.0 - w[1];                             \
        h(1,i) =  0.0;                                    \
        h(2,i) = -x   * q[i];                             \
      }                                                   \
      else if ( q[i] == 0 )                               \
      {                                                   \
        f[i]   = 0.5 * (1.0 - v[1]) * (1.0 + w[0]);       \
                                                          \
        g(0,i) = 0.5 * p[i] * (1.0 - v[1]);               \
        g(1,i) =      -y    * (1.0 + w[0]);               \
                                                          \
        h(0,i) =  0.0;                                    \
        h(1,i) = -1.0 - w[0];                             \
        h(2,i) = -y   * p[i];                             \
      }                                                   \
      else                                                \
      {                                                   \
        f[i]   = 0.25 * (1.0 + w[0]) * (1.0 + w[1]) *     \
                        (w[0] + w[1] - 1.0);              \
                                                          \
        g(0,i) = 0.25 * p[i] * (1.0 + w[1]) *             \
                        (2.0 * w[0] + w[1]);              \
        g(1,i) = 0.25 * q[i] * (1.0 + w[0]) *             \
                        (2.0 * w[1] + w[0]);              \
                                                          \
        h(0,i) = 0.50 * (1.0 + w[1]);                     \
        h(1,i) = 0.50 * (1.0 + w[0]);                     \
        h(2,i) = 0.50 * (0.5 + w[0] + w[1])               \
                      * p[i] * q[i];                      \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE9_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE9_FUNCS
#define JIVE_EVAL_SQUARE9_FUNCS( f, u )                   \
                                                          \
  do                                                      \
  {                                                       \
    const int*  iperm =                                   \
      jive::geom::Square9Funcs::IPERM;                    \
                                                          \
    jive::geom::Line3Funcs::FuncVector  f0, f1;           \
                                                          \
    jive::geom::Line3Funcs::evalFuncs ( f0, u[0] );       \
    jive::geom::Line3Funcs::evalFuncs ( f1, u[1] );       \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        f[iperm[3 * i + j]] = f1[i] * f0[j];              \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE9_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE9_GRADS
#define JIVE_EVAL_SQUARE9_GRADS( f, g, u )                \
                                                          \
  do                                                      \
  {                                                       \
    const int*  iperm =                                   \
      jive::geom::Square9Funcs::IPERM;                    \
                                                          \
    jive::geom::Line3Funcs::FuncVector  f0, f1;           \
    jive::geom::Line3Funcs::GradMatrix  g0, g1;           \
                                                          \
    jive::geom::Line3Funcs::evalGrads ( f0, g0, u[0] );   \
    jive::geom::Line3Funcs::evalGrads ( f1, g1, u[1] );   \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        int  k = iperm[3 * i + j];                        \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE9_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE9_GRADS2
#define JIVE_EVAL_SQUARE9_GRADS2( f, g, h, u )            \
                                                          \
  do                                                      \
  {                                                       \
    const int*  iperm =                                   \
      jive::geom::Square9Funcs::IPERM;                    \
                                                          \
    jive::geom::Line3Funcs::FuncVector   f0, f1;          \
    jive::geom::Line3Funcs::GradMatrix   g0, g1;          \
    jive::geom::Line3Funcs::Grad2Matrix  h0, h1;          \
                                                          \
    jive::geom::                                          \
      Line3Funcs::evalGrads2 ( f0, g0, h0, u[0] );        \
    jive::geom::                                          \
      Line3Funcs::evalGrads2 ( f1, g1, h1, u[1] );        \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        int  k = iperm[3 * i + j];                        \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE12_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE12_FUNCS
#define JIVE_EVAL_SQUARE12_FUNCS( f, u )                   \
                                                           \
  do                                                       \
  {                                                        \
    const int*     c  =                                    \
      jive::geom::Square12Funcs::VERTEX_COLORS;            \
    const double*  p  =                                    \
      jive::geom::Square12Funcs::VERTEX_COORDS[0];         \
    const double*  q  =                                    \
      jive::geom::Square12Funcs::VERTEX_COORDS[1];         \
                                                           \
    const double   a  = 1.0 / 32.0;                        \
    const double   b  = 9.0 / 32.0;                        \
    const double   x  = u[0];                              \
    const double   y  = u[1];                              \
    const double   x2 = x * x;                             \
    const double   y2 = y * y;                             \
                                                           \
    for ( int i = 0; i < 12; i++ )                         \
    {                                                      \
      double  v = p[i];                                    \
      double  w = q[i];                                    \
                                                           \
      switch ( c[i] )                                      \
      {                                                    \
      case 0:                                              \
        f[i] = a * (1.0 + x * v) * (1.0 + y * w)           \
                 * (9.0 * (x2 + y2) - 10.0);               \
        break;                                             \
                                                           \
      case 1:                                              \
        f[i] = b * (1.0 + y * w) * (1.0 - x2)              \
                 * (1.0 + 9.0 * x * v);                    \
        break;                                             \
                                                           \
      case 2:                                              \
        f[i] = b * (1.0 + x * v) * (1.0 - y2)              \
                 * (1.0 + 9.0 * y * w);                    \
        break;                                             \
      }                                                    \
    }                                                      \
  }                                                        \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE12_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE12_GRADS
#define JIVE_EVAL_SQUARE12_GRADS( f, g, u )                \
                                                           \
  do                                                       \
  {                                                        \
    const int*     c  =                                    \
      jive::geom::Square12Funcs::VERTEX_COLORS;            \
    const double*  p  =                                    \
      jive::geom::Square12Funcs::VERTEX_COORDS[0];         \
    const double*  q  =                                    \
      jive::geom::Square12Funcs::VERTEX_COORDS[1];         \
                                                           \
    const double   a  = 1.0 / 32.0;                        \
    const double   b  = 9.0 / 32.0;                        \
    const double   x  = u[0];                              \
    const double   y  = u[1];                              \
    const double   x2 = x * x;                             \
    const double   y2 = y * y;                             \
                                                           \
    for ( int i = 0; i < 12; i++ )                         \
    {                                                      \
      double  v = p[i];                                    \
      double  w = q[i];                                    \
                                                           \
      switch ( c[i] )                                      \
      {                                                    \
      case 0:                                              \
        f[i]   = a * (1.0 + x * v) * (1.0 + y * w)         \
                   * (9.0 * (x2 + y2) - 10.0);             \
        g(0,i) = a * (1.0 + y * w) * v                     \
                   * (9.0 * (x2 + y2) - 10.0) +            \
                 a * (1.0 + x * v) * (1.0 + y * w)         \
                   * (18.0 * x);                           \
        g(1,i) = a * (1.0 + x * v) * w                     \
                   * (9.0 * (x2 + y2) - 10.0) +            \
                 a * (1.0 + x * v) * (1.0 + y * w)         \
                   * (18.0 * y);                           \
        break;                                             \
                                                           \
      case 1:                                              \
        f[i]   = b * (1.0 + y * w) * (1.0 - x2)            \
                   * (1.0 + 9.0 * x * v);                  \
        g(0,i) = b * (1.0 + y * w) * (1.0 - x2)            \
                   * (9.0 * v) -                           \
                 b * (1.0 + y * w)                         \
                   * (2.0 * x) * (1.0 + 9.0 * x * v);      \
        g(1,i) = b * (1.0 - x2) * w                        \
                   * (1.0 + 9.0 * x * v);                  \
        break;                                             \
                                                           \
      case 2:                                              \
        f[i]   = b * (1.0 + x * v) * (1.0 - y2)            \
                   * (1.0 + 9.0 * y * w);                  \
        g(0,i) = b * (1.0 - y2) * v                        \
                   * (1.0 + 9.0 * y * w);                  \
        g(1,i) = b * (1.0 + x * v) * (1.0 - y2)            \
                   * (9.0 * w) -                           \
                 b * (1.0 + x * v) * (2.0 * y)             \
                   * (1.0 + 9.0 * y * w);                  \
        break;                                             \
      }                                                    \
    }                                                      \
  }                                                        \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE12_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE12_GRADS2
#define JIVE_EVAL_SQUARE12_GRADS2( f, g, h, u )            \
                                                           \
  do                                                       \
  {                                                        \
    const int*     c  =                                    \
      jive::geom::Square12Funcs::VERTEX_COLORS;            \
    const double*  p  =                                    \
      jive::geom::Square12Funcs::VERTEX_COORDS[0];         \
    const double*  q  =                                    \
      jive::geom::Square12Funcs::VERTEX_COORDS[1];         \
                                                           \
    const double   a  = 1.0 / 32.0;                        \
    const double   b  = 9.0 / 32.0;                        \
    const double   x  = u[0];                              \
    const double   y  = u[1];                              \
    const double   x2 = x * x;                             \
    const double   y2 = y * y;                             \
                                                           \
    for ( int i = 0; i < 12; i++ )                         \
    {                                                      \
      double  v = p[i];                                    \
      double  w = q[i];                                    \
                                                           \
      switch ( c[i] )                                      \
      {                                                    \
      case 0:                                              \
        f[i]   = a * (1.0 + x * v) * (1.0 + y * w)         \
                   * (9.0 * (x2 + y2) - 10.0);             \
        g(0,i) = a * (1.0 + y * w) * v                     \
                   * (9.0 * (x2 + y2) - 10.0) +            \
                 a * (1.0 + x * v) * (1.0 + y * w)         \
                   * (18.0 * x);                           \
        g(1,i) = a * (1.0 + x * v) * w                     \
                   * (9.0 * (x2 + y2) - 10.0) +            \
                 a * (1.0 + x * v) * (1.0 + y * w)         \
                   * (18.0 * y);                           \
        h(0,i) = a * (18.0 * (1.0 + y * w))                \
                   * (1.0 + 3.0 * x * v);                  \
        h(1,i) = a * (18.0 * (1.0 + x * v))                \
                   * (1.0 + 3.0 * y * w);                  \
        h(2,i) = a * (9.0 * (x2 + y2) - 10.0)              \
                   * (w * v) +                             \
                 a * (1.0 + y * w) * (18.0 * y * v) +      \
                 a * (1.0 + x * v) * (18.0 * x * w);       \
        break;                                             \
                                                           \
      case 1:                                              \
        f[i]   = b * (1.0 + y * w) * (1.0 - x2)            \
                   * (1.0 + 9.0 * x * v);                  \
        g(0,i) = b * (1.0 + y * w) * (1.0 - x2)            \
                   * (9.0 * v) -                           \
                 b * (1.0 + y * w)                         \
                   * (2.0 * x) * (1.0 + 9.0 * x * v);      \
        g(1,i) = b * (1.0 - x2) * w                        \
                   * (1.0 + 9.0 * x * v);                  \
        h(0,i) = b * (1.0 + y * w) * (-2.0)                \
                   * (1.0 + 27.0 * x * v);                 \
        h(1,i) =      0.0;                                 \
        h(2,i) = b * (9.0 * v * w) * (1.0 - x2) -          \
                 b * (2.0 * x * w)                         \
                   * (1.0 + 9.0 * x * v);                  \
        break;                                             \
                                                           \
      case 2:                                              \
        f[i]   = b * (1.0 + x * v) * (1.0 - y2)            \
                   * (1.0 + 9.0 * y * w);                  \
        g(0,i) = b * (1.0 - y2) * v                        \
                   * (1.0 + 9.0 * y * w);                  \
        g(1,i) = b * (1.0 + x * v) * (1.0 - y2)            \
                   * (9.0 * w) -                           \
                 b * (1.0 + x * v) * (2.0 * y)             \
                   * (1.0 + 9.0 * y * w);                  \
        h(0,i) =      0.0;                                 \
        h(1,i) = b * (1.0 + x * v) * (-2.0)                \
                   * (1.0 + 27.0 * y * w);                 \
        h(2,i) = b * (9.0 * v * w) * (1.0 - y2) -          \
                 b * (2.0 * v * y)                         \
                   * (1.0 + 9.0 * y * w);                  \
        break;                                             \
      }                                                    \
    }                                                      \
  }                                                        \
  while ( false )

//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE9B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE9B_FUNCS
#define JIVE_EVAL_SQUARE9B_FUNCS( f, u )                  \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector  f0, f1;          \
                                                          \
    jive::geom::Line3BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line3BFuncs::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        f[3 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE9B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE9B_GRADS
#define JIVE_EVAL_SQUARE9B_GRADS( f, g, u )               \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector  f0, f1;          \
    jive::geom::Line3BFuncs::GradMatrix  g0, g1;          \
                                                          \
    jive::geom::Line3BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line3BFuncs::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        int  k = 3 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE9B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE9B_GRADS2
#define JIVE_EVAL_SQUARE9B_GRADS2( f, g, h, u )           \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector   f0, f1;         \
    jive::geom::Line3BFuncs::GradMatrix   g0, g1;         \
    jive::geom::Line3BFuncs::Grad2Matrix  h0, h1;         \
                                                          \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        int  k = 3 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE16B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE16B_FUNCS
#define JIVE_EVAL_SQUARE16B_FUNCS( f, u )                 \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector  f0, f1;          \
                                                          \
    jive::geom::Line4BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line4BFuncs::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 4; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        f[4 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE16B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE16B_GRADS
#define JIVE_EVAL_SQUARE16B_GRADS( f, g, u )              \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector  f0, f1;          \
    jive::geom::Line4BFuncs::GradMatrix  g0, g1;          \
                                                          \
    jive::geom::Line4BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line4BFuncs::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 4; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        int  k = 4 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE16B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE16B_GRADS2
#define JIVE_EVAL_SQUARE16B_GRADS2( f, g, h, u )          \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector   f0, f1;         \
    jive::geom::Line4BFuncs::GradMatrix   g0, g1;         \
    jive::geom::Line4BFuncs::Grad2Matrix  h0, h1;         \
                                                          \
    jive::geom::                                          \
      Line4BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line4BFuncs::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 4; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        int  k = 4 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE25B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE25B_FUNCS
#define JIVE_EVAL_SQUARE25B_FUNCS( f, u )                 \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector  f0, f1;          \
                                                          \
    jive::geom::Line5BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line5BFuncs::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 5; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        f[5 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE25B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE25B_GRADS
#define JIVE_EVAL_SQUARE25B_GRADS( f, g, u )              \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector  f0, f1;          \
    jive::geom::Line5BFuncs::GradMatrix  g0, g1;          \
                                                          \
    jive::geom::Line5BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line5BFuncs::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 5; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        int  k = 5 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE25B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE25B_GRADS2
#define JIVE_EVAL_SQUARE25B_GRADS2( f, g, h, u )          \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector   f0, f1;         \
    jive::geom::Line5BFuncs::GradMatrix   g0, g1;         \
    jive::geom::Line5BFuncs::Grad2Matrix  h0, h1;         \
                                                          \
    jive::geom::                                          \
      Line5BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line5BFuncs::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 5; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        int  k = 5 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE3BX2_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE3BX2_FUNCS
#define JIVE_EVAL_SQUARE3BX2_FUNCS( f, u )                \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector  f0;              \
    jive::geom::Line2Funcs ::FuncVector  f1;              \
                                                          \
    jive::geom::Line3BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line2Funcs ::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        f[3 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE3BX2_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE3BX2_GRADS
#define JIVE_EVAL_SQUARE3BX2_GRADS( f, g, u )             \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector  f0;              \
    jive::geom::Line2Funcs ::FuncVector  f1;              \
    jive::geom::Line3BFuncs::GradMatrix  g0;              \
    jive::geom::Line2Funcs ::GradMatrix  g1;              \
                                                          \
    jive::geom::Line3BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line2Funcs ::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        int  k = 3 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE3BX2_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE3BX2_GRADS2
#define JIVE_EVAL_SQUARE3BX2_GRADS2( f, g, h, u )         \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line3BFuncs::FuncVector   f0;             \
    jive::geom::Line2Funcs ::FuncVector   f1;             \
    jive::geom::Line3BFuncs::GradMatrix   g0;             \
    jive::geom::Line2Funcs ::GradMatrix   g1;             \
    jive::geom::Line3BFuncs::Grad2Matrix  h0;             \
    jive::geom::Line2Funcs ::Grad2Matrix  h1;             \
                                                          \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line2Funcs ::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 3; j++ )                       \
      {                                                   \
        int  k = 3 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE4BX2_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4BX2_FUNCS
#define JIVE_EVAL_SQUARE4BX2_FUNCS( f, u )                \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector  f0;              \
    jive::geom::Line2Funcs ::FuncVector  f1;              \
                                                          \
    jive::geom::Line4BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line2Funcs ::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        f[4 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE4BX2_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4BX2_GRADS
#define JIVE_EVAL_SQUARE4BX2_GRADS( f, g, u )             \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector  f0;              \
    jive::geom::Line2Funcs ::FuncVector  f1;              \
    jive::geom::Line4BFuncs::GradMatrix  g0;              \
    jive::geom::Line2Funcs ::GradMatrix  g1;              \
                                                          \
    jive::geom::Line4BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line2Funcs ::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        int  k = 4 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE4BX2_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4BX2_GRADS2
#define JIVE_EVAL_SQUARE4BX2_GRADS2( f, g, h, u )         \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector   f0;             \
    jive::geom::Line2Funcs ::FuncVector   f1;             \
    jive::geom::Line4BFuncs::GradMatrix   g0;             \
    jive::geom::Line2Funcs ::GradMatrix   g1;             \
    jive::geom::Line4BFuncs::Grad2Matrix  h0;             \
    jive::geom::Line2Funcs ::Grad2Matrix  h1;             \
                                                          \
    jive::geom::                                          \
      Line4BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line2Funcs ::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        int  k = 4 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE4BX3B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4BX3B_FUNCS
#define JIVE_EVAL_SQUARE4BX3B_FUNCS( f, u )               \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector  f0;              \
    jive::geom::Line3BFuncs::FuncVector  f1;              \
                                                          \
    jive::geom::Line4BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line3BFuncs::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        f[4 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE4BX3B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4BX3B_GRADS
#define JIVE_EVAL_SQUARE4BX3B_GRADS( f, g, u )            \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector  f0;              \
    jive::geom::Line3BFuncs::FuncVector  f1;              \
    jive::geom::Line4BFuncs::GradMatrix  g0;              \
    jive::geom::Line3BFuncs::GradMatrix  g1;              \
                                                          \
    jive::geom::Line4BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line3BFuncs::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        int  k = 4 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE4BX3B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE4BX3B_GRADS2
#define JIVE_EVAL_SQUARE4BX3B_GRADS2( f, g, h, u )        \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line4BFuncs::FuncVector   f0;             \
    jive::geom::Line3BFuncs::FuncVector   f1;             \
    jive::geom::Line4BFuncs::GradMatrix   g0;             \
    jive::geom::Line3BFuncs::GradMatrix   g1;             \
    jive::geom::Line4BFuncs::Grad2Matrix  h0;             \
    jive::geom::Line3BFuncs::Grad2Matrix  h1;             \
                                                          \
    jive::geom::                                          \
      Line4BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 4; j++ )                       \
      {                                                   \
        int  k = 4 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE5BX2_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE5BX2_FUNCS
#define JIVE_EVAL_SQUARE5BX2_FUNCS( f, u )                \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector  f0;              \
    jive::geom::Line2Funcs ::FuncVector  f1;              \
                                                          \
    jive::geom::Line5BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line2Funcs ::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        f[5 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE5BX2_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE5BX2_GRADS
#define JIVE_EVAL_SQUARE5BX2_GRADS( f, g, u )             \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector  f0;              \
    jive::geom::Line2Funcs ::FuncVector  f1;              \
    jive::geom::Line5BFuncs::GradMatrix  g0;              \
    jive::geom::Line2Funcs ::GradMatrix  g1;              \
                                                          \
    jive::geom::Line5BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line2Funcs ::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        int  k = 5 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE5BX2_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE5BX2_GRADS2
#define JIVE_EVAL_SQUARE5BX2_GRADS2( f, g, h, u )         \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector   f0;             \
    jive::geom::Line2Funcs ::FuncVector   f1;             \
    jive::geom::Line5BFuncs::GradMatrix   g0;             \
    jive::geom::Line2Funcs ::GradMatrix   g1;             \
    jive::geom::Line5BFuncs::Grad2Matrix  h0;             \
    jive::geom::Line2Funcs ::Grad2Matrix  h1;             \
                                                          \
    jive::geom::                                          \
      Line5BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line2Funcs ::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 2; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        int  k = 5 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE5BX3B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE5BX3B_FUNCS
#define JIVE_EVAL_SQUARE5BX3B_FUNCS( f, u )               \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector  f0;              \
    jive::geom::Line3BFuncs ::FuncVector f1;              \
                                                          \
    jive::geom::Line5BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line3BFuncs::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        f[5 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE5BX3B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE5BX3B_GRADS
#define JIVE_EVAL_SQUARE5BX3B_GRADS( f, g, u )            \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector  f0;              \
    jive::geom::Line3BFuncs::FuncVector  f1;              \
    jive::geom::Line5BFuncs::GradMatrix  g0;              \
    jive::geom::Line3BFuncs::GradMatrix  g1;              \
                                                          \
    jive::geom::Line5BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line3BFuncs::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        int  k = 5 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE5BX3B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE5BX3B_GRADS2
#define JIVE_EVAL_SQUARE5BX3B_GRADS2( f, g, h, u )        \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line5BFuncs::FuncVector   f0;             \
    jive::geom::Line3BFuncs::FuncVector   f1;             \
    jive::geom::Line5BFuncs::GradMatrix   g0;             \
    jive::geom::Line3BFuncs::GradMatrix   g1;             \
    jive::geom::Line5BFuncs::Grad2Matrix  h0;             \
    jive::geom::Line3BFuncs::Grad2Matrix  h1;             \
                                                          \
    jive::geom::                                          \
      Line5BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 5; j++ )                       \
      {                                                   \
        int  k = 5 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE6BX3B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE6BX3B_FUNCS
#define JIVE_EVAL_SQUARE6BX3B_FUNCS( f, u )               \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line6BFuncs::FuncVector  f0;              \
    jive::geom::Line3BFuncs ::FuncVector f1;              \
                                                          \
    jive::geom::Line6BFuncs::evalFuncs ( f0, u[0] );      \
    jive::geom::Line3BFuncs::evalFuncs ( f1, u[1] );      \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 6; j++ )                       \
      {                                                   \
        f[6 * i + j] = f1[i] * f0[j];                     \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE6BX3B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE6BX3B_GRADS
#define JIVE_EVAL_SQUARE6BX3B_GRADS( f, g, u )            \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line6BFuncs::FuncVector  f0;              \
    jive::geom::Line3BFuncs::FuncVector  f1;              \
    jive::geom::Line6BFuncs::GradMatrix  g0;              \
    jive::geom::Line3BFuncs::GradMatrix  g1;              \
                                                          \
    jive::geom::Line6BFuncs::evalGrads ( f0, g0, u[0] );  \
    jive::geom::Line3BFuncs::evalGrads ( f1, g1, u[1] );  \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 6; j++ )                       \
      {                                                   \
        int  k = 6 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_SQUARE6BX3B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_SQUARE6BX3B_GRADS2
#define JIVE_EVAL_SQUARE6BX3B_GRADS2( f, g, h, u )        \
                                                          \
  do                                                      \
  {                                                       \
    jive::geom::Line6BFuncs::FuncVector   f0;             \
    jive::geom::Line3BFuncs::FuncVector   f1;             \
    jive::geom::Line6BFuncs::GradMatrix   g0;             \
    jive::geom::Line3BFuncs::GradMatrix   g1;             \
    jive::geom::Line6BFuncs::Grad2Matrix  h0;             \
    jive::geom::Line3BFuncs::Grad2Matrix  h1;             \
                                                          \
    jive::geom::                                          \
      Line6BFuncs::evalGrads2 ( f0, g0, h0, u[0] );       \
    jive::geom::                                          \
      Line3BFuncs::evalGrads2 ( f1, g1, h1, u[1] );       \
                                                          \
    for ( int i = 0; i < 3; i++ )                         \
    {                                                     \
      for ( int j = 0; j < 6; j++ )                       \
      {                                                   \
        int  k = 6 * i + j;                               \
                                                          \
        f[k]   = f1[i]   * f0[j];                         \
                                                          \
        g(0,k) = g0(0,j) * f1[i];                         \
        g(1,k) = f0[j]   * g1(0,i);                       \
                                                          \
        h(0,k) = h0(0,j) * f1[i];                         \
        h(1,k) = f0[j]   * h1(0,i);                       \
        h(2,k) = g0(0,j) * g1(0,i);                       \
      }                                                   \
    }                                                     \
  }                                                       \
  while ( false )


//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Square4Funcs
//=======================================================================


inline void Square4Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4_FUNCS( f, u );
}


inline void Square4Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4_GRADS( f, g, u );
}


inline void Square4Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square6Funcs
//=======================================================================


inline void Square6Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE6_FUNCS( f, u );
}


inline void Square6Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE6_GRADS( f, g, u );
}


inline void Square6Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE6_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square8Funcs
//=======================================================================


inline void Square8Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE8_FUNCS( f, u );
}


inline void Square8Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE8_GRADS( f, g, u );
}


inline void Square8Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE8_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square9Funcs
//=======================================================================


inline void Square9Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE9_FUNCS( f, u );
}


inline void Square9Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE9_GRADS( f, g, u );
}


inline void Square9Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE9_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square12Funcs
//=======================================================================


inline void Square12Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE12_FUNCS( f, u );
}


inline void Square12Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE12_GRADS( f, g, u );
}


inline void Square12Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE12_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square9BFuncs
//=======================================================================


inline void Square9BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE9B_FUNCS( f, u );
}


inline void Square9BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE9B_GRADS( f, g, u );
}


inline void Square9BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE9B_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square16BFuncs
//=======================================================================


inline void Square16BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE16B_FUNCS( f, u );
}


inline void Square16BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE16B_GRADS( f, g, u );
}


inline void Square16BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE16B_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square3Bx2Funcs
//=======================================================================


inline void Square3Bx2Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE3BX2_FUNCS( f, u );
}


inline void Square3Bx2Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE3BX2_GRADS( f, g, u );
}


inline void Square3Bx2Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE3BX2_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square4Bx2Funcs
//=======================================================================


inline void Square4Bx2Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4BX2_FUNCS( f, u );
}


inline void Square4Bx2Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4BX2_GRADS( f, g, u );
}


inline void Square4Bx2Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4BX2_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square4Bx3BFuncs
//=======================================================================


inline void Square4Bx3BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4BX3B_FUNCS( f, u );
}


inline void Square4Bx3BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4BX3B_GRADS( f, g, u );
}


inline void Square4Bx3BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE4BX3B_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square5Bx2Funcs
//=======================================================================


inline void Square5Bx2Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE5BX2_FUNCS( f, u );
}


inline void Square5Bx2Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE5BX2_GRADS( f, g, u );
}


inline void Square5Bx2Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE5BX2_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square5Bx3BFuncs
//=======================================================================


inline void Square5Bx3BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE5BX3B_FUNCS( f, u );
}


inline void Square5Bx3BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE5BX3B_GRADS( f, g, u );
}


inline void Square5Bx3BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE5BX3B_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Square6Bx3BFuncs
//=======================================================================


inline void Square6Bx3BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_SQUARE6BX3B_FUNCS( f, u );
}


inline void Square6Bx3BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_SQUARE6BX3B_GRADS( f, g, u );
}


inline void Square6Bx3BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_SQUARE6BX3B_GRADS2( f, g, h, u );
}


JIVE_END_PACKAGE( geom )

#endif
