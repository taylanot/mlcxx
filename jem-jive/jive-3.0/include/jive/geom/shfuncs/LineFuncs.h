
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

#ifndef JIVE_GEOM_LINEFUNCS_H
#define JIVE_GEOM_LINEFUNCS_H

#include <jive/geom/shfuncs/ShapeFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class Line2Funcs
//-----------------------------------------------------------------------


class Line2Funcs : public ShapeFuncs<1,2>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalFuncs

    ( FuncVector&       f,
      double            u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      double            u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      double            u );

};


//-----------------------------------------------------------------------
//   Line2C1Funcs
//-----------------------------------------------------------------------


class Line2C1Funcs : public ShapeFuncs<1,4>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalFuncs

    ( FuncVector&       f,
      double            u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      double            u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      double            u );

};


//-----------------------------------------------------------------------
//   class Line3Funcs
//-----------------------------------------------------------------------


class Line3Funcs : public ShapeFuncs<1,3>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalFuncs

    ( FuncVector&       f,
      double            u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      double            u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      double            u );

};


//-----------------------------------------------------------------------
//   class Line4Funcs
//-----------------------------------------------------------------------


class Line4Funcs : public ShapeFuncs<1,4>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalFuncs

    ( FuncVector&       f,
      double            u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      double            u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      double            u );

};


//-----------------------------------------------------------------------
//   class Line3BFuncs
//-----------------------------------------------------------------------


class Line3BFuncs : public ShapeFuncs<1,3>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalFuncs

    ( FuncVector&       f,
      double            u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      double            u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      double            u );

};


//-----------------------------------------------------------------------
//   class Line4BFuncs
//-----------------------------------------------------------------------


class Line4BFuncs : public ShapeFuncs<1,4>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalFuncs

    ( FuncVector&       f,
      double            u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      double            u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      double            u );

};


//-----------------------------------------------------------------------
//   class Line5BFuncs
//-----------------------------------------------------------------------


class Line5BFuncs : public ShapeFuncs<1,5>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalFuncs

    ( FuncVector&       f,
      double            u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      double            u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      double            u );

};


//-----------------------------------------------------------------------
//   class Line6BFuncs
//-----------------------------------------------------------------------


class Line6BFuncs : public ShapeFuncs<1,6>
{
 public:

  static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

  static inline void  evalFuncs

    ( FuncVector&       f,
      double            u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

  static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      double            u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

  static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      double            u );

};


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE2_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE2_FUNCS
#define JIVE_EVAL_LINE2_FUNCS( f, u )           \
                                                \
  f[0] = 0.5 * (1.0 - u);                       \
  f[1] = 0.5 * (1.0 + u)


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE2_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE2_GRADS
#define JIVE_EVAL_LINE2_GRADS( f, g, u )        \
                                                \
  f[0]   =  0.5 * (1.0 - u);                    \
  f[1]   =  0.5 * (1.0 + u);                    \
                                                \
  g(0,0) = -0.5;                                \
  g(0,1) =  0.5


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE2_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE2_GRADS2
#define JIVE_EVAL_LINE2_GRADS2( f, g, h, u )    \
                                                \
  f[0]   =  0.5 * (1.0 - u);                    \
  f[1]   =  0.5 * (1.0 + u);                    \
                                                \
  g(0,0) = -0.5;                                \
  g(0,1) =  0.5;                                \
                                                \
  h(0,0) = 0.0;                                 \
  h(0,1) = 0.0


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE2C1_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE2C1_FUNCS
#define JIVE_EVAL_LINE2C1_FUNCS( f, u )          \
                                                 \
  do                                             \
  {                                              \
    double  a = u - 1.0;                         \
    double  b = u + 1.0;                         \
    double  c = u - 2.0;                         \
    double  d = u + 2.0;                         \
                                                 \
    f[0] = (a * a * d) /  4.0;                   \
    f[1] = (a * a * b) /  8.0;                   \
    f[2] = (b * b * c) / -4.0;                   \
    f[3] = (b * b * a) /  8.0;                   \
  }                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE2C1_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE2C1_GRADS
#define JIVE_EVAL_LINE2C1_GRADS( f, g, u )       \
                                                 \
  JIVE_EVAL_LINE2C1_FUNCS( f, u );               \
                                                 \
  do                                             \
  {                                              \
    double  a =       u - 1.0;                   \
    double  b =       u + 1.0;                   \
    double  c = 3.0 * u - 1.0;                   \
    double  d = 3.0 * u + 1.0;                   \
                                                 \
    g(0,0) = (a * b * 3.0) /  4.0;               \
    g(0,1) = (a * d)       /  8.0;               \
    g(0,2) = (a * b * 3.0) / -4.0;               \
    g(0,3) = (b * c)       /  8.0;               \
  }                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE2C1_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE2C1_GRADS2
#define JIVE_EVAL_LINE2C1_GRADS2( f, g, h, u )   \
                                                 \
  JIVE_EVAL_LINE2C1_GRADS( f, g, u );            \
                                                 \
  do                                             \
  {                                              \
    double  a = 6.0 * u;                         \
                                                 \
    h(0,0) =  a        /  4.0;                   \
    h(0,1) = (a - 2.0) /  8.0;                   \
    h(0,2) =  a        / -4.0;                   \
    h(0,3) = (a + 2.0) /  8.0;                   \
  }                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE3_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE3_FUNCS
#define JIVE_EVAL_LINE3_FUNCS( f, u )            \
                                                 \
  do                                             \
  {                                              \
    double u2 = u * u;                           \
                                                 \
    f[0] = 0.5 * (u2 - u);                       \
    f[1] = 1.0 - u2;                             \
    f[2] = 0.5 * (u2 + u);                       \
  }                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE3_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE3_GRADS
#define JIVE_EVAL_LINE3_GRADS( f, g, u )        \
                                                \
  JIVE_EVAL_LINE3_FUNCS( f, u );                \
                                                \
  g(0,0) = u - 0.5;                             \
  g(0,1) = -2.0 * u;                            \
  g(0,2) = u + 0.5


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE3_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE3_GRADS2
#define JIVE_EVAL_LINE3_GRADS2( f, g, h, u )    \
                                                \
  JIVE_EVAL_LINE3_GRADS( f, g, u );             \
                                                \
  h(0,0) =  1.0;                                \
  h(0,1) = -2.0;                                \
  h(0,2) =  1.0


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE4_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE4_FUNCS
#define JIVE_EVAL_LINE4_FUNCS( f, u )            \
                                                 \
  do                                             \
  {                                              \
    const double  a  = 1.0 / 16.0;               \
    const double  b  = 9.0 / 16.0;               \
    const double  u2 = u * u;                    \
                                                 \
    f[0] = a * (1.0 - u)  * (9.0 * u2 - 1.0);    \
    f[1] = b * (1.0 - u2) * (1.0 - 3.0 * u);     \
    f[2] = b * (1.0 - u2) * (1.0 + 3.0 * u);     \
    f[3] = a * (1.0 + u)  * (9.0 * u2 - 1.0);    \
  }                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE4_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE4_GRADS
#define JIVE_EVAL_LINE4_GRADS( f, g, u )         \
                                                 \
  do                                             \
  {                                              \
    const double  a  = 1.0 / 16.0;               \
    const double  b  = 9.0 / 16.0;               \
    const double  u2 = u * u;                    \
                                                 \
    f[0]   =  a * (1.0 - u)  * (9.0 * u2 - 1.0); \
    f[1]   =  b * (1.0 - u2) * (1.0 - 3.0 * u);  \
    f[2]   =  b * (1.0 - u2) * (1.0 + 3.0 * u);  \
    f[3]   =  a * (1.0 + u)  * (9.0 * u2 - 1.0); \
                                                 \
    g(0,0) =  a * (1.0 + 18.0 * u - 27.0 * u2);  \
    g(0,1) = -b * (3.0 +  2.0 * u -  9.0 * u2);  \
    g(0,2) =  b * (3.0 -  2.0 * u -  9.0 * u2);  \
    g(0,3) = -a * (1.0 - 18.0 * u - 27.0 * u2);  \
  }                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE4_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE4_GRADS2
#define JIVE_EVAL_LINE4_GRADS2( f, g, h, u )     \
                                                 \
  do                                             \
  {                                              \
    const double  a  = 1.0 / 16.0;               \
    const double  b  = 9.0 / 16.0;               \
    const double  u2 = u * u;                    \
                                                 \
    f[0]   =  a * (1.0 - u)  * (9.0 * u2 - 1.0); \
    f[1]   =  b * (1.0 - u2) * (1.0 - 3.0 * u);  \
    f[2]   =  b * (1.0 - u2) * (1.0 + 3.0 * u);  \
    f[3]   =  a * (1.0 + u)  * (9.0 * u2 - 1.0); \
                                                 \
    g(0,0) =  a * (1.0 + 18.0 * u - 27.0 * u2);  \
    g(0,1) = -b * (3.0 +  2.0 * u -  9.0 * u2);  \
    g(0,2) =  b * (3.0 -  2.0 * u -  9.0 * u2);  \
    g(0,3) = -a * (1.0 - 18.0 * u - 27.0 * u2);  \
                                                 \
    h(0,0) =  a * 18.0 * (1.0 - 3.0 * u);        \
    h(0,1) = -b *  2.0 * (1.0 - 9.0 * u);        \
    h(0,2) = -b *  2.0 * (1.0 + 9.0 * u);        \
    h(0,3) =  a * 18.0 * (1.0 + 3.0 * u);        \
  }                                              \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE3B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE3B_FUNCS
#define JIVE_EVAL_LINE3B_FUNCS( f, u )          \
                                                \
  do                                            \
  {                                             \
    double  a = (1.0 - u) / 2.0;                \
    double  b = (1.0 + u) / 2.0;                \
                                                \
    f[0] = a * a;                               \
    f[1] = a * b * 2.0;                         \
    f[2] = b * b;                               \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE3B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE3B_GRADS
#define JIVE_EVAL_LINE3B_GRADS( f, g, u )       \
                                                \
  do                                            \
  {                                             \
    double  a = (1.0 - u) / 2.0;                \
    double  b = (1.0 + u) / 2.0;                \
                                                \
    f[0]   = a * a;                             \
    f[1]   = a * b * 2.0;                       \
    f[2]   = b * b;                             \
                                                \
    g(0,0) = -a;                                \
    g(0,1) = -u;                                \
    g(0,2) =  b;                                \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE3B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE3B_GRADS2
#define JIVE_EVAL_LINE3B_GRADS2( f, g, h, u )   \
                                                \
  do                                            \
  {                                             \
    double  a = (1.0 - u) / 2.0;                \
    double  b = (1.0 + u) / 2.0;                \
                                                \
    f[0]   = a * a;                             \
    f[1]   = a * b * 2.0;                       \
    f[2]   = b * b;                             \
                                                \
    g(0,0) = -a;                                \
    g(0,1) = -u;                                \
    g(0,2) =  b;                                \
                                                \
    h(0,0) =  0.5;                              \
    h(0,1) = -1.0;                              \
    h(0,2) =  0.5;                              \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE4B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE4B_FUNCS
#define JIVE_EVAL_LINE4B_FUNCS( f, u )          \
                                                \
  do                                            \
  {                                             \
    double  a  = (1.0 - u) / 2.0;               \
    double  b  = (1.0 + u) / 2.0;               \
    double  a2 = a * a     / 2.0;               \
    double  b2 = b * b     / 2.0;               \
                                                \
    f[0] = 2.0 * a * a2;                        \
    f[1] = 6.0 * b * a2;                        \
    f[2] = 6.0 * a * b2;                        \
    f[3] = 2.0 * b * b2;                        \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE4B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE4B_GRADS
#define JIVE_EVAL_LINE4B_GRADS( f, g, u )       \
                                                \
  do                                            \
  {                                             \
    double  a  = (1.0 - u) / 2.0;               \
    double  b  = (1.0 + u) / 2.0;               \
    double  a2 = a * a     / 2.0;               \
    double  b2 = b * b     / 2.0;               \
                                                \
    f[0]   =  2.0 * a * a2;                     \
    f[1]   =  6.0 * b * a2;                     \
    f[2]   =  6.0 * a * b2;                     \
    f[3]   =  2.0 * b * b2;                     \
                                                \
    g(0,0) = -3.0 *  a2;                        \
    g(0,1) =  3.0 * (a2 - a * b);               \
    g(0,2) = -3.0 * (b2 - a * b);               \
    g(0,3) =  3.0 *  b2;                        \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE4B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE4B_GRADS2
#define JIVE_EVAL_LINE4B_GRADS2( f, g, h, u )   \
                                                \
  do                                            \
  {                                             \
    double  a  = (1.0 - u) / 2.0;               \
    double  b  = (1.0 + u) / 2.0;               \
    double  a2 = a * a     / 2.0;               \
    double  b2 = b * b     / 2.0;               \
                                                \
    f[0]   =  2.0 * a * a2;                     \
    f[1]   =  6.0 * b * a2;                     \
    f[2]   =  6.0 * a * b2;                     \
    f[3]   =  2.0 * b * b2;                     \
                                                \
    g(0,0) = -3.0 *  a2;                        \
    g(0,1) =  3.0 * (a2 - a * b);               \
    g(0,2) = -3.0 * (b2 - a * b);               \
    g(0,3) =  3.0 *  b2;                        \
                                                \
    h(0,0) =  3.0 *  a     / 2.0;               \
    h(0,1) = -3.0 * (a - b / 2.0);              \
    h(0,2) = -3.0 * (b - a / 2.0);              \
    h(0,3) =  3.0 *  b     / 2.0;               \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE5B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE5B_FUNCS
#define JIVE_EVAL_LINE5B_FUNCS( f, u )          \
                                                \
  do                                            \
  {                                             \
    double  a  = (1.0 - u) / 2.0;               \
    double  b  = (1.0 + u) / 2.0;               \
    double  a2 = a  * a;                        \
    double  a3 = a2 * a;                        \
    double  b2 = b  * b;                        \
    double  b3 = b2 * b;                        \
                                                \
    f[0] =       a  * a3;                       \
    f[1] = 4.0 * b  * a3;                       \
    f[2] = 6.0 * a2 * b2;                       \
    f[3] = 4.0 * a  * b3;                       \
    f[4] =       b  * b3;                       \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE5B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE5B_GRADS
#define JIVE_EVAL_LINE5B_GRADS( f, g, u )       \
                                                \
  do                                            \
  {                                             \
    double  a  = (1.0 - u) / 2.0;               \
    double  b  = (1.0 + u) / 2.0;               \
    double  a2 = a  * a;                        \
    double  a3 = a2 * a;                        \
    double  b2 = b  * b;                        \
    double  b3 = b2 * b;                        \
                                                \
    f[0] =       a  * a3;                       \
    f[1] = 4.0 * b  * a3;                       \
    f[2] = 6.0 * a2 * b2;                       \
    f[3] = 4.0 * a  * b3;                       \
    f[4] =       b  * b3;                       \
                                                \
    g(0,0) = -2.0 * a3;                         \
    g(0,1) =  2.0 * a2 * (a - 3.0 * b);         \
    g(0,2) =  6.0 * a  *  b * (a - b);          \
    g(0,3) = -2.0 * b2 * (b - 3.0 * a);         \
    g(0,4) =  2.0 * b3;                         \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE5B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE5B_GRADS2
#define JIVE_EVAL_LINE5B_GRADS2( f, g, h, u )   \
                                                \
  do                                            \
  {                                             \
    double  a  = (1.0 - u) / 2.0;               \
    double  b  = (1.0 + u) / 2.0;               \
    double  a2 = a  * a;                        \
    double  a3 = a2 * a;                        \
    double  b2 = b  * b;                        \
    double  b3 = b2 * b;                        \
                                                \
    f[0] =       a  * a3;                       \
    f[1] = 4.0 * b  * a3;                       \
    f[2] = 6.0 * a2 * b2;                       \
    f[3] = 4.0 * a  * b3;                       \
    f[4] =       b  * b3;                       \
                                                \
    g(0,0) = -2.0 * a3;                         \
    g(0,1) =  2.0 * a2 * (a - 3.0 * b);         \
    g(0,2) =  6.0 * a  *  b * (a - b);          \
    g(0,3) = -2.0 * b2 * (b - 3.0 * a);         \
    g(0,4) =  2.0 * b3;                         \
                                                \
    h(0,0) =  3.0 * a2;                         \
    h(0,1) =  6.0 * a * (b - a);                \
    h(0,2) =  3.0 * (a2 - 4.0 * a * b + b2);    \
    h(0,3) =  6.0 * b * (a - b);                \
    h(0,4) =  3.0 * b2;                         \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE6B_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE6B_FUNCS
#define JIVE_EVAL_LINE6B_FUNCS( f, u )          \
                                                \
  do                                            \
  {                                             \
    double  a  = (1.0 - u) / 2.0;               \
    double  b  = (1.0 + u) / 2.0;               \
    double  a2 = a  * a;                        \
    double  a3 = a2 * a;                        \
    double  a4 = a2 * a2;                       \
    double  b2 = b  * b;                        \
    double  b3 = b2 * b;                        \
    double  b4 = b2 * b2;                       \
                                                \
    f[0] =        a  * a4;                      \
    f[1] =  5.0 * b  * a4;                      \
    f[2] = 10.0 * a3 * b2;                      \
    f[3] = 10.0 * a2 * b3;                      \
    f[4] =  5.0 * a  * b4;                      \
    f[5] =        b  * b4;                      \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE6B_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE6B_GRADS
#define JIVE_EVAL_LINE6B_GRADS( f, g, u )       \
                                                \
  do                                            \
  {                                             \
    double  a  = (1.0 - u) / 2.0;               \
    double  b  = (1.0 + u) / 2.0;               \
    double  a2 = a  * a;                        \
    double  a3 = a2 * a;                        \
    double  a4 = a2 * a2;                       \
    double  b2 = b  * b;                        \
    double  b3 = b2 * b;                        \
    double  b4 = b2 * b2;                       \
                                                \
    f[0] =        a  * a4;                      \
    f[1] =  5.0 * b  * a4;                      \
    f[2] = 10.0 * a3 * b2;                      \
    f[3] = 10.0 * a2 * b3;                      \
    f[4] =  5.0 * a  * b4;                      \
    f[5] =        b  * b4;                      \
                                                \
    g(0,0) = -2.5 * a4;                         \
    g(0,1) =  2.5 * a3 *     (a - 4.0 * b);     \
    g(0,2) = 10.0 * a2 * b * (a - 1.5 * b);     \
    g(0,3) = 10.0 * b2 * a * (1.5 * a - b);     \
    g(0,4) = -2.5 * b3 *     (b - 4.0 * a);     \
    g(0,5) =  2.5 * b4;                         \
  }                                             \
  while ( false )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_LINE6B_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_LINE6B_GRADS2
#define JIVE_EVAL_LINE6B_GRADS2( f, g, h, u )           \
                                                        \
  do                                                    \
  {                                                     \
    double  a  = (1.0 - u) / 2.0;                       \
    double  b  = (1.0 + u) / 2.0;                       \
    double  a2 = a  * a;                                \
    double  a3 = a2 * a;                                \
    double  a4 = a2 * a2;                               \
    double  b2 = b  * b;                                \
    double  b3 = b2 * b;                                \
    double  b4 = b2 * b2;                               \
                                                        \
    f[0] =        a  * a4;                              \
    f[1] =  5.0 * b  * a4;                              \
    f[2] = 10.0 * a3 * b2;                              \
    f[3] = 10.0 * a2 * b3;                              \
    f[4] =  5.0 * a  * b4;                              \
    f[5] =        b  * b4;                              \
                                                        \
    g(0,0) = -2.5 * a4;                                 \
    g(0,1) =  2.5 * a3 *     (a - 4.0 * b);             \
    g(0,2) = 10.0 * a2 * b * (a - 1.5 * b);             \
    g(0,3) = 10.0 * b2 * a * (1.5 * a - b);             \
    g(0,4) = -2.5 * b3 *     (b - 4.0 * a);             \
    g(0,5) =  2.5 * b4;                                 \
                                                        \
    h(0,0) =  5.0 * a3;                                 \
    h(0,1) = 10.0 * a2 * (1.5 * b - a);                 \
    h(0,2) =  5.0 * a  * (a2 - 6.0 * b * a + 3.0 * b2); \
    h(0,3) =  5.0 * b  * (b2 - 6.0 * a * b + 3.0 * a2); \
    h(0,4) = 10.0 * b2 * (1.5 * a - b);                 \
    h(0,5) =  5.0 *  b3;                                \
  }                                                     \
  while ( false )


//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Line2Funcs
//=======================================================================


inline void Line2Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE2_FUNCS( f, x );
}


inline void Line2Funcs::evalFuncs

  ( FuncVector&  f,
    double       u )

{
  JIVE_EVAL_LINE2_FUNCS( f, u );
}


inline void Line2Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE2_GRADS( f, g, x );
}


inline void Line2Funcs::evalGrads

  ( FuncVector&  f,
    GradMatrix&  g,
    double       u )

{
  JIVE_EVAL_LINE2_GRADS( f, g, u );
}


inline void Line2Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE2_GRADS2( f, g, h, x );
}


inline void Line2Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    double        u )

{
  JIVE_EVAL_LINE2_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Line2C1Funcs
//=======================================================================


inline void Line2C1Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE2C1_FUNCS( f, x );
}


inline void Line2C1Funcs::evalFuncs

  ( FuncVector&  f,
    double       u )

{
  JIVE_EVAL_LINE2C1_FUNCS( f, u );
}


inline void Line2C1Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE2C1_GRADS( f, g, x );
}


inline void Line2C1Funcs::evalGrads

  ( FuncVector&  f,
    GradMatrix&  g,
    double       u )

{
  JIVE_EVAL_LINE2C1_GRADS( f, g, u );
}


inline void Line2C1Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE2C1_GRADS2( f, g, h, x );
}


inline void Line2C1Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    double        u )

{
  JIVE_EVAL_LINE2C1_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Line3Funcs
//=======================================================================


inline void Line3Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE3_FUNCS( f, x );
}


inline void Line3Funcs::evalFuncs

  ( FuncVector&  f,
    double       u )

{
  JIVE_EVAL_LINE3_FUNCS( f, u );
}


inline void Line3Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE3_GRADS( f, g, x );
}


inline void Line3Funcs::evalGrads

  ( FuncVector&  f,
    GradMatrix&  g,
    double       u )

{
  JIVE_EVAL_LINE3_GRADS( f, g, u );
}


inline void Line3Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE3_GRADS2( f, g, h, x );
}


inline void Line3Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    double        u )

{
  JIVE_EVAL_LINE3_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Line4Funcs
//=======================================================================


inline void Line4Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE4_FUNCS( f, x );
}


inline void Line4Funcs::evalFuncs

  ( FuncVector&  f,
    double       u )

{
  JIVE_EVAL_LINE4_FUNCS( f, u );
}


inline void Line4Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE4_GRADS( f, g, x );
}


inline void Line4Funcs::evalGrads

  ( FuncVector&  f,
    GradMatrix&  g,
    double       u )

{
  JIVE_EVAL_LINE4_GRADS( f, g, u );
}


inline void Line4Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE4_GRADS2( f, g, h, x );
}


inline void Line4Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    double        u )

{
  JIVE_EVAL_LINE4_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Line3BFuncs
//=======================================================================


inline void Line3BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE3B_FUNCS( f, x );
}


inline void Line3BFuncs::evalFuncs

  ( FuncVector&  f,
    double       u )

{
  JIVE_EVAL_LINE3B_FUNCS( f, u );
}


inline void Line3BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE3B_GRADS( f, g, x );
}


inline void Line3BFuncs::evalGrads

  ( FuncVector&  f,
    GradMatrix&  g,
    double       u )

{
  JIVE_EVAL_LINE3B_GRADS( f, g, u );
}


inline void Line3BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE3B_GRADS2( f, g, h, x );
}


inline void Line3BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    double        u )

{
  JIVE_EVAL_LINE3B_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Line4BFuncs
//=======================================================================


inline void Line4BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE4B_FUNCS( f, x );
}


inline void Line4BFuncs::evalFuncs

  ( FuncVector&  f,
    double       u )

{
  JIVE_EVAL_LINE4B_FUNCS( f, u );
}


inline void Line4BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE4B_GRADS( f, g, x );
}


inline void Line4BFuncs::evalGrads

  ( FuncVector&  f,
    GradMatrix&  g,
    double       u )

{
  JIVE_EVAL_LINE4B_GRADS( f, g, u );
}


inline void Line4BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE4B_GRADS2( f, g, h, x );
}


inline void Line4BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    double        u )

{
  JIVE_EVAL_LINE4B_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Line5BFuncs
//=======================================================================


inline void Line5BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE5B_FUNCS( f, x );
}


inline void Line5BFuncs::evalFuncs

  ( FuncVector&  f,
    double       u )

{
  JIVE_EVAL_LINE5B_FUNCS( f, u );
}


inline void Line5BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE5B_GRADS( f, g, x );
}


inline void Line5BFuncs::evalGrads

  ( FuncVector&  f,
    GradMatrix&  g,
    double       u )

{
  JIVE_EVAL_LINE5B_GRADS( f, g, u );
}


inline void Line5BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE5B_GRADS2( f, g, h, x );
}


inline void Line5BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    double        u )

{
  JIVE_EVAL_LINE5B_GRADS2( f, g, h, u );
}


//=======================================================================
//   class Line6BFuncs
//=======================================================================


inline void Line6BFuncs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE6B_FUNCS( f, x );
}


inline void Line6BFuncs::evalFuncs

  ( FuncVector&  f,
    double       u )

{
  JIVE_EVAL_LINE6B_FUNCS( f, u );
}


inline void Line6BFuncs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE6B_GRADS( f, g, x );
}


inline void Line6BFuncs::evalGrads

  ( FuncVector&  f,
    GradMatrix&  g,
    double       u )

{
  JIVE_EVAL_LINE6B_GRADS( f, g, u );
}


inline void Line6BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  const double  x = u[0];

  JIVE_EVAL_LINE6B_GRADS2( f, g, h, x );
}


inline void Line6BFuncs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    double        u )

{
  JIVE_EVAL_LINE6B_GRADS2( f, g, h, u );
}


JIVE_END_PACKAGE( geom )

#endif
