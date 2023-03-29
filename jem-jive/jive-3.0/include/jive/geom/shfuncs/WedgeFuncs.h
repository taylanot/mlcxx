
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

#ifndef JIVE_GEOM_WEDGEFUNCS_H
#define JIVE_GEOM_WEDGEFUNCS_H

#include <jive/geom/shfuncs/PrismFuncs.h>
#include <jive/geom/shfuncs/LineFuncs.h>
#include <jive/geom/shfuncs/TriangFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class Wedge6Funcs
//-----------------------------------------------------------------------


class Wedge6Funcs : public ShapeFuncs<3,6>
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
//   class Wedge18Funcs
//-----------------------------------------------------------------------


class Wedge18Funcs : public ShapeFuncs<3,18>
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
//   macro JIVE_EVAL_WEDGE6_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE6_FUNCS
#define JIVE_EVAL_WEDGE6_FUNCS( f, u )                    \
                                                          \
  JIVE_EVAL_PRISM_FUNCS( f, u,                            \
                         jive::geom::Triang3Funcs,        \
                         jive::geom::Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_WEDGE6_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE6_GRADS
#define JIVE_EVAL_WEDGE6_GRADS( f, g, u )                 \
                                                          \
  JIVE_EVAL_PRISM_GRADS( f, g, u,                         \
                         jive::geom::Triang3Funcs,        \
                         jive::geom::Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_WEDGE6_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE6_GRADS2
#define JIVE_EVAL_WEDGE6_GRADS2( f, g, h, u )             \
                                                          \
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u,                     \
                          jive::geom::Triang3Funcs,       \
                          jive::geom::Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_WEDGE18_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE18_FUNCS
#define JIVE_EVAL_WEDGE18_FUNCS( f, u )                   \
                                                          \
  JIVE_EVAL_PRISM_FUNCS( f, u,                            \
                         jive::geom::Triang6Funcs,        \
                         jive::geom::Line3Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_WEDGE18_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE18_GRADS
#define JIVE_EVAL_WEDGE18_GRADS( f, g, u )                \
                                                          \
  JIVE_EVAL_PRISM_GRADS( f, g, u,                         \
                         jive::geom::Triang6Funcs,        \
                         jive::geom::Line3Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_WEDGE18_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE18_GRADS2
#define JIVE_EVAL_WEDGE18_GRADS2( f, g, h, u )            \
                                                          \
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u,                     \
                          jive::geom::Triang6Funcs,       \
                          jive::geom::Line3Funcs )





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Wedge6Funcs
//=======================================================================


inline void Wedge6Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_PRISM_FUNCS( f, u, Triang3Funcs, Line2Funcs );
}


inline void Wedge6Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS( f, g, u, Triang3Funcs, Line2Funcs );
}


inline void Wedge6Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u, Triang3Funcs, Line2Funcs );
}


//=======================================================================
//   class Wedge18Funcs
//=======================================================================


inline void Wedge18Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_PRISM_FUNCS( f, u, Triang6Funcs, Line3Funcs );
}


inline void Wedge18Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS( f, g, u, Triang6Funcs, Line3Funcs );
}


inline void Wedge18Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u, Triang6Funcs, Line3Funcs );
}


JIVE_END_PACKAGE( geom )

#endif
