
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

#ifndef JIVE_GEOM_HYPERCUBEFUNCS_H
#define JIVE_GEOM_HYPERCUBEFUNCS_H

#include <jive/geom/shfuncs/PrismFuncs.h>
#include <jive/geom/shfuncs/LineFuncs.h>
#include <jive/geom/shfuncs/CubeFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class HyperCube16Funcs
//-----------------------------------------------------------------------


class HyperCube16Funcs : public ShapeFuncs<4,16>
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
//   macro JIVE_EVAL_HYPERCUBE16_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_HYPERCUBE16_FUNCS
#define JIVE_EVAL_HYPERCUBE16_FUNCS( f, u )             \
                                                        \
  JIVE_EVAL_PRISM_FUNCS( f, u, Cube8Funcs, Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_HYPERCUBE16_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_HYPERCUBE16_GRADS
#define JIVE_EVAL_HYPERCUBE16_GRADS( f, g, u )          \
                                                        \
  JIVE_EVAL_PRISM_GRADS( f, g, u, Cube8Funcs, Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_HYPERCUBE16_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_HYPERCUBE16_GRADS2
#define JIVE_EVAL_HYPERCUBE16_GRADS2( f, g, h, u )              \
                                                                \
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u, Cube8Funcs, Line2Funcs )





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class HyperCube16Funcs
//=======================================================================


inline void HyperCube16Funcs::evalFuncs

  ( FuncVector&   f,
    const Point&  u )

{
  JIVE_EVAL_PRISM_FUNCS( f, u, Cube8Funcs, Line2Funcs );
}


inline void HyperCube16Funcs::evalGrads

  ( FuncVector&   f,
    GradMatrix&   g,
    const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS( f, g, u, Cube8Funcs, Line2Funcs );
}


inline void HyperCube16Funcs::evalGrads2

  ( FuncVector&   f,
    GradMatrix&   g,
    Grad2Matrix&  h,
    const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u, Cube8Funcs, Line2Funcs );
}


JIVE_END_PACKAGE( geom )

#endif
