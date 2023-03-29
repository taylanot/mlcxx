
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

#ifndef JIVE_GEOM_SHAPEFUNCS_H
#define JIVE_GEOM_SHAPEFUNCS_H

#include <jive/geom/Tuple.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class ShapeFuncs
//-----------------------------------------------------------------------


template <int R, int N>

  class ShapeFuncs

{
 public:

  static const int                RANK        = R;
  static const int                FUNC_COUNT  = N;
  static const int                GRAD_COUNT  = R;
  static const int                GRAD2_COUNT = (R * R + R) / 2;


  typedef Tuple
    < double, RANK >              Point;
  typedef Tuple
    < double, FUNC_COUNT >        FuncVector;
  typedef Tuple
    < double,
      GRAD_COUNT, FUNC_COUNT >    GradMatrix;
  typedef Tuple
    < double,
      GRAD2_COUNT, FUNC_COUNT >   Grad2Matrix;

};


//#######################################################################
//  Implementation
//#######################################################################


template <int R, int N>

  const int ShapeFuncs<R,N>::RANK;

template <int R, int N>

  const int ShapeFuncs<R,N>::FUNC_COUNT;

template <int R, int N>

  const int ShapeFuncs<R,N>::GRAD_COUNT;

template <int R, int N>

  const int ShapeFuncs<R,N>::GRAD2_COUNT;


JIVE_END_PACKAGE( geom )

#endif
