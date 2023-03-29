
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

#ifndef JIVE_SOLVER_GMRESUTILS_H
#define JIVE_SOLVER_GMRESUTILS_H

#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class GMRESUtils
//-----------------------------------------------------------------------


class GMRESUtils
{
 public:

  static inline void    rotate

    ( double&             x,
      double&             y,
      double              c,
      double              s );

  static void           rotate

    ( const Vector&       h,
      const Matrix&       g );

  static void           backsub

    ( const Vector&       a,
      const Vector&       h );

  static void           storeVector

    ( Matrix&             store,
      const Vector&       vec,
      idx_t               j,
      idx_t               maxVecs );

  static void           getSolution

    ( const Vector&       lhs,
      const Vector&       h,
      const Vector&       e,
      const Matrix&       v );

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   rotate
//-----------------------------------------------------------------------


inline void GMRESUtils::rotate

  ( double&  x,
    double&  y,
    double   c,
    double   s )

{
  double a = x;
  double b = y;

  x = c * a - s * b;
  y = s * a + c * b;
}



JIVE_END_PACKAGE( solver )

#endif
