
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


#include <jive/geom/shfuncs/SquareFuncs.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int     Square4Funcs::VERTEX_COORDS[2][4]   =

  { { -1,  1,  1, -1 },
    { -1, -1,  1,  1 } };


const int     Square6Funcs::IPERM[6]              =

  { 0, 1, 2, 5, 4, 3 };


const int     Square8Funcs::VERTEX_COORDS[2][8]   =

  { { -1,  0,  1,  1,  1,  0, -1, -1 },
    { -1, -1, -1,  0,  1,  1,  1,  0 } };


const int     Square9Funcs::IPERM[9]              =

  { 0, 1, 2, 7, 8, 3, 6, 5, 4 };


const int     Square12Funcs::VERTEX_COLORS   [12] =

  { 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 2, 2 };


const double  Square12Funcs::VERTEX_COORDS[2][12] =

  { { -1.0,       -1.0 / 3.0,  1.0 / 3.0,  1.0,
       1.0,        1.0,        1.0,        1.0 / 3.0,
      -1.0 / 3.0, -1.0,       -1.0,       -1.0 },
    { -1.0,       -1.0,       -1.0,       -1.0,
      -1.0 / 3.0,  1.0 / 3.0,  1.0,        1.0,
       1.0,        1.0,        1.0 / 3.0, -1.0 / 3.0 } };


JIVE_END_PACKAGE( geom )
