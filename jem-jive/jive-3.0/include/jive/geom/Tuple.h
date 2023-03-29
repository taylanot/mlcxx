
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

#ifndef JIVE_GEOM_TUPLE_H
#define JIVE_GEOM_TUPLE_H

#include <jem/base/tuple/Tuple.h>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   Common tuple types
//-----------------------------------------------------------------------


using jem::Tuple;

typedef jem::Tuple<double,1>    Vec1;
typedef jem::Tuple<double,2>    Vec2;
typedef jem::Tuple<double,3>    Vec3;
typedef jem::Tuple<double,4>    Vec4;

typedef jem::Tuple<idx_t,1>     IdxVec1;
typedef jem::Tuple<idx_t,2>     IdxVec2;
typedef jem::Tuple<idx_t,3>     IdxVec3;
typedef jem::Tuple<idx_t,4>     IdxVec4;

typedef jem::Tuple<double,1,1>  Mat1x1;
typedef jem::Tuple<double,2,2>  Mat2x2;
typedef jem::Tuple<double,3,2>  Mat3x2;
typedef jem::Tuple<double,3,3>  Mat3x3;
typedef jem::Tuple<double,4,4>  Mat4x4;
typedef jem::Tuple<double,6,3>  Mat6x3;
typedef jem::Tuple<double,6,6>  Mat6x6;


JIVE_END_PACKAGE( geom )

#endif
