
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

#ifndef JIVE_ARRAY_H
#define JIVE_ARRAY_H

#include <jem/base/array/Array.h>
#include <jive/defines.h>


//-----------------------------------------------------------------------
//   common Array types
//-----------------------------------------------------------------------


namespace jive
{
  typedef jem::Array<double>    Vector;
  typedef jem::Array<double,2>  Matrix;
  typedef jem::Array<double,3>  Cubix;

  typedef jem::Array<int>       IntVector;
  typedef jem::Array<int,2>     IntMatrix;
  typedef jem::Array<int,3>     IntCubix;

  typedef jem::Array<idx_t>     IdxVector;
  typedef jem::Array<idx_t,2>   IdxMatrix;
  typedef jem::Array<idx_t,3>   IdxCubix;

  typedef jem::Array<bool>      BoolVector;
  typedef jem::Array<bool,2>    BoolMatrix;
  typedef jem::Array<bool,2>    BoolCubix;

  typedef jem::Array<String>    StringVector;

  typedef jem::Array
    < jem::Array<double,2> >    MatrixVector;

  using jem::ALL;
  using jem::BEGIN;
  using jem::END;
  using jem::slice;
  using jem::Slice;
  using jem::SliceAll;
}


#endif
