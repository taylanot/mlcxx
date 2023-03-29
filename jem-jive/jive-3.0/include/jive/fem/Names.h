
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

#ifndef JIVE_FEM_NAMES_H
#define JIVE_FEM_NAMES_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class Names
//-----------------------------------------------------------------------


class Names
{
 public:

  static const char*    IROWS;
  static const char*    JCOLS;
  static const char*    VALUES;
  static const char*    WEIGHTS;

};


//-----------------------------------------------------------------------
//   class PropertyNames
//-----------------------------------------------------------------------


class PropertyNames
{
 public:

  static const char*    BLOCK_COUNT;
  static const char*    BOUNDARY_ELEMS;
  static const char*    CHECK_BLOCKS;
  static const char*    DBASE_FILTER;
  static const char*    DOMAINS;
  static const char*    FILE;
  static const char*    FUNC_FILTER;
  static const char*    INTERFACE_ELEMS;
  static const char*    MAP_FILE;
  static const char*    MAX_PARTS;
  static const char*    MAX_PRECISION;
  static const char*    MIN_DEGREE;
  static const char*    OVERLAP;
  static const char*    PARTITIONER;
  static const char*    PRECISION;
  static const char*    REORDER;
  static const char*    SHAPE_TABLE;
  static const char*    TABLE_FILTER;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef PropertyNames   PropNames;


JIVE_END_PACKAGE( fem )

#endif
