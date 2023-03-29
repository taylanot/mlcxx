
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

#ifndef JIVE_SOLVER_NAMES_H
#define JIVE_SOLVER_NAMES_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class PropertyNames
//-----------------------------------------------------------------------


class PropertyNames
{
 public:

  static const char*    DIAG_SHIFT;
  static const char*    DOFS;
  static const char*    DROP_TOL;
  static const char*    GRAM_SCHMIDT;
  static const char*    INNER_SOLVER;
  static const char*    LENIENT;
  static const char*    MATRIX;
  static const char*    MAX_FILL;
  static const char*    MAX_ITER;
  static const char*    MAX_VECTORS;
  static const char*    MAX_ZERO_PIVOTS;
  static const char*    MIN_NODES;
  static const char*    MIN_SIZE;
  static const char*    NOISE_LEVEL;
  static const char*    OPERATION;
  static const char*    OUTER_SOLVER;
  static const char*    PIVOT_THRESHOLD;
  static const char*    PRECISION;
  static const char*    PRECONDITIONER;
  static const char*    PRINT_INTERVAL;
  static const char*    PRINT_PIVOTS;
  static const char*    QUALITY;
  static const char*    REORDER;
  static const char*    RESTART_ITER;
  static const char*    RESTRICTOR;
  static const char*    RESTRICTORS;
  static const char*    SMOOTH;
  static const char*    SOLVER;
  static const char*    SYMMETRIC;
  static const char*    TABLE;
  static const char*    TYPE;
  static const char*    UPDATE_POLICY;
  static const char*    USE_THREADS;
  static const char*    ZERO_THRESHOLD;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef PropertyNames   PropNames;


JIVE_END_PACKAGE( solver )

#endif
