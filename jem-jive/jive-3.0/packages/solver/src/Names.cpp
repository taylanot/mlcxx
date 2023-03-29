
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


#include <jive/solver/Names.h>


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class PropertyNames
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  PropertyNames::DIAG_SHIFT      = "diagShift";
const char*  PropertyNames::DOFS            = "dofs";
const char*  PropertyNames::DROP_TOL        = "dropTol";
const char*  PropertyNames::GRAM_SCHMIDT    = "gramSchmidt";
const char*  PropertyNames::INNER_SOLVER    = "innerSolver";
const char*  PropertyNames::LENIENT         = "lenient";
const char*  PropertyNames::MATRIX          = "matrix";
const char*  PropertyNames::MAX_FILL        = "maxFill";
const char*  PropertyNames::MAX_ITER        = "maxIter";
const char*  PropertyNames::MAX_VECTORS     = "maxVectors";
const char*  PropertyNames::MAX_ZERO_PIVOTS = "maxZeroPivots";
const char*  PropertyNames::MIN_NODES       = "minNodes";
const char*  PropertyNames::MIN_SIZE        = "minSize";
const char*  PropertyNames::NOISE_LEVEL     = "noiseLevel";
const char*  PropertyNames::OPERATION       = "operation";
const char*  PropertyNames::OUTER_SOLVER    = "outerSolver";
const char*  PropertyNames::PIVOT_THRESHOLD = "pivotThreshold";
const char*  PropertyNames::PRECISION       = "precision";
const char*  PropertyNames::PRECONDITIONER  = "precon";
const char*  PropertyNames::PRINT_INTERVAL  = "printInterval";
const char*  PropertyNames::PRINT_PIVOTS    = "printPivots";
const char*  PropertyNames::QUALITY         = "quality";
const char*  PropertyNames::REORDER         = "reorder";
const char*  PropertyNames::RESTART_ITER    = "restartIter";
const char*  PropertyNames::RESTRICTOR      = "restrictor";
const char*  PropertyNames::RESTRICTORS     = "restrictors";
const char*  PropertyNames::SMOOTH          = "smooth";
const char*  PropertyNames::SOLVER          = "solver";
const char*  PropertyNames::SYMMETRIC       = "symmetric";
const char*  PropertyNames::TABLE           = "table";
const char*  PropertyNames::TYPE            = "type";
const char*  PropertyNames::UPDATE_POLICY   = "updatePolicy";
const char*  PropertyNames::USE_THREADS     = "useThreads";
const char*  PropertyNames::ZERO_THRESHOLD  = "zeroThreshold";


JIVE_END_PACKAGE( solver )
