
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


#include <jive/implict/Names.h>


JIVE_BEGIN_PACKAGE( implict )


//=======================================================================
//   class PropertyNames
//=======================================================================


const char*  PropertyNames::APPLY_CONS     = "applyCons";
const char*  PropertyNames::ARC_FUNC       = "arcFunc";
const char*  PropertyNames::BOUNDS         = "bounds";
const char*  PropertyNames::DELTA_CONS     = "deltaCons";
const char*  PropertyNames::DELTA_TIME     = "deltaTime";
const char*  PropertyNames::DOF_TYPE       = "dofType";
const char*  PropertyNames::EIGEN_COUNTS   = "eigenCounts";
const char*  PropertyNames::GAMMA          = "gamma";
const char*  PropertyNames::LEFT_MATRIX    = "leftMatrix";
const char*  PropertyNames::LINE_SEARCH    = "lineSearch";
const char*  PropertyNames::LOAD_INCR      = "loadIncr";
const char*  PropertyNames::LOAD_SCALE     = "loadScale";
const char*  PropertyNames::LOWER_BOUND    = "lowerBound";
const char*  PropertyNames::MATRIX         = "matrix";
const char*  PropertyNames::MAX_DTIME      = "maxDTime";
const char*  PropertyNames::MAX_INCR       = "maxIncr";
const char*  PropertyNames::MAX_ITER       = "maxIter";
const char*  PropertyNames::MIN_DTIME      = "minDTime";
const char*  PropertyNames::MIN_INCR       = "minIncr";
const char*  PropertyNames::MODEL          = "model";
const char*  PropertyNames::ORDER          = "order";
const char*  PropertyNames::PRECISION      = "precision";
const char*  PropertyNames::PRINT_VALUES   = "printValues";
const char*  PropertyNames::RAYLEIGH1      = "rayleigh1";
const char*  PropertyNames::RAYLEIGH2      = "rayleigh2";
const char*  PropertyNames::RIGHT_MATRIX   = "rightMatrix";
const char*  PropertyNames::SOLVER         = "solver";
const char*  PropertyNames::STORE_VECTORS  = "storeVectors";
const char*  PropertyNames::TIME_STEPPER   = "timeStepper";
const char*  PropertyNames::TINY           = "tiny";
const char*  PropertyNames::UPDATE_COND    = "updateWhen";
const char*  PropertyNames::UPPER_BOUND    = "upperBound";
const char*  PropertyNames::WEIGHT_TABLE   = "weightTable";


JIVE_END_PACKAGE( implict )
