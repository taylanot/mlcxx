
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


#include <jive/fem/Names.h>


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class Names
//=======================================================================


const char*  Names::IROWS   = "irows";
const char*  Names::JCOLS   = "jcols";
const char*  Names::VALUES  = "values";
const char*  Names::WEIGHTS = "weights";


//=======================================================================
//   class PropertyNames
//=======================================================================


const char*  PropertyNames::BLOCK_COUNT     = "blockCount";
const char*  PropertyNames::BOUNDARY_ELEMS  = "boundaryElems";
const char*  PropertyNames::CHECK_BLOCKS    = "checkBlocks";
const char*  PropertyNames::DBASE_FILTER    = "dbaseFilter";
const char*  PropertyNames::DOMAINS         = "domains";
const char*  PropertyNames::FILE            = "file";
const char*  PropertyNames::FUNC_FILTER     = "funcFilter";
const char*  PropertyNames::INTERFACE_ELEMS = "interfaceElems";
const char*  PropertyNames::MAP_FILE        = "mapFile";
const char*  PropertyNames::MAX_PARTS       = "maxParts";
const char*  PropertyNames::MAX_PRECISION   = "maxPrecision";
const char*  PropertyNames::MIN_DEGREE      = "minDegree";
const char*  PropertyNames::OVERLAP         = "overlap";
const char*  PropertyNames::PARTITIONER     = "partitioner";
const char*  PropertyNames::PRECISION       = "precision";
const char*  PropertyNames::REORDER         = "reorder";
const char*  PropertyNames::SHAPE_TABLE     = "shapeTable";
const char*  PropertyNames::TABLE_FILTER    = "tableFilter";


JIVE_END_PACKAGE( fem )
