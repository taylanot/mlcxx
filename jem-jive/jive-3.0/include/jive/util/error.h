
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

#ifndef JIVE_UTIL_ERROR_H
#define JIVE_UTIL_ERROR_H

#include <jem/base/String.h>
#include <jem/base/tuple/Tuple.h>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   error functions
//-----------------------------------------------------------------------


void                          indexError

  ( const String&               context,
    idx_t                       index,
    idx_t                       ubound );

void                          indexError

  ( const String&               context,
    const String&               name,
    idx_t                       index,
    idx_t                       ubound );

void                          sizeError

  ( const String&               context,
    const String&               name,
    idx_t                       actualSize,
    idx_t                       formalSize );

void                          shapeError

  ( const String&               context,
    const String&               name,
    const jem::Tuple<idx_t,2>&  actualShape,
    const jem::Tuple<idx_t,2>&  formalShape );

void                          shapeError

  ( const String&               context,
    const String&               name,
    const jem::Tuple<idx_t,3>&  actualShape,
    const jem::Tuple<idx_t,3>&  formalShape );

void                          shapeError

  ( const String&               context,
    const String&               name,
    const jem::Tuple<idx_t,4>&  actualShape,
    const jem::Tuple<idx_t,4>&  formalShape );

void                          nonSquareMatrixError

  ( const String&               context,
    const jem::Tuple<idx_t,2>&  shape );

void                          nonSquareMatrixError

  ( const String&               context,
    const String&               name,
    const jem::Tuple<idx_t,2>&  shape );

void                          classCastError

  ( const String&               context,
    const String&               actualType,
    const String&               formalType );


JIVE_END_PACKAGE( util )

#endif
