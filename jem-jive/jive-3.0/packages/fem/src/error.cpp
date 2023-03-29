
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


#include <jem/base/IllegalInputException.h>
#include <jive/fem/error.h>


using jem::IllegalInputException;


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   elemNodeCountError
//-----------------------------------------------------------------------


void              elemNodeCountError

  ( const String&   context,
    idx_t           elemID,
    idx_t           actualNodeCount,
    idx_t           formalNodeCount )

{
  throw IllegalInputException (
    context,
    String::format (
      "element %d has %d nodes but should have %d nodes",
      elemID,
      actualNodeCount,
      formalNodeCount
    )
  );
}


//-----------------------------------------------------------------------
//   localBoundaryIndexError
//-----------------------------------------------------------------------


void              localBoundaryIndexError

  ( const String&   context,
    idx_t           boundaryID,
    idx_t           localIndex,
    idx_t           elemBndCount )

{
  throw IllegalInputException (
    context,
    String::format (
      "boundary %d: invalid element boundary index: %d; "
      "valid range is [0,%d)",
      boundaryID,
      localIndex,
      elemBndCount
    )
  );
}


JIVE_END_PACKAGE( fem )
