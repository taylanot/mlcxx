
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

#ifndef JIVE_GEOM_BOXES_H
#define JIVE_GEOM_BOXES_H

#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


void                      getBoundingBox

  ( const Vector&           bbox,
    const Matrix&           coords );

void                      findMinMaxBoxes

  ( idx_t&                  jmin,
    idx_t&                  jmax,
    const Matrix&           boxes );

double                    sizeofBox

  ( const Vector&           box );

double                    sizeofSmallestBox

  ( const Matrix&           boxes );

double                    sizeofLargestBox

  ( const Matrix&           boxes );

void                      getBoxSizes

  ( const Vector&           sizes,
    const Matrix&           boxes );

void                      getBoxCenter

  ( const Vector&           center,
    const Vector&           box );

void                      getBoxCenters

  ( const Matrix&           centers,
    const Matrix&           boxes );

void                      makeBox

  ( const Vector&           box,
    const Vector&           center,
    double                  size );

void                      makeBoxes

  ( const Matrix&           boxes,
    const Matrix&           centers,
    double                  size );

void                      makeBoxes

  ( const Matrix&           boxes,
    const Matrix&           centers,
    const Vector&           sizes );

void                      makeEmptyBox

  ( const Vector&           box );

void                      scaleBox

  ( const Vector&           box,
    double                  scale );

void                      scaleBoxes

  ( const Matrix&           boxes,
    double                  scale );

void                      scaleBoxes

  ( const Matrix&           boxes,
    const Vector&           scales );

inline int                overlap

  ( double                  first1,
    double                  last1,
    double                  first2,
    double                  last2 );

bool                      isInBox

  ( const Vector&           point,
    const Vector&           box );

idx_t                     findEnclosingBoxes

  ( const IdxVector&        iboxes,
    const Vector&           point,
    const Matrix&           boxes );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   overlap
//-----------------------------------------------------------------------


inline int  overlap

  ( double    first1,
    double    last1,
    double    first2,
    double    last2 )

{
  // Note that this returns 0 (if no overlap) or 1 (if overlap)
  // according to the C++ standard (a bool is cast to 0 or 1).

  return (int) ((last2 - first1) * (last1 - first2) >= 0.0);
}


JIVE_END_PACKAGE( geom )

#endif
