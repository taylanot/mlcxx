
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

#ifndef JIVE_ALGEBRA_UTILITIES_H
#define JIVE_ALGEBRA_UTILITIES_H

#include <jive/Array.h>
#include <jive/SparseMatrix.h>
#include <jive/algebra/import.h>


JIVE_BEGIN_PACKAGE( algebra )


class AbstractMatrix;


//-----------------------------------------------------------------------
//   functions
//-----------------------------------------------------------------------


void                      gather

  ( const Vector&           lhs,
    const Slice&            slice,
    const Vector&           rhs );

void                      gather

  ( const Matrix&           lhs,
    const Slice&            rows,
    const Slice&            cols,
    const Matrix&           rhs );

void                      scatter

  ( const Vector&           lhs,
    const Slice&            slice,
    const Vector&           rhs,
    double                  alpha = 1.0 );

void                      scatter

  ( const Matrix&           lhs,
    const Slice&            rows,
    const Slice&            cols,
    const Matrix&           rhs,
    double                  alpha = 1.0 );

SparseStruct              denseToSparse

  ( idx_t                   m,
    idx_t                   n );

SparseMatrix              denseToSparse

  ( const Matrix&           matrix );

void                      markBorderRows

  ( const BoolVector&       mask,
    VectorExchanger&        vex,
    const SparseStruct&     mstruct );

void                      markBorderRows

  ( const BoolVector&       mask,
    VectorExchanger&        vex,
    const AbstractMatrix&   matrix );


JIVE_END_PACKAGE( algebra )

#endif
