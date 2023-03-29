
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


#include <cmath>
#include <jem/base/assert.h>
#include <jive/mp/utilities.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/utilities.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/SparseMatrixExtension.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   markBorderRows
//-----------------------------------------------------------------------


void                      markBorderRows

  ( const BoolVector&       mask,
    VectorExchanger&        vex,
    const SparseStruct&     mstruct )

{
  JEM_PRECHECK2 ( mask.size() == mstruct.size(0),
                  "mask array has wrong size" );

  const idx_t  rowCount = mstruct.size (0);

  IdxVector    offsets  = mstruct.getRowOffsets ();

  Matrix       buf ( rowCount, 2 );

  Vector       x   ( buf[0] );
  Vector       y   ( buf[1] );


  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    x[irow] = (double) (offsets[irow + 1] - offsets[irow]);
    y[irow] = x[irow];
  }

  vex.exchange ( y );

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    if ( std::fabs( x[irow] - y[irow] ) > 0.1 )
    {
      mask[irow] = true;
    }
  }
}


void                      markBorderRows

  ( const BoolVector&       mask,
    VectorExchanger&        vex,
    const AbstractMatrix&   matrix )

{
  using jive::mp::markBorderDofs;

  SparseMatrixExt*  smex = matrix.getExtension<SparseMatrixExt> ();

  if ( smex )
  {
    markBorderRows ( mask, vex, smex->getStructure() );
  }
  else
  {
    DofSpace*  dofs = vex.getDofSpace ();

    markBorderDofs ( mask, vex.getRecvBorders(), *dofs );
  }
}


JIVE_END_PACKAGE( algebra )
