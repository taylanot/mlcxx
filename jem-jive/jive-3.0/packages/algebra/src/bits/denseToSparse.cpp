
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


#include <jem/base/array/utilities.h>
#include <jive/algebra/utilities.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   denseToSparse
//-----------------------------------------------------------------------


SparseStruct      denseToSparse

  ( idx_t           m,
    idx_t           n )

{
  using jem::shape;

  JEM_PRECHECK2 ( m >= 0 && n >= 0, "invalid matrix shape" );

  IdxVector  istore  ( m + 1 + m * n );

  IdxVector  offsets = istore[slice(BEGIN,m + 1)];
  IdxVector  indices = istore[slice(m + 1,END)];

  idx_t      k = 0;

  for ( idx_t i = 0; i < m; i++ )
  {
    offsets[i] = k;

    for ( idx_t j = 0; j < n; j++, k++ )
    {
      indices[k] = j;
    }
  }

  offsets[m] = k;

  return SparseStruct ( shape( m, n ), offsets, indices );
}


SparseMatrix      denseToSparse

  ( const Matrix&   matrix )

{
  using jem::flatten;

  const idx_t  m = matrix.size (0);
  const idx_t  n = matrix.size (1);

  Vector       values;

  if ( matrix.stride(0) == (n * matrix.stride(1)) )
  {
    values.ref ( flatten( matrix.transpose() ) );
  }
  else
  {
    idx_t  k = 0;

    values.resize ( m * n );

    for ( idx_t i = 0; i < m; i++ )
    {
      for ( idx_t j = 0; j < n; j++, k++ )
      {
        values[k] = matrix(i,j);
      }
    }
  }

  return SparseMatrix ( denseToSparse( m, n ), values );
}


JIVE_END_PACKAGE( algebra )
