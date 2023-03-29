
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


#include <jem/base/tuple/utilities.h>
#include <jive/algebra/utilities.h>


JIVE_BEGIN_PACKAGE( algebra )


using jem::equal;


//-----------------------------------------------------------------------
//   gather (vector)
//-----------------------------------------------------------------------


void                gather

  ( const Vector&     lhs,
    const Slice&      slice,
    const Vector&     rhs )

{
  JEM_ASSERT2 ( lhs.size() == rhs[slice].size(),
                "Array size mismatch" );

  const double* JEM_RESTRICT  rp = rhs.addr ();
  double*       JEM_RESTRICT  lp = lhs.addr ();

  const idx_t   rst = rhs.stride() * slice.stride();
  const idx_t   n   = lhs.size ();

  idx_t         i   = slice.first ();


  rp += i * rhs.stride();

  if ( lhs.stride() == 1_idx )
  {
    for ( i = 0; i < n; i++ )
    {
      lp[i] = rp[i * rst];
    }
  }
  else
  {
    const idx_t  lst = lhs.stride ();

    for ( i = 0; i < n; i++ )
    {
      lp[i * lst] = rp[i * rst];
    }
  }
}


//-----------------------------------------------------------------------
//   gather (matrix)
//-----------------------------------------------------------------------


void                gather

  ( const Matrix&     lhs,
    const Slice&      rows,
    const Slice&      cols,
    const Matrix&     rhs )

{
  JEM_ASSERT2 ( equal( lhs.shape(), rhs(rows,cols).shape() ),
                "Array shape mismatch" );

  const double* JEM_RESTRICT  rp = rhs.addr ();
  double*       JEM_RESTRICT  lp = lhs.addr ();

  const idx_t   rst = rhs.stride(0) * rows.stride();
  const idx_t   m   = lhs.size (0);
  const idx_t   n   = lhs.size (1);

  idx_t         i   = rows.first ();
  idx_t         j   = cols.first ();


  rp += i * rhs.stride(0) + j * rhs.stride(1);

  if ( lhs.stride(0) == 1_idx )
  {
    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        lp[i] = rp[i * rst];
      }

      lp += lhs.stride(1);
      rp += rhs.stride(1) * cols.stride();
    }
  }
  else
  {
    const idx_t  lst = lhs.stride (0);

    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        lp[i * lst] = rp[i * rst];
      }

      lp += lhs.stride(1);
      rp += rhs.stride(1) * cols.stride();
    }
  }
}


JIVE_END_PACKAGE( algebra )
