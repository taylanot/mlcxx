
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
//   scatter (vector)
//-----------------------------------------------------------------------


void                scatter

  ( const Vector&     lhs,
    const Slice&      slice,
    const Vector&     rhs,
    double            alpha )

{
  JEM_ASSERT2 ( lhs[slice].size() == rhs.size(),
                "Array size mismatch" );

  const double* JEM_RESTRICT  rp = rhs.addr ();
  double*       JEM_RESTRICT  lp = lhs.addr ();

  const idx_t   lst = lhs.stride() * slice.stride();
  const idx_t   n   = rhs.size ();

  idx_t         i   = slice.first ();


  lp += i * lhs.stride();

  if ( rhs.stride() == 1_idx )
  {
    for ( i = 0; i < n; i++ )
    {
      lp[i * lst] += alpha * rp[i];
    }
  }
  else
  {
    const idx_t  rst = rhs.stride ();

    for ( i = 0; i < n; i++ )
    {
      lp[i * lst] += alpha * rp[i * rst];
    }
  }
}


//-----------------------------------------------------------------------
//   scatter (matrix)
//-----------------------------------------------------------------------


void                scatter

  ( const Matrix&     lhs,
    const Slice&      rows,
    const Slice&      cols,
    const Matrix&     rhs,
    double            alpha )

{
  JEM_ASSERT2 ( equal( lhs(rows,cols).shape(), rhs.shape() ),
                "Array shape mismatch" );

  const double* JEM_RESTRICT  rp = rhs.addr ();
  double*       JEM_RESTRICT  lp = lhs.addr ();

  const idx_t   lst = lhs.stride(0) * rows.stride();
  const idx_t   m   = rhs.size (0);
  const idx_t   n   = rhs.size (1);

  idx_t         i   = rows.first ();
  idx_t         j   = cols.first ();


  lp += i * lhs.stride(0) + j * lhs.stride(1);

  if ( rhs.stride(0) == 1_idx )
  {
    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        lp[i * lst] += alpha * rp[i];
      }

      lp += lhs.stride(1) * cols.stride();
      rp += rhs.stride(1);
    }
  }
  else
  {
    const idx_t  rst = rhs.stride (0);

    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        lp[i * lst] += alpha * rp[i * rst];
      }

      lp += lhs.stride(1) * cols.stride();
      rp += rhs.stride(1);
    }
  }
}


JIVE_END_PACKAGE( algebra )
