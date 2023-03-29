
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


#include <jem/numeric/algebra/matmul.h>
#include <jive/solver/GMRESUtils.h>


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class GMRESUtils
//=======================================================================

//-----------------------------------------------------------------------
//   rotate
//-----------------------------------------------------------------------


void GMRESUtils::rotate

  ( const Vector&  h,
    const Matrix&  g )

{
  const idx_t  n = g.size (1);

  for ( idx_t j = 0; j < n; j++ )
  {
    rotate ( h[j], h[j + 1], g(0,j), g(1,j) );
  }
}


//-----------------------------------------------------------------------
//   backsub
//-----------------------------------------------------------------------


void GMRESUtils::backsub

  ( const Vector&  a,
    const Vector&  h )

{
  const idx_t  n = a.size ();

  idx_t        k = h.size() - 1;


  if ( a.isContiguous() && h.isContiguous() )
  {
    const double* JEM_RESTRICT  hp = h.addr ();
    double*       JEM_RESTRICT  ap = a.addr ();

    for ( idx_t i = n - 1; i >= 0; i-- )
    {
      double  x = ap[i] = ap[i] / hp[k];

      k--;

      for ( idx_t j = i - 1; j >= 0; j--, k-- )
      {
        ap[j] -= x * hp[k];
      }
    }
  }
  else
  {
    for ( idx_t i = n - 1; i >= 0; i-- )
    {
      double  x = a[i] = a[i] / h[k];

      k--;

      for ( idx_t j = i - 1; j >= 0; j--, k-- )
      {
        a[j] -= x * h[k];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   storeVector
//-----------------------------------------------------------------------


void GMRESUtils::storeVector

  ( Matrix&             store,
    const Vector&       vec,
    idx_t               j,
    idx_t               maxVecs )

{
  JEM_PRECHECK ( j < maxVecs );

  const idx_t  m = vec  .size ();
  const idx_t  n = store.size (1);

  if ( j >= n )
  {
    idx_t  k = (idx_t) (1.5 * (double) n) + 4;

    if ( k <= j )
    {
      k = j + 1;
    }
    if ( k > maxVecs )
    {
      k = maxVecs;
    }

    store.reshape ( m, k );
  }

  store[j] = vec;
}


//-----------------------------------------------------------------------
//   getSolution
//-----------------------------------------------------------------------


void GMRESUtils::getSolution

  ( const Vector&  lhs,
    const Vector&  h,
    const Vector&  e,
    const Matrix&  v )

{
  using jem::numeric::matmul;

  JEM_PRECHECK ( e.size() == v.size(1) );

  Vector  a = e.clone ();

  backsub ( a, h );
  matmul  ( lhs, v, a );
}


JIVE_END_PACKAGE( solver )
