
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


#include <jem/base/assert.h>
#include <jem/base/array/utilities.h>
#include <jive/util/utilities.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   makeUnique
//-----------------------------------------------------------------------


IdxVector makeUnique ( const IdxVector& ivec )
{
  const idx_t  n = ivec.size ();

  idx_t        lbound;
  idx_t        ubound;


  if ( n <= 1 )
  {
    return ivec;
  }

  lbound = ubound = ivec[0];

  for ( idx_t i = 1; i < n; i++ )
  {
    idx_t  j = ivec[i];

    if ( j < lbound )
    {
      lbound = j;
    }
    else if ( j > ubound )
    {
      ubound = j;
    }
  }

  return makeUnique ( ivec, lbound, ubound );
}


IdxVector           makeUnique

  ( const IdxVector&  ivec,
    idx_t             lbound,
    idx_t             ubound )

{
  const idx_t  m = ubound - lbound + 1;
  const idx_t  n = ivec.size ();

  idx_t        i, j;


  if ( n <= 1 )
  {
    return ivec;
  }

  if ( m <= (2 * n) && n > 32 )
  {
    BoolVector  mask ( m );
    idx_t       k;

    mask = false;
    j    = 0;

    for ( i = 0; i < n; i++ )
    {
      k = ivec[i] - lbound;

      if ( ! mask[k] )
      {
        ivec[j++] = ivec[i];
        mask[k]   = true;
      }
    }
  }
  else
  {
    jem::sort ( ivec );

    j = 0;

    for ( i = 1; i < n; i++ )
    {
      if ( ivec[j] != ivec[i] )
      {
        ivec[++j] = ivec[i];
      }
    }

    j++;
  }

  return ivec[slice(BEGIN,j)];
}


JIVE_END_PACKAGE( util )
