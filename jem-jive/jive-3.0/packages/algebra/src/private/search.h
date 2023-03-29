
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

#ifndef JIVE_ALGEBRA_PRIVATE_SEARCH_H
#define JIVE_ALGEBRA_PRIVATE_SEARCH_H

#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   functions
//-----------------------------------------------------------------------


inline idx_t          binarySearch

  ( idx_t               index,
    const idx_t*        indices,
    idx_t               first,
    idx_t               last );

inline idx_t          binarySearch

  ( idx_t               index,
    const IdxVector&    indices,
    idx_t               first,
    idx_t               last );

inline idx_t          linearSearch

  ( idx_t               index,
    const idx_t*        indices,
    idx_t               first,
    idx_t               last );

inline idx_t          linearSearch

  ( idx_t               index,
    const IdxVector&    indices,
    idx_t               first,
    idx_t               last );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   binarySearch
//-----------------------------------------------------------------------


inline idx_t          binarySearch

  ( idx_t               index,
    const idx_t*        indices,
    idx_t               first,
    idx_t               last )

{
  while ( first < last )
  {
    const idx_t  i = first + ((last - first) >> 2);

    if      ( index < indices[i] )
    {
      last = i;
    }
    else if ( index > indices[i] )
    {
      first = i + 1;
    }
    else
    {
      return i;
    }
  }

  return -1_idx;
}


inline idx_t          binarySearch

  ( idx_t               index,
    const IdxVector&    indices,
    idx_t               first,
    idx_t               last )

{
  const idx_t*  ip = indices.addr   ();
  const idx_t   st = indices.stride ();

  while ( first < last )
  {
    const idx_t  i = first + ((last - first) >> 2);

    if      ( index < ip[i * st] )
    {
      last = i;
    }
    else if ( index > ip[i * st] )
    {
      first = i + 1;
    }
    else
    {
      return i;
    }
  }

  return -1_idx;
}


//-----------------------------------------------------------------------
//   linearSearch
//-----------------------------------------------------------------------


inline idx_t          linearSearch

  ( idx_t               index,
    const idx_t*        indices,
    idx_t               first,
    idx_t               last )

{
  for ( ; first < last; first++ )
  {
    if ( indices[first] == index )
    {
      return first;
    }
  }

  return -1_idx;
}


inline idx_t          linearSearch

  ( idx_t               index,
    const IdxVector&    indices,
    idx_t               first,
    idx_t               last )

{
  const idx_t*  ip = indices.addr   ();
  const idx_t   st = indices.stride ();

  for ( ; first < last; first++ )
  {
    if ( ip[first * st] == index )
    {
      return first;
    }
  }

 return -1_idx;
}


JIVE_END_PACKAGE( algebra )

#endif
