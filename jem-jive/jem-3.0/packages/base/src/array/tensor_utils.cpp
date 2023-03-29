
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <jem/base/RuntimeException.h>
#include <jem/base/array/tensor_utils.h>


JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( tensor )


//=======================================================================
//   private utility functions
//=======================================================================

//-----------------------------------------------------------------------
//   uboundsError
//-----------------------------------------------------------------------


void      uboundsError

  ( idx_t   ub1,
    idx_t   ub2 )

{
  throw RuntimeException (
    JEM_FUNC,
    String::format (
      "incompatible tensor index bounds: %d and %d",
      ub1,
      ub2
    )
  );
}


//-----------------------------------------------------------------------
//   unboundedError
//-----------------------------------------------------------------------


void unboundedError ()
{
  throw RuntimeException (
    JEM_FUNC,
    "unbounded index in tensor reduction function"
  );
}


//-----------------------------------------------------------------------
//   usedIndexError
//-----------------------------------------------------------------------


void usedIndexError ()
{
  throw RuntimeException (
    JEM_FUNC,
    "repeated index in tensor expression"
  );
}



//-----------------------------------------------------------------------
//   incrIndex
//-----------------------------------------------------------------------


bool              incrIndex

  ( idx_t* const*   index,
    const idx_t*    ubound,
    int             rank )

{
  const int  k = rank - 1;

  if ( k < 0 )
  {
    return false;
  }

  for ( int i = 0; i < k; i++ )
  {
    (*index[i]) = 0_idx;

    if ( (++(*index[i + 1])) < ubound[i + 1] )
    {
      return true;
    }
  }

  return ( (*index[k]) < ubound[k] );
}


//-----------------------------------------------------------------------
//   zeroIndex
//-----------------------------------------------------------------------


void              zeroIndex

  ( idx_t* const*   index,
    int             rank )

{
  for ( int i = 0; i < rank; i++ )
  {
    (*index[i]) = 0_idx;
  }
}


//-----------------------------------------------------------------------
//   checkUnique
//-----------------------------------------------------------------------


void              checkUnique

  ( idx_t* const*   index,
    int             rank )

{
  for ( int i = 0; i < rank; i++ )
  {
    for ( int j = i + 1; j < rank; j++ )
    {
      if ( index[i] == index[j] )
      {
        usedIndexError ();
      }
    }
  }
}


JEM_END_NAMESPACE( tensor )
JEM_END_PACKAGE_BASE
