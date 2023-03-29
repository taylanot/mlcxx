
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


#include <jem/base/Error.h>
#include <jem/base/array/base.h>


JEM_BEGIN_PACKAGE_BASE

JEM_BEGIN_NAMESPACE( array )


//=======================================================================
//   class Errors
//=======================================================================


const char*  Errors::INVALID_DIM    = "invalid Array dimension";
const char*  Errors::INVALID_SIZE   = "invalid Array size";
const char*  Errors::INVALID_SHAPE  = "invalid Array shape";
const char*  Errors::INVALID_SLICE  = "invalid Array slice";
const char*  Errors::INVALID_INDEX  = "invalid Array index";
const char*  Errors::INVALID_INDEX0 = "invalid Array index in "
                                      "dimension 0";
const char*  Errors::INVALID_INDEX1 = "invalid Array index in "
                                      "dimension 1";
const char*  Errors::INVALID_INDEX2 = "invalid Array index in "
                                      "dimension 2";
const char*  Errors::INVALID_INDEX3 = "invalid Array index in "
                                      "dimension 3";
const char*  Errors::INVALID_INDEX4 = "invalid Array index in "
                                      "dimension 4";
const char*  Errors::INVALID_INDEX5 = "invalid Array index in "
                                      "dimension 5";
const char*  Errors::INVALID_SELECT = "invalid Array selection "
                                      "(invalid index)";
const char*  Errors::SIZE_MISMATCH  = "Array size mismatch";
const char*  Errors::SHAPE_MISMATCH = "Array shape mismatch";
const char*  Errors::EMPTY_ARRAY    = "empty Array";
const char*  Errors::RESHAPE_ERROR  = "invalid Array reshape "
                                      "operation";


//=======================================================================
//   private utility functions
//=======================================================================


//-----------------------------------------------------------------------
//   getFastError
//-----------------------------------------------------------------------


void getFastError ()
{
  throw Error (
    JEM_FUNC,
    "method `getFast\' called for a non-contiguous array"
  );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t           size

  ( const idx_t*  shape,
    int           rank )

{
  idx_t  k = 1;

  for ( int i = 0; i < rank; i++ )
  {
    k *= shape[i];
  }

  return k;
}


//-----------------------------------------------------------------------
//   incrIndex
//-----------------------------------------------------------------------


bool            incrIndex

  ( idx_t*        index,
    const idx_t*  shape,
    int           rank )

{
  const int  k = rank - 1;

  if ( k < 0 )
  {
    return false;
  }

  for ( int i = 0; i < k; i++ )
  {
    index[i] = 0;

    if ( (++index[i + 1]) < shape[i + 1] )
    {
      return true;
    }
  }

  return (index[k] < shape[k]);
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


bool            isContiguous

  ( const idx_t*  shape,
    const idx_t*  stride,
    int           rank )

{
  idx_t  k = 1_idx;

  for ( int i = 0; i < rank; i++ )
  {
    if ( stride[i] != k )
    {
      return false;
    }

    k *= shape[i];
  }

  return true;
}


//-----------------------------------------------------------------------
//   checkShape
//-----------------------------------------------------------------------


bool            checkShape

  ( const idx_t*  shape,
    int           rank )

{
  for ( int i = 0; i < rank; i++ )
  {
    if ( shape[i] < 0_idx )
    {
      return false;
    }
  }

  return true;
}


JEM_END_NAMESPACE( array )

JEM_END_PACKAGE_BASE
