
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


#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class SparseUtils
//=======================================================================

//-----------------------------------------------------------------------
//   sumOffsets
//-----------------------------------------------------------------------


idx_t SparseUtils::sumOffsets ( idx_t* offsets, idx_t n )
{
  idx_t  i, j, k;

  j = 0;

  for ( i = 0; i < n; i++ )
  {
    k          = offsets[i];
    offsets[i] = j;
    j         += k;
  }

  return j;
}


idx_t SparseUtils::sumOffsets ( const Array<idx_t>& offsets )
{
  if ( offsets.isContiguous() )
  {
    return sumOffsets ( offsets.addr(), offsets.size() );
  }
  else
  {
    const idx_t  n = offsets.size ();

    idx_t        i, j, k;

    j = 0;

    for ( i = 0; i < n; i++ )
    {
      k          = offsets[i];
      offsets[i] = j;
      j         += k;
    }

    return j;
  }
}


//-----------------------------------------------------------------------
//   shiftOffsets
//-----------------------------------------------------------------------


void SparseUtils::shiftOffsets ( idx_t* offsets, idx_t n )
{
  for ( idx_t i = n - 1; i > 0; i-- )
  {
    offsets[i] = offsets[i - 1];
  }

  if ( n > 0 )
  {
    offsets[0] = 0;
  }
}


void SparseUtils::shiftOffsets ( const Array<idx_t>& offsets )
{
  if ( offsets.isContiguous() )
  {
    shiftOffsets ( offsets.addr(), offsets.size() );
  }
  else
  {
    shift ( offsets, 1 );

    if ( offsets.size() > 0 )
    {
      offsets[0] = 0;
    }
  }
}


//-----------------------------------------------------------------------
//   findSuperRows
//-----------------------------------------------------------------------


void SparseUtils::findSuperRows

  ( Array<idx_t>&        superRows,
    idx_t                maxSize,
    const Array<idx_t>&  rowOffsets,
    const Array<idx_t>&  colIndices )

{
  const idx_t  rowCount = rowOffsets.size() - 1;

  idx_t        size;
  idx_t        irow;
  idx_t        isup;
  idx_t        i, j, k;


  superRows.resize ( rowCount + 1 );

  size =  0;
  isup = -1;

  if ( rowCount > 0 )
  {
    size++;
    isup++;

    superRows[isup] = 0;
  }

  for ( irow = 1; irow < rowCount; irow++ )
  {
    i = rowOffsets[irow - 1];
    j = rowOffsets[irow];
    k = rowOffsets[irow + 1];

    if ( size < maxSize &&
         equal( colIndices[slice(i,j)], colIndices[slice(j,k)] ) )
    {
      size++;
    }
    else
    {
      isup++;

      superRows[isup] = irow;
      size            = 1;
    }
  }

  k = isup + 1;

  superRows[k] = rowCount;

  if ( superRows.size() > k + 1 )
  {
    superRows.reshape ( k + 1 );
  }
}


JEM_END_PACKAGE( numeric )
