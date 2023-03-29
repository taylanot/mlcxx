
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
#include <jem/base/IllegalIndexException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/numeric/sparse/select.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class SparseSelector
//=======================================================================

//-----------------------------------------------------------------------
//   select
//-----------------------------------------------------------------------


SparseStruct SparseSelector::select

  ( const SparseStruct&  src,
    const Array<idx_t>&  rowMap,
    const Array<idx_t>&  colMap )

{
  const idx_t   srcColCount = src.size (1);

  const idx_t   outRowCount = rowMap.size ();
  const idx_t   outColCount = colMap.size ();

  Array<idx_t>  colBuf      ( srcColCount );

  idx_t*        JEM_RESTRICT  invMap = colBuf.addr ();
  idx_t*        JEM_RESTRICT  colIdx = 0;

  Array<idx_t>  outOffsets;
  Array<idx_t>  outIndices;

  idx_t         nnz = 0;


  for ( idx_t i = 0; i < srcColCount; i++ )
  {
    invMap[i] = -1;
  }

  for ( idx_t i = 0; i < outColCount; i++ )
  {
    invMap[colMap[i]] = i;
  }

  if ( src.isContiguous() )
  {
    const idx_t* JEM_RESTRICT  srcOffsets = src.getOffsetPtr ();
    const idx_t* JEM_RESTRICT  srcIndices = src.getIndexPtr  ();

    for ( idx_t jrow = 0; jrow < outRowCount; jrow++ )
    {
      idx_t  irow = rowMap    [jrow];
      idx_t  i    = srcOffsets[irow];
      idx_t  n    = srcOffsets[irow + 1];

      for ( ; i < n; i++ )
      {
        idx_t  jcol = invMap[srcIndices[i]];

        if ( jcol >= 0 )
        {
          nnz++;
        }
      }
    }

    outOffsets.resize ( outRowCount + 1 );
    outIndices.resize ( nnz );

    colIdx = outIndices.addr ();
    nnz    = 0;

    for ( idx_t jrow = 0; jrow < outRowCount; jrow++ )
    {
      outOffsets[jrow] = nnz;

      idx_t   irow = rowMap    [jrow];
      idx_t  i    = srcOffsets[irow];
      idx_t  n    = srcOffsets[irow + 1];

      for ( ; i < n; i++ )
      {
        idx_t  jcol = invMap[srcIndices[i]];

        if ( jcol >= 0 )
        {
          colIdx[nnz++] = jcol;
        }
      }
    }
  }
  else
  {
    const Array<idx_t>&  srcOffsets = src.getRowOffsets    ();
    const Array<idx_t>&  srcIndices = src.getColumnIndices ();

    for ( idx_t jrow = 0; jrow < outRowCount; jrow++ )
    {
      idx_t  irow = rowMap    [jrow];
      idx_t  i    = srcOffsets[irow];
      idx_t  n    = srcOffsets[irow + 1];

      for ( ; i < n; i++ )
      {
        idx_t  jcol = invMap[srcIndices[i]];

        if ( jcol >= 0 )
        {
          nnz++;
        }
      }
    }

    outOffsets.resize ( outRowCount + 1 );
    outIndices.resize ( nnz );

    colIdx = outIndices.addr ();
    nnz    = 0;

    for ( idx_t jrow = 0; jrow < outRowCount; jrow++ )
    {
      outOffsets[jrow] = nnz;

      idx_t  irow = rowMap    [jrow];
      idx_t  i    = srcOffsets[irow];
      idx_t  n    = srcOffsets[irow + 1];

      for ( ; i < n; i++ )
      {
        idx_t  jcol = invMap[srcIndices[i]];

        if ( jcol >= 0 )
        {
          colIdx[nnz++] = jcol;
        }
      }
    }
  }

  outOffsets[outRowCount] = nnz;

  return SparseStructure ( shape ( outRowCount, outColCount ),
                           outOffsets, outIndices );
}


//-----------------------------------------------------------------------
//   makeMap (Array<idx_t>)
//-----------------------------------------------------------------------


Array<idx_t> SparseSelector::makeMap

  ( idx_t                size,
    const String&      kind,
    const Array<idx_t>&  map )

{
  const idx_t  n = map.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  idx = map[i];

    if ( idx < 0 || idx >= size )
    {
      throw IllegalIndexException (
        JEM_FUNC,
        String::format (
          "illegal %s index: %d; valid range is [0,%d)",
          kind,
          idx,
          size
        )
      );
    }
  }

  return map;
}


//-----------------------------------------------------------------------
//   makeMap (Array<bool>)
//-----------------------------------------------------------------------


Array<idx_t> SparseSelector::makeMap

  ( idx_t               size,
    const String&       kind,
    const Array<bool>&  mask )

{
  if ( size != mask.size() )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "invalid %s mask array; length is %d but should be %d",
        kind,
        mask.size (),
        size
      )
    );
  }

  Array<idx_t>  map ( count( mask ) );
  idx_t         i, j;


  for ( i = j = 0; i < size; i++ )
  {
    if ( mask[i] )
    {
      map[j++] = i;
    }
  }

  return map;
}


//-----------------------------------------------------------------------
//   makeMap (Slice)
//-----------------------------------------------------------------------


Array<idx_t>  SparseSelector::makeMap

  ( idx_t          size,
    const String&  kind,
    const Slice&   slice )

{
  idx_t  first  = slice.first  ();
  idx_t  last   = slice.last   ( size );
  idx_t  stride = slice.stride ();

  if ( first >= last || first >= size )
  {
    return Array<idx_t> ();
  }

  Array<idx_t>  map ( (last - first + stride - 1) / stride );

  for ( idx_t i = 0; first < last; i++, first += stride )
  {
    map[i] = first;
  }

  return map;
}


//-----------------------------------------------------------------------
//   makeMap (SliceAll)
//-----------------------------------------------------------------------


Array<idx_t>  SparseSelector::makeMap

  ( idx_t             size,
    const String&     kind,
    const SliceAll&   slice )

{
   Array<idx_t>  map ( size );

   for ( idx_t i = 0; i < size; i++ )
   {
     map[i] = i;
   }

   return map;
}


//-----------------------------------------------------------------------
//   makeMap (SliceFromTo)
//-----------------------------------------------------------------------


Array<idx_t>  SparseSelector::makeMap

  ( idx_t               size,
    const String&       kind,
    const SliceFromTo&  slice )

{
  idx_t  first = slice.first ();
  idx_t  last  = slice.last  ( size );

  if ( first >= last || first >= size )
  {
    return Array<idx_t> ();
  }

  Array<idx_t>  map ( last - first );

  for ( idx_t i = 0; first < last; i++, first++ )
  {
    map[i] = first;
  }

  return map;
}


JEM_END_PACKAGE( numeric )
