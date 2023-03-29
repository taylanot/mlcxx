
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


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class SparseSelector
//=======================================================================

//-----------------------------------------------------------------------
//   select (SparseMatrix)
//-----------------------------------------------------------------------


template <class T>

  SparseMatrix<T> SparseSelector::select

    ( const SparseMatrix<T>&  src,
      const Array<idx_t>&       rowMap,
      const Array<idx_t>&       colMap )

{
  SparseMatrix<T>  out ( select( src.getStructure(),
                                 rowMap, colMap ) );

  const idx_t  srcColCount = src.size (1);
  const idx_t  outRowCount = out.size (0);
  const idx_t  outColCount = out.size (1);

  Array<bool>  mbuf ( srcColCount );

  bool*       JEM_RESTRICT  mask      = mbuf.addr       ();
  T*          JEM_RESTRICT  outValues = out.getValuePtr ();

  idx_t       j = 0;


  for ( idx_t i = 0; i < srcColCount; i++ )
  {
    mask[i] = false;
  }

  for ( idx_t i = 0; i < outColCount; i++ )
  {
    mask[colMap[i]] = true;
  }

  if ( src.isContiguous() )
  {
    const idx_t* JEM_RESTRICT  srcOffsets = src.getOffsetPtr ();
    const idx_t* JEM_RESTRICT  srcIndices = src.getIndexPtr  ();
    const T*     JEM_RESTRICT  srcValues  = src.getValuePtr  ();


    for ( idx_t jrow = 0; jrow < outRowCount; jrow++ )
    {
      idx_t  irow = rowMap    [jrow];
      idx_t  i    = srcOffsets[irow];
      idx_t  n    = srcOffsets[irow + 1];

      for ( ; i < n; i++ )
      {
        if ( mask[srcIndices[i]] )
        {
          outValues[j++] = srcValues[i];
        }
      }
    }
  }
  else
  {
    const Array<idx_t>&  srcOffsets = src.getRowOffsets    ();
    const Array<idx_t>&  srcIndices = src.getColumnIndices ();
    const Array<T>&      srcValues  = src.getValues        ();


    for ( idx_t jrow = 0; jrow < outRowCount; jrow++ )
    {
      idx_t  irow = rowMap    [jrow];
      idx_t  i    = srcOffsets[irow];
      idx_t  n    = srcOffsets[irow + 1];

      for ( ; i < n; i++ )
      {
        if ( mask[srcIndices[i]] )
        {
          outValues[j++] = srcValues[i];
        }
      }
    }
  }

  return out;
}


//-----------------------------------------------------------------------
//   select (SparseStruct)
//-----------------------------------------------------------------------


template <class R, class C>

  SparseStruct          select

  ( const SparseStruct&   src,
    const R&              rows,
    const C&              cols )

{
  Array<idx_t>  rowMap =

    SparseSelector::makeMap ( src.size(0), "row",    rows );

  Array<idx_t>  colMap =

    SparseSelector::makeMap ( src.size(1), "column", cols );

  return SparseSelector::select ( src, rowMap, colMap );
}


//-----------------------------------------------------------------------
//   select (SparseMatrix)
//-----------------------------------------------------------------------


template <class T, class R, class C>

  SparseMatrix<T>         select

  ( const SparseMatrix<T>&  src,
    const R&                rows,
    const C&                cols )

{
  Array<idx_t>  rowMap =

    SparseSelector::makeMap ( src.size(0), "row",    rows );

  Array<idx_t>  colMap =

    SparseSelector::makeMap ( src.size(1), "column", cols );

  return SparseSelector::select ( src, rowMap, colMap );
}


JEM_END_PACKAGE( numeric )
