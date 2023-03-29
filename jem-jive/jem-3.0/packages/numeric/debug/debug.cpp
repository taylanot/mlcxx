
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


#include <cstdio>
#include <jem/io/utilities.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/intrinsics.h>
#include <jem/numeric/sparse/SparseMatrix.h>


using namespace jem;
using namespace jem::numeric;

using jem::io::getWidthOf;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   printInfo_
//-----------------------------------------------------------------------


static void printInfo_ ( const SparseStructure& s )
{
  std::printf ( "number of rows       : %ld\n",
                (long) s.size(0) );
  std::printf ( "number of columns    : %ld\n",
                (long) s.size(1) );
  std::printf ( "number of non-zeroes : %ld\n",
                (long) s.nonZeroCount() );
}


//-----------------------------------------------------------------------
//   dprint_ ( SparseStructure )
//-----------------------------------------------------------------------


static void dprint_ ( const SparseStructure& s )
{
  Array<idx_t>  rowOffsets = s.getRowOffsets    ();
  Array<idx_t>  colIndices = s.getColumnIndices ();

  const idx_t   rowCount   = s.size (0);

  int           w1, w2;
  idx_t         i,  j, n;


  printInfo_ ( s );

  if ( s.nonZeroCount() == 0 )
  {
    return;
  }

  std::puts ( "\nnon-zero entries [ (irow,jcol) ]:\n" );

  w1 = getWidthOf ( rowCount  );
  w2 = getWidthOf ( s.size(1) );

  for ( i = j = 0; i < rowCount; i++ )
  {
    n = rowOffsets[i + 1];

    for ( ; j < n; j++ )
    {
      std::printf ( "  (%*ld,%*ld)\n",
                    w1, (long) i, w2, (long) colIndices[j] );
    }
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( SparseMatrix<int> )
//-----------------------------------------------------------------------


static void dprint_ ( const SparseMatrix<int>& s )
{
  Array<idx_t>  rowOffsets = s.getRowOffsets    ();
  Array<idx_t>  colIndices = s.getColumnIndices ();
  Array<int>    values     = s.getValues        ();

  const idx_t   rowCount   = s.size (0);

  int           w1, w2, w3;
  idx_t         i,  j,  n;


  printInfo_ ( s.getStructure() );

  if ( s.nonZeroCount() == 0 )
  {
    return;
  }

  std::puts ( "\nnon-zero entries [ (irow,jcol) : value ]:\n" );

  w1 = getWidthOf ( rowCount  );
  w2 = getWidthOf ( s.size(1) );
  w3 = getWidthOf ( max( abs( values ) ) );

  if ( min( values ) < 0 )
  {
    w3++;
  }

  for ( i = j = 0; i < rowCount; i++ )
  {
    n = rowOffsets[i + 1];

    for ( ; j < n; j++ )
    {
      std::printf ( "  (%*ld,%*ld) : %*d\n",
                    w1, (long) i, w2, (long) colIndices[j],
                    w3, values[j] );
    }
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( SparseMatrix<double> )
//-----------------------------------------------------------------------


static void dprint_ ( const SparseMatrix<double>& s )
{
  Array<idx_t>   rowOffsets = s.getRowOffsets    ();
  Array<idx_t>   colIndices = s.getColumnIndices ();
  Array<double>  values     = s.getValues        ();

  const idx_t    rowCount   = s.size (0);

  int            w1, w2;
  idx_t          i,  j, n;


  printInfo_ ( s.getStructure() );

  if ( s.nonZeroCount() == 0 )
  {
    return;
  }

  std::puts ( "\nnon-zero entries [ (irow,jcol) : value ]:\n" );

  w1 = getWidthOf ( rowCount  );
  w2 = getWidthOf ( s.size(1) );

  for ( i = j = 0; i < rowCount; i++ )
  {
    n = rowOffsets[i + 1];

    for ( ; j < n; j++ )
    {
      std::printf ( "  (%*ld,%*ld) : %+e\n",
                    w1, (long) i, w2, (long) colIndices[j], values[j] );
    }
  }
}


//-----------------------------------------------------------------------
//   dprintp_ ( const T* )
//-----------------------------------------------------------------------


template <class T>

  static inline void dprintp_ ( const T* t )

{
  if ( t )
  {
    dprint_ ( *t );
  }
  else
  {
    std::puts ( "null" );
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   dprint ( SparseStructure )
//-----------------------------------------------------------------------


extern void dprint ( const SparseStructure* s )
{
  dprintp_ ( s );
}


//-----------------------------------------------------------------------
//   dprint ( SparseMatrix<int> )
//-----------------------------------------------------------------------


extern void dprint ( const SparseMatrix<int>* s )
{
  dprintp_ ( s );
}


//-----------------------------------------------------------------------
//   dprint ( SparseMatrix<double> )
//-----------------------------------------------------------------------


extern void dprint ( const SparseMatrix<double>* s )
{
  dprintp_ ( s );
}
