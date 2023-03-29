
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/util/error.h>
#include <jem/util/SparseArray.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   private SparseArray stuff
//=======================================================================

JEM_BEGIN_NAMESPACE( sparse )

//-----------------------------------------------------------------------
//   class Errors
//-----------------------------------------------------------------------


const char*  Errors::INVALID_SHAPE = "invalid SparseArray shape";
const char*  Errors::INVALID_INDEX = "invalid SparseArray index";


//-----------------------------------------------------------------------
//   getTableBits
//-----------------------------------------------------------------------


ulint getTableBits ( ulint n )
{
  ulint  i, k;

  k = 2_ulint;

  for ( i = 1; i < MAX_TABLE_BITS; i++ )
  {
    k = k << 1_ulint;

    if ( k >= n )
    {
      break;
    }
  }

  return i;
}


//-----------------------------------------------------------------------
//   largerPrime
//-----------------------------------------------------------------------


ulint largerPrime ( idx_t n )
{
  static const lint PRIMES[] =
    {
      3,          7,          13,        23,
      53,         97,         193,       389,
      769,        1543,       3079,      6151,
      12289,      24593,      49157,     98317,
      196613,     393241,     786433,    1572869,
      3145739,    6291469,    12582917,  25165843,
      50331653,   100663319,  201326611, 402653189,
      805306457,  1610612741, -1
    };

  lint  k = (lint) n;
  int   i = 0;

  while ( PRIMES[i] <= k && PRIMES[i + 1] > 0 )
  {
    i++;
  }

  return (ulint) PRIMES[i];
}


//-----------------------------------------------------------------------
//   modmul
//-----------------------------------------------------------------------


ulint modmul ( ulint i, ulint j, ulint n )
{
  JEM_ASSERT ( n < MAX_TABLE_BITS );

  static const ulint  UL1   = 1_ulint;
  static const ulint  MASK1 = (UL1 << 1_ulint) - UL1;
  static const ulint  MASK2 = (UL1 << 2_ulint) - UL1;
  static const ulint  MASK3 = (UL1 << 3_ulint) - UL1;
  static const ulint  MASK4 = (UL1 << 4_ulint) - UL1;
  static const ulint  MASK5 = (UL1 << 5_ulint) - UL1;
  static const ulint  MASK6 = (UL1 << 6_ulint) - UL1;
  static const ulint  MASK7 = (UL1 << 7_ulint) - UL1;
  static const ulint  MASK8 = (UL1 << 8_ulint) - UL1;
  static const ulint  MASK9 = (UL1 << 9_ulint) - UL1;

  const ulint  mask  = (UL1 << n) - UL1;

  ulint  k;
  ulint  r;

  i = i & mask;
  j = j & mask;

  r = i * (j & MASK1);
  k = r & mask;
  i = (i << 1_ulint) & mask;
  j =  j >> 1_ulint;

  r =  j * (i & MASK2);
  k = (k + (r & mask)) & mask;
  j = (j << 2_ulint) & mask;
  i =  i >> 2_ulint;

  r =  i * (j & MASK3);
  k = (k + (r & mask)) & mask;
  i = (i << 3_ulint) & mask;
  j =  j >> 3_ulint;

  r =  j * (i & MASK4);
  k = (k + (r & mask)) & mask;
  j = (j << 4_ulint) & mask;
  i =  i >> 4_ulint;

  r =  i * (j & MASK5);
  k = (k + (r & mask)) & mask;
  i = (i << 5_ulint) & mask;
  j =  j >> 5_ulint;

  if ( ! (i | j) )
  {
    return k;
  }

  r =  j * (i & MASK6);
  k = (k + (r & mask)) & mask;
  j = (j << 6_ulint) & mask;
  i =  i >> 6_ulint;

  r =  i * (j & MASK7);
  k = (k + (r & mask)) & mask;
  i = (i << 7_ulint) & mask;
  j =  j >> 7_ulint;

  r =  j * (i & MASK8);
  k = (k + (r & mask)) & mask;
  j = (j << 8_ulint) & mask;
  i =  i >> 8_ulint;

  r =  i * (j & MASK9);
  k = (k + (r & mask)) & mask;
  i = (i << 9_ulint) & mask;
  j =  j >> 9_ulint;

  return k;
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint              hashValue

  ( const idx_t*    i,
    int             rank )

{
  if ( rank == 0 )
  {
    return 0_lint;
  }
  else
  {
    ulint  k = (ulint) i[0] + JEM_ULINT_C(0x9e3779b9);

    for ( int j = 1; j < rank; j++ )
    {
      k ^= (ulint) i[j] + JEM_ULINT_C(0x9e3779b9) + (k << 6) + (k >> 2);
    }

    return (lint) k;
  }
}


ulint             hashValue

  ( const idx_t*    i,
    const ulint*    p,
    ulint           n,
    ulint           mask,
    int             rank )

{
  ulint  k;

  if ( rank == 0 )
  {
    k = 0_ulint;
  }
  else
  {
    k = (ulint) i[0] & mask;

    for ( int j = 0; j < rank; j++ )
    {
      k = (k + modmul( (ulint) i[j], p[j], n )) & mask;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   largerEqual
//-----------------------------------------------------------------------


bool              largerEqual

  ( const idx_t*    lhs,
    const idx_t*    rhs,
    int             rank )

{
  for ( int j = 0; j < rank; j++ )
  {
    if ( lhs[j] >= rhs[j] )
    {
      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   updateShape
//-----------------------------------------------------------------------


void              updateShape

  ( idx_t*          sh,
    const idx_t*    i,
    int             rank )

{
  for ( int j = 0; j < rank; j++ )
  {
    if ( i[j] >= sh[j] )
    {
      if ( i[j] >= maxOf<int>() )
      {
        overflowError ( JEM_FUNC, i[j] );
      }

      sh[j] = i[j] + 1;
    }
  }
}


//-----------------------------------------------------------------------
//   updateBounds
//-----------------------------------------------------------------------


void              updateBounds

  ( idx_t*          b,
    const idx_t*    sh,
    double          x,
    int             rank )

{
  double  y;

  if ( rank > 0 )
  {
    for ( int j = 0; j < rank - 1; j++ )
    {
      y = (double) sh[j] * x;

      if ( y > (0.99 * (double) maxOf( b[0] )) )
      {
        b[j] = maxOf ( b[0] );
      }
      else
      {
        b[j] = jem::max ( sh[j], (idx_t) y );
      }
    }

    b[rank - 1] = maxOf ( b[0] );
  }
}


//-----------------------------------------------------------------------
//   updateHash
//-----------------------------------------------------------------------


void              updateHash

  ( ulint*          p,
    const idx_t*    b,
    ulint           n,
    int             rank )

{
  if ( rank > 0 )
  {
    ulint  k;

    p[0] = 1_ulint;

    for ( int j = 0; j < rank - 1; j++ )
    {
      k        = largerPrime ( b[j] );
      p[j + 1] = modmul      ( p[j], k, n );
    }
  }
}


JEM_END_NAMESPACE( sparse )

JEM_END_PACKAGE( util )
