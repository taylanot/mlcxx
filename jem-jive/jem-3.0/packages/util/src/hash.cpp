
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


#include <jem/util/hash.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   private stuff
//=======================================================================

//-----------------------------------------------------------------------
//   typedef ub4
//-----------------------------------------------------------------------


template <bool Expr>

  class HashInt

{
 public:

  typedef ulint          UInt;

};


template <>

  class HashInt<true>

{
 public:

  typedef unsigned int   UInt;

};


typedef HashInt<(sizeof(int) >= 4)>::UInt  ub4;


//-----------------------------------------------------------------------
//   hashMix
//-----------------------------------------------------------------------


inline void  hashMix ( ub4& a, ub4& b, ub4& c ) noexcept
{
  a -= b; a -= c; a ^= (c >> 13);
  b -= c; b -= a; b ^= (a <<  8);
  c -= a; c -= b; c ^= (b >> 13);
  a -= b; a -= c; a ^= (c >> 12);
  b -= c; b -= a; b ^= (a << 16);
  c -= a; c -= b; c ^= (b >>  5);
  a -= b; a -= c; a ^= (c >>  3);
  b -= c; b -= a; b ^= (a << 10);
  c -= a; c -= b; c ^= (b >> 15);
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   hash
//-----------------------------------------------------------------------


lint hash ( const void* buf, idx_t len, lint ini ) noexcept
{
  const byte* JEM_RESTRICT  key = (const byte*) buf;

  ub4    a, b, c;
  idx_t  n;


  // Set up the internal state

  n = len;
  a = (ub4) 0x9e3779b9;
  b = a;
  c = (ub4) ini;

  while ( n >= 12 )
  {
    a += ( key[0] + (((ub4) key[ 1] <<  8) +
                     ((ub4) key[ 2] << 16) +
                     ((ub4) key[ 3] << 24)) );

    b += ( key[4] + (((ub4) key[ 5] <<  8) +
                     ((ub4) key[ 6] << 16) +
                     ((ub4) key[ 7] << 24)) );

    c += ( key[8] + (((ub4) key[ 9] <<  8) +
                     ((ub4) key[10] << 16) +
                     ((ub4) key[11] << 24)) );

    hashMix ( a, b, c );

    key += 12;
    n   -= 12;
  }

  // Handle the last 11 bytes. Note that all case statements fall
  // through.

  c = (ub4) ((idx_t) c + len);

  switch ( n )
  {
  case 11: c += ((ub4) key[10] << 24);
  case 10: c += ((ub4) key[ 9] << 16);
  case  9: c += ((ub4) key[ 8] <<  8);

    // The first byte of c is reserved for the length

  case  8: b += ((ub4) key[ 7] << 24);
  case  7: b += ((ub4) key[ 6] << 16);
  case  6: b += ((ub4) key[ 5] <<  8);
  case  5: b +=        key[ 4];
  case  4: a += ((ub4) key[ 3] << 24);
  case  3: a += ((ub4) key[ 2] << 16);
  case  2: a += ((ub4) key[ 1] <<  8);
  case  1: a +=        key[ 0];

    // Case 0: nothing left to add

  }

  hashMix ( a, b, c );

  return (lint) c;
}


//-----------------------------------------------------------------------
//   hashBits
//-----------------------------------------------------------------------


int hashBits ( idx_t count ) noexcept
{
  int  n = 2;

  count = count / 2;

  while ( hashSize(n) < count )
  {
    n++;
  }

  return n;
}


JEM_END_PACKAGE( util )
