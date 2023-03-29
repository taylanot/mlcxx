
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


#include <climits>
#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/io/TextOutput.h>
#include <jem/util/error.h>
#include <jem/util/BitSet.h>


#if ( CHAR_BIT != 8 )
  #error "Number of bits per byte must be eight -- sorry"
#endif


JEM_BEGIN_PACKAGE( util )


using jem::io::DataInput;
using jem::io::DataOutput;
using jem::io::TextOutput;


//=======================================================================
//   class BitSet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int    BitSet::BITS_PER_BYTE_ = 8;
const idx_t  BitSet::SHIFT_BITS_    = 3;
const idx_t  BitSet::BYTE_MASK_     = 7;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BitSet::BitSet () noexcept
{
  size_     = 0;
  capacity_ = 0;
  data_     = nullptr;
}


BitSet::BitSet ( idx_t cap )
{
  JEM_PRECHECK2 ( cap >= 0, "invalid capacity" );

  const idx_t  n = byteCount_ ( cap );

  size_     = 0;
  capacity_ = cap;
  data_     = (byte*) MemCache::alloc ( (size_t) n );

  std::memset ( data_, 0, (size_t) n );
}


BitSet::BitSet ( const BitSet& rhs )
{
  const idx_t  n = byteCount_ ( rhs.size_ );

  size_     = rhs.size_;
  capacity_ = size_;
  data_     = (byte*) MemCache::alloc ( (size_t) n );

  std::memcpy ( data_, rhs.data_, (size_t) n );
}


BitSet::BitSet ( BitSet&& rhs ) noexcept
{
  size_         = rhs.size_;
  capacity_     = rhs.capacity_;
  data_         = rhs.data_;
  rhs.size_     = 0;
  rhs.capacity_ = 0;
  rhs.data_     = nullptr;
}


BitSet::~BitSet ()
{
  const idx_t  n = byteCount_ ( capacity_ );

  MemCache::dealloc ( data_, (size_t) n );

  data_ = nullptr;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void BitSet::readFrom ( DataInput& in )
{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  clear   ();
  reserve ( n );

  size_ = n;
  n     = byteCount_ ( size_ );

  in.decode ( data_, n );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void BitSet::writeTo ( DataOutput& out ) const
{
  encode     ( out, size_ );
  out.encode ( data_, byteCount_( size_ ) );
}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


BitSet& BitSet::operator = ( const BitSet& rhs )
{
  if ( this != & rhs )
  {
    if ( rhs.size_ > capacity_ )
    {
      realloc_ ( rhs.size_ );
    }

    idx_t  m = byteCount_ ( rhs.size_ );

    std::memcpy ( data_, rhs.data_, (size_t) m );

    if ( rhs.size_ < size_ )
    {
      idx_t  n = byteCount_ ( size_ );

      std::memset ( data_ + m, 0, (size_t) (n - m) );
    }

    size_ = rhs.size_;
  }

  return *this;
}


BitSet& BitSet::operator = ( BitSet&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void BitSet::swap ( BitSet& rhs ) noexcept
{
  jem::swap ( size_,     rhs.size_     );
  jem::swap ( capacity_, rhs.capacity_ );
  jem::swap ( data_,     rhs.data_     );
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


void BitSet::set ( idx_t i )
{
  if ( i < 0 )
  {
    indexError_ ( JEM_FUNC, i );
  }

  if ( i >= size_ )
  {
    if ( i >= capacity_ )
    {
      expand_ ( i + 1 );
    }

    size_ = i + 1;
  }

  idx_t  j = i >> SHIFT_BITS_;
  idx_t  k = i &  BYTE_MASK_;

  data_[j] = (byte) (data_[j] | (1 << (int) k));
}


void BitSet::set ( idx_t i, bool value )
{
  if ( value )
  {
    set   ( i );
  }
  else
  {
    clear ( i );
  }
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


bool BitSet::get ( idx_t i ) const
{
  if ( i >= size_ )
  {
    return false;
  }
  else
  {
    if ( i < 0 )
    {
      indexError_ ( JEM_FUNC, i );
    }

    idx_t  j = i >> SHIFT_BITS_;
    idx_t  k = i &  BYTE_MASK_;

    return ((data_[j] & (1 << (int) k)) != 0);
  }
}


//-----------------------------------------------------------------------
//   flip
//-----------------------------------------------------------------------


void BitSet::flip ( idx_t i )
{
  if ( i >= size_ )
  {
    set ( i );
  }
  else
  {
    if ( i < 0 )
    {
      indexError_ ( JEM_FUNC, i );
    }

    idx_t  j = i >> SHIFT_BITS_;
    idx_t  k = i &  BYTE_MASK_;
    int    m = 1 << (int) k;

    if ( data_[j] & m )
    {
      data_[j] = (byte) (data_[j] & (~m));
    }
    else
    {
      data_[j] = (byte) (data_[j] | m);
    }
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void BitSet::clear ()
{
  const idx_t  n = byteCount_ ( size_ );

  std::memset ( data_, 0, (size_t) n );

  size_ = 0;
}


void BitSet::clear ( idx_t i )
{
  if ( i < 0 )
  {
    indexError_ ( JEM_FUNC, i );
  }

  if ( i < size_ )
  {
    idx_t  j = i >> SHIFT_BITS_;
    idx_t  k = i &  BYTE_MASK_;

    data_[j] = (byte) (data_[j] & (~(1 << (int) k)));

    if ( i + 1 == size_ )
    {
      updateSize_ ( j, k );
    }
  }
}


void BitSet::clear ( const BitSet& mask )
{
  idx_t  n = byteCount_ ( min( size_, mask.size_ ) );

  for ( idx_t i = 0; i < n; i++ )
  {
    data_[i] = (byte) (data_[i] & (~mask.data_[i]));
  }

  if ( mask.size_ >= size_ )
  {
    updateSize_ ();
  }
}


//-----------------------------------------------------------------------
//   andWith
//-----------------------------------------------------------------------


void BitSet::andWith ( const BitSet& rhs )
{
  idx_t  n = byteCount_ ( size_ );

  if ( size_ > rhs.size_ )
  {
    idx_t  m = byteCount_ ( rhs.size_ );

    std::memset ( data_ + m, 0, (size_t) (n - m) );

    size_ = rhs.size_;
    n     = m;
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    data_[i] = (byte) (data_[i] & rhs.data_[i]);
  }

  updateSize_ ();
}


//-----------------------------------------------------------------------
//   orWith
//-----------------------------------------------------------------------


void BitSet::orWith ( const BitSet& rhs )
{
  idx_t  n = byteCount_ ( rhs.size_ );

  if ( rhs.size_ > size_ )
  {
    idx_t  m = byteCount_ ( size_ );

    if ( rhs.size_ > capacity_ )
    {
      realloc_ ( rhs.size_ );
    }

    std::memcpy ( data_ + m, rhs.data_ + m, (size_t) (n - m) );

    size_ = rhs.size_;
    n     = m;
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    data_[i] = (byte) (data_[i] | rhs.data_[i]);
  }
}


//-----------------------------------------------------------------------
//   xorWith
//-----------------------------------------------------------------------


void BitSet::xorWith ( const BitSet& rhs )
{
  idx_t  n = byteCount_ ( rhs.size_ );

  if ( rhs.size_ > size_ )
  {
    idx_t  m = byteCount_ ( size_ );

    if ( rhs.size_ > capacity_ )
    {
      realloc_ ( rhs.size_ );
    }

    for ( idx_t i = 0; i < m; i++ )
    {
      data_[i] = (byte) (data_[i] ^ rhs.data_[i]);
    }

    std::memcpy ( data_ + m, rhs.data_ + m, (size_t) (n - m) );

    size_ = rhs.size_;
  }

  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      data_[i] = (byte) (data_[i] ^ rhs.data_[i]);
    }

    if ( size_ == rhs.size_ )
    {
      updateSize_ ();
    }
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void BitSet::reserve ( idx_t n )
{
  JEM_PRECHECK2 ( n >= 0, "invalid reserve count" );

  if ( n < size_ )
  {
    n = size_;
  }

  if ( n != capacity_ )
  {
    realloc_ ( n );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void BitSet::trimToSize ()
{
  if ( capacity_ > size_ )
  {
    realloc_ ( size_ );
  }
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


idx_t BitSet::count () const noexcept
{
  const idx_t  k = size_ >> SHIFT_BITS_;
  const idx_t  m = size_ &  BYTE_MASK_;

  idx_t        n = 0;

  for ( idx_t i = 0; i < k; i++ )
  {
    int  b = data_[i];

    if ( b )
    {
      for ( int j = 0; j < BITS_PER_BYTE_; j++ )
      {
        if ( b & 1 )
        {
          n++;
        }

        b = b >> 1;
      }
    }
  }

  if ( m > 0 && data_[k] )
  {
    int  b = data_[k];

    for ( idx_t j = 0; j < m; j++ )
    {
      if ( b & 1 )
      {
        n++;
      }

      b = b >> 1;
    }
  }

  return n;
}


//-----------------------------------------------------------------------
//   expand_
//-----------------------------------------------------------------------


void BitSet::expand_ ( idx_t cap )
{
  idx_t  n = (idx_t) ((double) maxOf( capacity_ ) / 1.501) - 4;

  if ( capacity_ >= n )
  {
    n = maxOf ( capacity_ );
  }
  else
  {
    n = jem::max ( (idx_t) (1.5 * (double) capacity_) + 4, cap );
  }

  if ( cap < 0 || n < cap )
  {
    overflowError ( JEM_FUNC, capacity_ );
  }

  realloc_ ( n );
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


void BitSet::realloc_ ( idx_t cap )
{
  if ( cap < size_ )
  {
    cap = size_;
  }

  idx_t  oldLen = byteCount_ ( capacity_ );
  idx_t  newLen = byteCount_ ( cap );

  data_ = (byte*) MemCache::realloc ( data_,
                                      (size_t) oldLen,
                                      (size_t) newLen );

  if ( newLen > oldLen )
  {
    std::memset ( data_ + oldLen, 0, (size_t) (newLen - oldLen) );
  }

  capacity_ = cap;
}


//-----------------------------------------------------------------------
//   updateSize_
//-----------------------------------------------------------------------


void BitSet::updateSize_ ()
{
  if ( size_ > 0 )
  {
    const idx_t  last = size_ - 1;

    updateSize_ ( last >> SHIFT_BITS_, last & BYTE_MASK_ );
  }
}


void BitSet::updateSize_ ( idx_t i, idx_t j )
{
  byte  b = data_[i];

  if ( b )
  {
    int  mask = 1 << j;

    while ( mask )
    {
      if ( b & mask )
      {
        return;
      }

      size_--;

      mask = mask >> 1;
    }
  }
  else
  {
    size_ -= j;
    size_--;
  }

  for ( i--; i >= 0; i-- )
  {
    b = data_[i];

    if ( b )
    {
      int  mask = 128;

      while ( mask )
      {
        if ( b & mask )
        {
          return;
        }

        size_--;

        mask = mask >> 1;
      }
    }
    else
    {
      size_ -= BITS_PER_BYTE_;
    }
  }
}


//-----------------------------------------------------------------------
//   byteCount_
//-----------------------------------------------------------------------


inline idx_t BitSet::byteCount_ ( idx_t nbits )
{
  return (((nbits + 15) >> SHIFT_BITS_) - 1);
}


//-----------------------------------------------------------------------
//   indexError_
//-----------------------------------------------------------------------


void BitSet::indexError_

  ( const char*  where,
    idx_t        index )

{
  throw IllegalIndexException (
    where,
    String::format ( "invalid BitSet index: %d", index )
  );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


void              print

  ( TextOutput&     out,
    const BitSet&   bs )

{
  for ( idx_t i = 0; i < bs.size(); i++ )
  {
    if ( bs.get(i) )
    {
      print ( out, '1' );
    }
    else
    {
      print ( out, '0' );
    }
  }
}


JEM_END_PACKAGE( util )


//=======================================================================
//   class TypeTraits < util::BitSet >
//=======================================================================


JEM_BEGIN_PACKAGE_BASE


String TypeTraits < util::BitSet >::whatis ()
{
  return "a bit set";
}


String TypeTraits < util::BitSet >::typeName ()
{
  return "jem::util::BitSet";
}


JEM_END_PACKAGE_BASE
