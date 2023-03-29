
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


#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/io/ArrayOutputStream.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class ArrayOutputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ArrayOutputStream::ArrayOutputStream ()
{
  pos_ = 0_idx;
}


ArrayOutputStream::ArrayOutputStream

  ( const Array<byte>& buf ) :

    buf_ ( buf, 0_idx )

{
  pos_ = 0_idx;
}


ArrayOutputStream::ArrayOutputStream ( const Self& rhs ) :

  buf_ ( rhs.buf_ )

{
  pos_ = rhs.pos_;
}


ArrayOutputStream::~ArrayOutputStream ()
{
  pos_ = -666_idx;
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<OutputStream> ArrayOutputStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void ArrayOutputStream::write ( const void* buf, idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  const byte*  first = (byte*) buf;

  if ( pos_ < buf_.size() )
  {
    idx_t        k   = jem::min ( n, buf_.size() - pos_ );
    Array<byte>  buf = buf_[slice(pos_,pos_ + k)];

    if ( buf.isContiguous() )
    {
      std::memcpy ( buf.addr(), first, (size_t) k );
    }
    else
    {
      for ( idx_t i = 0; i < k; i++ )
      {
        buf[i] = first[i];
      }
    }

    pos_  += k;
    first += k;
    n     -= k;
  }

  if ( n > 0 )
  {
    const byte*  last = first + n;

    buf_.pushBack ( first, last );

    pos_ += n;
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void ArrayOutputStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  if ( n > 0_idx )
  {
    write ( buf, n );
  }
}


//-----------------------------------------------------------------------
//   setPosition
//-----------------------------------------------------------------------


void ArrayOutputStream::setPosition ( idx_t pos )
{
  JEM_PRECHECK2 ( (pos >= 0) && (pos <= size()),
                  "invalid array buffer position (out of bounds)" );
  pos_ = pos;
}


JEM_END_PACKAGE( io )
