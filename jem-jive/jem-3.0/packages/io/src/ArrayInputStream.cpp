
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


#include <utility>
#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/io/EOFException.h>
#include <jem/io/ArrayInputStream.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class ArrayInputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ArrayInputStream::ArrayInputStream ( const Array<byte>& buf ) :

  buf_ ( buf ),
  pos_ ( 0_idx )

{}


ArrayInputStream::ArrayInputStream ( Array<byte>&& buf ) :

  buf_ ( std::move( buf ) ),
  pos_ ( 0_idx )

{}


ArrayInputStream::ArrayInputStream ( const Self& rhs ) :

  buf_ ( rhs.buf_.clone() )

{
  pos_ = rhs.pos_;
}


ArrayInputStream::~ArrayInputStream ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<InputStream> ArrayInputStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t ArrayInputStream::poll ( const Time& timeout )
{
  return max ( 0_idx, buf_.size() - pos_ );
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t ArrayInputStream::read ( void* buf, idx_t n )
{
  JEM_PRECHECK( n >= 0 );

  if ( pos_ >= buf_.size() )
  {
    return 0_idx;
  }

  n = min ( n, buf_.size() - pos_ );

  if ( buf_.isContiguous() )
  {
    std::memcpy ( buf, buf_.addr( pos_ ), (size_t) n );
  }
  else
  {
    copy ( (byte*) buf, buf_.begin() + pos_,
                        buf_.begin() + (pos_ + n) );
  }

  pos_ += n;

  return n;
}


//-----------------------------------------------------------------------
//   readAll
//-----------------------------------------------------------------------


void ArrayInputStream::readAll ( void* buf, idx_t n )
{
  JEM_PRECHECK( n >= 0 );

  if ( (pos_ + n) > buf_.size() )
  {
    throw EOFException (
      JEM_FUNC,
      String::format (
        "unexpected end of input; expected %d more bytes",
        (pos_ + n) - buf_.size ()
      )
    );
  }

  if ( buf_.isContiguous() )
  {
    std::memcpy ( buf, buf_.addr( pos_ ), (size_t) n );
  }
  else
  {
    copy ( (byte*) buf, buf_.begin() + pos_,
                        buf_.begin() + (pos_ + n) );
  }

  pos_ += n;
}


//-----------------------------------------------------------------------
//   skip
//-----------------------------------------------------------------------


idx_t ArrayInputStream::skip ( idx_t n )
{
  if ( pos_ < buf_.size() )
  {
    n     = min ( n, buf_.size() - pos_ );
    pos_ += n;

    return n;
  }
  else
  {
    return 0_idx;
  }
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void ArrayInputStream::reset ()
{
  pos_ = 0_idx;
}


void ArrayInputStream::reset ( const Array<byte>& buf )
{
  buf_.ref ( buf );

  pos_ = 0_idx;
}


void ArrayInputStream::reset ( Array<byte>&& buf )
{
  buf_.ref ( std::move( buf ) );

  pos_ = 0_idx;
}


JEM_END_PACKAGE( io )
