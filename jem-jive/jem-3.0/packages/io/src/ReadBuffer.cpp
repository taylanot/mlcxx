
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


#include <cctype>
#include <cstring>
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/base/MemoryError.h>
#include <jem/io/params.h>
#include <jem/io/ReadBuffer.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class ReadBuffer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int   ReadBuffer::STACK_SIZE_   = 16;
const char  ReadBuffer::CANARY_VALUE_ = 123;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ReadBuffer::ReadBuffer ( idx_t sz )
{
  if ( sz <= 0 )
  {
    sz = DEFAULT_BUFSIZE;
  }

  stack_   = STACK_SIZE_;
  bufsize_ = sz + stack_;
  buffer_  = (char*) MemCache::alloc ( (size_t) bufsize_ + 1 );
  first_   = stack_;
  last_    = first_;

  buffer_[bufsize_] = CANARY_VALUE_;
}


ReadBuffer::~ReadBuffer ()
{
  if ( buffer_ )
  {
    MemCache::dealloc ( buffer_, (size_t) bufsize_ + 1 );
  }

  buffer_ = 0;
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void ReadBuffer::reset ()
{
  first_ = last_ = stack_;
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


idx_t ReadBuffer::read ( Reader& in, char* dest, idx_t n )
{
  if ( first_ == last_ )
  {
    if ( n > bufsize_ )
    {
      return in.read ( dest, n );
    }
    else
    {
      fillBuffer_ ( in );
    }
  }

  n = min ( last_ - first_, n );

  std::memcpy ( dest, buffer_ + first_, (size_t) n );

  first_ += n;

  return n;
}


//-----------------------------------------------------------------------
//   readUntil
//-----------------------------------------------------------------------


idx_t ReadBuffer::readUntil

  ( Reader&  in,
    char*    buf,
    idx_t    n,
    int      delim )

{
  idx_t  i = 0;

  do
  {
    while ( first_ != last_ )
    {
      if ( i >= n || buffer_[first_] == delim )
      {
        return i;
      }
      else
      {
        buf[i] = buffer_[first_];
        i++; first_++;
      }
    }

    fillBuffer_ ( in );
  }
  while ( first_ != last_ );

  return i;
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void ReadBuffer::pushBack ( Reader& in )
{
  while ( last_ > first_ )
  {
    in.pushBack ( buffer_[--last_] );
  }
}


//-----------------------------------------------------------------------
//   skipUntil
//-----------------------------------------------------------------------


void ReadBuffer::skipUntil ( Reader& in, int delim )
{
  do
  {
    while ( first_ != last_ )
    {
      if ( buffer_[first_] == delim )
      {
        return;
      }
      else
      {
        first_++;
      }
    }

    fillBuffer_ ( in );
  }
  while ( first_ != last_ );
}


//-----------------------------------------------------------------------
//   skipWhite
//-----------------------------------------------------------------------


void ReadBuffer::skipWhite ( Reader& in )
{
  do
  {
    while ( first_ != last_ )
    {
      if ( ! std::isspace( buffer_[first_] ) )
      {
        return;
      }
      else
      {
        first_++;
      }
    }

    fillBuffer_ ( in );
  }
  while ( first_ != last_ );
}


//-----------------------------------------------------------------------
//   skip
//-----------------------------------------------------------------------


idx_t ReadBuffer::skip ( Reader& in, idx_t n )
{
  idx_t  k, m;

  k = 0;

  while ( n > 0 )
  {
    if ( first_ == last_ )
    {
      fillBuffer_ ( in );

      if ( first_ == last_ )
      {
        break;
      }
    }

    m       = min( n, last_ - first_ );
    first_ += m;
    k      += m;
    n      -= m;
  }

  return k;
}


//-----------------------------------------------------------------------
//   scan
//-----------------------------------------------------------------------


bool ReadBuffer::scan ( Reader& in, const char* str, idx_t n )
{
  if ( first_ + n <= last_ )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      if ( buffer_[first_ + i] != str[i] )
      {
        return false;
      }
    }

    first_ += n;
  }
  else
  {
    int    c = read ( in );
    idx_t  i = 0;

    while ( c >= 0 && i < n && c == str[i] )
    {
      c = read ( in );
      i++;
    }

    pushBack ( in, c );

    if ( i < n )
    {
      while ( i > 0 )
      {
        pushBack ( in, str[--i] );
      }

      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   read_
//-----------------------------------------------------------------------


int ReadBuffer::read_ ( Reader& in )
{
  fillBuffer_ ( in );

  if ( first_ != last_ )
  {
    return (unsigned char) buffer_[first_++];
  }
  else
  {
    return -1;
  }
}


//-----------------------------------------------------------------------
//   pushBack_
//-----------------------------------------------------------------------


void ReadBuffer::pushBack_ ( int c )
{
  if ( first_ == 0 )
  {
    growStack_ ();
  }

  buffer_[--first_] = (char) c;
}


//-----------------------------------------------------------------------
//   fillBuffer_
//-----------------------------------------------------------------------


void ReadBuffer::fillBuffer_ ( Reader& in )
{
  if ( buffer_[bufsize_] != CANARY_VALUE_ )
  {
    throw MemoryError ( JEM_FUNC );
  }

  first_ = stack_;
  last_  = stack_;
  last_ += in.read ( buffer_ + first_, bufsize_ - stack_ );
}


//-----------------------------------------------------------------------
//   growStack_
//-----------------------------------------------------------------------


void ReadBuffer::growStack_ ()
{
  char*  buffer;
  idx_t  first;
  idx_t  last;
  idx_t  delta;


  delta  = stack_ + 1;
  buffer = (char*) MemCache::alloc ( (size_t) (bufsize_ + delta + 1) );
  first  = first_ + delta;
  last   = last_  + delta;

  std::memcpy ( buffer + first, buffer_ + first_,
                (size_t) (last_ - first_) );

  MemCache::dealloc ( buffer_, (size_t) bufsize_ + 1 );

  buffer_   = buffer;
  bufsize_ += delta;
  stack_   += delta;
  first_    = first;
  last_     = last;

  buffer_[bufsize_] = CANARY_VALUE_;
}


JEM_END_PACKAGE( io )
