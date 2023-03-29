
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
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/io/TextInput.h>
#include <jem/io/TextOutput.h>
#include <jem/base/Integer.h>
#include <jem/base/Float.h>
#include <jem/base/Character.h>
#include <jem/base/StringBuffer.h>


JEM_BEGIN_PACKAGE_BASE


using jem::io::DataInput;
using jem::io::DataOutput;
using jem::io::TextInput;
using jem::io::TextOutput;


//=======================================================================
//   class StringBuffer
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


StringBuffer::StringBuffer () noexcept
{
  size_  = 0;
  owner_ = true;
}


StringBuffer::StringBuffer ( idx_t size ) :

  str_ ( NIL, size )

{
  size_  = size;
  owner_ = true;
}


StringBuffer::StringBuffer ( const String& str ) :

  str_ ( str )

{
  size_  = str.size_;
  owner_ = false;
}


StringBuffer::StringBuffer ( const Self& rhs ) :

  str_ ( rhs.str_ )

{
  size_  = rhs.size_;
  owner_ = rhs.owner_;

  if ( owner_ )
  {
    clone_ ();
  }
}


StringBuffer::StringBuffer ( Self&& rhs ) noexcept
{
  size_  = 0;
  owner_ = true;

  swap ( rhs );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void StringBuffer::write ( int c )
{
  char*  buf = expand ( 1_idx );

  buf[0] = (char) c;
}


void StringBuffer::write

  ( int    c,
    idx_t  n )

{
  JEM_PRECHECK2 ( n >= 0, "invalid repeat count" );

  char*  buf = expand ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    buf[i] = (char) c;
  }
}


void StringBuffer::write

  ( const char*  str,
    idx_t        len )

{
  JEM_PRECHECK2 ( len >= 0, "invalid character array length" );

  // Check for overlapping memory. This will be the case when (part
  // of) a StringBuffer if written to itself.

  if ( str_.str_ <= str && str < (str_.str_ + str_.size_) )
  {
    // Memory overlaps, so make a temporary copy.

    operator += ( String( str, len ) );
  }
  else
  {
    std::memcpy ( expand( len ), str, (size_t) len );
  }
}


//-----------------------------------------------------------------------
//   printByte
//-----------------------------------------------------------------------


void StringBuffer::printByte ( byte b )
{
  const char*  digits = Character::LOWERCASE_DIGITS;

  const byte   MASK   = 15;

  char*  buf = expand ( 4_idx );

  buf[0] = '0';
  buf[1] = 'x';
  buf[2] = digits[(b >> 4) & MASK];
  buf[3] = digits[b & MASK];
}


//-----------------------------------------------------------------------
//   printBool
//-----------------------------------------------------------------------


void StringBuffer::printBool ( bool b )
{
  idx_t  len = b ? 4_idx : 5_idx;
  char*  buf = expand ( len );

  if ( b )
  {
    buf[0] = 't';
    buf[1] = 'r';
    buf[2] = 'u';
    buf[3] = 'e';
  }
  else
  {
    buf[0] = 'f';
    buf[1] = 'a';
    buf[2] = 'l';
    buf[3] = 's';
    buf[4] = 'e';
  }
}


//-----------------------------------------------------------------------
//   printInt
//-----------------------------------------------------------------------


void StringBuffer::printInt ( lint i )
{
  idx_t  len = (idx_t) (sizeof(i) * 4_idx);
  char*  buf = xalloc ( len );

  len    = Integer::print ( buf, len, i );
  size_ += len;
}


//-----------------------------------------------------------------------
//   printFloat
//-----------------------------------------------------------------------


void StringBuffer::printFloat ( double x )
{
  idx_t  len = 64_idx;
  char*  buf = xalloc ( len );

  len    = Float::print ( buf, len, x );
  size_ += len;
}


//-----------------------------------------------------------------------
//   operator =
//-----------------------------------------------------------------------


StringBuffer& StringBuffer::operator = ( const Self& rhs )
{
  if ( &rhs != this )
  {
    if ( ! owner_ || rhs.size_ > str_.size_ )
    {
      realloc_ ( rhs.size_ );
    }

    std::memcpy ( str_.str_, rhs.str_.cstr_, (size_t) rhs.size_ );

    size_ = rhs.size_;
  }

  return *this;
}


StringBuffer& StringBuffer::operator = ( Self&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


StringBuffer& StringBuffer::operator = ( const String& rhs )
{
  if ( owner_ )
  {
    if ( rhs.size_ > str_.size_ )
    {
      realloc_ ( rhs.size_ );
    }

    std::memcpy ( str_.str_, rhs.cstr_, (size_t) rhs.size_ );
  }
  else
  {
    str_ = rhs;
  }

  size_ = rhs.size_;

  return *this;
}


//-----------------------------------------------------------------------
//   operator +=
//-----------------------------------------------------------------------


StringBuffer& StringBuffer::operator += ( const String& rhs )
{
  std::memcpy ( expand( rhs.size_ ), rhs.cstr_, (size_t) rhs.size_ );

  return *this;
}


StringBuffer& StringBuffer::operator += ( char rhs )
{
  char*  buf = expand ( 1_idx );

  buf[0] = rhs;

  return *this;
}


//-----------------------------------------------------------------------
//   slice operator
//-----------------------------------------------------------------------


String StringBuffer::operator [] ( const Slice&  s ) const
{
  JEM_ASSERT2 ( s.first() <= s.last( size() ),
                "invalid StringBuffer slice" );

  idx_t  first  = s.first  ();
  idx_t  last   = s.last   ( size_ );
  idx_t  stride = s.stride ();

  idx_t  len    = last - first;

  if ( len > 0_idx && stride > 1_idx )
  {
    len = 1_idx + (len - 1_idx) / stride;
  }

  String  tmp ( NIL, len );

  for ( idx_t i = 0; i < len; i++, first += stride )
  {
    tmp.str_[i] = str_.cstr_[first];
  }

  return tmp;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void StringBuffer::swap ( StringBuffer& rhs ) noexcept
{
  str_.swap ( rhs.str_ );
  jem::swap ( size_,  rhs.size_ );
  jem::swap ( owner_, rhs.owner_ );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void StringBuffer::clear ()
{
  size_ = 0_idx;

  if ( ! owner_ )
  {
    str_   = String ();
    owner_ = true;
  }
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void StringBuffer::resize ( idx_t size )
{
  JEM_PRECHECK2 ( size >= 0, "invalid StringBuffer size" );

  if ( size != this->size_ )
  {
    if ( size > str_.size_ )
    {
      realloc_ ( size );
    }

    this->size_ = size;
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void StringBuffer::reserve ( idx_t cap )
{
  if ( cap > str_.size_ )
  {
    realloc_ ( cap );
  }
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String StringBuffer::toString ()
{
  if ( size_ != str_.size_ )
  {
    realloc_ ( size_ );
  }

  owner_ = false;

  return str_;
}


//-----------------------------------------------------------------------
//   clone_
//-----------------------------------------------------------------------


void StringBuffer::clone_ ()
{
  String  tmp ( NIL, str_.size_ );

  std::memcpy ( tmp.str_, str_.cstr_, (size_t) size_ );
  str_.swap   ( tmp );

  owner_ = true;
}


//-----------------------------------------------------------------------
//   xalloc_
//-----------------------------------------------------------------------


char* StringBuffer::xalloc_ ( idx_t len )
{
  idx_t  cap = len + size_ + size_ / 2_idx;

  if ( owner_ )
  {
    str_.realloc_ ( cap );
  }
  else
  {
    String  tmp ( NIL, cap );

    std::memcpy ( tmp.str_, str_.cstr_, (size_t) size_ );
    str_.swap   ( tmp );

    owner_ = true;
  }

  return str_.str_ + size_;
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


void StringBuffer::realloc_ ( idx_t cap )
{
  if ( cap < size_ )
  {
    size_ = cap;
  }

  if ( owner_ )
  {
    str_.realloc_ ( cap );
  }
  else
  {
    String  tmp ( NIL, cap );

    std::memcpy ( tmp.str_, str_.cstr_, (size_t) size_ );
    str_.swap   ( tmp );

    owner_ = true;
  }
}


//=======================================================================
//   Related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


void                    decode

  ( DataInput&            in,
    StringBuffer&         sb )

{
  uidx_t  sz;

  decode ( in, sz );

  sb.resize ( (idx_t) sz );
  in.decode ( sb.addr(), (idx_t) sz );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


void                    encode

  ( DataOutput&           out,
    const StringBuffer&   sb )

{
  out.encode ( sb.size() );
  out.encode ( sb.addr(), sb.size() );
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


void                    read

  ( TextInput&            in,
    StringBuffer&         sb )

{
  int  c;

  in.skipWhite ();
  c = in.read  ();

  while ( c >= 0 && ! std::isspace( c ) )
  {
    sb += (char) c;
    c   = in.read ();
  }

  in.pushBack ( c );
}


JEM_END_PACKAGE_BASE
