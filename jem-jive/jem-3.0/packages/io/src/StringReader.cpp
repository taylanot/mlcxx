
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
#include <utility>
#include <jem/base/ClassTemplate.h>
#include <jem/io/IOException.h>
#include <jem/io/StringReader.h>
#include <jem/util/Flex.h>


JEM_DEFINE_CLASS( jem::io::StringReader );


JEM_BEGIN_PACKAGE( io )


using util::Flex;


//=======================================================================
//   class StringReader
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StringReader::StringReader () : pos_( 0 )
{}


StringReader::StringReader ( const String& str ) :

  str_ ( str ),
  pos_ (   0 )

{}


StringReader::StringReader ( String&& str ) :

  str_ ( std::move( str ) ),
  pos_ ( 0 )

{}


StringReader::StringReader ( const Self& rhs ) :

  str_ ( rhs.str_ ),
  pos_ ( rhs.pos_ )

{}


StringReader::~StringReader ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Reader> StringReader::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t StringReader::poll ( const Time& timeout )
{
  return (str_.size() - pos_);
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


int StringReader::read ()
{
  if ( pos_ < str_.size() )
  {
    return (unsigned char) str_[pos_++];
  }
  else
  {
    return -1;
  }
}


idx_t StringReader::read ( char* buf, idx_t n )
{
  JEM_PRECHECK( n >= 0 );

  n = min ( n, str_.size() - pos_ );

  std::memcpy ( buf, str_.addr( pos_ ), (size_t) n );

  pos_ += n;

  return n;
}


//-----------------------------------------------------------------------
//   readUntil
//-----------------------------------------------------------------------


idx_t StringReader::readUntil ( char* buf, idx_t n, int delim )
{
  idx_t  i = 0;

  while ( i < n && pos_ < str_.size() && str_[pos_] != delim )
  {
    buf[i] = str_[pos_];

    i++;
    pos_++;
  }

  return i;
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void StringReader::pushBack ( int c )
{
  if ( c < 0 )
  {
    return;
  }

  if ( pos_ == 0 )
  {
    throw IOException ( JEM_FUNC, "failed to push back a character" );
  }

  pos_--;
}


//-----------------------------------------------------------------------
//   skipUntil
//-----------------------------------------------------------------------


void StringReader::skipUntil ( int delim )
{
  while ( pos_ < str_.size() && str_[pos_] != delim )
  {
    pos_++;
  }
}


//-----------------------------------------------------------------------
//   skipWhite
//-----------------------------------------------------------------------


void StringReader::skipWhite ()
{
  while ( pos_ < str_.size() && std::isspace( str_[pos_] ) )
  {
    pos_++;
  }
}


//-----------------------------------------------------------------------
//   skip
//-----------------------------------------------------------------------


idx_t StringReader::skip ( idx_t n )
{
  JEM_PRECHECK( n >= 0 );

  idx_t  k = min ( n, (str_.size() - pos_) );

  pos_ += k;

  return k;
}


//-----------------------------------------------------------------------
//   setString
//-----------------------------------------------------------------------


void StringReader::setString ( const String& str )
{
  pos_ = 0;
  str_ = str;
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void StringReader::reset ()
{
  pos_ = 0;
}


JEM_END_PACKAGE( io )
