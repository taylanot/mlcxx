
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
#include <jem/io/BufferedReader.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class BufferedReader
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BufferedReader::BufferedReader

  ( const Ref<Reader>&  in,
    idx_t               bufsize ) :

    input_  ( in      ),
    buffer_ ( bufsize )

{
  JEM_PRECHECK ( in );
}


BufferedReader::BufferedReader ( const Self& rhs ) :

  input_  ( rhs.input_->dup()  ),
  buffer_ ( rhs.buffer_.size() )

{}


BufferedReader::~BufferedReader ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Reader> BufferedReader::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void BufferedReader::close ()
{
  input_->close ();
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t BufferedReader::poll ( const Time& timeout )
{
  return buffer_.poll ( *input_, timeout );
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


int BufferedReader::read ()
{
  return buffer_.read ( *input_ );
}


idx_t BufferedReader::read ( char* dest, idx_t n )
{
  return buffer_.read ( *input_, dest, n );
}


//-----------------------------------------------------------------------
//   readUntil
//-----------------------------------------------------------------------


idx_t BufferedReader::readUntil ( char* buf, idx_t n, int delim )
{
  return buffer_.readUntil ( *input_, buf, n, delim );
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void BufferedReader::pushBack ( int c )
{
  buffer_.pushBack ( *input_, c );
}


//-----------------------------------------------------------------------
//   skipUntil
//-----------------------------------------------------------------------


void BufferedReader::skipUntil ( int delim )
{
  buffer_.skipUntil ( *input_, delim );
}


//-----------------------------------------------------------------------
//   skipWhite
//-----------------------------------------------------------------------


void BufferedReader::skipWhite ()
{
  buffer_.skipWhite ( *input_ );
}


//-----------------------------------------------------------------------
//   skip
//-----------------------------------------------------------------------


idx_t BufferedReader::skip ( idx_t n )
{
  return buffer_.skip ( *input_, n );
}


JEM_END_PACKAGE( io )
