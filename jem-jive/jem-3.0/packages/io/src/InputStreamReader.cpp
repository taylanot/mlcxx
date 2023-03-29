
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
#include <jem/base/ClassTemplate.h>
#include <jem/io/IOException.h>
#include <jem/io/InputStreamReader.h>


JEM_DEFINE_CLASS( jem::io::InputStreamReader );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class InputStreamReader
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


InputStreamReader::InputStreamReader ()
{}


InputStreamReader::InputStreamReader ( Ref<InputStream> in )
{
  JEM_PRECHECK ( in );

  input_ = in;
}


InputStreamReader::InputStreamReader ( const Self& rhs ) :

  input_ ( rhs.input_->dup() )

{}


InputStreamReader::~InputStreamReader ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Reader> InputStreamReader::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void InputStreamReader::close ()
{
  input_->close ();
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t InputStreamReader::poll ( const Time& timeout )
{
  if ( stack_.size() )
  {
    return stack_.size ();
  }
  else
  {
    return input_->poll ( timeout );
  }
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


int InputStreamReader::read ()
{
  if ( stack_.size() > 0 )
  {
    char  tmp = stack_.back ();

    stack_.popBack ();

    return tmp;
  }
  else
  {
    byte   b;
    idx_t  n;

    n = input_->read ( &b, 1 );

    if ( n != 1 )
    {
      return -1;
    }
    else
    {
      return b;
    }
  }
}


idx_t InputStreamReader::read ( char* buf, idx_t n )
{
  JEM_PRECHECK( n >= 0 );

  idx_t  k = 0;

  if ( stack_.size() > 0 )
  {
    while ( stack_.size() > 0 && k < n )
    {
      buf[k++] = stack_.back ();

      stack_.popBack ();
    }

    buf += k;
    n   -= k;
  }

  k += input_->read ( buf, n );

  return k;
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void InputStreamReader::pushBack ( int c )
{
  if ( c >= 0 )
  {
    stack_.pushBack( (char) c );
  }
}


JEM_END_PACKAGE( io )
