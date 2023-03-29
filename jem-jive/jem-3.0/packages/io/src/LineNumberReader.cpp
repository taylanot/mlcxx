
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
#include <jem/io/LineNumberReader.h>


JEM_DEFINE_CLASS( jem::io::LineNumberReader );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class LineNumberReader
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


LineNumberReader::LineNumberReader

  ( const Ref<Reader>&  in,
    lint                lineno ) :

    input_ ( in )

{
  JEM_PRECHECK ( in );

  lineno_ = lineno;
}


LineNumberReader::LineNumberReader ( const Self& rhs ) :

  input_ ( rhs.input_->dup() )

{
  lineno_ = rhs.lineno_;
}


LineNumberReader::~LineNumberReader ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Reader> LineNumberReader::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void LineNumberReader::close ()
{
  input_->close ();
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t LineNumberReader::poll ( const Time& timeout )
{
  return input_->poll ( timeout );
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


int LineNumberReader::read ()
{
  int  c = input_->read();

  if ( c == '\n' )
  {
    lineno_++;
  }

  return c;
}


idx_t LineNumberReader::read ( char* buf, idx_t n )
{
  idx_t  i;

  for ( i = 0; i < n; i++ )
  {
    int  c = input_->read ();

    if      ( c == '\n' )
    {
      lineno_++;
    }
    else if ( c < 0 )
    {
      break;
    }

    buf[i] = (char) c;
  }

  return i;
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void LineNumberReader::pushBack ( int c )
{
  if ( c == '\n' )
  {
    lineno_--;
  }

  input_->pushBack ( c );
}


//-----------------------------------------------------------------------
//   get/setLineNumber
//-----------------------------------------------------------------------


lint LineNumberReader::getLineNumber () const noexcept
{
  return lineno_;
}


void LineNumberReader::setLineNumber ( lint lineno )
{
  lineno_ = lineno;
}


JEM_END_PACKAGE( io )
