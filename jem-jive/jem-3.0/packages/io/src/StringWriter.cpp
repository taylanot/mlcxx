
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


#include <jem/base/ClassTemplate.h>
#include <jem/io/StringWriter.h>


JEM_DEFINE_CLASS( jem::io::StringWriter );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class StringWriter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StringWriter::StringWriter ( idx_t cap )
{
  buffer_.reserve ( cap );
}


StringWriter::StringWriter ( const Self& rhs ) :

  buffer_ ( rhs.buffer_ )

{}


StringWriter::~StringWriter ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> StringWriter::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void StringWriter::write ( int c )
{
  buffer_ += ( (char) c );
}


void StringWriter::write ( int c, idx_t n )
{
  buffer_.write ( c, n );
}


void StringWriter::write ( const char* buf, idx_t n )
{
  buffer_.write ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void StringWriter::writeNoThrow ( const char* buf, idx_t n ) noexcept
{
  if ( n > 0 )
  {
    buffer_.write ( buf, n );
  }
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String StringWriter::toString () const
{
  return const_cast<Self*>( this ) -> buffer_.toString ();
}


JEM_END_PACKAGE( io )
