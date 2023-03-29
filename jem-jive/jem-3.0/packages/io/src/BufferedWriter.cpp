
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
#include <jem/io/BufferedWriter.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class BufferedWriter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BufferedWriter::BufferedWriter

  ( const Ref<Writer>&  out,
    idx_t               bufsize ) :

    output_ (     out ),
    buffer_ ( bufsize )

{
  JEM_PRECHECK ( out );
}


BufferedWriter::BufferedWriter ( const Self& rhs ) :

  output_ ( rhs.output_->dup() ),
  buffer_ ( rhs.buffer_.size() )

{}


BufferedWriter::~BufferedWriter ()
{
  buffer_.flushNoThrow ( *output_ );
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> BufferedWriter::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void BufferedWriter::close ()
{
  buffer_ .flush ( *output_ );
  output_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void BufferedWriter::flush ()
{
  buffer_ .flush ( *output_ );
  output_->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void BufferedWriter::write ( int c )
{
  buffer_.write ( *output_, c );
}


void BufferedWriter::write ( int c, idx_t n )
{
  buffer_.write ( *output_, c, n );
}


void BufferedWriter::write ( const char* buf, idx_t n )
{
  buffer_.write ( *output_, buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void BufferedWriter::writeNoThrow ( const char* buf, idx_t n ) noexcept
{
  buffer_.writeNoThrow ( *output_, buf, n );
}


//-----------------------------------------------------------------------
//   flushBuffer
//-----------------------------------------------------------------------


void BufferedWriter::flushBuffer ()
{
  buffer_.flush ( *output_ );
}


JEM_END_PACKAGE( io )
