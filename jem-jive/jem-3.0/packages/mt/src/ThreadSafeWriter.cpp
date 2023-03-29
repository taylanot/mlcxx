
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
#include <jem/base/utilities.h>
#include <jem/mt/IOMutex.h>
#include <jem/mt/ThreadSafeWriter.h>


JEM_BEGIN_PACKAGE( mt )


//=======================================================================
//   class ThreadSafeWriter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ThreadSafeWriter::ThreadSafeWriter ()
{}


ThreadSafeWriter::ThreadSafeWriter ( const Ref<Writer>& out )
{
  JEM_PRECHECK ( out );

  output_ = out;
  mutex_  = newInstance<IOMutex> ();
}


ThreadSafeWriter::ThreadSafeWriter

  ( const Ref<Writer>&   out,
    const Ref<IOMutex>&  mutex )

{
  JEM_PRECHECK ( out != nullptr && mutex != nullptr );

  output_ = out;
  mutex_  = mutex;
}


ThreadSafeWriter::ThreadSafeWriter ( const Self& rhs ) :

  output_ ( rhs.output_->dup() ),
  mutex_  ( rhs.mutex_ )

{}


ThreadSafeWriter::~ThreadSafeWriter ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> ThreadSafeWriter::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   lock & unlock
//-----------------------------------------------------------------------


void ThreadSafeWriter::lock ()
{
  mutex_->lock ();
}


void ThreadSafeWriter::unlock ()
{
  mutex_->unlock ();
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void ThreadSafeWriter::close ()
{
  Lock<IOMutex> lock ( *mutex_ );

  output_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void ThreadSafeWriter::flush ()
{
  Lock<IOMutex> lock ( *mutex_ );

  output_->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void ThreadSafeWriter::write ( int c )
{
  Lock<IOMutex> lock ( *mutex_ );

  output_->write ( c );
}


void ThreadSafeWriter::write ( int c, idx_t n )
{
  Lock<IOMutex> lock ( *mutex_ );

  output_->write ( c, n );
}


void ThreadSafeWriter::write ( const char* buf, idx_t n )
{
  Lock<IOMutex> lock ( *mutex_ );

  output_->write ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void ThreadSafeWriter::writeNoThrow

  ( const char*  buf,
    idx_t        n ) noexcept

{
  Lock<IOMutex> lock ( *mutex_ );

  output_->writeNoThrow ( buf, n );
}


JEM_END_PACKAGE( mt )
