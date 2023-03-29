
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
#include <jem/mt/ThreadSafeOutputStream.h>


JEM_BEGIN_PACKAGE( mt )


//=======================================================================
//   ThreadSafeOutputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ThreadSafeOutputStream::ThreadSafeOutputStream ()
{}


ThreadSafeOutputStream::ThreadSafeOutputStream

  ( Ref<OutputStream> out ) :

    output_ ( out )

{
  JEM_PRECHECK ( out );

  mutex_ = newInstance<IOMutex> ();
}


ThreadSafeOutputStream::ThreadSafeOutputStream

  ( Ref<OutputStream>    out,
    const Ref<IOMutex>&  mutex ) :

    output_ (   out ),
    mutex_  ( mutex )

{
  JEM_PRECHECK ( out && mutex );
}


ThreadSafeOutputStream::ThreadSafeOutputStream ( const Self& rhs ) :

  output_ ( rhs.output_->dup() ),
  mutex_  ( rhs.mutex_ )

{}


ThreadSafeOutputStream::~ThreadSafeOutputStream ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<OutputStream> ThreadSafeOutputStream::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   lock & unlock
//-----------------------------------------------------------------------


void ThreadSafeOutputStream::lock ()
{
  mutex_->lock ();
}


void ThreadSafeOutputStream::unlock ()
{
  mutex_->unlock ();
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void ThreadSafeOutputStream::close ()
{
  Lock<IOMutex> lock ( *mutex_ );

  output_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void ThreadSafeOutputStream::flush ()
{
  Lock<IOMutex> lock ( *mutex_ );

  output_->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void ThreadSafeOutputStream::write

  ( const void*  buf,
    idx_t        n )

{
  Lock<IOMutex> lock ( *mutex_ );

  output_->write ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void ThreadSafeOutputStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  Lock<IOMutex> lock ( *mutex_ );

  output_->writeNoThrow ( buf, n );
}


JEM_END_PACKAGE( mt )
