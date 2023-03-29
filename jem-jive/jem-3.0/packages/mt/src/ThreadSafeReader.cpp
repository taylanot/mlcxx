
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
#include <jem/mt/ThreadSafeReader.h>


JEM_BEGIN_PACKAGE( mt )


//=======================================================================
//   ThreadSafeReader
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ThreadSafeReader::ThreadSafeReader ()
{}


ThreadSafeReader::ThreadSafeReader ( const Ref<Reader>&  in ) :

  input_ ( in )

{
  JEM_PRECHECK ( in );

  mutex_ = newInstance<IOMutex> ();
}


ThreadSafeReader::ThreadSafeReader

  ( const Ref<Reader>&   in,
    const Ref<IOMutex>&  mutex ) :

    input_ (    in ),
    mutex_ ( mutex )

{
  JEM_PRECHECK ( in != nullptr && mutex != nullptr );
}


ThreadSafeReader::ThreadSafeReader ( const Self& rhs ) :

  input_ ( rhs.input_->dup() ),
  mutex_ ( rhs.mutex_ )

{}


ThreadSafeReader::~ThreadSafeReader ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Reader> ThreadSafeReader::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   lock & unlock
//-----------------------------------------------------------------------


void ThreadSafeReader::lock ()
{
  mutex_->lock ();
}


void ThreadSafeReader::unlock ()
{
  mutex_->unlock ();
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void ThreadSafeReader::close ()
{
  Lock<IOMutex> lock ( *mutex_ );

  input_->close ();
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t ThreadSafeReader::poll ( const Time& timeout )
{
  Lock<IOMutex> lock ( *mutex_ );

  return input_->poll ( timeout );
}


//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------


int ThreadSafeReader::read ()
{
  Lock<IOMutex> lock ( *mutex_ );

  return input_->read ();
}


idx_t ThreadSafeReader::read ( char* buf, idx_t n )
{
  Lock<IOMutex> lock ( *mutex_ );

  return input_->read ( buf, n );
}


//-----------------------------------------------------------------------
//   readUntil
//-----------------------------------------------------------------------


idx_t ThreadSafeReader::readUntil ( char* buf, idx_t n, int delim )
{
  Lock<IOMutex> lock ( *mutex_ );

  return input_->readUntil ( buf, n, delim );
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void ThreadSafeReader::pushBack ( int ch )
{
  Lock<IOMutex> lock ( *mutex_ );

  input_->pushBack ( ch );
}


//-----------------------------------------------------------------------
//   skipUntil
//-----------------------------------------------------------------------


void ThreadSafeReader::skipUntil ( int delim )
{
  Lock<IOMutex> lock ( *mutex_ );

  input_->skipUntil ( delim );
}


//-----------------------------------------------------------------------
//   skipWhite
//-----------------------------------------------------------------------


void ThreadSafeReader::skipWhite ()
{
  Lock<IOMutex> lock ( *mutex_ );

  input_->skipWhite ();
}


//-----------------------------------------------------------------------
//   skip
//-----------------------------------------------------------------------


idx_t ThreadSafeReader::skip ( idx_t n )
{
  Lock<IOMutex> lock ( *mutex_ );

  return input_->skip ( n );
}


JEM_END_PACKAGE( mt )
