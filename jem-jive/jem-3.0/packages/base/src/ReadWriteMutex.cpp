
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
#include <jem/base/thread/ThreadLib.h>
#include <jem/base/thread/ThreadQueue.h>
#include <jem/base/ReadWriteMutex.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class ReadWriteMutex
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  ReadWriteMutex::MAX_WRITE_COUNT_ = 10;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ReadWriteMutex::ReadWriteMutex ()
{
  ThreadLib::init ();

  readQueue_  = nullptr;
  writeQueue_ = nullptr;
  readCount_  = 0;
  writeCount_ = 0;
}


ReadWriteMutex::~ReadWriteMutex ()
{
  readQueue_  = nullptr;
  writeQueue_ = nullptr;
}


//-----------------------------------------------------------------------
//   tryLockRead
//-----------------------------------------------------------------------


bool ReadWriteMutex::tryLockRead () noexcept
{
  bool  result = false;


  spinlock_.lock ();

  if ( writeCount_ < MAX_WRITE_COUNT_ )
  {
    if ( writeQueue_ )
    {
      writeCount_++;
    }

    readCount_++;
    result = true;
  }

  spinlock_.unlock ();

  return result;
}


//-----------------------------------------------------------------------
//   lockRead
//-----------------------------------------------------------------------


void ReadWriteMutex::lockRead ()
{
  spinlock_.lock ();

  if ( writeCount_ < MAX_WRITE_COUNT_ )
  {
    readCount_++;

    if ( writeQueue_ )
    {
      writeCount_++;
    }
  }
  else
  {
    ThreadNode  self;

    ThreadQueue::push ( readQueue_, &self );
    self.waitNoCancel ( spinlock_ );

    if ( writeQueue_ )
    {
      writeCount_++;
    }

    if ( readQueue_ && writeCount_ < MAX_WRITE_COUNT_ )
    {
      ThreadNode*  node = ThreadQueue::pop ( readQueue_ );

      readCount_++;

      node->notify ();
    }
  }

  spinlock_.unlock ();
}


//-----------------------------------------------------------------------
//   unlockRead
//-----------------------------------------------------------------------


void ReadWriteMutex::unlockRead ()
{
  JEM_ASSERT_NOTHROW ( lockedRead() );

  // Note that a reader thread never has to signal another reader
  // thread because reader threads can not block each other.

  spinlock_.lock ();

  readCount_--;

  if ( readCount_ == 0 && writeQueue_ )
  {
    ThreadNode*  writer = ThreadQueue::pop ( writeQueue_ );

    writeCount_ = MAX_WRITE_COUNT_;

    writer->notify ();
  }

  spinlock_.unlock ();
}


//-----------------------------------------------------------------------
//   tryLockWrite
//-----------------------------------------------------------------------


bool ReadWriteMutex::tryLockWrite () noexcept
{
  bool  result = false;


  spinlock_.lock ();

  if ( readCount_ == 0 && writeCount_ == 0 )
  {
    writeCount_ = MAX_WRITE_COUNT_;
    result      = true;
  }

  spinlock_.unlock ();

  return result;
}


//-----------------------------------------------------------------------
//   lockWrite
//-----------------------------------------------------------------------


void ReadWriteMutex::lockWrite ()
{
  spinlock_.lock ();

  if ( readCount_ || writeCount_ )
  {
    ThreadNode  self;

    ThreadQueue::push ( writeQueue_, &self );
    self.waitNoCancel ( spinlock_ );
  }
  else
  {
    writeCount_ = MAX_WRITE_COUNT_;
  }

  spinlock_.unlock ();
}


//-----------------------------------------------------------------------
//   unlockWrite
//-----------------------------------------------------------------------


void ReadWriteMutex::unlockWrite ()
{
  JEM_ASSERT_NOTHROW ( lockedWrite() );

  ThreadNode*  node;


  spinlock_.lock ();

  if ( readQueue_ )
  {
    node = ThreadQueue::pop ( readQueue_ );

    readCount_++;
    writeCount_ = 0;

    node->notify ();
  }
  else if ( writeQueue_ )
  {
    node = ThreadQueue::pop ( writeQueue_ );

    node->notify ();
  }
  else
  {
    writeCount_ = 0;
  }

  spinlock_.unlock ();
}


JEM_END_PACKAGE_BASE
