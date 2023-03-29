
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
#include <jem/base/CancelledException.h>
#include <jem/base/thread/ThreadLib.h>
#include <jem/base/thread/ThreadQueue.h>
#include <jem/base/Mutex.h>
#include <jem/base/FastCondition.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class FastCondition
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


FastCondition::FastCondition ()
{
  ThreadLib::init ();

  queue_ = nullptr;
}


FastCondition::~FastCondition ()
{
  queue_ = nullptr;
}


//-----------------------------------------------------------------------
//   wait_
//-----------------------------------------------------------------------


void FastCondition::wait_

  ( Mutex&       mutex,
    bool*        cancelled,
    const Time*  timeout )

{
  JEM_ASSERT_NOTHROW ( mutex.locked() );

  ThreadNode  self;
  byte        signal;


  mutex.spinlock_.lock ();

  ThreadQueue::push ( queue_, &self );

  if ( mutex.queue_ )
  {
    ThreadNode*  node = ThreadQueue::pop ( mutex.queue_ );

    node->notify ();
  }
  else
  {
    mutex.locked_ = false;
  }

  signal = self.wait ( mutex.spinlock_, timeout );

  if ( ! (signal & SigThread::NOTIFY) )
  {
    if ( ! ThreadQueue::remove( queue_, &self ) )
    {
      self.waitNoCancel ( mutex.spinlock_ );
    }
    else if ( mutex.locked_ )
    {
      ThreadQueue::push ( mutex.queue_, &self );
      self.waitNoCancel ( mutex.spinlock_ );
    }
    else
    {
      mutex.locked_ = true;
    }
  }

  mutex.spinlock_.unlock ();

  if ( signal & SigThread::CANCEL )
  {
    if ( cancelled )
    {
      *cancelled = true;
    }
    else
    {
      throw CancelledException ( JEM_FUNC, "wait cancelled" );
    }
  }
}


//-----------------------------------------------------------------------
//   waitNoCancel_
//-----------------------------------------------------------------------


void FastCondition::waitNoCancel_ ( Mutex& mutex )
{
  JEM_ASSERT_NOTHROW ( mutex.locked() );

  ThreadNode  self;


  mutex.spinlock_.lock ();

  ThreadQueue::push ( queue_, &self );

  if ( mutex.queue_ )
  {
    ThreadNode*  node = ThreadQueue::pop ( mutex.queue_ );

    node->notify ();
  }
  else
  {
    mutex.locked_ = false;
  }

  self.waitNoCancel ( mutex.spinlock_ );

  mutex.spinlock_.unlock ();
}


//-----------------------------------------------------------------------
//   notify_
//-----------------------------------------------------------------------


void FastCondition::notify_ ( Mutex& mutex )
{
  JEM_ASSERT_NOTHROW ( mutex.locked() );

  mutex.spinlock_.lock ();

  if ( queue_ )
  {
    ThreadNode*  node = ThreadQueue::pop ( queue_ );

    ThreadQueue::push ( mutex.queue_, node );
  }

  mutex.spinlock_.unlock ();
}


//-----------------------------------------------------------------------
//   notifyAll_
//-----------------------------------------------------------------------


void FastCondition::notifyAll_ ( Mutex& mutex )
{
  JEM_ASSERT_NOTHROW ( mutex.locked() );

  ThreadNode*  node;


  mutex.spinlock_.lock ();

  while ( queue_ )
  {
    node = ThreadQueue::pop ( queue_ );

    ThreadQueue::push ( mutex.queue_, node );
  }

  mutex.spinlock_.unlock ();
}


JEM_END_PACKAGE_BASE
