
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
#include <jem/base/Mutex.h>
#include <jem/base/CancelledException.h>
#include <jem/base/thread/ThreadLib.h>
#include <jem/base/thread/ThreadQueue.h>
#include <jem/base/Condition.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Condition
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Condition::Condition ()
{
  ThreadLib::init ();

  queue_ = nullptr;
}


Condition::~Condition ()
{
  queue_ = nullptr;
}


//-----------------------------------------------------------------------
//   notify
//-----------------------------------------------------------------------


void Condition::notify ()
{
  ThreadNode*  waiter = nullptr;


  spinlock_.lock ();

  if ( queue_ )
  {
    waiter = ThreadQueue::pop ( queue_ );
  }

  spinlock_.unlock ();

  if ( waiter )
  {
    Mutex*  mutex = static_cast<Mutex*> ( waiter->getData() );

    mutex->spinlock_.lock ();

    if ( mutex->locked_ )
    {
      ThreadQueue::push ( mutex->queue_, waiter );
    }
    else
    {
      waiter->notify ();
    }

    mutex->spinlock_.unlock ();
  }
}


//-----------------------------------------------------------------------
//   notifyAll
//-----------------------------------------------------------------------


void Condition::notifyAll ()
{
  ThreadNode*  queue = nullptr;
  ThreadNode*  waiter;
  Mutex*       mutex;


  spinlock_.lock ();

  queue  = queue_;
  queue_ = nullptr;

  spinlock_.unlock ();

  while ( queue )
  {
    waiter = ThreadQueue::pop    ( queue );
    mutex  = static_cast<Mutex*> ( waiter->getData() );

    mutex->spinlock_.lock ();

    if ( mutex->locked_ )
    {
      ThreadQueue::push ( mutex->queue_, waiter );
    }
    else
    {
      waiter->notify ();
    }

    mutex->spinlock_.unlock ();
  }
}


//-----------------------------------------------------------------------
//   wait_
//-----------------------------------------------------------------------


void Condition::wait_

  ( Mutex&       mutex,
    bool*        cancelled,
    const Time*  timeout )

{
  JEM_ASSERT_NOTHROW ( mutex.locked() );

  ThreadNode  self ( &mutex );

  bool        removed;
  byte        signal;


  spinlock_.lock    ();

  ThreadQueue::push ( queue_, &self );

  spinlock_.unlock  ();

  mutex.spinlock_.lock ();

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
    spinlock_.lock   ();

    removed = ThreadQueue::remove ( queue_, &self );

    spinlock_.unlock ();

    if      ( ! removed )
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


void Condition::waitNoCancel_ ( Mutex& mutex )
{
  JEM_ASSERT_NOTHROW ( mutex.locked() );

  ThreadNode  self ( &mutex );


  spinlock_.lock    ();

  ThreadQueue::push ( queue_, &self );

  spinlock_.unlock  ();

  mutex.spinlock_.lock ();

  mutex.locked_ = false;

  if ( mutex.queue_ )
  {
    ThreadNode*  node = ThreadQueue::pop ( mutex.queue_ );

    node->notify ();
  }

  self.waitNoCancel ( mutex.spinlock_ );

  mutex.locked_ = true;

  mutex.spinlock_.unlock ();
}


JEM_END_PACKAGE_BASE
