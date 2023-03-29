
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
#include <jem/base/RecursiveMonitor.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class RecursiveMonitor
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


RecursiveMonitor::RecursiveMonitor ()
{
  queue_ = nullptr;
}


RecursiveMonitor::~RecursiveMonitor ()
{
  queue_ = nullptr;
}


//-----------------------------------------------------------------------
//   waitNoCancel
//-----------------------------------------------------------------------


void RecursiveMonitor::waitNoCancel ()
{
  JEM_ASSERT_NOTHROW ( locked() );

  const int   count  = mutex_.count_;
  Thread*     self   = mutex_.owner_;

  ThreadNode  myNode ( self );


  mutex_.count_ = 0;

  mutex_.spinlock_.lock ();

  ThreadQueue::push ( queue_, &myNode );

  if ( mutex_.queue_ )
  {
    ThreadNode*  node = ThreadQueue::pop ( mutex_.queue_ );

    mutex_.owner_ = static_cast<Thread*> ( node->getData() );

    node->notify ();
  }
  else
  {
    mutex_.owner_ = nullptr;
  }

  myNode.waitNoCancel ( mutex_.spinlock_ );

  mutex_.spinlock_.unlock ();

  mutex_.count_ = count;
}


//-----------------------------------------------------------------------
//   wait_
//-----------------------------------------------------------------------


void RecursiveMonitor::wait_ ( const Time* timeout )
{
  JEM_ASSERT_NOTHROW ( locked() );

  const int   count  = mutex_.count_;
  Thread*     self   = mutex_.owner_;

  ThreadNode  myNode ( self );

  byte        sig;


  mutex_.count_ = 0;

  mutex_.spinlock_.lock ();

  ThreadQueue::push ( queue_, &myNode );

  if ( mutex_.queue_ )
  {
    ThreadNode*  node = ThreadQueue::pop ( mutex_.queue_ );

    mutex_.owner_ = static_cast<Thread*> ( node->getData() );

    node->notify ();
  }
  else
  {
    mutex_.owner_ = nullptr;
  }

  sig = myNode.wait ( mutex_.spinlock_, timeout );

  if ( ! (sig & SigThread::NOTIFY) )
  {
    if ( ! ThreadQueue::remove( queue_, &myNode ) )
    {
      myNode.waitNoCancel ( mutex_.spinlock_ );
    }
    else if ( mutex_.owner_ && mutex_.owner_ != self )
    {
      ThreadQueue::push   ( mutex_.queue_, &myNode );
      myNode.waitNoCancel ( mutex_.spinlock_ );
    }
    else
    {
      mutex_.owner_ = self;
    }
  }

  mutex_.spinlock_.unlock ();

  mutex_.count_ = count;

  if ( sig & SigThread::CANCEL )
  {
    throw CancelledException ( JEM_FUNC, "wait cancelled" );
  }
}


//-----------------------------------------------------------------------
//   notify_
//-----------------------------------------------------------------------


void RecursiveMonitor::notify_ ()
{
  JEM_ASSERT_NOTHROW ( locked() );

  mutex_.spinlock_.lock ();

  if ( queue_ )
  {
    ThreadQueue::push ( mutex_.queue_, ThreadQueue::pop( queue_ ) );
  }

  mutex_.spinlock_.unlock ();
}


//-----------------------------------------------------------------------
//   notifyAll_
//-----------------------------------------------------------------------


void RecursiveMonitor::notifyAll_ ()
{
  JEM_ASSERT_NOTHROW ( locked() );

  mutex_.spinlock_.lock ();

  while ( queue_ )
  {
    ThreadQueue::push ( mutex_.queue_, ThreadQueue::pop( queue_ ) );
  }

  mutex_.spinlock_.unlock ();
}


JEM_END_PACKAGE_BASE
