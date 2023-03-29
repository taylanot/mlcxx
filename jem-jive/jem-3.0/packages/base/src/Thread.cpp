
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
#include <jem/base/Time.h>
#include <jem/base/AutoPointer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/CancelledException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/thread/ThreadLib.h>
#include <jem/base/thread/ThreadLocal.h>
#include <jem/base/thread/ThreadStart.h>
#include <jem/base/thread/ThreadState.h>
#include <jem/base/Thread.h>


JEM_DEFINE_CUSTOM_CLASS( jem::Thread );


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Thread
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Thread::Thread ()
{
  AutoPointer<ThreadState>  state = new ThreadState ();

  ThreadLib::init ();

  Thread*  self = ThreadLib::getThread ();

  state->setSigmask ( self->state_->getSigmask() );

  state_ = state.release ();
}


Thread::Thread ( MainTag_ )
{
  // Because this constructor is called by ThreadLib there
  // is no need to call ThreadLib::init().

  AutoPointer<ThreadState>  state = new ThreadState ();

  state->alive  = true;
  state->local  = newInstance<ThreadLocal>  ();
  state->native = newInstance<NativeThread> ();

  state_ = state.release ();
}


Thread::~Thread ()
{
  delete state_;
  state_ = nullptr;
}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Thread::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis ( "a thread" );

  return myClass;
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void Thread::start ()
{
  AutoPointer<ThreadCache>  cache;
  Ref<ThreadLocal>          local;
  ThreadStart*              start;
  bool                      alive;


  // Switch to multi-thread mode:

  ThreadLib::initMulti   ();

  // Collect dead threads:

  ThreadLib::reapZombies ();

  local = ThreadLib::getThread()->state_->local->clone ();
  cache = ThreadLib::getCache ()               ->clone ();
  start = nullptr;

  try
  {
    state_->lock ();

    alive         = state_->alive;
    state_->alive = true;

    state_->clearSignal ( SigThread::ALL );

    state_->unlock ();

    if ( ! alive )
    {
      start         =   new ThreadStart ();
      start->start  = & ThreadLib::entryPoint;
      start->thread =   this;
      start->local  =   local;
      start->cache  =   cache;

      NativeThread::start ( start );
    }
  }
  catch ( ... )
  {
    delete start;
    die_ ();
    throw;
  }
}


//-----------------------------------------------------------------------
//   join
//-----------------------------------------------------------------------


void Thread::join ()
{
  join_ ( nullptr );
}


void Thread::join ( const Time& timeout )
{
  if ( timeout > Time::zero() )
  {
    join_ ( &timeout );
  }
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void Thread::cancel ()
{
  state_->lock ();

  if ( state_->notify( SigThread::CANCEL ) &&
       state_->native != nullptr )
  {
    state_->native->interrupt ();
  }

  state_->unlock ();
}


//-----------------------------------------------------------------------
//   isCancelled
//-----------------------------------------------------------------------


bool Thread::isCancelled () const
{
  byte  sigmask = state_->getSigmask ();
  byte  signal  = state_->getSignal  ();

  signal &= sigmask;

  return ((signal & SigThread::CANCEL) != 0);
}


//-----------------------------------------------------------------------
//   isAlive
//-----------------------------------------------------------------------


bool Thread::isAlive () const
{
  return state_->alive;
}


//-----------------------------------------------------------------------
//   getCurrent
//-----------------------------------------------------------------------


Thread* Thread::getCurrent ()
{
  ThreadLib::init ();

  return ThreadLib::getThread ();
}


//-----------------------------------------------------------------------
//   cancelled
//-----------------------------------------------------------------------


bool Thread::cancelled ()
{
  ThreadLib::init ();

  ThreadState&  state   = * ThreadLib::getThread()->state_;

  byte          sigmask = state.getSigmask ();
  byte          signal  = state.getSignal  ();

  bool          result;


  signal &= sigmask;

  if ( ! (signal & SigThread::CANCEL) )
  {
    result = false;
  }
  else
  {
    Lock<ThreadState>  lock ( state );

    sigmask =           state.getSigmask ();
    signal  = sigmask & state.getSignal  ();
    result  = ((signal & SigThread::CANCEL) != 0);

    state.clearSignal ( SigThread::CANCEL );

    if ( state.native )
    {
      state.native->clearInterrupt ();
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   allowCancel
//-----------------------------------------------------------------------

bool Thread::allowCancel ( bool toggle )
{
  ThreadLib::init ();

  ThreadState&  state = * ThreadLib::getThread()->state_;

  byte          sigmask;


  state.lock ();

  sigmask = state.enableSignal ( SigThread::CANCEL,
                                 toggle );

  state.unlock ();

  return ((sigmask & SigThread::CANCEL) != 0);
}


//-----------------------------------------------------------------------
//   sleep
//-----------------------------------------------------------------------


void Thread::sleep ( const Time&  timeout )
{
  ThreadLib::init ();

  if ( timeout > Time::zero() )
  {
    ThreadState&  state = * ThreadLib::getThread()->state_;
    byte          signal;

    state.lock ();

    signal = state.wait ( SigThread::ALL, timeout );

    state.unlock ();

    if ( signal & SigThread::CANCEL )
    {
      throw CancelledException ( JEM_FUNC, "sleep cancelled" );
    }
  }
}


//-----------------------------------------------------------------------
//   newLocalKey
//-----------------------------------------------------------------------


int Thread::newLocalKey ()
{
  ThreadLib::init ();

  return ThreadLib::newLocalKey ();
}


//-----------------------------------------------------------------------
//   getLocal
//-----------------------------------------------------------------------


Ref<Object> Thread::getLocal ( int key )
{
  JEM_ASSERT_NOTHROW ( key >= 0 && key < ThreadLocal::MAX_KEYS );

  return ThreadLib::getThread()->state_->local->get ( key );
}


//-----------------------------------------------------------------------
//   setLocal
//-----------------------------------------------------------------------


void Thread::setLocal ( int key, const Ref<Object>& obj )
{
  if ( key < 0 || key >= ThreadLocal::MAX_KEYS || !
       ThreadLib::initialized() )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "illegal thread local key: " + String( key )
    );
  }

  ThreadLib::getThread()->state_->local->set ( key, obj );
}


//-----------------------------------------------------------------------
//   join_
//-----------------------------------------------------------------------


void Thread::join_ ( const Time* timeout )
{
  ThreadState*  self  = ThreadLib::getThread()->state_;
  ThreadState*  other = state_;

  byte          signal;


  if ( self == state_ )
  {
    return;
  }

  other->lock ();

  if ( ! other->alive )
  {
    other->unlock ();
    return;
  }

  self ->nextJoiner = other->joinQueue;
  other->joinQueue  = self;

  other->unlock ();
  self ->lock   ();

  if ( timeout )
  {
    signal = self->wait ( SigThread::ALL, *timeout );
  }
  else
  {
    signal = self->wait ( SigThread::ALL );
  }

  self->unlock ();

  if ( ! (signal & SigThread::NOTIFY) )
  {
    ThreadState*  prev = nullptr;
    ThreadState*  next;

    other->lock ();

    next = other->joinQueue;

    while ( next )
    {
      if ( next == self )
      {
        if ( prev )
        {
          prev->nextJoiner = self->nextJoiner;
        }
        else
        {
          other->joinQueue = self->nextJoiner;
        }

        break;
      }

      prev = next;
      next = next->nextJoiner;
    }

    other->unlock ();

    if ( ! next )
    {
      self->lock   ();
      self->wait   ( SigThread::NOTIFY );
      self->unlock ();
    }
  }

  ThreadLib::reapZombies ();

  self->nextJoiner = nullptr;

  if ( signal & SigThread::CANCEL )
  {
    throw CancelledException ( JEM_FUNC, "join cancelled" );
  }
}


//-----------------------------------------------------------------------
//   die_
//-----------------------------------------------------------------------


void Thread::die_ ()
{
  ThreadState*  joinQueue;
  ThreadState*  next;


  state_->lock ();

  joinQueue         = state_->joinQueue;
  state_->joinQueue = nullptr;
  state_->alive     = false;

  state_->unlock ();

  while ( joinQueue )
  {
    next = joinQueue->nextJoiner;

    joinQueue->lock   ();
    joinQueue->notify ( SigThread::NOTIFY );
    joinQueue->unlock ();

    joinQueue = next;
  }
}


JEM_END_PACKAGE_BASE
