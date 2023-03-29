
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


#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <jem/version.h>
#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/base/Mutex.h>
#include <jem/base/Thread.h>
#include <jem/base/EnvParams.h>
#include <jem/base/SerialSection.h>
#include <jem/base/SystemException.h>
#include <jem/base/CancelledException.h>
#include <jem/base/native/System.h>
#include <jem/base/native/Thread.h>
#include <jem/base/thread/ThreadStart.h>
#include <jem/base/thread/ThreadLocal.h>
#include <jem/base/thread/ThreadLib.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class MainThread
//=======================================================================


class MainThread : public Thread
{
 public:

                    MainThread ();

  virtual void      run        () override;


 protected:

  virtual          ~MainThread ();

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MainThread::MainThread () : Thread ( Thread::MAIN_TAG_ )
{}


MainThread::~MainThread ()
{}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void MainThread::run ()
{}


//=======================================================================
//   class ThreadLib::Lock_
//=======================================================================


class ThreadLib::Lock_
{
 public:

  inline            Lock_ ();
  inline           ~Lock_ ();


 public:

  static Mutex*     mutex;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Mutex* ThreadLib::Lock_::mutex = nullptr;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline ThreadLib::Lock_::Lock_ ()
{
  mutex->lock ();
}


inline ThreadLib::Lock_::~Lock_ ()
{
  mutex->unlock ();
}


//=======================================================================
//   class ThreadLib
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


ThreadLib  ThreadLib::instance_;

lint       ThreadLib::SPIN_COUNT   = 0;
byte       ThreadLib::initialized_ = 0;
Thread*    ThreadLib::mainThread_  = nullptr;
int        ThreadLib::nextKey_     = 0;
Thread**   ThreadLib::zombies_     = nullptr;
idx_t      ThreadLib::maxZombies_  = 0;
idx_t      ThreadLib::zombieCount_ = 0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ThreadLib::ThreadLib ()
{}


ThreadLib::~ThreadLib ()
{
  if ( initialized_ )
  {
    mainThread_->die_ ();

    try
    {
      reapZombies ();
    }
    catch ( ... )
    {}
  }
}


//-----------------------------------------------------------------------
//   newLocalKey
//-----------------------------------------------------------------------


int ThreadLib::newLocalKey ()
{
  JEM_ASSERT_NOTHROW ( initialized() );

  Lock_  lock;
  int    k;

  if ( nextKey_ >= ThreadLocal::MAX_KEYS )
  {
    throw SystemException (
      JEM_FUNC,
      "no more thread local keys available"
    );
  }

  k = nextKey_++;

  return k;
}


//-----------------------------------------------------------------------
//   entryPoint
//-----------------------------------------------------------------------


void ThreadLib::entryPoint ( ThreadStart*  start )
{
  JEM_ASSERT_NOTHROW ( start );

  AutoPointer<Cache>  cache  = start->cache;
  Ref<Thread>         thread = start->thread;
  Ref<ThreadLocal>    local  = start->local;
  Ref<Native>         native;


  delete start;

  Native::setCache ( cache.get() );

  native = newInstance<Native> ();

  thread->state_->lock ();

  thread->state_->local  = local;
  thread->state_->native = native;

  thread->state_->unlock ();

  Native::setThread ( thread.get() );

  try
  {
    thread->run ();
  }
  catch ( const CancelledException& )
  {
    // Ignore this exception
  }
  catch ( const Throwable& ex )
  {
    jem::abort ( ex );
  }
  catch ( ... )
  {
    jem::abort ( JEM_FUNC,
                 "uncaught exception in thread entry point" );
  }

  local  = nullptr;
  native = nullptr;

  thread->state_->lock ();

  thread->state_->local .swap ( local );
  thread->state_->native.swap ( native );

  thread->state_->unlock ();

  local  = nullptr;
  native = nullptr;
  cache  = nullptr;

  kill_ ( thread );

  Native::setThread ( nullptr );
  Native::setCache  ( nullptr );
}


//-----------------------------------------------------------------------
//   reapZombies
//-----------------------------------------------------------------------


void ThreadLib::reapZombies ()
{
  if ( zombieCount_ )
  {
    Lock_  lock;

    while ( zombieCount_ > 0 )
    {
      --zombieCount_;

      Collectable::delRef ( zombies_[zombieCount_] );

      zombies_[zombieCount_] = nullptr;
    }
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void ThreadLib::init_ ()
{
  lint  cacheSize;

  try
  {
    initialized_ = 1;
    cacheSize    = Cache::DEFAULT_SIZE;

    igetenv ( SPIN_COUNT, EnvParams::SPIN_COUNT );
    igetenv ( cacheSize,  EnvParams::CACHE_SIZE );

    if ( SPIN_COUNT < 0 )
    {
      SPIN_COUNT = 0;
    }
    if ( cacheSize < 0 )
    {
      cacheSize = 0;
    }

    Lock_::mutex = new Mutex ();

    Native::init     ();
    Native::setCache ( new Cache( (size_t) cacheSize ) );

    Ref<Thread>  mt = newInstance<MainThread> ();

    mainThread_ = mt.get ();

    Collectable::addRef ( *mt );
    Native::setThread   ( mainThread_ );
  }
  catch ( const Throwable& ex )
  {
    jem::abort ( ex );
  }
  catch ( ... )
  {
    jem::abort ( JEM_FUNC, "caught an unexpected exception" );
  }
}


//-----------------------------------------------------------------------
//   initMulti_
//-----------------------------------------------------------------------


void ThreadLib::initMulti_ ()
{
  if ( ! initialized_ )
  {
    init_ ();
  }

  initialized_ = 2;

  SerialSection::init_ ();
}


//-----------------------------------------------------------------------
//   kill_
//-----------------------------------------------------------------------


void ThreadLib::kill_ ( Ref<Thread>& thread )
{
  Lock_  lock;


  if ( zombieCount_ >= maxZombies_ )
  {
    Thread**  z;
    idx_t     n;


    n = 2 * maxZombies_ + 1;

    z = (Thread**)

      ::realloc ( zombies_, (size_t) n * sizeof(Thread*) );

    if ( z )
    {
      zombies_    = z;
      maxZombies_ = n;
    }
  }

  if ( zombieCount_ < maxZombies_ )
  {
    zombies_[zombieCount_++] = thread.get ();

    Collectable::addRef ( thread.get() );
  }

  thread->die_();

  thread = nullptr;
}


JEM_END_PACKAGE_BASE
