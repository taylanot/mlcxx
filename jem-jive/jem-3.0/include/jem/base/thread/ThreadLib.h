
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

#ifndef JEM_BASE_THREAD_THREADLIB_H
#define JEM_BASE_THREAD_THREADLIB_H

#include <jem/base/utilities.h>
#include <jem/base/Thread.h>
#include <jem/base/native/Thread.h>
#include <jem/base/thread/ThreadState.h>


JEM_BEGIN_PACKAGE_BASE


class ThreadStart;
class ThreadCache;
class ThreadLocal;


//-----------------------------------------------------------------------
//   class ThreadLib
//-----------------------------------------------------------------------


class ThreadLib
{
 public:

  typedef ThreadLib         Self;
  typedef ThreadCache       Cache;
  typedef ThreadMonitor     Monitor;
  typedef NativeThread      Native;
  typedef ThreadLocal       Local;


  static lint               SPIN_COUNT;


  static inline void        init          ();
  static inline void        initMulti     ();
  static inline byte        initialized   () noexcept;

  static inline Thread*     getThread     () noexcept;
  static inline Monitor*    getMonitor    ();

  static inline Monitor*    getMonitor

    ( Thread&                 thread );

  static inline Native*     getNative     ();
  static inline Cache*      getCache      ();
  static inline Local*      getLocal      ();

  static inline void        yield

    ( lint                    dt = 0 )       noexcept;

  static int                newLocalKey   ();

  static void               entryPoint

    ( ThreadStart*            start );

  static void               reapZombies   ();


 private:

  friend class              Nil;

  class                     Lock_;


                            ThreadLib     ();
                           ~ThreadLib     ();

  static void               init_         ();
  static void               initMulti_    ();

  static void               kill_

    ( Ref<Thread>&            thread );

  static void               runError_

    ( const String&           name,
      const String&           where,
      const String&           what );


 private:

                            ThreadLib     ( const Self& );
  ThreadLib&                operator =    ( const Self& );


 private:

  static ThreadLib          instance_;
  static byte               initialized_;
  static Thread*            mainThread_;
  static int                nextKey_;
  static Thread**           zombies_;
  static idx_t              maxZombies_;
  static idx_t              zombieCount_;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


inline void ThreadLib::init ()
{
  if ( ! initialized_ )
  {
    init_ ();
  }
}


//-----------------------------------------------------------------------
//   initMulti
//-----------------------------------------------------------------------


inline void ThreadLib::initMulti ()
{
  if ( initialized_ < 2 )
  {
    initMulti_ ();
  }
}


//-----------------------------------------------------------------------
//   initialized
//-----------------------------------------------------------------------


inline byte ThreadLib::initialized () noexcept
{
  return initialized_;
}


//-----------------------------------------------------------------------
//   getThread
//-----------------------------------------------------------------------


inline Thread* ThreadLib::getThread () noexcept
{
  return NativeThread::getThread ();
}


//-----------------------------------------------------------------------
//   getMonitor
//-----------------------------------------------------------------------


inline ThreadMonitor* ThreadLib::getMonitor ()
{
  return NativeThread::getThread()->state_;
}


inline ThreadMonitor* ThreadLib::getMonitor ( Thread& thread )
{
  return thread.state_;
}


//-----------------------------------------------------------------------
//   getNative
//-----------------------------------------------------------------------


inline NativeThread* ThreadLib::getNative ()
{
  return NativeThread::getThread()->state_->native.get ();
}


//-----------------------------------------------------------------------
//   getCache
//-----------------------------------------------------------------------


inline ThreadCache* ThreadLib::getCache ()
{
  return NativeThread::getCache ();
}


//-----------------------------------------------------------------------
//   getLocal
//-----------------------------------------------------------------------


inline ThreadLocal* ThreadLib::getLocal ()
{
  return NativeThread::getThread()->state_->local.get ();
}


//-----------------------------------------------------------------------
//   yield
//-----------------------------------------------------------------------


inline void ThreadLib::yield ( lint timeout ) noexcept
{
  NativeThread::yield ( timeout );
}


JEM_END_PACKAGE_BASE

#endif
