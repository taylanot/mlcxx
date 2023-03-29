
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

#ifndef JEM_BASE_THREAD_THREADMONITOR_H
#define JEM_BASE_THREAD_THREADMONITOR_H

#include <jem/base/native/Monitor.h>
#include <jem/base/thread/SigThread.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ThreadMonitor
//-----------------------------------------------------------------------


class ThreadMonitor
{
 public:

  typedef ThreadMonitor   Self;


  inline                  ThreadMonitor ();

  inline void             lock          ();
  inline void             unlock        ();

  inline byte             wait

    ( byte                  sigmask );

  inline byte             wait

    ( byte                  sigmask,
      const Time&           timeout );

  inline bool             notify

    ( byte                  signal );

  inline byte             getSignal     () const noexcept;

  inline void             clearSignal

    ( byte                  sigmask )            noexcept;

  inline byte             getSigmask    () const noexcept;

  inline byte             setSigmask

    ( byte                  sigmask )            noexcept;

  inline byte             enableSignal

    ( byte                  signal,
      bool                  enable = true )      noexcept;


 protected:

  inline                 ~ThreadMonitor ();


 private:

                          ThreadMonitor ( const Self& );
  ThreadMonitor&          operator =    ( const Self& );


 private:

  NativeMonitor           monitor_;

  bool                    sleeping_;
  byte                    sigmask_;
  byte                    signal_;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline ThreadMonitor::ThreadMonitor () :

  sleeping_ ( false ),
  sigmask_  ( SigThread::NOTIFY ),
  signal_   ( SigThread::NONE   )

{}


inline ThreadMonitor::~ThreadMonitor ()
{}


//-----------------------------------------------------------------------
//   lock
//-----------------------------------------------------------------------


inline void ThreadMonitor::lock ()
{
  monitor_.lock ();
}


//-----------------------------------------------------------------------
//   unlock
//-----------------------------------------------------------------------


inline void ThreadMonitor::unlock ()
{
  monitor_.unlock ();
}


//-----------------------------------------------------------------------
//   wait (without timeout)
//-----------------------------------------------------------------------


inline byte ThreadMonitor::wait ( byte sigmask )
{
  byte  signal;


  sigmask &= sigmask_;
  signal   = signal_ & sigmask;

  while ( ! signal )
  {
    sleeping_ = true;

    monitor_.wait ();

    sleeping_ = false;
    signal    = signal_ & sigmask;
  }

  signal_ = (byte) (signal_ & (~sigmask));

  return signal;
}


//-----------------------------------------------------------------------
//   wait (with timeout)
//-----------------------------------------------------------------------


inline byte ThreadMonitor::wait

  ( byte         sigmask,
    const Time&  timeout )

{
  byte  signal;


  sigmask &= sigmask_;
  signal   = signal_ & sigmask;

  if ( ! signal )
  {
    sleeping_ = true;

    monitor_.wait ( timeout );

    sleeping_ = false;
    signal    = signal_ & sigmask;
  }

  signal_ = (byte) (signal_ & (~sigmask));

  return signal;
}


//-----------------------------------------------------------------------
//   notify
//-----------------------------------------------------------------------


inline bool ThreadMonitor::notify ( byte signal )
{
  signal_ |= signal;

  if ( signal & sigmask_ )
  {
    if ( sleeping_ )
    {
      monitor_.notify ();
    }

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getSignal
//-----------------------------------------------------------------------


inline byte ThreadMonitor::getSignal () const noexcept
{
  return signal_;
}


//-----------------------------------------------------------------------
//   clearSignal
//-----------------------------------------------------------------------


inline void ThreadMonitor::clearSignal ( byte sigmask ) noexcept
{
  signal_ = (byte) (signal_ & (~sigmask));
}


//-----------------------------------------------------------------------
//   getSigmask
//-----------------------------------------------------------------------


inline byte ThreadMonitor::getSigmask () const noexcept
{
  return sigmask_;
}


//-----------------------------------------------------------------------
//   setSigmask
//-----------------------------------------------------------------------


inline byte ThreadMonitor::setSigmask ( byte sigmask ) noexcept
{
  byte  tmp = sigmask_;

  sigmask_  = sigmask;

  return tmp;
}


//-----------------------------------------------------------------------
//   enableSignal
//-----------------------------------------------------------------------


inline byte ThreadMonitor::enableSignal

  ( byte  signal,
    bool  enable ) noexcept

{
  byte  tmp = sigmask_;

  if ( enable )
  {
    sigmask_ = (byte) (sigmask_ | signal);
  }
  else
  {
    sigmask_ = (byte) (sigmask_ & (~signal));
  }

  return tmp;
}


JEM_END_PACKAGE_BASE

#endif
