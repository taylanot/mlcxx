
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

#ifndef JEM_BASE_RECURSIVEMONITOR_H
#define JEM_BASE_RECURSIVEMONITOR_H

#include <jem/base/RecursiveMutex.h>


JEM_BEGIN_PACKAGE_BASE


class Time;


//-----------------------------------------------------------------------
//   class RecursiveMonitor
//-----------------------------------------------------------------------


class RecursiveMonitor
{
 public:

                      RecursiveMonitor  ();
                     ~RecursiveMonitor  ();

  inline bool         locked            () const noexcept;
  inline bool         tryLock           ()       noexcept;
  inline void         lock              ();
  inline void         unlock            ();
  inline void         wait              ();

  inline void         wait

    ( const Time&       timeout );

  void                waitNoCancel      ();

  inline void         notify            ();
  inline void         notifyAll         ();


 private:

  void                wait_

    ( const Time*       timeout );

  void                notify_           ();
  void                notifyAll_        ();


 private:

  RecursiveMutex      mutex_;
  ThreadNode*         queue_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   locked
//-----------------------------------------------------------------------


inline bool RecursiveMonitor::locked () const noexcept
{
  return mutex_.locked ();
}


//-----------------------------------------------------------------------
//   tryLock
//-----------------------------------------------------------------------


inline bool RecursiveMonitor::tryLock () noexcept
{
  return mutex_.tryLock ();
}


//-----------------------------------------------------------------------
//   lock
//-----------------------------------------------------------------------


inline void RecursiveMonitor::lock ()
{
  mutex_.lock ();
}


//-----------------------------------------------------------------------
//   unlock
//-----------------------------------------------------------------------


inline void RecursiveMonitor::unlock ()
{
  mutex_.unlock ();
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


inline void RecursiveMonitor::wait ()
{
  wait_ ( nullptr );
}


inline void RecursiveMonitor::wait ( const Time& timeout )
{
  wait_ ( & timeout );
}


//-----------------------------------------------------------------------
//   notify
//-----------------------------------------------------------------------


inline void RecursiveMonitor::notify ()
{
  if ( queue_ )
  {
    notify_ ();
  }
}


//-----------------------------------------------------------------------
//   notifyAll
//-----------------------------------------------------------------------


inline void RecursiveMonitor::notifyAll ()
{
  if ( queue_ )
  {
    notifyAll_ ();
  }
}


JEM_END_PACKAGE_BASE

#endif
