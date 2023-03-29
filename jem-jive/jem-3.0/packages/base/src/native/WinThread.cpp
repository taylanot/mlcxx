
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


#include <jem/base/utilities.h>
#include <jem/base/SystemException.h>
#include <jem/base/thread/ThreadStart.h>
#include "native/WinThread.h"


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   non-member functions
//=======================================================================

JEM_BEGIN_NAMESPACE( win )

//-----------------------------------------------------------------------
//   interruptAPC
//-----------------------------------------------------------------------


extern "C"
{
  static void interruptAPC ( ULONG_PTR arg )
  {
    // Does nothing.
  }
}


//-----------------------------------------------------------------------
//   threadFunc
//-----------------------------------------------------------------------


extern "C"
{
  static DWORD threadFunc ( void* arg )
  {
    // Make sure that no exceptions propagate beyond this
    // function, since this is the upper most stack frame
    // of the thread.

    try
    {
      ((ThreadStart*) arg)->start ( (ThreadStart*) arg );
    }
    catch ( ... )
    {
      abort ( JEM_FUNC,
              "uncaught exception in WIN thread entry point" );
    }

    return 0;
  }
}

JEM_END_NAMESPACE( win )

//=======================================================================
//   class WinThread
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


thread_local void*  WinThread::thread_ = nullptr;
thread_local void*  WinThread::cache_  = nullptr;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


WinThread::WinThread ()
{
  self_ = GetCurrentThread ();
}


WinThread::~WinThread ()
{
  self_ = nullptr;
}


//-----------------------------------------------------------------------
//   interrupt
//-----------------------------------------------------------------------


void WinThread::interrupt ()
{
  QueueUserAPC ( (PAPCFUNC) win::interruptAPC, self_, 0 );
}


//-----------------------------------------------------------------------
//   clearInterrupt
//-----------------------------------------------------------------------


void WinThread::clearInterrupt ()
{
  // Does nothing.
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void WinThread::init ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void WinThread::start ( ThreadStart* start )
{
  HANDLE  thread =

    CreateThread ( nullptr,
                   0,
                   (LPTHREAD_START_ROUTINE) &win::threadFunc,
                   start,
                   0,
                   nullptr );

  if ( ! thread )
  {
    throw SystemException (
      JEM_FUNC,
      "CreateThread() failed (too many threads?)"
    );
  }
}


//-----------------------------------------------------------------------
//   yield
//-----------------------------------------------------------------------


void WinThread::yield ( lint nsecs ) noexcept
{
  if ( nsecs <= 0 )
  {
    SwitchToThread ();
  }
  else
  {
    SleepEx ( (DWORD) (nsecs / 1000000_lint), TRUE );
  }
}


JEM_END_PACKAGE_BASE
