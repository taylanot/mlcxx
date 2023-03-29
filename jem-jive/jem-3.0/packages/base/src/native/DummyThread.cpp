
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
#include <jem/base/Time.h>
#include <jem/base/SystemException.h>
#include "native/DummyThread.h"


#if defined(JEM_OS_POSIX)

  extern "C"
  {
    #include <unistd.h>
  }

#endif


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class DummyThread
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Thread*       DummyThread::current_ = nullptr;
ThreadCache*  DummyThread::cache_   = nullptr;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DummyThread::DummyThread ()
{}


DummyThread::~DummyThread ()
{}


//-----------------------------------------------------------------------
//   interrupt
//-----------------------------------------------------------------------


void DummyThread::interrupt ()
{}


//-----------------------------------------------------------------------
//   clearInterrupt
//-----------------------------------------------------------------------


void DummyThread::clearInterrupt ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void DummyThread::init ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void DummyThread::start ( ThreadStart* )
{
  throw SystemException (
    JEM_FUNC,
    "thread creation failed because this version of Jem has been "
    "compiled without thread support"
  );
}


//-----------------------------------------------------------------------
//   yield
//-----------------------------------------------------------------------


void DummyThread::yield ( lint nsecs )
{
  if ( nsecs > 0 )
  {
    sleep ( Time( 0, nsecs ) );
  }
}


//-----------------------------------------------------------------------
//   sleep
//-----------------------------------------------------------------------


void DummyThread::sleep ( const Time& dt )
{
  if ( dt <= Time::zero() )
  {
    return;
  }

#if defined(JEM_OS_UNIX98)

  struct timespec  ts;

  ts.tv_sec  = (time_t) dt.sec  ();
  ts.tv_nsec =          dt.nsec ();

  ::nanosleep ( &ts, nullptr );

#elif defined(JEM_OS_POSIX)

  ::sleep ( dt.sec() );

#else
  #error "Unsupported operating system -- sorry"
#endif

}


JEM_END_PACKAGE_BASE
