
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


#include <jem/base/Time.h>
#include <jem/base/SystemException.h>
#include "native/PosixMonitor.h"


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class PosixMonitor
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PosixMonitor::PosixMonitor ()
{
  if ( pthread_mutex_init( &mutex_, nullptr ) != 0 )
  {
    throw SystemException ( JEM_FUNC, "pthread_mutex_init() failed" );
  }
  if ( pthread_cond_init( &cond_, nullptr ) != 0 )
  {
    throw SystemException ( JEM_FUNC, "pthread_cond_init() failed" );
  }
}


PosixMonitor::~PosixMonitor ()
{
  pthread_cond_destroy  ( &cond_  );
  pthread_mutex_destroy ( &mutex_ );
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


void PosixMonitor::wait ( const Time& timeout )
{
  if ( timeout <= Time::zero() )
  {
    return;
  }

  struct timespec  ts;
  Time             t;

  t = Time::now() + timeout;

  ts.tv_sec  = (time_t) t.sec  ();
  ts.tv_nsec =          t.nsec ();

  pthread_cond_timedwait ( &cond_, &mutex_, &ts );
}


JEM_END_PACKAGE_BASE
