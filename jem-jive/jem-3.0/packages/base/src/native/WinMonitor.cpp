
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
#include "native/WinMonitor.h"


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class WinMonitor
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


WinMonitor::WinMonitor ()
{
  waiting_ = false;

  mutex_   = CreateMutex ( nullptr, FALSE, nullptr );
  event_   = CreateEvent ( nullptr, FALSE, FALSE, nullptr );

  if ( ! mutex_ )
  {
    throw SystemException ( JEM_FUNC, "CreateMutex() failed" );
  }

  if ( ! event_ )
  {
    throw SystemException ( JEM_FUNC, "CreateEvent() failed" );
  }
}


WinMonitor::~WinMonitor ()
{
  CloseHandle ( mutex_ );
  CloseHandle ( event_ );
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


void WinMonitor::wait ( const Time& timeout )
{
  if ( timeout <= Time::zero() )
  {
    return;
  }

  DWORD  millis = (DWORD)

    (timeout.sec() * 1000_lint + timeout.nsec() / 1000000_lint);

  waiting_ = true;

  SignalObjectAndWait ( mutex_, event_, millis, TRUE );

  waiting_ = false;
}


JEM_END_PACKAGE_BASE
