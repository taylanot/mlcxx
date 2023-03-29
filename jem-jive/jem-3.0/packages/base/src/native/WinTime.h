
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

#ifndef JEM_BASE_NATIVE_WINTIME_H
#define JEM_BASE_NATIVE_WINTIME_H

#include <ctime>
#include <jem/base/Time.h>
#include <jem/base/SystemException.h>

extern "C"
{
  #include <windows.h>
}

#include <jem/base/native/winclean.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class WinTime
//-----------------------------------------------------------------------


class WinTime
{
 public:

  static const LONGLONG   NSEC100 = 10000000LL;


  static inline Time      getTime       ();

  static inline Time      getTime

    ( const FILETIME&       ft );

  static inline void      getLocalTime

    ( struct tm&            result,
      const Time&           t );

  static inline double    getCPUTime    ();
  static inline double    getCPUTick    ();
  static void             initCPUTime   ();


 private:

  static double           CPU_TICK_;

};


typedef WinTime           NativeTime;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getTime
//-----------------------------------------------------------------------


inline Time WinTime::getTime ()
{
  FILETIME  ft;

  GetSystemTimeAsFileTime ( &ft );

  return getTime ( ft );
}


inline Time WinTime::getTime ( const FILETIME& ft )
{
  ULARGE_INTEGER  utmp;
  LONGLONG        rest;
  lint            sec;
  lint            nsec;


  utmp.HighPart = ft.dwHighDateTime;
  utmp.LowPart  = ft.dwLowDateTime;

  rest  = (LONGLONG) utmp.QuadPart;
  rest -= 0x19db1ded53e8000LL;

  sec  = (lint) (rest / NSEC100);
  nsec = (lint) (rest % NSEC100) * 100_lint;

  return Time ( sec, nsec );
}


//-----------------------------------------------------------------------
//   getLocalTime
//-----------------------------------------------------------------------


inline void WinTime::getLocalTime

  ( struct tm&   result,
    const Time&  t )

{
  // Note that localtime() is thread safe in Windows.

  time_t      tt  = (time_t) t.sec ();
  struct tm*  tmp = localtime ( &tt );

  if ( ! tmp )
  {
    throw SystemException ( JEM_FUNC, "localtime() failed" );
  }

  result = *tmp;
}


//-----------------------------------------------------------------------
//   getCPUTime
//-----------------------------------------------------------------------


inline double WinTime::getCPUTime ()
{
  HANDLE          self = GetCurrentProcess ();

  ULARGE_INTEGER  utmp;
  FILETIME        start;
  FILETIME        endt;
  FILETIME        syst;
  FILETIME        usrt;
  double          cput;


  if ( ! GetProcessTimes( self, &start, &endt, &syst, &usrt ) )
  {
    throw SystemException ( JEM_FUNC, "GetProcessTimes() failed" );
  }

  utmp.HighPart = syst.dwHighDateTime;
  utmp.LowPart  = syst.dwLowDateTime;

  cput  = (double) utmp.QuadPart;

  utmp.HighPart = usrt.dwHighDateTime;
  utmp.LowPart  = usrt.dwLowDateTime;

  cput += (double) utmp.QuadPart;

  return (1.0e-7 * cput);
}


//-----------------------------------------------------------------------
//   getCPUTick
//-----------------------------------------------------------------------


inline double WinTime::getCPUTick ()
{
  return CPU_TICK_;
}


JEM_END_PACKAGE_BASE

#endif
