
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

#ifndef JEM_BASE_NATIVE_POSIXTIME_H
#define JEM_BASE_NATIVE_POSIXTIME_H

#include <ctime>
#include <jem/base/Time.h>
#include <jem/base/SystemException.h>

extern "C"
{
  #include <sys/time.h>
  #include <sys/times.h>
}


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class PosixTime
//-----------------------------------------------------------------------


class PosixTime
{
 public:

  static inline Time    getTime       ();

  static inline void    getLocalTime

    ( struct tm&          result,
      const Time&         t );

#if defined(JEM_OS_POSIX_REALTIME)

  static inline Time    getClockTick  ();

#endif

  static inline double  getCPUTime    ();
  static inline double  getCPUTick    ();
  static void           initCPUTime   ();


 private:

  static double         TICKS_PER_SEC_;

};


typedef PosixTime       NativeTime;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getTime
//-----------------------------------------------------------------------


inline Time PosixTime::getTime ()
{
  lint  sec, nsec;

#if defined(JEM_OS_POSIX_REALTIME)

  struct timespec  ts;

  if ( clock_gettime( CLOCK_REALTIME, &ts ) == 0 )
  {
    sec  = (lint) ts.tv_sec;
    nsec = ts.tv_nsec;
  }
  else
  {
    throw SystemException ( JEM_FUNC, "clock_gettime() failed" );
  }

#else

  struct timeval  tv;

  if ( gettimeofday( &tv, 0 ) == 0 )
  {
    sec  = (lint) tv.tv_sec;
    nsec = 1000_lint * tv.tv_usec;
  }
  else
  {
    throw SystemException ( JEM_FUNC, "gettimeofday() failed" );
  }

#endif

  return Time ( sec, nsec );
}


//-----------------------------------------------------------------------
//   getLocalTime
//-----------------------------------------------------------------------


inline void PosixTime::getLocalTime

  ( struct tm&   result,
    const Time&  t )

{
  time_t  tt = (time_t) t.sec ();

#if defined(JEM_OS_UNIX98)

  localtime_r ( &tt, &result );

#else

  struct tm*  tmp;

  tmp    = localtime ( &tt );
  result = *tmp;

#endif
}


//-----------------------------------------------------------------------
//   getClockTick
//-----------------------------------------------------------------------


#if defined(JEM_OS_POSIX_REALTIME)

inline Time PosixTime::getClockTick ()
{
  struct timespec  ts;
  lint             sec, nsec;


  if ( clock_getres( CLOCK_REALTIME, &ts ) == 0 )
  {
    sec  = (lint) ts.tv_sec;
    nsec = (lint) ts.tv_nsec;
  }
  else
  {
    throw SystemException ( JEM_FUNC, "clock_getres() failed" );
  }

  return Time ( sec, nsec );
}

#endif


//-----------------------------------------------------------------------
//   getCPUTime
//-----------------------------------------------------------------------


inline double PosixTime::getCPUTime ()
{
  struct tms  ts;

  if ( times( &ts ) == (clock_t) -1 )
  {
    throw SystemException ( JEM_FUNC, "times() failed" );
  }

  double  ticks = (double) ts.tms_utime + (double) ts.tms_stime;

  return ticks * TICKS_PER_SEC_;
}


//-----------------------------------------------------------------------
//   getCPUTick
//-----------------------------------------------------------------------


inline double PosixTime::getCPUTick ()
{
  return TICKS_PER_SEC_;
}


JEM_END_PACKAGE_BASE

#endif
