
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
#include <jem/base/limits.h>
#include <jem/base/Once.h>
#include <jem/base/CString.h>
#include <jem/base/Time.h>
#include <jem/base/native/Time.h>


JEM_BEGIN_PACKAGE_BASE



//=======================================================================
//   class Time
//=======================================================================

//-----------------------------------------------------------------------
//   static data members
//-----------------------------------------------------------------------


const lint  Time::NANOS_PER_SEC_ =  1000000000;

lint        Time::tick_          = -1;
lint        Time::nanoTick_      = -1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Time::Time ( lint sec, lint nsec ) :

  sec_ (  sec ),
  nsec_( nsec )

{
  normalize_ ();
}


Time::Time ( double sec )
{
  sec_  = (lint)   sec;
  sec  -= (double) sec_;
  nsec_ = (lint)  (sec * (double) NANOS_PER_SEC_);

  normalize_ ();
}


//-----------------------------------------------------------------------
//   computed assignment operators
//-----------------------------------------------------------------------


void Time::operator += ( const Time& rhs )
{
  sec_  += rhs.sec_;
  nsec_ += rhs.nsec_;

  if ( nsec_ )
  {
    normalize_ ();
  }
}


void Time::operator -= ( const Time& rhs )
{
  sec_  -= rhs.sec_;
  nsec_ -= rhs.nsec_;

  if ( nsec_ )
  {
    normalize_ ();
  }
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Time::toString () const
{
  const int   N = 256;

  struct tm   t;
  char        buf[N];
  int         n;

  NativeTime::getLocalTime ( t, *this );

  n = (int) strftime ( buf, N, "%a %b %d %H:%M:%S %Z %Y", & t );

  return String ( buf, n );
}


String Time::toString ( const String& format ) const
{
  const int   N = 256;

  struct tm   t;
  char        buf[N];
  int         n;

  NativeTime::getLocalTime ( t, *this );

  n = (int) strftime ( buf, N, makeCString( format ), & t );

  return String ( buf, n );
}


//-----------------------------------------------------------------------
//   now
//-----------------------------------------------------------------------


Time Time::now ()
{
  return NativeTime::getTime ();
}


//-----------------------------------------------------------------------
//   resolution
//-----------------------------------------------------------------------


Time Time::resolution ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, initResolution_ );

  return Time ( tick_, nanoTick_ );
}


//-----------------------------------------------------------------------
//   normalize_
//-----------------------------------------------------------------------


void Time::normalize_ ()
{
  int  sign;

  if ( nsec_ < 0_lint )
  {
    sign  = -1;
    nsec_ = -nsec_;
  }
  else
  {
    sign = 1;
  }

  sec_  = sec_ + sign * (nsec_ / NANOS_PER_SEC_);
  nsec_ = nsec_ % NANOS_PER_SEC_;

  if ( sign < 0 )
  {
    sec_  = sec_ - 1_lint;
    nsec_ = NANOS_PER_SEC_ - nsec_;
  }
}


//-----------------------------------------------------------------------
//   initResolution_
//-----------------------------------------------------------------------


void Time::initResolution_ ()
{
  Time  tick;

#if defined(JEM_OS_POSIX_REALTIME)

  tick = PosixTime::getClockTick ();

#else

  Time  res;
  Time  t, delta;
  bool  success;
  int   i, j;


  res     = Time ( maxOf<lint>(), 0 );
  success = false;

  // Probe for the clock resolution

  for ( i = 0; i < 10; i++ )
  {
    t = NativeTime::getTime ();

    for ( j = 10000; j > 0; j-- )
    {
      delta = NativeTime::getTime() - t;

      if ( delta > zero() )
      {
        break;
      }
    }

    if ( j > 0 && delta < res )
    {
      res     = delta;
      success = true;
    }
  }

  // If the clock resolution could not be determined, a default
  // resolution of one second is assumed.

  if ( success )
  {
    tick = res;
  }
  else
  {
    tick = Time ( 1_lint );
  }

#endif

  tick_     = tick.sec  ();
  nanoTick_ = tick.nsec ();
}




//=======================================================================
//   Non-member functions for class Time
//=======================================================================

//-----------------------------------------------------------------------
//   arithmetic operators
//-----------------------------------------------------------------------


Time            operator -

  ( const Time&   rhs )

{
  if ( rhs.nsec() )
  {
    return Time ( -rhs.sec(), -rhs.nsec() );
  }
  else
  {
    return Time ( -rhs.sec() );
  }
}


Time            operator -

  ( const Time&   lhs,
    const Time&   rhs )

{
  Time  t ( lhs );

  t -= rhs;

  return t;
}


Time            operator +

  ( const Time&   lhs,
    const Time&   rhs )

{
  Time  t ( lhs );

  t += rhs;

  return t;
}


//-----------------------------------------------------------------------
//   comparison operators
//-----------------------------------------------------------------------


bool            operator <

  ( const Time&   lhs,
    const Time&   rhs ) noexcept

{
  if      ( lhs.sec() < rhs.sec() )
  {
    return true;
  }
  else if ( lhs.sec() == rhs.sec() )
  {
    return (lhs.nsec() < rhs.nsec());
  }
  else
  {
    return false;
  }
}


JEM_END_PACKAGE_BASE
