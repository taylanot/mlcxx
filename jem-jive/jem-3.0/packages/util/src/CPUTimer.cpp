
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
#include <jem/base/Once.h>
#include <jem/base/native/Time.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/util/CPUTimer.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class CPUTimer
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


CPUTimer::CPUTimer ()
{
  running_ = false;
  t0_      = 0.0;
  dt_      = 0.0;

  initOnce_ ();
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void CPUTimer::swap ( CPUTimer& rhs ) noexcept
{
  jem::swap ( running_, rhs.running_ );
  jem::swap ( t0_,      rhs.t0_      );
  jem::swap ( dt_,      rhs.dt_      );
}


//-----------------------------------------------------------------------
//   time
//-----------------------------------------------------------------------


Time CPUTimer::time () const
{
  return Time ( toDouble() );
}


//-----------------------------------------------------------------------
//   toDouble
//-----------------------------------------------------------------------


double CPUTimer::toDouble () const
{
  if ( running_ )
  {
    return (dt_ + (NativeTime::getCPUTime() - t0_));
  }
  else
  {
    return dt_;
  }
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void CPUTimer::start ()
{
  if ( ! running_ )
  {
    t0_      = NativeTime::getCPUTime ();
    running_ = true;
  }
}


//-----------------------------------------------------------------------
//   stop
//-----------------------------------------------------------------------


void CPUTimer::stop ()
{
  if ( running_ )
  {
    dt_     += (NativeTime::getCPUTime() - t0_);
    running_ = false;
  }
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void CPUTimer::reset()
{
  dt_ = 0.0;

  if ( running_ )
  {
    t0_ = NativeTime::getCPUTime ();
  }
}


//-----------------------------------------------------------------------
//   isRunning
//-----------------------------------------------------------------------


bool CPUTimer::isRunning () const noexcept
{
  return running_;
}


//-----------------------------------------------------------------------
//   resolution
//-----------------------------------------------------------------------


Time CPUTimer::resolution ()
{
  initOnce_ ();

  return Time ( NativeTime::getCPUTick() );
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void CPUTimer::readFrom ( io::DataInput& in )
{
  decode ( in, running_, t0_, dt_ );
}


void CPUTimer::writeTo ( io::DataOutput& out ) const
{
  encode ( out, running_, t0_, dt_ );
}


//-----------------------------------------------------------------------
//   initOnce_
//-----------------------------------------------------------------------


void CPUTimer::initOnce_ ()
{
  static Once once = JEM_ONCE_INITIALIZER;

  runOnce ( once, NativeTime::initCPUTime );
}


JEM_END_PACKAGE( util )
