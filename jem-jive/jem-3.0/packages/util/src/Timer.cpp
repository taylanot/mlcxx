
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
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/util/Timer.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class Timer
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


Timer::Timer () noexcept :

  running_ ( false ),
  t0_      ( 0_lint ),
  dt_      ( 0_lint )

{}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void Timer::swap ( Timer& rhs ) noexcept
{
  jem::swap ( running_, rhs.running_ );
  jem::swap (      t0_,      rhs.t0_ );
  jem::swap (      dt_,      rhs.dt_ );
}


//-----------------------------------------------------------------------
//   time
//-----------------------------------------------------------------------


Time Timer::time () const
{
  if ( running_ )
  {
    return ( dt_ + (Time::now() - t0_) );
  }
  else
  {
    return dt_;
  }
}


//-----------------------------------------------------------------------
//   toDouble
//-----------------------------------------------------------------------


double Timer::toDouble () const
{
  return time().toDouble ();
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void Timer::start ()
{
  if ( ! running_ )
  {
    t0_      = Time::now ();
    running_ = true;
  }
}


//-----------------------------------------------------------------------
//   stop
//-----------------------------------------------------------------------


void Timer::stop ()
{
  if ( running_ )
  {
    dt_      += (Time::now() - t0_);
    running_  = false;
  }
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void Timer::reset ()
{
  dt_ = 0_sec;

  if ( running_ )
  {
    t0_ = Time::now ();
  }
}


//-----------------------------------------------------------------------
//   isRunning
//-----------------------------------------------------------------------


bool Timer::isRunning () const noexcept
{
  return running_;
}


//-----------------------------------------------------------------------
//   resolution
//-----------------------------------------------------------------------


Time Timer::resolution ()
{
  return Time::resolution ();
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void Timer::readFrom ( io::DataInput& in )
{
  decode ( in, running_, t0_, dt_ );
}


void Timer::writeTo ( io::DataOutput& out ) const
{
  encode ( out, running_, t0_, dt_ );
}


JEM_END_PACKAGE( util )
