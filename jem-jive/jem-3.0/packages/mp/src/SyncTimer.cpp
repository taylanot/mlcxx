
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/mp/SyncTimer.h>


JEM_BEGIN_PACKAGE( mp )


//=======================================================================
//   class SyncTimer
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SyncTimer::SyncTimer

  ( const Ref<Context>&  ctx,
    double               dt ) :

    context_ ( ctx )

{
  JEM_PRECHECK ( ctx );

  interval_   = dt;
  mustSync_   = (ctx->size() > 1);
  syncCount_  = 1;
  icount_     = 0;
}


SyncTimer::~SyncTimer ()
{}


//-----------------------------------------------------------------------
//   check
//-----------------------------------------------------------------------


bool SyncTimer::check ()
{
  bool  result = false;

  if ( ! timer_.isRunning() )
  {
    return result;
  }

  if ( mustSync_ )
  {
    icount_++;

    if ( icount_ >= syncCount_ )
    {
      result = syncAndCheck_ ();
    }
  }
  else
  {
    if ( timer_.toDouble() > interval_ )
    {
      timer_.reset ();

      result = true;
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SyncTimer::start ()
{
  timer_.start ();
}


//-----------------------------------------------------------------------
//   stop
//-----------------------------------------------------------------------


void SyncTimer::stop ()
{
  timer_.stop ();
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void SyncTimer::reset ()
{
  timer_.reset ();
  icount_ = 0;
}


//-----------------------------------------------------------------------
//   setInterval
//-----------------------------------------------------------------------


void SyncTimer::setInterval ( double dt )
{
  interval_  = dt;
  icount_    = 0;
  syncCount_ = 1;

  timer_.reset ();
}


//-----------------------------------------------------------------------
//   syncAndCheck_
//-----------------------------------------------------------------------


bool SyncTimer::syncAndCheck_ ()
{
  double  dt;
  double  n;

  int     ibuf[2];


  if ( context_->myRank() == 0 )
  {
    dt = timer_.toDouble ();

    if ( dt <= 0.0 )
    {
      ibuf[0] = maxOf ( ibuf[0] );
    }
    else
    {
      if ( interval_ > 100.0 )
      {
        n = icount_ * (100.0 / dt);
      }
      else
      {
        n = icount_ * (interval_ / dt);
      }

      if ( n > (double) maxOf( ibuf[0] ) )
      {
        ibuf[0] = maxOf ( ibuf[0] );
      }
      else
      {
        ibuf[0] = (int) n;
      }
    }

    if ( dt >= interval_ )
    {
      ibuf[1] = 1;
    }
    else
    {
      ibuf[1] = 0;
    }

    context_->broadcast ( SendBuffer( ibuf, 2 ) );
  }
  else
  {
    context_->broadcast ( RecvBuffer( ibuf, 2 ), 0 );
  }

  if      ( ibuf[0] / 2 > syncCount_ )
  {
    syncCount_ *= 2;
  }
  else if ( ibuf[0] < syncCount_ / 4 )
  {
    syncCount_ /= 4;
  }
  else
  {
    syncCount_  = ibuf[0];
  }

  if ( syncCount_ <= 0 )
  {
    syncCount_ = 1;
  }

  if ( ibuf[1] )
  {
    timer_.reset ();
    icount_ = 0;
  }

  return (ibuf[1] != 0);
}


JEM_END_PACKAGE( mp )
