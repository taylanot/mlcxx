
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


#include <jem/base/thread/spin.h>
#include <jem/base/thread/ThreadLib.h>
#include <jem/base/thread/ThreadMonitor.h>
#include "Rendezvous.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


//=======================================================================
//   class Rendezvous
//=======================================================================

//-----------------------------------------------------------------------
//   notify
//-----------------------------------------------------------------------


void Rendezvous::notify ( void* data )
{
  Rendezvous&     rz  = * ( static_cast<Rendezvous*>( data ) );
  ThreadMonitor*  mon = 0;


  rz.spinlock_.lock ();

  rz.count_++;

  if ( rz.count_ >= rz.waitCount_ )
  {
    mon         = rz.monitor_;
    rz.monitor_ = 0;
  }

  rz.spinlock_.unlock ();

  if ( mon )
  {
    mon->lock   ();
    mon->notify ( SigThread::NOTIFY );
    mon->unlock ();
  }
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


byte Rendezvous::wait

  ( idx_t  waitCount,
    lint   spinCount )

{
  ThreadMonitor*  mon;
  byte            sig;
  lint            i;


  JEM_SPIN_WHILE ( count_ < waitCount, i, spinCount );

  spinlock_.lock ();

  if ( count_ >= waitCount )
  {
    spinlock_.unlock ();
    return SigThread::NOTIFY;
  }

  monitor_ = mon = ThreadLib::getMonitor ();
  waitCount_     = waitCount;

  spinlock_.unlock ();

  mon->lock   ();
  sig = mon->wait ( SigThread::ALL );
  mon->unlock ();

  if ( ! (sig & SigThread::NOTIFY) )
  {
    ThreadMonitor*  tmp;

    spinlock_.lock ();

    tmp      = monitor_;
    monitor_ = 0;

    spinlock_.unlock ();

    if ( ! tmp )
    {
      mon->lock    ();
      sig = (byte) (sig | mon->wait( SigThread::NOTIFY ));
      mon->unlock  ();
    }
  }

  return sig;
}


//-----------------------------------------------------------------------
//   waitNoCancel
//-----------------------------------------------------------------------


void Rendezvous::waitNoCancel

  ( idx_t  waitCount,
    lint   spinCount )

{
  ThreadMonitor*  mon;
  lint            i;


  JEM_SPIN_WHILE ( count_ < waitCount, i, spinCount );

  spinlock_.lock ();

  if ( count_ >= waitCount )
  {
    spinlock_.unlock ();
    return;
  }

  monitor_ = mon = ThreadLib::getMonitor ();
  waitCount_     = waitCount;

  spinlock_.unlock ();

  mon->lock   ();
  mon->wait   ( SigThread::NOTIFY );
  mon->unlock ();
}


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )
