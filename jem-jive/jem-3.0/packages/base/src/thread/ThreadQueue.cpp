
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
#include <jem/base/thread/ThreadQueue.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class ThreadNode
//=======================================================================

//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


byte ThreadNode::wait

  ( SpinLock&    mutex,
    const Time*  timeout )

{
  const lint  n = ThreadLib::SPIN_COUNT;

  Monitor*    mon;
  byte        sig;
  lint        i;


  if ( n > 0 && ! timeout )
  {
    mutex.unlock ();

    JEM_SPIN_WHILE ( ! notified_, i, n );

    mutex.lock   ();
  }

  if ( notified_ )
  {
    notified_ = false;
    return SigThread::NOTIFY;
  }

  mon = monitor_ = ThreadLib::getMonitor ();

  mutex.unlock ();
  mon ->lock   ();

  if ( timeout )
  {
    sig = mon->wait ( SigThread::ALL, *timeout );
  }
  else
  {
    sig = mon->wait ( SigThread::ALL );
  }

  mon->unlock ();
  mutex.lock  ();

  if ( ! (sig & SigThread::NOTIFY) )
  {
    Monitor*  tmp;

    tmp      = monitor_;
    monitor_ = nullptr;

    if ( ! tmp )
    {
      mutex.unlock ();
      mon ->lock   ();

      sig = (byte) (sig | mon->wait( SigThread::NOTIFY ));

      mon->unlock  ();
      mutex.lock   ();
    }
  }

  notified_ = false;

  return sig;
}


//-----------------------------------------------------------------------
//   waitNoCancel
//-----------------------------------------------------------------------


void ThreadNode::waitNoCancel ( SpinLock& mutex )
{
  const lint  n = ThreadLib::SPIN_COUNT;

  Monitor*    mon;
  lint        i;


  if ( n > 0 )
  {
    mutex.unlock ();

    JEM_SPIN_WHILE ( ! notified_, i, n );

    mutex.lock   ();
  }

  if ( notified_ )
  {
    notified_ = false;
    return;
  }

  mon = monitor_ = ThreadLib::getMonitor ();

  mutex.unlock ();
  mon ->lock   ();
  mon ->wait   ( SigThread::NOTIFY );
  mon ->unlock ();
  mutex.lock   ();

  notified_ = false;
}


//=======================================================================
//   class ThreadQueue
//=======================================================================

//-----------------------------------------------------------------------
//   remove
//-----------------------------------------------------------------------


bool ThreadQueue::remove ( Node*& queue, Node* node ) noexcept
{
  if      ( ! queue )
  {
    return false;
  }
  else if ( queue == node )
  {
    // The node to be removed is at the head of the queue.

    if ( node->next_ == node )
    {
      // The queue only contains the node to be removed.

      queue = nullptr;
    }
    else
    {
      // Adjust the last node in the queue and remove the head.

      Node*  helper = node->next_;

      while ( helper->next_ != node )
      {
        helper = helper->next_;
      }

      queue = helper->next_ = node->next_;
      node->next_           = node;
    }

    return true;
  }
  else
  {
    // The node to be removed might be somewhere
    // further in the queue.

    Node*  helper = queue;

    while ( helper->next_ != queue )
    {
      if ( helper->next_ == node )
      {
        helper->next_ = node->next_;
        node  ->next_ = node;

        return true;
      }

      helper = helper->next_;
    }

    return false;
  }
}


JEM_END_PACKAGE_BASE
