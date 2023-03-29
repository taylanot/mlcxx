
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
#include <jem/mp/params.h>
#include "Message.h"
#include "MessagePool.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


using jem::util::ListNode;


//=======================================================================
//   class MessagePool
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MessagePool::MessagePool ()
{}


MessagePool::~MessagePool ()
{}


//-----------------------------------------------------------------------
//   queueSend
//-----------------------------------------------------------------------


bool MessagePool::queueSend ( Message& smsg )
{
  Queue_&    mqueue = mqueues_[smsg.rank & QUEUE_MASK_];

  ListNode*  list;
  ListNode*  node;


  JEM_ASSERT_NOTHROW ( ! smsg.enqueued );

  list = & mqueue.recvList;

  spinlock_.lock ();

  // Check the recv list.

  for ( node = list->next(); node != list; node = node->next() )
  {
    Message&  rmsg = static_cast<Message&> ( *node );

    if ( rmsg.rank == smsg.rank &&
         (rmsg.tag == smsg.tag  || rmsg.tag == ANY_TAG) )
    {
      rmsg     .unlink ();
      spinlock_.unlock ();
      rmsg     .copy   ( smsg );
      rmsg     .notify ();

      return false;
    }
  }

  // Check the any-source recv list.

  list = & anyList_;

  for ( node = list->next(); node != list; node = node->next() )
  {
    Message&  rmsg = static_cast<Message&> ( *node );

    if ( rmsg.tag == smsg.tag || rmsg.tag == ANY_TAG )
    {
      rmsg     .unlink ();
      spinlock_.unlock ();
      rmsg     .copy   ( smsg );
      rmsg     .notify ();

      return false;
    }
  }

  // Add the message to the send list.

  smsg.enqueued = true;
  mqueue.sendList.pushFront ( & smsg );
  spinlock_.unlock ();

  return true;
}


//-----------------------------------------------------------------------
//   queueRecv
//-----------------------------------------------------------------------


bool MessagePool::queueRecv ( Message& rmsg )
{
  Queue_&    mqueue = mqueues_[rmsg.rank & QUEUE_MASK_];

  ListNode*  list;
  ListNode*  node;


  JEM_ASSERT_NOTHROW ( ! rmsg.enqueued );

  list = & mqueue.sendList;

  spinlock_.lock ();

  // Check the send list.

  for ( node = list->next(); node != list; node = node->next() )
  {
    Message&  smsg = static_cast<Message&> ( *node );

    if ( smsg.rank == rmsg.rank &&
         (rmsg.tag == smsg.tag || rmsg.tag == ANY_TAG) )
    {
      smsg     .unlink ();
      spinlock_.unlock ();
      rmsg     .copy   ( smsg );
      smsg     .notify ();

      return false;
    }
  }

  // Add the message to the recv list.

  rmsg.enqueued = true;
  mqueue.recvList.pushFront ( & rmsg );
  spinlock_.unlock ();

  return true;
}


//-----------------------------------------------------------------------
//   queueAnyRecv
//-----------------------------------------------------------------------


bool MessagePool::queueAnyRecv ( Message& rmsg )
{
  Queue_*    mqueue;
  ListNode*  list;
  ListNode*  node;
  int        i;


  JEM_ASSERT_NOTHROW ( ! rmsg.enqueued );

  spinlock_.lock ();

  // Check all send lists.

  for ( i = 0; i < QUEUE_COUNT_; i++ )
  {
    mqueue = mqueues_ + i;
    list   = & mqueue->sendList;

    for ( node = list->next(); node != list; node = node->next() )
    {
      Message&  smsg = static_cast<Message&> ( *node );

      if ( rmsg.tag == smsg.tag || rmsg.tag == ANY_TAG )
      {
        smsg     .unlink ();
        spinlock_.unlock ();
        rmsg     .copy   ( smsg );
        smsg     .notify ();

        return false;
      }
    }
  }

  // Add the message to the any-source recv list.

  rmsg.enqueued = true;
  anyList_ .pushFront ( & rmsg );
  spinlock_.unlock    ();

  return true;
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void MessagePool::cancel ( Message& msg )
{
  ListNode*  node     = & msg;
  bool       dequeued = false;


  spinlock_.lock ();

  if ( node->next() != node )
  {
    node->unlink ();
    dequeued = true;
  }

  spinlock_.unlock ();

  if ( dequeued )
  {
    msg.enqueued = false;
  }
  else
  {
    msg.waitNoCancel ();
  }
}


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )
