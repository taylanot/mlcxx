
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

#ifndef JEM_MP_MT_MESSAGEPOOL_H
#define JEM_MP_MT_MESSAGEPOOL_H

#include <jem/base/Object.h>
#include <jem/base/SpinLock.h>
#include <jem/util/ListNode.h>


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


class Message;


//-----------------------------------------------------------------------
//   class MessagePool
//-----------------------------------------------------------------------


class MessagePool : public Object
{
 public:

  typedef MessagePool   Self;
  typedef Object        Super;


                        MessagePool  ();

  bool                  queueSend

    ( Message&            msg );

  bool                  queueRecv

    ( Message&            msg );

  bool                  queueAnyRecv

    ( Message&            msg );

  void                  cancel

    ( Message&            msg );


 protected:

  virtual              ~MessagePool ();


 private:

  static const int      QUEUE_COUNT_ = 8;
  static const int      QUEUE_MASK_  = QUEUE_COUNT_ - 1;


  class                 Queue_
  {
   public:

    util::ListNode        sendList;
    util::ListNode        recvList;

  };


  Queue_                mqueues_ [ QUEUE_COUNT_ ];
  util::ListNode        anyList_;
  SpinLock              spinlock_;

};


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )

#endif
