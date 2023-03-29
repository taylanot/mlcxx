
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

#ifndef JEM_MP_MT_MESSAGE_H
#define JEM_MP_MT_MESSAGE_H

#include <cstring>
#include <jem/base/SpinLock.h>
#include <jem/util/ListNode.h>
#include <jem/mp/Status.h>
#include <jem/mp/Buffer.h>
#include "error.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


//-----------------------------------------------------------------------
//   class Message
//-----------------------------------------------------------------------


class Message : public jem::util::ListNode
{
 public:

  typedef void        (*NotifyHandler) ( void* );


  inline                Message      () noexcept;

  inline                Message

    ( const SendBuffer&   buf,
      int                 rnk,
      int                 tag )         noexcept;

  inline                Message

    ( const RecvBuffer&   buf,
      int                 rnk,
      int                 tag )         noexcept;

  inline void           copy

    ( Message&            rhs );

  inline void           notify       ();
  byte                  wait         ();
  void                  waitNoCancel ();


 public:

  SpinLock              spinlock;
  volatile bool         enqueued;

  NotifyHandler         notifyHandler;
  void*                 notifyData;

  int                   rank;
  int                   tag;

  Status                status;
  Buffer                buffer;


 private:

  static void           myNotifier_

    ( void*               data );

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline Message::Message () noexcept :

  enqueued      ( false ),
  notifyHandler ( nullptr ),
  notifyData    ( nullptr ),
  rank          ( -1 ),
  tag           ( -1 )

{}


inline Message::Message

  ( const SendBuffer&  buf,
    int                rnk,
    int                tag ) noexcept :

    enqueued      ( false ),
    notifyHandler ( nullptr ),
    notifyData    ( nullptr ),
    rank          ( rnk ),
    tag           ( tag ),
    buffer        ( buf )

{}


inline Message::Message

  ( const RecvBuffer&  buf,
    int                rnk,
    int                tag ) noexcept :

    enqueued      ( false ),
    notifyHandler ( nullptr ),
    notifyData    ( nullptr ),
    rank          ( rnk ),
    tag           ( tag ),
    buffer        ( buf )

{}


//-----------------------------------------------------------------------
//   copy
//-----------------------------------------------------------------------


inline void Message::copy ( Message& rhs )
{
  status.source = rhs.rank;
  status.tag    = rhs.tag;
  status.size   = rhs.buffer.size ();

  if ( buffer.type() != rhs.buffer.type() )
  {
    status.error = rhs.status.error = BUFFER_TYPE_ERROR;
  }
  else if ( buffer.size() < rhs.buffer.size() )
  {
    status.error = rhs.status.error = BUFFER_SIZE_ERROR;
  }
  else
  {
    std::memcpy (
      buffer    .addr(),
      rhs.buffer.addr(),
      (size_t) (sizeOf( buffer.type() ) * rhs.buffer.size())
    );
  }
}


//-----------------------------------------------------------------------
//   notify
//-----------------------------------------------------------------------


inline void Message::notify ()
{
  spinlock.lock ();

  if ( notifyHandler )
  {
    NotifyHandler  handler = notifyHandler;
    void*          data    = notifyData;

    notifyHandler = 0;
    notifyData    = 0;
    enqueued      = false;

    spinlock.unlock ();

    handler ( data );
  }
  else
  {
    enqueued = false;
    spinlock.unlock ();
  }
}


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )

#endif
