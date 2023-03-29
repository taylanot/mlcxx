
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
#include <jem/base/WeakRef.h>
#include <jem/gl/CallbackQueue.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class CallbackQueue::Entry_
//=======================================================================


class CallbackQueue::Entry_
{
 public:

  inline                  Entry_      ();

  inline                  Entry_

    ( Callback              cb,
      const Ref<Object>&    obj,
      const Time&           tm );

  inline                 ~Entry_      ();

  inline bool             isGarbage   () const;
  inline void             clear       ();

  inline void             init

    ( Callback              cb,
      const Ref<Object>&    obj,
      const Time&           tm );

  inline void             insert

    ( Entry_*               e );

  inline void             unlink      ();


 public:

  Callback                callback;
  WeakRef<Object>         object;
  Time                    deadline;

  bool                    hasObject;
  bool                    isActive;

  Entry_*                 next;
  Entry_*                 prev;

};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline CallbackQueue::Entry_::Entry_ ()
{
  callback  = 0;
  deadline  = Time ( maxOf<lint>() );
  hasObject = false;
  isActive  = false;
  next      = this;
  prev      = this;
}


inline CallbackQueue::Entry_::Entry_

  ( Callback            cb,
    const Ref<Object>&  obj,
    const Time&         tm ) :

    callback (  cb ),
    object   ( obj ),
    deadline (  tm )

{
  hasObject = (obj != nullptr);
  isActive  = true;
  next      = this;
  prev      = this;
}


CallbackQueue::Entry_::~Entry_ ()
{
  unlink ();
}


//-----------------------------------------------------------------------
//   isGarbage
//-----------------------------------------------------------------------


inline bool CallbackQueue::Entry_::isGarbage () const
{
  if ( hasObject )
  {
    return (object.get() == nullptr);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void CallbackQueue::Entry_::clear ()
{
  object   = nullptr;
  callback = 0;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


inline void CallbackQueue::Entry_::init

  ( Callback            cb,
    const Ref<Object>&  obj,
    const Time&         tm )

{
  callback  = cb;
  object    = obj;
  deadline  = tm;
  hasObject = (obj != nullptr);
  isActive  = true;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


inline void CallbackQueue::Entry_::insert ( Entry_* e )
{
  e->next    = this;
  e->prev    = prev;
  prev->next = e;
  prev       = e;
}


//-----------------------------------------------------------------------
//   unlink
//-----------------------------------------------------------------------


inline void CallbackQueue::Entry_::unlink ()
{
  next->prev  = prev;
  prev->next  = next;
  next = prev = this;
}


//=======================================================================
//   class CallbackQueue::Invocation_
//=======================================================================


class CallbackQueue::Invocation_
{
 public:

  explicit inline         Invocation_

    ( CallbackQueue*        queue );

                         ~Invocation_ ();


 private:

  CallbackQueue*          cbQueue_;
  bool                    invoking_;

};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline CallbackQueue::Invocation_::Invocation_

  ( CallbackQueue*  queue )

{
  cbQueue_         = queue;
  invoking_        = queue->invoking_;
  queue->invoking_ = true;
}


CallbackQueue::Invocation_::~Invocation_ ()
{
  if ( ! invoking_ )
  {
    Entry_*  tail = cbQueue_->tail_;
    Entry_*  e;

    for ( e = tail->next; e != tail; e = e->next )
    {
      e->isActive = true;
    }
  }

  cbQueue_->invoking_ = invoking_;
}


//=======================================================================
//   class CallbackQueue
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


CallbackQueue::CallbackQueue ()
{
  tail_      = new Entry_ ();
  cache_     = nullptr;
  queueSize_ = 0;
  cacheSize_ = 0;
  invoking_  = false;
}


CallbackQueue::~CallbackQueue ()
{
  while ( tail_->next != tail_ )
  {
    delete tail_->next;
  }

  delete tail_;

  tail_ = nullptr;

  if ( cache_ )
  {
    while ( cache_->next != cache_ )
    {
      delete cache_->next;
    }

    delete cache_;

    cache_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void CallbackQueue::clear ()
{
  while ( tail_->next != tail_ )
  {
    delete tail_->next;
  }

  queueSize_ = 0;
}


//-----------------------------------------------------------------------
//   timeLeft
//-----------------------------------------------------------------------


Time CallbackQueue::timeLeft ( const Time& now ) const
{
  Entry_*  e;

  for ( e = tail_->next; e != tail_; e = e->next )
  {
    if ( e->isActive )
    {
      return (e->deadline - now);
    }
  }

  return Time ( maxOf<lint>() );
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void CallbackQueue::insert

  ( Callback            cb,
    const Ref<Object>&  obj,
    const Time&         tm )

{
  JEM_PRECHECK ( cb );

  Entry_*  e;


  if ( cache_ )
  {
    e = cache_;

    if ( (--cacheSize_) == 0 )
    {
      cache_ = nullptr;
    }
    else
    {
      cache_ = e->next;
      e->unlink ();
    }

    e->init ( cb, obj, tm );
  }
  else
  {
    e = new Entry_ ( cb, obj, tm );
  }

  if ( invoking_ )
  {
    e->isActive = false;
  }

  insert_ ( e );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


idx_t CallbackQueue::erase ( Callback cb )
{
  Entry_*  e = tail_->next;
  idx_t    k = 0;


  while ( e != tail_ )
  {
    // Collect garbage

    if ( e->isGarbage() )
    {
      e = erase_ ( e );
      continue;
    }

    if ( e->callback == cb )
    {
      e = erase_ ( e );
      k++;
    }
    else
    {
      e = e->next;
    }
  }

  return k;
}


bool CallbackQueue::erase

  ( Callback            cb,
    const Ref<Object>&  obj )

{
  Entry_*  e = tail_->next;
  bool     r = false;


  while ( e != tail_ )
  {
    // Collect garbage

    if ( e->isGarbage() )
    {
      e = erase_ ( e );
      continue;
    }

    if ( e->callback == cb && e->object.get() == obj )
    {
      e = erase_ ( e );
      r = true;
    }
    else
    {
      e = e->next;
    }
  }

  return r;
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void CallbackQueue::update

  ( Callback            cb,
    const Ref<Object>&  obj,
    const Time&         tm )

{
  Entry_*  e = tail_->next;


  while ( e != tail_ )
  {
    // Collect garbage

    if ( e->isGarbage() )
    {
      e = erase_ ( e );
      continue;
    }

    if ( e->callback == cb && e->object.get() == obj )
    {
      e->unlink ();

      e->deadline = tm;

      insert_ ( e );

      return;
    }

    e = e->next;
  }

  insert ( cb, obj, tm );
}


//-----------------------------------------------------------------------
//   invoke
//-----------------------------------------------------------------------


void CallbackQueue::invoke

  ( Time&           tleft,
    RenderContext&  rtx )

{
  if ( tail_->next == tail_ )
  {
    tleft = Time ( maxOf<lint>() );
    return;
  }
  else
  {
    Time     now = Time::now ();
    Entry_*  e;

    invoke_ ( now, rtx );

    for ( e = tail_->next; e != tail_; e = e->next )
    {
      if ( e->isActive )
      {
        tleft = (e->deadline - now);
        return;
      }
    }

    tleft = Time ( maxOf<lint>() );
  }
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


idx_t CallbackQueue::contains ( Callback cb ) const
{
  Entry_*  e = tail_->next;
  idx_t    k = 0;


  while ( e != tail_ )
  {
    if ( e->isGarbage() )
    {
      continue;
    }

    if ( e->callback == cb )
    {
      k++;
    }

    e = e->next;
  }

  return k;
}


bool CallbackQueue::contains

  ( Callback            cb,
    const Ref<Object>&  obj ) const

{
  Entry_*  e = tail_->next;
  bool     r = false;


  while ( e != tail_ )
  {
    if ( e->isGarbage() )
    {
      continue;
    }

    if ( e->callback == cb && e->object.get() == obj )
    {
      r = true;
    }

    e = e->next;
  }

  return r;
}


//-----------------------------------------------------------------------
//   insert_
//-----------------------------------------------------------------------


void CallbackQueue::insert_ ( Entry_* e )
{
  Entry_*  n;


  n = tail_->next;

  while ( n != tail_ )
  {
    // Collect garbage

    if ( n->isGarbage() )
    {
      n = erase_ ( n );
      continue;
    }

    if ( n->deadline > e->deadline )
    {
      break;
    }

    n = n->next;
  }

  n->insert ( e );

  queueSize_++;
}


//-----------------------------------------------------------------------
//   erase_
//-----------------------------------------------------------------------


CallbackQueue::Entry_* CallbackQueue::erase_ ( Entry_* e )
{
  Entry_*  n = e->next;

  if ( cacheSize_ > 10 )
  {
    delete e;
  }
  else
  {
    e->clear  ();
    e->unlink ();

    if ( cache_ )
    {
      cache_->insert ( e );
    }
    else
    {
      cache_ = e;
    }

    cacheSize_++;
  }

  queueSize_--;

  return n;
}


//-----------------------------------------------------------------------
//   invoke_
//-----------------------------------------------------------------------


void CallbackQueue::invoke_

  ( Time&           now,
    RenderContext&  rtx )

{
  Invocation_  inv ( this );

  Entry_*      e;


  while ( true )
  {
    for ( e = tail_->next; e != tail_; e = e->next )
    {
      if ( e->isActive )
      {
        break;
      }
    }

    if ( e == tail_ )
    {
      break;
    }

    if ( e->deadline > now )
    {
      while ( e != tail_ && e->isGarbage() )
      {
        e = erase_ ( e );
      }

      break;
    }
    else
    {
      Ref<Object>  obj = e->object.get ();
      Callback     cb  = e->callback;

      if ( obj == nullptr && e->hasObject )
      {
        e = erase_ ( e );
      }
      else
      {
        erase_ ( e );

        cb ( rtx, obj );

        now = Time::now ();
      }
    }
  }
}


JEM_END_PACKAGE( gl )
