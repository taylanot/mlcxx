
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
#include <jem/gl/EventQueue.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class EventQueue
//=======================================================================


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


EventQueue::EventQueue ()
{
  tail_ = new GfxEvent ();
  size_ = 0;
}


EventQueue::EventQueue ( EventQueue&& rhs ) : EventQueue ()
{
  swap ( rhs );
}


EventQueue::~EventQueue ()
{
  clear ();
  delete tail_;

  tail_ = 0;
}


//-----------------------------------------------------------------------
//   push
//-----------------------------------------------------------------------


idx_t EventQueue::push ( GfxEvent* ev )
{
  JEM_ASSERT ( ev );

  GfxEvent*  last = tail_->prev ();


  switch ( ev->type )
  {
  case GfxEvent::MOTION:

    if ( last->type == GfxEvent::MOTION )
    {
      // Multiple successive motion events typically indicate that
      // the system is not being able to process those events fast
      // enough. The previous motion event is dropped to give
      // the system enough time to suck in some fresh air :-)

      MotionEvent&  a = static_cast<MotionEvent&> ( *ev   );
      MotionEvent&  b = static_cast<MotionEvent&> ( *last );

      b.xPos = a.xPos;
      b.yPos = a.yPos;

      delete ev;
      return size_;
    }

    break;

  case GfxEvent::REDRAW:

    if ( last->type == GfxEvent::REDRAW )
    {
      RedrawEvent&  a = static_cast<RedrawEvent&> ( *ev   );
      RedrawEvent&  b = static_cast<RedrawEvent&> ( *last );

      if ( a.scene == b.scene )
      {
        b.flags = a.flags;

        delete ev;
        return size_;
      }
    }

    break;

  case GfxEvent::RESIZE:

    {
      GfxEvent*  e = tail_->next ();
      GfxEvent*  n;

      // Delete all current events except an INIT event

      while ( e != tail_ )
      {
        n = e->next ();

        if ( e->type != GfxEvent::INIT )
        {
          delete e;
          size_--;
        }

        e = n;
      }
    }

    break;

  default:

    ; // Nothing to do
  }

  tail_->insert ( ev );

  if ( (++size_) > MAX_EVENTS )
  {
    prune_ ();
  }

  return size_;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void EventQueue::clear ()
{
  while ( size_ )
  {
    pop ();
  }
}


//-----------------------------------------------------------------------
//   prune_
//-----------------------------------------------------------------------


void EventQueue::prune_ ()
{
  GfxEvent*  e;
  GfxEvent*  n;


  e = tail_->next ();

  // First pass: preserve RESIZE and MOUSE events.

  while ( size_ > MAX_EVENTS && e != tail_ )
  {
    n = e->next ();

    if ( e->type != GfxEvent::RESIZE &&
         e->type != GfxEvent::MOUSE )
    {
      delete e;
      size_--;
    }

    e = n;
  }

  // Second pass: only preserve RESIZE events.

  while ( size_ > MAX_EVENTS && e != tail_ )
  {
    n = e->next ();

    if ( e->type != GfxEvent::RESIZE )
    {
      delete e;
      size_--;
    }

    e = n;
  }
}


JEM_END_PACKAGE( gl )
