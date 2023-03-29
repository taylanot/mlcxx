
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

#ifndef JEM_GL_EVENTQUEUE_H
#define JEM_GL_EVENTQUEUE_H

#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/gl/Event.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class EventQueue
//-----------------------------------------------------------------------


class EventQueue
{
 public:

  static const int          MAX_EVENTS  = 256;


                            EventQueue  ();

                            EventQueue

    ( EventQueue&&            rhs );

                           ~EventQueue  ();

  inline EventQueue&        operator =

    ( EventQueue&&            rhs )              noexcept;

  inline void               swap

    ( EventQueue&             rhs )              noexcept;

  inline idx_t              size        () const noexcept;
  inline GfxEvent*          peek        () const noexcept;
  inline EventPtr           pop         ();

  idx_t                     push

    ( GfxEvent*               ev );

  void                      clear       ();


 private:

                            EventQueue

    ( const EventQueue&       rhs );

  EventQueue&               operator =

    ( const EventQueue&       rhs );

  void                      prune_      ();


 private:

  GfxEvent*                 tail_;
  idx_t                     size_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   move operator
//-----------------------------------------------------------------------


EventQueue& EventQueue::operator = ( EventQueue&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void EventQueue::swap ( EventQueue& rhs ) noexcept
{
  jem::swap ( tail_, rhs.tail_ );
  jem::swap ( size_, rhs.size_ );
}

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t EventQueue::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   peek
//-----------------------------------------------------------------------


inline GfxEvent* EventQueue::peek () const noexcept
{
  if ( tail_ )
  {
    return tail_->next ();
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   pop
//-----------------------------------------------------------------------


inline EventPtr EventQueue::pop ()
{
  JEM_ASSERT ( size() );

  GfxEvent*  ev = tail_->next ();

  ev->unlink ();

  size_--;

  return ev;
}


JEM_END_PACKAGE( gl )

#endif
