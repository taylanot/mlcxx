
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

#ifndef JEM_GL_CALLBACKQUEUE_H
#define JEM_GL_CALLBACKQUEUE_H

#include <jem/base/Time.h>
#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE( gl )


class RenderContext;


//-----------------------------------------------------------------------
//   class CallbackQueue
//-----------------------------------------------------------------------


class CallbackQueue
{
 public:

  typedef CallbackQueue     Self;

  typedef void            (*Callback)

    ( RenderContext&          rtx,
      const Ref<Object>&      obj );


                            CallbackQueue   ();
                           ~CallbackQueue   ();

  inline idx_t              size            () const noexcept;
  void                      clear           ();
  inline Time               timeLeft        () const;

  Time                      timeLeft

    ( const Time&             now )            const;

  void                      insert

    ( Callback                cb,
      const Ref<Object>&      obj,
      const Time&             tm );

  idx_t                     erase

    ( Callback                cb );

  bool                      erase

    ( Callback                cb,
      const Ref<Object>&      obj );

  void                      update

    ( Callback                cb,
      const Ref<Object>&      obj,
      const Time&             tm );

  void                      invoke

    ( Time&                   tleft,
      RenderContext&          rtx );

  idx_t                     contains

    (  Callback               cb )             const;

  bool                      contains

    ( Callback                cb,
      const Ref<Object>&      obj )            const;


 private:

  class                     Entry_;
  class                     Invocation_;
  friend class              Invocation_;


                            CallbackQueue

    ( const Self&             rhs );

  CallbackQueue&            operator =

    ( const Self&             rhs );

  void                      insert_

    ( Entry_*                 e );

  Entry_*                   erase_

    ( Entry_*                 e );

  void                      invoke_

    ( Time&                   now,
      RenderContext&          rtx );


 private:

  Entry_*                   tail_;
  Entry_*                   cache_;

  idx_t                     queueSize_;
  idx_t                     cacheSize_;
  bool                      invoking_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t CallbackQueue::size () const noexcept
{
  return queueSize_;
}


//-----------------------------------------------------------------------
//   timeLeft
//-----------------------------------------------------------------------


inline Time CallbackQueue::timeLeft () const
{
  return timeLeft ( Time::now() );
}


JEM_END_PACKAGE( gl )

#endif
