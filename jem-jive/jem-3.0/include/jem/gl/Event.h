
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

#ifndef JEM_GL_EVENT_H
#define JEM_GL_EVENT_H

#include <jem/base/Ref.h>
#include <jem/base/MemCache.h>
#include <jem/base/AutoPointer.h>


JEM_BEGIN_PACKAGE( gl )


class GfxNode;


//-----------------------------------------------------------------------
//   class GfxEvent
//-----------------------------------------------------------------------


class GfxEvent
{
 public:

  enum                      TypeID
  {
                              NOP,
                              REDRAW,
                              RESIZE,
                              INIT,
                              MOUSE,
                              MOTION,
                              CLICK2,
                              KEY,
                              COMMAND
  };


  inline                    GfxEvent  ()       noexcept;
  virtual  inline          ~GfxEvent  ();
  inline GfxEvent*          next      () const noexcept;
  inline GfxEvent*          prev      () const noexcept;
  inline void               unlink    ()       noexcept;

  inline void               insert

    ( GfxEvent*               ev );

  static inline void*       operator new

    ( size_t                  size );

  static inline void        operator delete

    ( void*                   addr,
      size_t                  size );


 public:

  const TypeID              type;


 protected:

  explicit inline           GfxEvent

    ( TypeID                  tid )            noexcept;


 private:

                            GfxEvent

    ( const GfxEvent&         rhs );

  GfxEvent&                 operator =

    ( const GfxEvent&         rhs );


 private:

  GfxEvent*                 next_;
  GfxEvent*                 prev_;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef
  AutoPointer<GfxEvent>     EventPtr;


//-----------------------------------------------------------------------
//   class RedrawEvent
//-----------------------------------------------------------------------


class RedrawEvent : public GfxEvent
{
 public:

  typedef GfxEvent          Super;

  static const int          FAST     = 1 << 0;
  static const int          EXTERNAL = 1 << 1;


  inline                    RedrawEvent () noexcept;

  explicit inline           RedrawEvent

    ( const Ref<GfxNode>&     scene );


 public:

  Ref<GfxNode>              scene;
  int                       flags;

};


typedef AutoPointer
  <RedrawEvent>             RedrawEventPtr;


//-----------------------------------------------------------------------
//   class ResizeEvent
//-----------------------------------------------------------------------


class ResizeEvent : public GfxEvent
{
 public:

  typedef GfxEvent          Super;


  inline                    ResizeEvent () noexcept;

};


//-----------------------------------------------------------------------
//   InitEvent
//-----------------------------------------------------------------------


class InitEvent : public GfxEvent
{
 public:

  typedef GfxEvent          Super;


  inline                    InitEvent () noexcept;

};


//-----------------------------------------------------------------------
//   MouseEvent
//-----------------------------------------------------------------------


class MouseEvent : public GfxEvent
{
 public:

  typedef GfxEvent          Super;


  inline                    MouseEvent

    ( int                     button,
      int                     mods,
      bool                    down,
      int                     xPos,
      int                     yPos );


 public:

  int                       button;
  int                       mods;
  bool                      down;
  int                       xPos;
  int                       yPos;

};


//-----------------------------------------------------------------------
//   class MotionEvent
//-----------------------------------------------------------------------


class MotionEvent : public GfxEvent
{
 public:

  typedef GfxEvent          Super;


  inline                    MotionEvent

    ( int                     xPos,
      int                     yPos );


 public:

  int                       xPos;
  int                       yPos;

};


//-----------------------------------------------------------------------
//   Click2Event
//-----------------------------------------------------------------------


class Click2Event : public GfxEvent
{
 public:

  typedef GfxEvent          Super;


  inline                    Click2Event

    ( int                     button,
      int                     mods,
      int                     xPos,
      int                     yPos );


 public:

  int                       button;
  int                       mods;
  int                       xPos;
  int                       yPos;

};


//-----------------------------------------------------------------------
//   class KeyEvent
//-----------------------------------------------------------------------


class KeyEvent : public GfxEvent
{
 public:

  typedef GfxEvent          Super;


  inline                    KeyEvent

    ( int                     key,
      int                     mods,
      int                     xPos,
      int                     yPos );


 public:

  int                       key;
  int                       mods;
  int                       xPos;
  int                       yPos;

};


//-----------------------------------------------------------------------
//   class CommandEvent
//-----------------------------------------------------------------------


class CommandEvent : public GfxEvent
{
 public:

  typedef GfxEvent          Super;


  explicit inline           CommandEvent

    ( const String&           cmd );


 public:

  String                    command;

};



//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class GfxEvent
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline GfxEvent::GfxEvent () noexcept : type ( NOP )
{
  next_ = prev_ = this;
}


inline GfxEvent::GfxEvent ( TypeID t ) noexcept : type ( t )
{
  next_ = prev_ = this;
}


inline GfxEvent::~GfxEvent ()
{
  unlink ();
}


//-----------------------------------------------------------------------
//   next & prev
//-----------------------------------------------------------------------


inline GfxEvent* GfxEvent::next () const noexcept
{
  return next_;
}


inline GfxEvent* GfxEvent::prev () const noexcept
{
  return prev_;
}


//-----------------------------------------------------------------------
//   unlink
//-----------------------------------------------------------------------


inline void GfxEvent::unlink () noexcept
{
  next_->prev_ = prev_;
  prev_->next_ = next_;
  next_        = this;
  prev_        = this;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


inline void GfxEvent::insert ( GfxEvent* ev )
{
  ev->next_    = this;
  ev->prev_    = prev_;
  prev_->next_ = ev;
  prev_        = ev;
}


//-----------------------------------------------------------------------
//   operator new & delete
//-----------------------------------------------------------------------


inline void* GfxEvent::operator new ( size_t size )
{
  return MemCache::alloc ( size );
}


inline void GfxEvent::operator delete ( void* p, size_t size )
{
  MemCache::dealloc ( p, size );
}


//=======================================================================
//   class RedrawEvent
//=======================================================================


inline RedrawEvent::RedrawEvent () noexcept : Super ( REDRAW )
{
  flags = 0;
}


inline RedrawEvent::RedrawEvent ( const Ref<GfxNode>& s ) :

  Super ( REDRAW ),
  scene ( s )

{
  flags = 0;
}


//=======================================================================
//   class ResizeEvent
//=======================================================================


inline ResizeEvent::ResizeEvent () noexcept : Super ( RESIZE )
{}


//=======================================================================
//   class InitEvent
//=======================================================================


inline InitEvent::InitEvent () noexcept : Super ( INIT )
{}


//=======================================================================
//   class MouseEvent
//=======================================================================


inline MouseEvent::MouseEvent

  ( int   b,
    int   m,
    bool  d,
    int   x,
    int   y ) :

    Super ( MOUSE )

{
  button = b;
  mods   = m;
  down   = d;
  xPos   = x;
  yPos   = y;
}


//=======================================================================
//   class MotionEvent
//=======================================================================


inline MotionEvent::MotionEvent ( int x, int y ) :

  Super ( MOTION )

{
  xPos = x;
  yPos = y;
}


//=======================================================================
//   class Click2Event
//=======================================================================


inline Click2Event::Click2Event

  ( int   b,
    int   m,
    int   x,
    int   y ) :

    Super ( CLICK2 )

{
  button = b;
  mods   = m;
  xPos   = x;
  yPos   = y;
}


//=======================================================================
//   class KeyEvent
//=======================================================================


inline KeyEvent::KeyEvent ( int k, int m, int x, int y ) :

  Super ( KEY )

{
  key  = k;
  mods = m;
  xPos = x;
  yPos = y;
}


//=======================================================================
//   class CommandEvent
//=======================================================================


inline CommandEvent::CommandEvent ( const String& cmd ) :

  Super   ( COMMAND ),
  command ( cmd )

{}


JEM_END_PACKAGE( gl )

#endif
