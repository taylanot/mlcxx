
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


#include <cstdlib>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/input.h>
#include <jem/gl/Box.h>
#include <jem/gl/Rect.h>
#include <jem/gl/Actions.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/EventNode.h>


JEM_DEFINE_CLASS( jem::gl::EventNode );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class EventNode
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


EventNode::EventNode

  ( const String&        name,
    const Ref<GfxNode>&  child ) :

    Super ( name, child )

{
  focus_ = false;

  mouse_.button =  0;
  mouse_.mods   =  0;
  mouse_.seqnr  = -1;
  mouse_.x      =  0;
  mouse_.y      =  0;
}


EventNode::~EventNode ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool EventNode::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if ( action == Actions::TAKE_FOCUS )
  {
    if ( ! focus_ )
    {
      focus_ = true;
      focusEvent.emit ( focus_ );
    }

    return true;
  }

  if ( action == Actions::GIVE_FOCUS )
  {
    passAction ( child_, action, params );

    if ( focus_ )
    {
      focus_        = false;
      mouse_.button = 0;

      params.set      ( ActionParams::FOCUS, this );
      focusEvent.emit ( focus_ );
    }

    return true;
  }

  bool  result = passAction ( child_, action, params );

  if ( action == Actions::GIVE_INPUT )
  {
    mouse_.button = 0;
    result        = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool EventNode::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  typedef GLActionParams  Params;

  bool  result = false;


  switch ( action )
  {
  case GLActions::REDRAW:
  case GLActions::REDRAW_FAST:

    result = passGLAction ( child_, rtx, action, params );

    if ( redrawEvent.isConnected() )
    {
      redrawEvent.emit ( rtx, action, params );

      result = true;
    }

    break;

  case GLActions::INIT_VIEW:

    result = passGLAction ( child_, rtx, action, params );

    if ( resizeEvent.isConnected() )
    {
      Rect  r;
      int   x, y, w, h;

      params.get ( x, Params::VIEW_XPOS   );
      params.get ( y, Params::VIEW_YPOS   );
      params.get ( w, Params::VIEW_WIDTH  );
      params.get ( h, Params::VIEW_HEIGHT );

      r.x      = (GLint)   x;
      r.y      = (GLint)   y;
      r.width  = (GLsizei) w;
      r.height = (GLsizei) h;

      resizeEvent.emit ( rtx, r );

      result = true;
    }

    break;

  case GLActions::KEY_EVENT:

    if ( focus_ )
    {
      result = passGLAction ( child_, rtx, action, params );

      if ( keyEvent.isConnected() )
      {
        KeyPress  key;

        params.get ( key.code, Params::KEY_CODE );
        params.get ( key.mods, Params::KEY_MODS );
        params.get ( key.x,    Params::KEY_XPOS );
        params.get ( key.y,    Params::KEY_YPOS );

        keyEvent.emit ( rtx, key );

        result = true;
      }
    }

    break;

  case GLActions::MOTION_EVENT:

    if ( focus_ )
    {
      result = passGLAction ( child_, rtx, action, params );

      if ( mouse_.button )
      {
        MouseDrag  drag;

        params.get ( drag.x, Params::MOUSE_XPOS );
        params.get ( drag.y, Params::MOUSE_YPOS );

        drag.dx = drag.x - mouse_.x;
        drag.dy = drag.y - mouse_.y;

        // Only start a mouse drag when the mouse has moved a minimum
        // distance.

        if ( mouse_.seqnr < 0 )
        {
          if ( ::abs(drag.dx) <= MouseDrag::THRESHOLD &&
               ::abs(drag.dy) <= MouseDrag::THRESHOLD )
          {
            return result;
          }
        }

        mouse_.x = drag.x;
        mouse_.y = drag.y;

        if ( drag.dx != 0 || drag.dy != 0 )
        {
          if ( mouse_.seqnr < maxOf( mouse_.seqnr ) )
          {
            mouse_.seqnr++;
          }

          if ( dragEvent.isConnected() )
          {
            drag.button = mouse_.button;
            drag.mods   = mouse_.mods;
            drag.seqnr  = mouse_.seqnr;

            dragEvent.emit ( rtx, drag );
          }
        }

        result = true;
      }
    }

    break;

  case GLActions::MOUSE_EVENT:

    if ( focus_ )
    {
      MouseClick  click;
      bool        down;

      result = passGLAction ( child_, rtx, action, params );

      params.get ( click.button, Params::MOUSE_BUTTON );
      params.get ( click.mods,   Params::MOUSE_MODS   );
      params.get ( click.x,      Params::MOUSE_XPOS   );
      params.get ( click.y,      Params::MOUSE_YPOS   );
      params.get ( down,         Params::MOUSE_DOWN   );

      if ( down )
      {
        click.what    =  MouseClick::DOWN;
        mouse_.button =  click.button;
        mouse_.mods   =  click.mods;
        mouse_.seqnr  = -1;
        mouse_.x      =  click.x;
        mouse_.y      =  click.y;
      }
      else
      {
        click.what     = MouseClick::UP;
        mouse_.button &= (~click.button);
      }

      if ( mouseEvent.isConnected() )
      {
        mouseEvent.emit ( rtx, click );

        if ( ! mouse_.button && mouse_.seqnr < 0 )
        {
          click.what = MouseClick::SINGLE;
          click.mods = mouse_.mods;
          click.x    = mouse_.x;
          click.y    = mouse_.y;

          mouseEvent.emit ( rtx, click );
        }
      }

      result = true;
    }

    break;

  case GLActions::EXE_COMMAND:

    if ( focus_ )
    {
      result = passGLAction ( child_, rtx, action, params );

      if ( commandEvent.isConnected() )
      {
        String  cmd;

        params.get ( cmd, Params::COMMAND );

        commandEvent.emit ( rtx, cmd );
      }

      result = true;
    }

    break;

  default:

    result = passGLAction ( child_, rtx, action, params );
  }

  return result;
}


//-----------------------------------------------------------------------
//   takeFocus
//-----------------------------------------------------------------------


void EventNode::takeFocus ()
{
  if ( ! focus_ )
  {
    focus_ = true;
    focusEvent.emit ( focus_ );
  }
}


//-----------------------------------------------------------------------
//   giveFocus
//-----------------------------------------------------------------------


void EventNode::giveFocus ()
{
  if ( child_ )
  {
    Ref<GfxNode>  child = child_;
    Properties    params;

    child->takeAction ( Actions::GIVE_FOCUS, params );
  }

  if ( focus_ )
  {
    focus_ = false;
    focusEvent.emit ( focus_ );
  }
}


JEM_END_PACKAGE( gl )
