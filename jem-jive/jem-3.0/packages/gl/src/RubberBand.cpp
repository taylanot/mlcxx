
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
#include <jem/base/Time.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/render.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Compound.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/RedrawRoot.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/RubberBand.h>


JEM_DEFINE_CLASS( jem::gl::RubberBand );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class RubberBand
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  RubberBand::CANCEL_    = "_RUBBER_BAND_CANCEL";

const int    RubberBand::ACTIVE_    = 1 << 0;
const int    RubberBand::TRANSIENT_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


RubberBand::RubberBand ( const String& name ) :

  Super ( name )

{
  state_   =  0;
  origin_  =  0;
  current_ =  0;
  aspect_  = -1.0;
}


RubberBand::~RubberBand ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool RubberBand::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if ( (state_ & ACTIVE_) && action == CANCEL_ )
  {
    cancel ();

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool RubberBand::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  typedef GLActionParams  Params;

  if ( ! (state_ & ACTIVE_) )
  {
    return false;
  }

  bool  result = true;

  switch ( action )
  {
  case GLActions::KEY_EVENT:
  {
    int  key;

    params.get ( key, Params::KEY_CODE );

    // Cancel when the ESC key has been hit.

    if ( key == Keys::ESCAPE )
    {
      cancel ( rtx );
    }

    break;
  }

  case GLActions::MOUSE_EVENT:
  {
    int   bttn;
    bool  down;

    params.get ( bttn, Params::MOUSE_BUTTON );
    params.get ( down, Params::MOUSE_DOWN   );

    if      ( bttn == mouse_.button && down == false )
    {
      finish_ ( rtx );
    }
    else if ( bttn != mouse_.button && down == true )
    {
      cancel  ( rtx );
    }

    break;
  }

  case GLActions::MOTION_EVENT:
  {
    int  x, y;

    params.get ( x, Params::MOUSE_XPOS );
    params.get ( y, Params::MOUSE_YPOS );

    if ( aspect_ > 0.0 )
    {
      int  w = x - (int) origin_[0];
      int  h = y - (int) origin_[1];

      if ( aspect_ > 1.0 )
      {
        int  sign = (h >= 0) ? 1 : -1;

        h = sign * (int) (::abs(w) * aspect_);
      }
      else
      {
        int  sign = (w >= 0) ? 1 : -1;

        w = sign * (int) (::abs(h) / aspect_);
      }

      x = w + (int) origin_[0];
      y = h + (int) origin_[1];
    }

    current_[0] = (GLint) x;
    current_[1] = (GLint) y;

    rtx.postFastRedraw ( this );

    break;
  }

  case GLActions::REDRAW:
  case GLActions::REDRAW_FAST:

    redraw_ ( rtx );

    break;

  case GLActions::INIT_VIEW:

    cancel ();

    break;

  default:

    result = false;
  }

  return result;
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void RubberBand::start

  ( RenderContext&    rtx,
    const MouseDrag&  mouse )

{
  int  savedState = state_;

  state_     = 0;
  mouse_     = mouse;;
  origin_[0] = (GLint) (mouse.x - mouse.dx);
  origin_[1] = (GLint) (mouse.y - mouse.dy);
  current_   = origin_;

  // Cancel all currently active RubberBand objects.

  try
  {
    GfxNode*  root = findRedrawRoot ( getParent() );

    if ( root )
    {
      Properties  params;

      root->takeAction ( CANCEL_, params );
    }
  }
  catch ( ... )
  {
    state_ = savedState;
    throw;
  }

  state_ = savedState | ACTIVE_;

  rtx.postFastRedraw ( this );
}


void RubberBand::start

  ( RenderContext&       rtx,
    const Ref<GfxNode>&  scene,
    const MouseDrag&     mouse )

{
  if ( state_ & ACTIVE_ )
  {
    cancel ();
  }

  if ( ! getParent() && addToScene( scene, this ) )
  {
    state_ |= TRANSIENT_;
  }

  start ( rtx, mouse );
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void RubberBand::cancel ()
{
  if ( state_ & TRANSIENT_ )
  {
    eraseFromParent ( this );
  }

  state_ = 0;
}


void RubberBand::cancel ( RenderContext& rtx )
{
  if ( state_ & ACTIVE_ )
  {
    // Redraw must be posted *before* deleting this node from its
    // parent.

    rtx.postRedraw   ( this );
    cancel           ();
    cancelEvent.emit ( rtx  );
  }
}


//-----------------------------------------------------------------------
//   finish_
//-----------------------------------------------------------------------


void RubberBand::finish_ ( RenderContext& rtx )
{
  Rect  r;

  if ( current_[0] > origin_[0] )
  {
    r.x      = origin_[0];
    r.width  = (GLsizei) (current_[0] - origin_[0]);
  }
  else
  {
    r.x      = current_[0];
    r.width  = (GLsizei) (origin_[0] - current_[0]);
  }

  if ( current_[1] > origin_[1] )
  {
    r.y      = origin_[1];
    r.height = (GLsizei) (current_[1] - origin_[1]);
  }
  else
  {
    r.y      = current_[1];
    r.height = (GLsizei) (origin_[1] - current_[1]);
  }

  cancel ();

  selectEvent.emit ( rtx, r, mouse_ );
}


//-----------------------------------------------------------------------
//   redraw_
//-----------------------------------------------------------------------


void RubberBand::redraw_ ( RenderContext& rtx )
{
  GLContext&     gtx     = rtx.getGLContext ();
  GLFunctions1*  funcs   = gtx.findFunctions<GLFunctions1> ();

  Vertex2i       origin  = origin_;
  Vertex2i       current = current_;

  CanvasScope    canvas  ( gtx );
  AttribScope    attribs ( gtx, GLContext::COLOR |
                                GLContext::LINE_WIDTH );

  gtx.setLineWidth ( 2.0F );

  // Make sure that the origin is located at the bottom left corner
  // of the rectangle. Otherwise, the rectangle may face away from
  // the camera. This is required for OpenGL Core profiles.

  if ( origin[0] > current[0] )
  {
    jem::swap ( origin[0], current[0] );
  }

  if ( origin[1] > current[1] )
  {
    jem::swap ( origin[1], current[1] );
  }

  if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLfloat  vertPos[15];

    vertPos[ 0] = (GLfloat) origin [0];
    vertPos[ 1] = (GLfloat) origin [1];
    vertPos[ 2] = 0.0F;

    vertPos[ 3] = (GLfloat) current[0];
    vertPos[ 4] = (GLfloat) origin [1];
    vertPos[ 5] = 0.0F;

    vertPos[ 6] = (GLfloat) current[0];
    vertPos[ 7] = (GLfloat) current[1];
    vertPos[ 8] = 0.0F;

    vertPos[ 9] = (GLfloat) origin [0];
    vertPos[10] = (GLfloat) current[1];
    vertPos[11] = 0.0F;

    vertPos[12] = (GLfloat) origin [0];
    vertPos[13] = (GLfloat) origin [1];
    vertPos[14] = 0.0F;

    ShaderDraw::draw ( gtx, 12, vertPos,
                       Color ( 1.0F, 0.0F, 0.0F, 0.1F ),
                       ShaderDraw::TRIANGLE_FAN );

    ShaderDraw::draw ( gtx, 15, vertPos,
                       Color::red(),
                       ShaderDraw::LINE_STRIP );
  }
  else if ( funcs )
  {
    gtx.setColor      ( Color( 1.0F, 0.0F, 0.0F, 0.1F ) );

    funcs->glRect     ( origin [0], origin [1],
                        current[0], current[1] );

    gtx.setColor      ( Color::red() );

    funcs->glBegin    ( GL_LINE_STRIP );
    {
      funcs->glVertex ( origin [0], origin [1] );
      funcs->glVertex ( origin [0], current[1] );
      funcs->glVertex ( current[0], current[1] );
      funcs->glVertex ( current[0], origin [1] );
      funcs->glVertex ( origin [0], origin [1] );
    }
    funcs->glEnd      ();
  }
}


JEM_END_PACKAGE( gl )
