
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
#include <jem/base/ClassTemplate.h>
#include <jem/io/OutputStream.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/gl/Rect.h>
#include <jem/gl/Actions.h>
#include <jem/gl/GLSetup.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/FrameGrabber.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Window.h>


JEM_DEFINE_CLASS( jem::gl::Window );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Window
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Window::Window

  ( const String&        name,
    const Ref<GfxNode>&  child,
    const Ref<GLSetup>&  setup ) :

    Super  ( name, child ),
    setup_ ( setup )

{
  using jem::util::connect;

  JEM_PRECHECK ( setup );

  bgColor_   = Color::white ();
  newSetup_  = true;
  clearMask_ = GL_COLOR_BUFFER_BIT;

  connect ( setup->changeEvent, this, & Self::setupChanged_ );
}


Window::~Window ()
{}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool Window::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  GLContext&     gtx         = rtx.getGLContext ();
  GLFuncs1Core*  funcs       = gtx.findFunctions<GLFuncs1Core> ();
  bool           myResult    = true;
  bool           childResult = false;

  int   w, h;


  if ( newSetup_ && action != (int) GLActions::INIT_GL )
  {
    initGL_ ( gtx );
  }

  switch ( action )
  {
  case GLActions::REDRAW:
  case GLActions::REDRAW_FAST:

    if ( funcs )
    {
      funcs->glClear ( clearMask_ );
    }

    break;

  case GLActions::INIT_GL:

    initGL_ ( gtx );

    break;

  case GLActions::INIT_VIEW:

    params.get ( w, GLActionParams::VIEW_WIDTH  );
    params.get ( h, GLActionParams::VIEW_HEIGHT );

    size_.width  = (GLsizei) w;
    size_.height = (GLsizei) h;

    gtx.setViewport  ( Rect( size_ ) );
    setup_->initView ( gtx, Rect( size_ ) );

    break;

  case GLActions::GRAB_FRAME:

    {
      Ref<OutputStream>  output;
      Ref<FrameGrabber>  grabber;

      Rect               vport ( size_ );

      params.get ( grabber, GLActionParams::FRAME_GRABBER );
      params.get ( output,  GLActionParams::FRAME_OUTPUT  );

      grabber->grabFrame ( *output, rtx, *this, vport );

      return true;
    }

    break;

  default:

    myResult = false;
  }

  childResult = passGLAction ( child_, rtx, action, params );

  checkGLError ( gtx, JEM_FUNC );

  return (myResult || childResult);
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Window::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    if ( bgColor_.configure( PropNames::BG_COLOR, myProps ) )
    {
      newSetup_ = true;
    }

    setup_->configure ( myProps );
  }

  configChild ( child_, props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Window::getConfig ( const Properties& conf ) const
{
  Properties  myConf = conf.makeProps ( myName_ );

  bgColor_.getConfig ( PropNames::BG_COLOR, myConf  );
  setup_ ->getConfig ( myConf );

  if ( child_ )
  {
    child_->getConfig ( conf );
  }
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void Window::listProps ( const Properties& conf ) const
{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::BG_COLOR,
               "The background color" );

  setup_->listProps ( myConf );

  if ( child_ )
  {
    child_->listProps ( conf );
  }
}


//-----------------------------------------------------------------------
//   setChild
//-----------------------------------------------------------------------


void Window::setChild ( const Ref<GfxNode>& child )
{
  Super::setChild_ ( child );
}


//-----------------------------------------------------------------------
//   getChild
//-----------------------------------------------------------------------


Ref<GfxNode> Window::getChild () const
{
  return child_;
}


//-----------------------------------------------------------------------
//   setBgColor
//-----------------------------------------------------------------------


void Window::setBgColor ( const Color& bg )
{
  bgColor_  = clamped ( bg );
  newSetup_ = true;
}


//-----------------------------------------------------------------------
//   initGL_
//-----------------------------------------------------------------------


void Window::initGL_ ( GLContext& gtx )
{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();
  const Color&   bg    = bgColor_;

  setup_->initGL ( gtx );

  if  ( funcs )
  {
    funcs->glClearColor ( bg[0], bg[1], bg[2], bg[3] );
  }

  clearMask_ = getClearMask ( gtx );
  newSetup_  = false;
}


//-----------------------------------------------------------------------
//   setupChanged_
//-----------------------------------------------------------------------


void Window::setupChanged_ ()
{
  newSetup_ = true;
}


JEM_END_PACKAGE( gl )
