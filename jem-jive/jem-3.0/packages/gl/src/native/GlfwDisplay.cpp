
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
#include <jem/base/Thread.h>
#include <jem/base/CString.h>
#include <jem/base/System.h>
#include <jem/base/SystemException.h>
#include <jem/gl/gl.h>
#include <jem/gl/input.h>
#include <jem/gl/import.h>
#include <jem/gl/Display.h>
#include "glfw.h"
#include "GlfwDisplay.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GlfwWindow
//=======================================================================


class GlfwWindow : public NativeWindow
{
 public:

  typedef GlfwWindow        Self;
  typedef NativeWindow      Super;


  explicit inline           GlfwWindow

    ( GLFWwindow*             wptr );

  inline                    GlfwWindow

    ( const String&           title,
      int                     width,
      int                     height );

  GLFWwindow*               get             () const;

  virtual void              close           ()       override;
  virtual void              setCurrent      ()       override;
  virtual bool              shouldClose     () const override;
  virtual void              swapBuffers     ()       override;
  virtual Size2i            getWindowSize   () const override;

  static GLFWwindow*        createWindow

    ( const String&           title,
      int                     width,
      int                     height );


 private:

  GLFWwindow*               window_;

#ifdef JEM_OS_DARWIN

  // This is a work around for a bug with GLFW on macOS;
  // see the function swapBuffers() below.

  bool                      moved_;

#endif

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline GlfwWindow::GlfwWindow ( GLFWwindow* wptr ) : window_ ( wptr )
{
#ifdef JEM_OS_DARWIN

  moved_ = false;

#endif
}


inline GlfwWindow::GlfwWindow

  ( const String&  title,
    int            width,
    int            height ) :

    GlfwWindow ( createWindow( title, width, height ) )

{}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


GLFWwindow* GlfwWindow::get () const
{
  return window_;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void GlfwWindow::close ()
{
  GlfwDisplay*  disp = GlfwDisplay::instance_;

  Super::close         ();
  disp->winlist_.erase ( this );

  if ( disp->last_ == this )
  {
    disp->last_ = nullptr;
  }

  glfwDestroyWindow ( window_ );

  window_ = nullptr;
}


//-----------------------------------------------------------------------
//   setCurrent
//-----------------------------------------------------------------------


void GlfwWindow::setCurrent ()
{
  if ( window_ )
  {
    GlfwDisplay*  disp = GlfwDisplay::instance_;

    glfwMakeContextCurrent ( window_ );

    disp->last_ = this;
  }
}


//-----------------------------------------------------------------------
//   shouldClose
//-----------------------------------------------------------------------


bool GlfwWindow::shouldClose () const
{
  bool  retval = false;

  if ( window_ )
  {
    int  rv = glfwWindowShouldClose ( window_ );

    retval = rv ? true : false;
  }

  return retval;
}


//-----------------------------------------------------------------------
//   swapBuffers
//-----------------------------------------------------------------------


void GlfwWindow::swapBuffers ()
{
  if ( Thread::getCurrent() != NativeDisplay::getThread() )
  {
    contextError ( JEM_FUNC );
  }

  if ( window_ )
  {
    glfwSwapBuffers ( window_ );

#ifdef JEM_OS_DARWIN

    if ( ! moved_ )
    {
      int  x, y;

      glfwGetWindowPos ( window_, &x, &y );
      glfwSetWindowPos ( window_, ++x, y );

      moved_ = true;
    }

#endif
  }
}


//-----------------------------------------------------------------------
//   getWindowSize
//-----------------------------------------------------------------------


Size2i GlfwWindow::getWindowSize () const
{
  Size2i  wSize;

  if ( window_ )
  {
    glfwGetFramebufferSize ( window_, &wSize.width, &wSize.height );
  }
  else
  {
    wSize.width  = 0;
    wSize.height = 0;
  }

  return wSize;
}


//-----------------------------------------------------------------------
//   createWindow
//-----------------------------------------------------------------------


GLFWwindow* GlfwWindow::createWindow

  ( const String&  title,
    int            width,
    int            height )

{
  CString      ctitle ( title );

  GLFWwindow*  win    = nullptr;

  glfwWindowHint ( GLFW_VISIBLE, GLFW_FALSE );
  glfwWindowHint ( GLFW_CONTEXT_VERSION_MAJOR, 3 );
  glfwWindowHint ( GLFW_CONTEXT_VERSION_MINOR, 3 );
  glfwWindowHint ( GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE );
  glfwWindowHint ( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

  win = glfwCreateWindow ( width, height, ctitle,
                           nullptr, nullptr );

  if ( ! win )
  {
    // Try again with an older OpenGL profile.

    glfwWindowHint ( GLFW_CONTEXT_VERSION_MAJOR, 1 );
    glfwWindowHint ( GLFW_CONTEXT_VERSION_MINOR, 1 );
    glfwWindowHint ( GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE );
    glfwWindowHint ( GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE );

    win = glfwCreateWindow ( width, height, ctitle,
                             nullptr, nullptr );
  }

  return win;
}


//=======================================================================
//   class GlfwDisplay
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


GlfwDisplay*  GlfwDisplay::instance_ = nullptr;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GlfwDisplay::GlfwDisplay ()
{
  instance_  = this;
  screenDPI_ = 0;
}


GlfwDisplay::~GlfwDisplay ()
{
  instance_ = nullptr;

  glfwTerminate ();
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void GlfwDisplay::init_

  ( int&    argc,
    char**  argv )

{
  glfwSetErrorCallback ( errorFunc_ );

  if ( ! glfwInit() )
  {
    throw SystemException (
      JEM_FUNC,
      "unable to initialize the GLFW library"
    );
  }
}


//-----------------------------------------------------------------------
//   mainLoop_
//-----------------------------------------------------------------------


void GlfwDisplay::mainLoop_ ()
{
  while ( instance_ )
  {
    popCommand_    ( 120.0 );
    glfwPollEvents ();
  }
}


//-----------------------------------------------------------------------
//   getScreenDPI_
//-----------------------------------------------------------------------


int GlfwDisplay::getScreenDPI_ ()
{
  if ( screenDPI_ <= 0 )
  {
    const GLFWvidmode*  mode = nullptr;
    GLFWmonitor*        mon  = nullptr;
    GLFWwindow*         win  = nullptr;

    int                 width,  width2;
    int                 height, height2;

    mon = glfwGetPrimaryMonitor ();

    if ( ! mon )
    {
      return 0;
    }

    mode = glfwGetVideoMode ( mon );

    if ( ! mode || (mode->height <= 0) )
    {
      return 0;
    }

    glfwGetMonitorPhysicalSize ( mon, &width, &height );

    if ( height <= 0 )
    {
      return 0;
    }

    screenDPI_ = (int) (0.5 + (25.4 * mode->height) /
                               (double) height);

    win = GlfwWindow::createWindow ( "hidden",
                                     mode->width  / 2,
                                     mode->height / 2 );

    if ( win )
    {
      glfwGetWindowSize      ( win, &width,  &height );
      glfwGetFramebufferSize ( win, &width2, &height2 );

      glfwDestroyWindow ( win );

      win = nullptr;

      if ( (height > 0) && (height2 > 0) )
      {
        screenDPI_ = (int) (screenDPI_ * ((double) height2 /
                                          (double) height));
      }
    }
  }

  return screenDPI_;
}


//-----------------------------------------------------------------------
//   newWindow_
//-----------------------------------------------------------------------


Ref<NativeWindow> GlfwDisplay::newWindow_

  ( const String&  title,
    int            width,
    int            height )

{
  Ref<GlfwWindow>  win;

  if ( ! (width > 0 && height > 0) )
  {
    const GLFWvidmode*  mode =

      glfwGetVideoMode ( glfwGetPrimaryMonitor() );

    width  = (int) (0.75 * (double) mode->width);
    height = (int) (0.75 * (double) mode->height);
  }

  win = newInstance<GlfwWindow> ( title, width, height );

  if ( ! win->get() )
  {
    throw SystemException (
      JEM_FUNC,
      "unable to create a window using the GLFW library"
    );
  }

  glfwSetFramebufferSizeCallback ( win->get(), resizeFunc_  );
  glfwSetWindowRefreshCallback   ( win->get(), redrawFunc_  );
  glfwSetMouseButtonCallback     ( win->get(), mouseFunc_   );
  glfwSetCursorPosCallback       ( win->get(), motionFunc_  );
  glfwSetKeyCallback             ( win->get(), keyFunc_     );
  glfwSetCharModsCallback        ( win->get(), charsFunc_   );

  if ( getMode() & Display::EXIT_ON_CLOSE )
  {
    glfwSetWindowCloseCallback ( win->get(), closeFunc_   );
  }

  winlist_.pushBack ( win.get() );
  win->setCurrent   ();
  glfwSwapInterval  ( 1 );
  glfwShowWindow    ( win->get() );

  return win;
}


//-----------------------------------------------------------------------
//   postEvent_
//-----------------------------------------------------------------------


inline void GlfwDisplay::postEvent_

  ( GLFWwindow*  wptr,
    GfxEvent*    ev )

{
  GlfwWindow*  win = getWindow_ ( wptr );

  if ( win )
  {
    Super::postEvent_ ( *win, ev );
  }
}


//-----------------------------------------------------------------------
//   getWindow_
//-----------------------------------------------------------------------


GlfwWindow* GlfwDisplay::getWindow_ ( GLFWwindow* wptr )
{
  GlfwWindow*  win = nullptr;

  if ( (last_ != nullptr) && (last_->get() == wptr) )
  {
    win = last_.get ();
  }
  else
  {
    const idx_t  n = winlist_.size ();

    for ( idx_t i = 0; i < n; i++ )
    {
      win = winlist_.getAs<GlfwWindow> ( i );

      if ( win->get() == wptr )
      {
        last_ = win;
        break;
      }
    }
  }

  return win;
}


//-----------------------------------------------------------------------
//   resizeFunc_
//-----------------------------------------------------------------------


void GlfwDisplay::resizeFunc_

  ( GLFWwindow*  wptr,
    int          width,
    int          height )

{
  try
  {
    instance_->postEvent_ ( wptr, new ResizeEvent() );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   redrawFunc_
//-----------------------------------------------------------------------


void GlfwDisplay::redrawFunc_ ( GLFWwindow* wptr )
{
  try
  {
    instance_->postEvent_ ( wptr, new RedrawEvent() );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   mouseFunc_
//-----------------------------------------------------------------------


void GlfwDisplay::mouseFunc_

  ( GLFWwindow*  wptr,
    int          button,
    int          action,
    int          mods )

{
  int     btn = 0;
  double  x   = 0.0;
  double  y   = 0.0;

  switch ( button )
  {
  case GLFW_MOUSE_BUTTON_LEFT:

    btn |= Buttons::LEFT;
    break;

  case GLFW_MOUSE_BUTTON_MIDDLE:

    btn |= Buttons::MIDDLE;
    break;

  case GLFW_MOUSE_BUTTON_RIGHT:

    btn |= Buttons::RIGHT;
    break;
  }

  glfwGetCursorPos ( wptr, &x, &y );
  toFramebuf_      ( wptr,  x,  y );

  try
  {
    instance_->postEvent_ (
      wptr,
      new MouseEvent (
        btn,
        getKeyMods_ ( wptr ),
        (action == GLFW_PRESS),
        (int) x, (int) y
      )
    );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   motionFunc_
//-----------------------------------------------------------------------


void GlfwDisplay::motionFunc_

  ( GLFWwindow*  wptr,
    double       x,
    double       y )

{
  toFramebuf_ ( wptr, x, y );

  try
  {
    instance_->postEvent_ (
      wptr,
      new MotionEvent ( (int) x, (int) y )
    );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   keyFunc_
//-----------------------------------------------------------------------


void GlfwDisplay::keyFunc_

  ( GLFWwindow*  wptr,
    int          key,
    int          code,
    int          action,
    int          mods )

{
  if ( (key > 255) && (action != GLFW_RELEASE) )
  {
    double  x = 0.0;
    double  y = 0.0;

    switch ( key )
    {
    case GLFW_KEY_ESCAPE:     key = Keys::ESCAPE;      break;
    case GLFW_KEY_F1:         key = Keys::F1;          break;
    case GLFW_KEY_F2:         key = Keys::F2;          break;
    case GLFW_KEY_F3:         key = Keys::F3;          break;
    case GLFW_KEY_F4:         key = Keys::F4;          break;
    case GLFW_KEY_F5:         key = Keys::F5;          break;
    case GLFW_KEY_F6:         key = Keys::F6;          break;
    case GLFW_KEY_F7:         key = Keys::F7;          break;
    case GLFW_KEY_F8:         key = Keys::F8;          break;
    case GLFW_KEY_F9:         key = Keys::F9;          break;
    case GLFW_KEY_F10:        key = Keys::F10;         break;
    case GLFW_KEY_F11:        key = Keys::F12;         break;
    case GLFW_KEY_F12:        key = Keys::F11;         break;
    case GLFW_KEY_LEFT:       key = Keys::LEFT;        break;
    case GLFW_KEY_RIGHT:      key = Keys::RIGHT;       break;
    case GLFW_KEY_UP:         key = Keys::UP;          break;
    case GLFW_KEY_DOWN:       key = Keys::DOWN;        break;
    case GLFW_KEY_PAGE_UP:    key = Keys::PAGE_UP;     break;
    case GLFW_KEY_PAGE_DOWN:  key = Keys::PAGE_DOWN;   break;
    case GLFW_KEY_HOME:       key = Keys::HOME;        break;
    case GLFW_KEY_END:        key = Keys::END;         break;

    default:                  return;
    }

    glfwGetCursorPos  ( wptr, &x, &y );
    toFramebuf_       ( wptr,  x,  y );

    try
    {
      mods = getKeyMods_ ( mods );

      instance_->postEvent_ (
        wptr,
        new KeyEvent ( (int) key, mods, (int) x, (int) y )
      );
    }
    catch ( ... ) {}
  }
}


//-----------------------------------------------------------------------
//   charsFunc_
//-----------------------------------------------------------------------


void GlfwDisplay::charsFunc_

  ( GLFWwindow*   wptr,
    unsigned int  code,
    int           mods )

{

  if ( code < 127U )
  {
    int     key = (int) code;
    double  x   = 0.0;
    double  y   = 0.0;

    glfwGetCursorPos  ( wptr, &x, &y );
    toFramebuf_       ( wptr,  x,  y );

    try
    {
      mods = getKeyMods_ ( mods );

      instance_->postEvent_ (
        wptr,
        new KeyEvent ( (int) key, mods, (int) x, (int) y )
      );
    }
    catch ( ... ) {}
  }
}

//-----------------------------------------------------------------------
//   closeFunc_
//-----------------------------------------------------------------------


void GlfwDisplay::closeFunc_ ( GLFWwindow* wptr )
{
  glfwTerminate ();
  std::exit     ( EXIT_SUCCESS );
}


//-----------------------------------------------------------------------
//   toFramebuf_
//-----------------------------------------------------------------------


void GlfwDisplay::toFramebuf_

  ( GLFWwindow*  wptr,
    double&      x,
    double&      y )

{
  int  winWidth, winHeight;
  int  bufWidth, bufHeight;

  glfwGetWindowSize      ( wptr, &winWidth, &winHeight );
  glfwGetFramebufferSize ( wptr, &bufWidth, &bufHeight );

  y = (double) winHeight - y;

  if ( (winWidth != bufWidth) && (winWidth > 0) )
  {
    x *= (double) bufWidth / (double) winWidth;
  }

  if ( (winHeight != bufHeight) && (winHeight > 0) )
  {
    y *= (double) bufHeight / (double) winHeight;
  }
}


//-----------------------------------------------------------------------
//   getKeyMods_
//-----------------------------------------------------------------------


inline int GlfwDisplay::getKeyMods_ ( GLFWwindow* wptr )
{
  int  r = 0;

  if ( glfwGetKey( wptr, GLFW_KEY_LEFT_SHIFT  ) ||
       glfwGetKey( wptr, GLFW_KEY_RIGHT_SHIFT ) )
  {
    r |= Keys::SHIFT;
  }
  if ( glfwGetKey( wptr, GLFW_KEY_LEFT_CONTROL  ) ||
       glfwGetKey( wptr, GLFW_KEY_RIGHT_CONTROL ) )
  {
    r |= Keys::CTRL;
  }
  if ( glfwGetKey( wptr, GLFW_KEY_LEFT_ALT  ) ||
       glfwGetKey( wptr, GLFW_KEY_RIGHT_ALT ) )
  {
    r |= Keys::ALT;
  }

  return r;
}


inline int GlfwDisplay::getKeyMods_ ( int mods )
{
  int  r = 0;

  if ( mods & GLFW_MOD_SHIFT )
  {
    r |= Keys::SHIFT;
  }
  if ( mods & GLFW_MOD_CONTROL )
  {
    r |= Keys::CTRL;
  }
  if ( mods & GLFW_MOD_ALT )
  {
    r |= Keys::ALT;
  }

  return r;
}


//-----------------------------------------------------------------------
//   errorFunc_()
//-----------------------------------------------------------------------


void GlfwDisplay::errorFunc_

  ( int          err,
    const char*  desc )

{
  try
  {
    System::err() << "GLFW ERROR: " << desc << "\n\n";
  }
  catch ( ... )
  {}
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newOnScreenDisplay
//-----------------------------------------------------------------------


NativeDisplay* newOnScreenDisplay ()
{
  return new GlfwDisplay ();
}


JEM_END_PACKAGE( gl )
