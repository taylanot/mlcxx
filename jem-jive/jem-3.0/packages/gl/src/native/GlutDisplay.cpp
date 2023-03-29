
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
#include <jem/base/Thread.h>
#include <jem/base/CString.h>
#include <jem/base/SystemException.h>
#include <jem/gl/gl.h>
#include <jem/gl/input.h>
#include <jem/gl/import.h>
#include "glut.h"
#include "GlutDisplay.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GlutWindow
//=======================================================================


class GlutWindow : public NativeWindow
{
 public:

  typedef GlutWindow        Self;
  typedef NativeWindow      Super;


  explicit inline           GlutWindow

    ( int                     wid );

  virtual void              close           ()       override;
  virtual void              setCurrent      ()       override;
  virtual void              swapBuffers     ()       override;
  virtual Size2i            getWindowSize   () const override;


 public:

  const int                 winID;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline GlutWindow::GlutWindow ( int id ) : winID ( id )
{}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void GlutWindow::close ()
{
  GlutDisplay*  disp = GlutDisplay::instance_;

  Super::close         ();
  disp->winlist_.erase ( this );

  if ( disp->last_ == this )
  {
    disp->last_ = nullptr;
  }

  if ( disp->winlist_.size() == 0 )
  {
    glutSetWindow  ( winID );
    glutHideWindow ();

    disp->zombie_ = newInstance<GlutWindow> ( winID );
  }
  else
  {
    glutDestroyWindow ( winID );
  }
}


//-----------------------------------------------------------------------
//   setCurrent
//-----------------------------------------------------------------------


void GlutWindow::setCurrent ()
{
  GlutDisplay*  disp = GlutDisplay::instance_;

  glutSetWindow ( winID );

  disp->last_ = this;
}


//-----------------------------------------------------------------------
//   getWindowSize
//-----------------------------------------------------------------------


Size2i GlutWindow::getWindowSize () const
{
  return Size2i ( (GLsizei) glutGet( GLUT_WINDOW_WIDTH ),
                  (GLsizei) glutGet( GLUT_WINDOW_HEIGHT ) );
}


//-----------------------------------------------------------------------
//   swapBuffers
//-----------------------------------------------------------------------


void GlutWindow::swapBuffers ()
{
  if ( Thread::getCurrent() != NativeDisplay::getThread() )
  {
    contextError ( JEM_FUNC );
  }

  glutSwapBuffers ();
}


//=======================================================================
//   class GlutDisplay
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


GlutDisplay*  GlutDisplay::instance_ = nullptr;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GlutDisplay::GlutDisplay ()
{
  instance_ = this;
}


GlutDisplay::~GlutDisplay ()
{
  instance_ = nullptr;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void GlutDisplay::init_

  ( int&    argc,
    char**  argv )

{
  int  mode = (GLUT_RGB    |
               GLUT_DOUBLE |
               GLUT_DEPTH);

  glutInit            ( & argc, argv );
  glutInitDisplayMode ( (unsigned int) mode );

#if defined(FREEGLUT)

  glutInitContextVersion ( 3, 3 );

#endif

#if defined(FREEGLUT) && defined(JEM_OS_WINDOWS)

  // Freeglut on Windows, requires a current window before
  // querying the display mode.

  int  winID  = glutCreateWindow ( "test" );

#endif

  int  result = glutGet ( GLUT_DISPLAY_MODE_POSSIBLE );

  if ( ! result )
  {
    // Try again without the accumulation buffers.

    mode &= ~GLUT_ACCUM;

    glutInitDisplayMode ( (unsigned int) mode );
  }

  result = glutGet ( GLUT_DISPLAY_MODE_POSSIBLE );

#if defined(FREEGLUT) && defined(JEM_OS_WINDOWS)

  glutDestroyWindow ( winID );

#endif

  if ( ! result )
  {
    // Give up.

    throw SystemException (
      JEM_FUNC,
      "unable to find a suitable OpenGL visual; "
      "glutInitDisplayMode() failed"
    );
  }

  glutIdleFunc ( idleFunc_ );
}


//-----------------------------------------------------------------------
//   mainLoop_
//-----------------------------------------------------------------------


void GlutDisplay::mainLoop_ ()
{
  glutMainLoop ();
}


//-----------------------------------------------------------------------
//   getScreenDPI_
//-----------------------------------------------------------------------


int GlutDisplay::getScreenDPI_ ()
{
  int  pixels = glutGet ( GLUT_SCREEN_HEIGHT );
  int  height = glutGet ( GLUT_SCREEN_HEIGHT_MM );

  if ( (pixels > 0) && (height > 0) )
  {
    return (int) (0.5 + (25.4 * pixels) / (double) height);
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   newWindow_
//-----------------------------------------------------------------------


Ref<NativeWindow> GlutDisplay::newWindow_

  ( const String&  title,
    int            width,
    int            height )

{
  Ref<GlutWindow>  win;

  if ( zombie_ )
  {
    win = zombie_;

    glutSetWindow      ( zombie_->winID );
    glutSetWindowTitle ( makeCString( title ) );

    if ( width > 0 && height > 0 )
    {
      glutReshapeWindow ( width, height );
    }

    glutShowWindow ();

    zombie_ = nullptr;
  }
  else
  {
    if ( width > 0 && height > 0 )
    {
      glutInitWindowSize ( width, height );
    }

    win = newInstance<GlutWindow> (
      glutCreateWindow ( makeCString( title ) )
    );

    glutReshapeFunc       ( resizeFunc_  );
    glutDisplayFunc       ( redrawFunc_  );
    glutMouseFunc         ( mouseFunc_   );
    glutMotionFunc        ( motionFunc_  );
    glutPassiveMotionFunc ( motionFunc_  );
    glutKeyboardFunc      ( keyFunc_     );
    glutSpecialFunc       ( specialFunc_ );
  }

  winlist_.pushBack ( win.get() );

  last_ = win;

  return win;
}


//-----------------------------------------------------------------------
//   postEvent_
//-----------------------------------------------------------------------


inline void GlutDisplay::postEvent_

  ( int        wid,
    GfxEvent*  ev )
{
  GlutWindow*  win = getWindow_ ( wid );

  if ( win )
  {
    Super::postEvent_ ( *win, ev );
  }
}


//-----------------------------------------------------------------------
//   getWindow_
//-----------------------------------------------------------------------


GlutWindow* GlutDisplay::getWindow_ ( int wid )
{
  GlutWindow*  win = nullptr;

  if ( last_ != nullptr && last_->winID == wid )
  {
    win = last_.get ();
  }
  else
  {
    const idx_t  n = winlist_.size ();

    for ( idx_t i = 0; i < n; i++ )
    {
      win = winlist_.getAs<GlutWindow> ( i );

      if ( win->winID == wid )
      {
        last_ = win;
        break;
      }
    }
  }

  return win;
}


//-----------------------------------------------------------------------
//   idleFunc_
//-----------------------------------------------------------------------


void GlutDisplay::idleFunc_ ()
{
  try
  {
    instance_->popCommand_ ( 0.2 );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   resizeFunc_
//-----------------------------------------------------------------------


void GlutDisplay::resizeFunc_ ( int width, int height )
{
  try
  {
    instance_->postEvent_ ( glutGetWindow(), new ResizeEvent() );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   redrawFunc_
//-----------------------------------------------------------------------


void GlutDisplay::redrawFunc_ ()
{
  try
  {
    instance_->postEvent_ ( glutGetWindow(), new RedrawEvent() );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   mouseFunc_
//-----------------------------------------------------------------------


void GlutDisplay::mouseFunc_ ( int button, int state, int x, int y )
{
  int  btn = 0;

  switch ( button )
  {
  case GLUT_LEFT_BUTTON:

    btn |= Buttons::LEFT;
    break;

  case GLUT_MIDDLE_BUTTON:

    btn |= Buttons::MIDDLE;
    break;

  case GLUT_RIGHT_BUTTON:

    btn |= Buttons::RIGHT;
    break;
  }

  y = glutGet( GLUT_WINDOW_HEIGHT ) - y;

  try
  {
    instance_->postEvent_ (
      glutGetWindow  (),
      new MouseEvent (
        btn,
        getKeyMods_ (),
        (state == GLUT_DOWN),
        x, y
      )
    );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   motionFunc_
//-----------------------------------------------------------------------


void GlutDisplay::motionFunc_ ( int x, int y )
{
  y = glutGet( GLUT_WINDOW_HEIGHT ) - y;

  try
  {
    instance_->postEvent_ ( glutGetWindow(),
                            new MotionEvent( x, y ) );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   keyFunc_
//-----------------------------------------------------------------------


void GlutDisplay::keyFunc_ ( unsigned char code, int x, int y )
{
  y = glutGet( GLUT_WINDOW_HEIGHT ) - y;

  try
  {
    int  key  = (int) code;
    int  mods = getKeyMods_ ();

    if ( (mods & Keys::CTRL) && key >= 1 && key <= 26 )
    {
      // Control ASCII key code; map to normal key code.

      key = (int) ('a' + key - 1);
    }

    if ( key == 27 )
    {
      key = Keys::ESCAPE;
    }

    instance_->postEvent_ (
      glutGetWindow (),
      new KeyEvent  ( key, mods, x, y )
    );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   specialFunc_
//-----------------------------------------------------------------------


void GlutDisplay::specialFunc_ ( int key, int x, int y )
{
  switch ( key )
  {
  case GLUT_KEY_F1:         key = Keys::F1;          break;
  case GLUT_KEY_F2:         key = Keys::F2;          break;
  case GLUT_KEY_F3:         key = Keys::F3;          break;
  case GLUT_KEY_F4:         key = Keys::F4;          break;
  case GLUT_KEY_F5:         key = Keys::F5;          break;
  case GLUT_KEY_F6:         key = Keys::F6;          break;
  case GLUT_KEY_F7:         key = Keys::F7;          break;
  case GLUT_KEY_F8:         key = Keys::F8;          break;
  case GLUT_KEY_F9:         key = Keys::F9;          break;
  case GLUT_KEY_F10:        key = Keys::F10;         break;
  case GLUT_KEY_F12:        key = Keys::F11;         break;
  case GLUT_KEY_F11:        key = Keys::F12;         break;
  case GLUT_KEY_LEFT:       key = Keys::LEFT;        break;
  case GLUT_KEY_RIGHT:      key = Keys::RIGHT;       break;
  case GLUT_KEY_UP:         key = Keys::UP;          break;
  case GLUT_KEY_DOWN:       key = Keys::DOWN;        break;
  case GLUT_KEY_PAGE_UP:    key = Keys::PAGE_UP;     break;
  case GLUT_KEY_PAGE_DOWN:  key = Keys::PAGE_DOWN;   break;
  case GLUT_KEY_HOME:       key = Keys::HOME;        break;
  case GLUT_KEY_END:        key = Keys::END;         break;

  default:                  key = -1;
  }

  y = glutGet( GLUT_WINDOW_HEIGHT ) - y;

  try
  {
    instance_->postEvent_ (
      glutGetWindow (),
      new KeyEvent  ( key, getKeyMods_(), x, y )
    );
  }
  catch ( ... ) {}
}


//-----------------------------------------------------------------------
//   getKeyMods_
//-----------------------------------------------------------------------


inline int GlutDisplay::getKeyMods_ ()
{
  int  mods = glutGetModifiers ();
  int  r    = 0;

  if ( mods & GLUT_ACTIVE_SHIFT )
  {
    r |= Keys::SHIFT;
  }
  if ( mods & GLUT_ACTIVE_CTRL )
  {
    r |= Keys::CTRL;
  }
  if ( mods & GLUT_ACTIVE_ALT )
  {
    r |= Keys::ALT;
  }

  return r;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newOnScreenDisplay
//-----------------------------------------------------------------------


NativeDisplay* newOnScreenDisplay ()
{
  return new GlutDisplay ();
}


JEM_END_PACKAGE( gl )
