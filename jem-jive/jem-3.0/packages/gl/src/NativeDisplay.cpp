
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
#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/io/OutputStream.h>
#include <jem/util/List.h>
#include <jem/util/Timer.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Actions.h>
#include <jem/gl/GfxNode.h>
#include <jem/gl/GfxError.h>
#include <jem/gl/Display.h>
#include <jem/gl/ErrorHandler.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/FrameGrabber.h>
#include "NativeDisplay.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class NativeDisplay::Command_
//=======================================================================


class NativeDisplay::Command_ : public Collectable
{
 public:

  typedef NativeDisplay     Display;


  void                      exec

    ( Display&                disp );

  void                      wait      ();

  virtual void              run

    ( Display&                disp )      = 0;


 protected:

  explicit inline           Command_

    ( const Ref<Window>&      win = nullptr );


 protected:

  Ref<Window>               win_;
  Ref<GfxError>             err_;

  Monitor                   mon_;
  bool                      ready_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline NativeDisplay::Command_::Command_

  ( const Ref<Window>& win ) : win_ ( win )

{
  ready_ = false;
}


//-----------------------------------------------------------------------
//   exec
//-----------------------------------------------------------------------


void NativeDisplay::Command_::exec ( Display& disp )
{
  try
  {
    run ( disp );
  }
  catch ( const Throwable& ex )
  {
    if ( win_ && win_->errHandler )
    {
      win_->errHandler->error ( ex );
    }
    else
    {
      err_ = newInstance<GfxError> ( ex );
    }
  }
  catch ( ... )
  {
    if ( win_ && win_->errHandler )
    {
      win_->errHandler->error ();
    }
    else
    {
      err_ = newInstance<GfxError> ();
    }
  }

  mon_.lock   ();

  ready_ = true;

  mon_.notify ();
  mon_.unlock ();
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


void NativeDisplay::Command_::wait ()
{
  Lock<Monitor>  lock ( mon_ );

  if ( ! ready_ )
  {
    mon_.wait ();
  }

  if ( err_ != nullptr )
  {
    err_->rethrow ();
  }
}


//=======================================================================
//   class InitCmd_
//=======================================================================


class NativeDisplay::InitCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline                    InitCmd_

    ( int*                    argc,
      char**                  argv );

  virtual void              run

    ( Display&                disp )  override;


 private:

  int*                      argc_;
  char**                    argv_;

};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline NativeDisplay::InitCmd_::InitCmd_

  ( int*    argc,
    char**  argv ) :

    argc_ ( argc ),
    argv_ ( argv )

{}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::InitCmd_::run ( Display& disp )
{
  disp.init_ ( *argc_, argv_ );
}


//=======================================================================
//   class GetDpiCmd_
//=======================================================================


class NativeDisplay::GetDpiCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline                    GetDpiCmd_   ();

  virtual void              run

    ( Display&                disp )        override;

  inline int                getDPI       () const;


 private:

  int                        dpi_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::GetDpiCmd_::GetDpiCmd_ ()
{
  dpi_ = 0;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::GetDpiCmd_::run ( Display& disp )
{
  dpi_ = disp.getScreenDPI_ ();
}


//-----------------------------------------------------------------------
//   getDPI
//-----------------------------------------------------------------------


inline int NativeDisplay::GetDpiCmd_::getDPI () const
{
  return dpi_;
}


//=======================================================================
//   class MainLoopCmd_
//=======================================================================


class NativeDisplay::MainLoopCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline                    MainLoopCmd_  ();

  virtual void              run

    ( Display&                disp )         override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::MainLoopCmd_::MainLoopCmd_ ()
{}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::MainLoopCmd_::run ( Display& disp )
{}


//=======================================================================
//   class NewWinCmd_
//=======================================================================


class NativeDisplay::NewWinCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline                    NewWinCmd_

    ( const String&           title,
      int                     width,
      int                     height,
      Ref<ErrorHandler>       handler );

  virtual void              run

    ( Display&                disp )     override;

  inline Window*            getWindow () const;


 private:

  String                    title_;
  int                       width_;
  int                       height_;
  Ref<ErrorHandler>         handler_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::NewWinCmd_::NewWinCmd_

  ( const String&      title,
    int                width,
    int                height,
    Ref<ErrorHandler>  handler ) :

    title_   ( title   ),
    width_   ( width   ),
    height_  ( height  ),
    handler_ ( handler )

{}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::NewWinCmd_::run ( Display& disp )
{
  win_ = disp.newWindow_ ( title_, width_, height_ );

  win_->errHandler = handler_;

  disp.winlist_.pushBack ( win_.get() );
}


//-----------------------------------------------------------------------
//   getWindow
//-----------------------------------------------------------------------


inline NativeWindow* NativeDisplay::NewWinCmd_::getWindow () const
{
  return win_.get ();
}


//=======================================================================
//   class NativeDisplay::FreezeCmd_
//=======================================================================


class NativeDisplay::FreezeCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  explicit inline           FreezeCmd_

    ( const Ref<Window>&      win );

  virtual void              run

    ( Display&                disp )  override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::FreezeCmd_::FreezeCmd_

  ( const Ref<Window>& win ) : Super ( win )

{
  JEM_ASSERT ( win );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::FreezeCmd_::run ( Display& disp )
{
  if ( win_->scene )
  {
    disp.flushEvents_ ( *win_, *win_->scene );

    err_        = win_->err;
    win_->err   = nullptr;
    win_->scene = nullptr;
  }
}


//=======================================================================
//   class NativeDisplay::ResumeCmd_
//=======================================================================


class NativeDisplay::ResumeCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline                    ResumeCmd_

    ( const Ref<Window>&      win,
      const Ref<GfxNode>&     scene );

  virtual void              run

    ( Display&                disp )  override;


 private:

  Ref<GfxNode>              scene_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


NativeDisplay::ResumeCmd_::ResumeCmd_

  (  const Ref<Window>&   win,
     const Ref<GfxNode>&  scene ) :

    Super  ( win   ),
    scene_ ( scene )

{
  JEM_ASSERT ( win != nullptr && scene != nullptr );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::ResumeCmd_::run ( Display& disp )
{
  win_->scene = scene_;
  win_->err   = nullptr;
}


//=======================================================================
//   class NativeDisplay::CloseWinCmd_
//=======================================================================


class NativeDisplay::CloseWinCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  explicit inline           CloseWinCmd_

    ( const Ref<Window>&      win );

  virtual void              run

    ( Display&                disp )  override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::CloseWinCmd_::CloseWinCmd_

  ( const Ref<Window>& win ) :

    Super ( win )

{
  JEM_ASSERT ( win );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::CloseWinCmd_::run ( Display& disp )
{
  win_        ->close ();
  disp.winlist_.erase ( win_.get() );

  win_ = nullptr;
  err_ = nullptr;
}


//=======================================================================
//   class NativeDisplay::SetSceneCmd_
//=======================================================================


class NativeDisplay::SetSceneCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline                    SetSceneCmd_

    ( const Ref<Window>&      win,
      const Ref<GfxNode>&     scene );

  virtual void              run

    ( Display&                disp )  override;


 private:

  Ref<GfxNode>              scene_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::SetSceneCmd_::SetSceneCmd_

  ( const Ref<Window>&   win,
    const Ref<GfxNode>&  scene ) :

  Super  ( win   ),
  scene_ ( scene )

{
  JEM_ASSERT ( win && scene );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::SetSceneCmd_::run ( Display& disp )
{
  win_->mon.lock ();

  win_->events.clear ();
  win_->events.push  ( new InitEvent  () );
  win_->events.push  ( new ResizeEvent() );
  win_->events.push  ( new RedrawEvent() );

  win_->mon.unlock ();

  win_->callbacks.clear ();

  win_->err = nullptr;

  if ( win_->scene )
  {
    win_->scene = scene_;
  }
}


//=======================================================================
//   class NativeDisplay::GrabFrameCmd_
//=======================================================================


class NativeDisplay::GrabFrameCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline                    GrabFrameCmd_

    ( const Ref<Window>&      win,
      const Ref<Output>&      out,
      const Ref<Grabber>&     grabber,
      const Ref<GfxNode>&     scene );

  virtual void              run

    ( Display&                disp )  override;


 private:

  Ref<Output>               output_;
  Ref<Grabber>              grabber_;
  Ref<GfxNode>              scene_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::GrabFrameCmd_::GrabFrameCmd_

  (  const Ref<Window>&   win,
     const Ref<Output>&   output,
     const Ref<Grabber>&  grabber,
     const Ref<GfxNode>&  scene ) :

    Super    ( win     ),
    output_  ( output  ),
    grabber_ ( grabber ),
    scene_   ( scene   )

{
  JEM_ASSERT ( win     != nullptr && output != nullptr &&
               grabber != nullptr && scene  != nullptr );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::GrabFrameCmd_::run ( Display& disp )
{
  typedef GLActionParams  Params;

  if ( win_->events.size() )
  {
    disp.flushEvents_ ( *win_, *scene_ );
  }

  if ( win_->err )
  {
    err_      = win_->err;
    win_->err = nullptr;

    return;
  }

  disp.params_.set   ( Params::FRAME_GRABBER, grabber_ );
  disp.params_.set   ( Params::FRAME_OUTPUT,  output_  );
  disp.takeAction_   ( *win_, *scene_, (int) GLActions::GRAB_FRAME );
  disp.params_.erase ( Params::FRAME_GRABBER );
  disp.params_.erase ( Params::FRAME_OUTPUT  );
}


//=======================================================================
//   class NativeDisplay::FlushCmd_
//=======================================================================


class NativeDisplay::FlushCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline                    FlushCmd_

    ( const Ref<Window>&      win,
      const Ref<GfxNode>&     scene );

  virtual void              run

    ( Display&                disp )  override;


 private:

  Ref<GfxNode>              scene_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::FlushCmd_::FlushCmd_

  ( const Ref<Window>&   win,
    const Ref<GfxNode>&  scene ) :

  Super  ( win   ),
  scene_ ( scene )

{
  JEM_ASSERT ( win != nullptr && scene != nullptr );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::FlushCmd_::run ( Display& disp )
{
  disp.flushEvents_ ( *win_, *scene_ );

  err_      = win_->err;
  win_->err = nullptr;
}


//=======================================================================
//   class TestCloseCmd_
//=======================================================================


class NativeDisplay::TestCloseCmd_ : public Command_
{
 public:

  typedef Command_          Super;


  inline explicit           TestCloseCmd_

    ( const Ref<Window>&      win );

  virtual void              run

    ( Display&                disp )        override;

  inline bool               shouldClose  () const;


 private:

  bool                      shouldClose_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NativeDisplay::TestCloseCmd_::TestCloseCmd_

  ( const Ref<Window>&  win ) :

    Super ( win )

{
  shouldClose_ = false;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::TestCloseCmd_::run ( Display& disp )
{
  shouldClose_ = win_->shouldClose ();
}


//-----------------------------------------------------------------------
//   shouldClose
//-----------------------------------------------------------------------


inline bool NativeDisplay::TestCloseCmd_::shouldClose () const
{
  return shouldClose_;
}


//=======================================================================
//   class NativeDisplay::CmdQueue_
//=======================================================================


class NativeDisplay::CmdQueue_ : public Collectable
{
 public:

  inline void               post

    ( const Ref<Command_>&    cmd );

  inline Ref<Command_>      poll      ();
  inline Ref<Command_>      wait      ();
  inline Ref<Command_>      wait

    ( double                  timeout );


 private:

  Monitor                   mon_;
  util::List
    < Ref<Command_> >       cmds_;

};


//-----------------------------------------------------------------------
//   post
//-----------------------------------------------------------------------


inline void NativeDisplay::CmdQueue_::post

  ( const Ref<Command_>& cmd )

{
  JEM_ASSERT ( cmd );

  Lock<Monitor>  lock ( mon_ );

  cmds_.pushBack ( cmd );

  mon_.notify ();
}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


inline Ref<NativeDisplay::Command_>

  NativeDisplay::CmdQueue_::poll ()

{
  Ref<Command_>  cmd;

  {
    Lock<Monitor>  lock ( mon_ );

    if ( cmds_.size() )
    {
      cmd = cmds_.front ();
      cmds_.popFront    ();
    }
  }

  return cmd;
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


inline Ref<NativeDisplay::Command_>

  NativeDisplay::CmdQueue_::wait ()

{
  Ref<Command_>  cmd;

  {
    Lock<Monitor>  lock ( mon_ );

    while ( cmds_.size() == 0 )
    {
      mon_.wait ();
    }

    cmd = cmds_.front ();
    cmds_.popFront    ();
  }

  return cmd;
}


inline Ref<NativeDisplay::Command_>

  NativeDisplay::CmdQueue_::wait ( double timeout )

{
  Ref<Command_>  cmd;

  {
    Lock<Monitor>  lock ( mon_ );

    if      ( cmds_.size() )
    {
      cmd = cmds_.front ();
      cmds_.popFront    ();
    }
    else if ( timeout > 0.01 )
    {
      mon_.wait ( Time( timeout ) );

      if ( cmds_.size() )
      {
        cmd = cmds_.front ();
        cmds_.popFront    ();
      }
    }
  }

  return cmd;
}


//=======================================================================
//   class NativeDisplay::Thread_
//=======================================================================


class NativeDisplay::Thread_ : public Thread
{
 public:

  typedef Thread            Super;
  typedef Thread_           Self;


                            Thread_   ();

  virtual void              run       () override;


 protected:

  virtual                  ~Thread_   ();

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NativeDisplay::Thread_::Thread_ ()
{
  NativeDisplay::thread_ = this;
}


NativeDisplay::Thread_::~Thread_ ()
{
  NativeDisplay::thread_ = 0;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void NativeDisplay::Thread_::run ()
{
  NativeDisplay*  disp = instance_;

  while ( true )
  {
    Ref<Command_> cmd = disp->cmds_->wait ();

    cmd->exec ( *disp );

    if ( dynamicCast<MainLoopCmd_*>( cmd ) )
    {
      break;
    }
  }

  disp->state_ |= IN_MAIN_LOOP_;

  // Wait until at least one window has been created.

  while ( disp->winlist_.size() == 0 )
  {
    Ref<Command_> cmd = disp->cmds_->wait ();

    cmd->exec ( *disp );
  }

  disp->mainLoop_ ();
}


//=======================================================================
//   class Display
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double    NativeDisplay::MAX_DRAW_DELAY_   = 0.1;
const double    NativeDisplay::MIN_DETAIL_LEVEL_ = 0.01;

const int       NativeDisplay::INITIALIZED_      = 1 << 0;
const int       NativeDisplay::IN_MAIN_LOOP_     = 1 << 1;

NativeDisplay*  NativeDisplay::instance_         = nullptr;
Thread*         NativeDisplay::thread_           = nullptr;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


NativeDisplay::NativeDisplay ()
{
  if ( instance_ )
  {
    throw IllegalOperationException (
      JEM_FUNC,
      "another NativeDisplay already exists"
    );
  }

  instance_ =  this;
  mode_     =  0;
  state_    =  0;
  cmds_     =  newInstance<CmdQueue_> ();
  lod_      = -1.0;
  avgLod_   = -1.0;
  delay_    =  2.0;
  sleep_    =  0.0;
}


NativeDisplay::~NativeDisplay ()
{
  instance_ = nullptr;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void NativeDisplay::init

  ( int&    argc,
    char**  argv,
    int     mode )

{
  if ( state_ & INITIALIZED_ )
  {
    return;
  }

  mode_   = mode;
  state_ |= INITIALIZED_;

  if ( mode & Display::NEW_THREAD )
  {
    Ref<Thread>    thr = newInstance<Thread_>  ();
    Ref<Command_>  cmd = newInstance<InitCmd_> ( &argc, argv );

    thr  ->start ();
    cmds_->post  ( cmd );
    cmd  ->wait  ();
  }
  else
  {
    thread_ = Thread::getCurrent ();

    init_ ( argc, argv );
  }
}


//-----------------------------------------------------------------------
//   getScreenDPI
//-----------------------------------------------------------------------


int NativeDisplay::getScreenDPI ()
{
  JEM_PRECHECK ( initialized() );

  if ( Thread::getCurrent() == thread_ )
  {
    return getScreenDPI_ ();
  }
  else
  {
    Ref<GetDpiCmd_>  cmd = newInstance<GetDpiCmd_> ();

    cmds_->post ( cmd );
    cmd  ->wait ();

    return cmd->getDPI ();
  }
}


//-----------------------------------------------------------------------
//   mainLoop
//-----------------------------------------------------------------------


void NativeDisplay::mainLoop ()
{
  JEM_PRECHECK ( initialized() );

  if ( Thread::getCurrent() == thread_ )
  {
    if ( state_ & IN_MAIN_LOOP_ )
    {
      return;
    }

    state_ |= IN_MAIN_LOOP_;

    // Wait until at least one window has been created.

    while ( winlist_.size() == 0 )
    {
      Ref<Command_>  cmd = cmds_->wait ();

      cmd->exec ( *this );
    }

    mainLoop_ ();
  }
  else
  {
    Ref<Command_>  cmd = newInstance<MainLoopCmd_> ();

    cmds_->post ( cmd );
    cmd  ->wait ();
  }
}


//-----------------------------------------------------------------------
//   newWindow
//-----------------------------------------------------------------------


Ref<NativeWindow> NativeDisplay::newWindow

  ( const String&      title,
    int                width,
    int                height,
    Ref<ErrorHandler>  handler )

{
  JEM_PRECHECK ( initialized() );

  Ref<NewWinCmd_>  cmd =

    newInstance<NewWinCmd_> ( title, width, height, handler );

  if ( Thread::getCurrent() == thread_ )
  {
    cmd->run ( *this );
  }
  else
  {
    cmds_->post ( cmd );
    cmd  ->wait ();
  }

  return cmd->getWindow ();
}


//-----------------------------------------------------------------------
//   freezeWindow
//-----------------------------------------------------------------------


void NativeDisplay::freezeWindow ( const Ref<Window>& win )
{
  JEM_PRECHECK ( initialized() );

  Ref<Command_>  cmd = newInstance<FreezeCmd_> ( win );

  if ( Thread::getCurrent() == thread_ )
  {
    cmd->run ( *this );
  }
  else
  {
    cmds_->post ( cmd );
    cmd  ->wait ();
  }
}


//-----------------------------------------------------------------------
//   resumeWindow
//-----------------------------------------------------------------------


void NativeDisplay::resumeWindow

  ( const Ref<Window>&   win,
    const Ref<GfxNode>&  node )

{
  JEM_PRECHECK ( initialized() );

  Ref<Command_>  cmd = newInstance<ResumeCmd_> ( win, node );

  if ( Thread::getCurrent() == thread_ )
  {
    cmd->run ( *this );
  }
  else
  {
    cmds_->post ( cmd );
  }
}


//-----------------------------------------------------------------------
//   closeWindow
//-----------------------------------------------------------------------


void NativeDisplay::closeWindow ( const Ref<Window>& win )
{
  JEM_PRECHECK ( initialized() );

  Ref<Command_>  cmd = newInstance<CloseWinCmd_> ( win );

  if ( Thread::getCurrent() == thread_ )
  {
    cmd->run ( *this );
  }
  else
  {
    cmds_->post ( cmd );
    cmd  ->wait ();
  }
}


//-----------------------------------------------------------------------
//   setScene
//-----------------------------------------------------------------------


void NativeDisplay::setScene

  ( const Ref<Window>&   win,
    const Ref<GfxNode>&  scene )

{
  JEM_PRECHECK ( initialized() );

  Ref<Command_>  cmd = newInstance<SetSceneCmd_> ( win, scene );

  if ( Thread::getCurrent() == thread_ )
  {
    cmd->run ( *this );
  }
  else
  {
    cmds_->post ( cmd );
    cmd  ->wait ();
  }
}


//-----------------------------------------------------------------------
//   grabFrame
//-----------------------------------------------------------------------


void NativeDisplay::grabFrame

  ( const Ref<Window>&   win,
    const Ref<Output>&   out,
    const Ref<Grabber>&  grabber,
    const Ref<GfxNode>&  scene )

{
  JEM_PRECHECK ( initialized() );

  Ref<Command_>  cmd =

    newInstance<GrabFrameCmd_> ( win, out, grabber, scene );

  if ( Thread::getCurrent() == thread_ )
  {
    cmd->run ( *this );
  }
  else
  {
    cmds_->post ( cmd );
    cmd  ->wait ();
  }
}


//-----------------------------------------------------------------------
//   flushEvents
//-----------------------------------------------------------------------


void NativeDisplay::flushEvents

  ( const Ref<Window>&   win,
    const Ref<GfxNode>&  node )

{
  JEM_PRECHECK ( initialized() );

  Ref<Command_>  cmd = newInstance<FlushCmd_> ( win, node );

  if ( Thread::getCurrent() == thread_ )
  {
    cmd->run ( *this );
  }
  else
  {
    cmds_->post ( cmd );
    cmd  ->wait ();
  }
}


//-----------------------------------------------------------------------
//   hasWindows
//-----------------------------------------------------------------------


bool NativeDisplay::hasWindows () const
{
  return (winlist_.size() > 0);
}


//-----------------------------------------------------------------------
//   shouldClose
//-----------------------------------------------------------------------


bool NativeDisplay::shouldClose ( const Ref<Window>& win )
{
  JEM_PRECHECK ( initialized() );

  bool  retval = false;

  if ( win )
  {
    Ref<TestCloseCmd_>  cmd = newInstance<TestCloseCmd_> ( win );

    if ( Thread::getCurrent() == thread_ )
    {
      cmd->run ( *this );
    }
    else
    {
      cmds_->post ( cmd );
      cmd  ->wait ();
    }

    retval = cmd->shouldClose ();
  }

  return retval;
}


//-----------------------------------------------------------------------
//   popCommand_
//-----------------------------------------------------------------------


void NativeDisplay::popCommand_ ( double sleep )
{
  JEM_ASSERT ( Thread::getCurrent() == getThread() );

  Ref<Command_>  cmd;
  Window*        win;
  Time           tleft;
  double         dt;


  for ( idx_t i = 0; i < winlist_.size(); i++ )
  {
    win = winlist_.getAs<Window> ( i );

    // Skip frozen windows.

    if ( ! win->scene )
    {
      continue;
    }

    // To preserve the order of events they must be flushed before the
    // callbacks are invoked

    if ( win->events.size() )
    {
      flushEvents_ ( *win, *win->scene );
    }
    else
    {
      // Make sure that the work load is updated.

      win->stopWork ();
    }

    win->callbacks.invoke ( tleft, *win );

    dt = tleft.toDouble ();

    if ( dt < sleep_ )
    {
      sleep_ = dt;

      if ( sleep_ < 0.0 )
      {
        sleep_ = 0.0;
      }
    }
  }

  cmd = cmds_->wait ( jem::min( sleep_, sleep ) );

  if      ( cmd )
  {
    sleep_ = 0.0;

    cmd->exec ( *this );
  }
  else if ( sleep_ < 10.0 )
  {
    sleep_ += 0.02;
  }
}


//-----------------------------------------------------------------------
//   postEvent_
//-----------------------------------------------------------------------


void NativeDisplay::postEvent_ ( Window& win, GfxEvent* ev )
{
  JEM_ASSERT ( Thread::getCurrent() == getThread() );


  sleep_ = 0.0;

  if ( win.err )
  {
    delete ev;
  }
  else
  {
    win.mon.lock ();

    idx_t  n = win.events.push ( ev );

    win.mon.notify ();
    win.mon.unlock ();

    // Events are flushed in the idle loop, except when there is a
    // significant backlog.

    if ( win.scene != nullptr && n > 5 )
    {
      flushEvents_ ( win, *win.scene );
    }
  }
}


//-----------------------------------------------------------------------
//   redrawFast_
//-----------------------------------------------------------------------


void NativeDisplay::redrawFast_

  ( Window&   win,
    GfxNode&  scene )

{
  GLContext&  gtx = win.getGLContext ();

  double      newLod;


  timer_.reset ();
  timer_.start ();

  if ( lod_ < 0.0 )
  {
    lod_ = 0.05;

    params_.set ( GLActionParams::DETAIL_LEVEL, lod_ );
    takeAction_ ( win, scene, (int) GLActions::REDRAW_FAST );
    finishGLOps ( gtx );

    lod_ *= MAX_DRAW_DELAY_ / (timer_.toDouble() + 0.001);

    clamp ( lod_, MIN_DETAIL_LEVEL_, 1.0 );

    timer_.reset ();
  }

  params_.set ( GLActionParams::DETAIL_LEVEL, lod_ );
  takeAction_ ( win, scene, (int) GLActions::REDRAW_FAST );
  finishGLOps ( gtx );
  timer_.stop ();

  win.callbacks.update ( redrawCallback_, & scene,
                         Time::now() + Time( delay_ ) );

  newLod = lod_ * MAX_DRAW_DELAY_ / (timer_.toDouble() + 0.001);

  if ( avgLod_ < 0.0 )
  {
    avgLod_ = 0.5 * (newLod + lod_);
    lod_    = clamped ( avgLod_, MIN_DETAIL_LEVEL_, 1.0 );
  }
  else
  {
    avgLod_ = 0.75 * avgLod_ + 0.25 * newLod;

    if ( 1.1 * avgLod_ < lod_ )
    {
      lod_ = clamped ( avgLod_, MIN_DETAIL_LEVEL_, 1.0 );
    }
  }
}


//-----------------------------------------------------------------------
//   takeAction_
//-----------------------------------------------------------------------


inline bool NativeDisplay::takeAction_

  ( Window&   win,
    GfxNode&  node,
    int       action )

{
  win.incrActionID ();

  return node.takeGLAction ( win, action, params_ );
}


//-----------------------------------------------------------------------
//   flushEvents_
//-----------------------------------------------------------------------


void NativeDisplay::flushEvents_ ( Window& win, GfxNode& scene )
{
  if ( win.err )
  {
    return;
  }

  win.startWork ();

  try
  {
    sendEvents_ ( win, scene );
  }
  catch ( const Throwable& ex )
  {
    if ( ! win.errHandler )
    {
      win.err = newInstance<GfxError> ( ex );
    }
    else
    {
      win.errHandler->error ( ex );
    }
  }
  catch ( ... )
  {
    if ( ! win.errHandler )
    {
      win.err = newInstance<GfxError> ();
    }
    else
    {
      win.errHandler->error ();
    }
  }

  win.stopWork ();
}


//-----------------------------------------------------------------------
//   sendEvents_
//-----------------------------------------------------------------------


void NativeDisplay::sendEvents_

  ( Window&   win,
    GfxNode&  scene )

{
  typedef GLActionParams  Params;

  EventPtr  ev;


  win.setCurrent ();

  if ( events_.size() )
  {
    events_.clear ();
  }

  win.mon   .lock   ();
  win.events.swap   ( events_ );
  win.mon   .unlock ();

  while ( events_.size() )
  {
    ev = events_.pop ();

    switch ( ev->type )
    {
    case GfxEvent::NOP:

      break;

    case GfxEvent::REDRAW:

      {
        RedrawEvent&  rev = static_cast<RedrawEvent&> ( *ev );
        GfxNode*      rsc = rev.scene.get ();

        if ( ! rsc )
        {
          rsc = & scene;
        }

        // Convert external redraw events to fast redraw events if a
        // normal redraw is pending. This is to improve the response
        // time of the viewer window.

        if ( (rev.flags & RedrawEvent::FAST)     == 0 &&
             (rev.flags & RedrawEvent::EXTERNAL) != 0 &&
             win.callbacks.contains( redrawCallback_, rsc ) )
        {
          rev.flags |= RedrawEvent::FAST;
        }

        if ( (rev.flags & RedrawEvent::FAST) &&
             (delay_ >= (2.0 * MAX_DRAW_DELAY_)) )
        {
          redrawFast_ ( win, *rsc );
        }
        else
        {
          timer_.reset ();
          timer_.start ();
          takeAction_  ( win, *rsc, (int) GLActions::REDRAW );
          finishGLOps  ( win.getGLContext() );
          timer_.stop  ();

          win.callbacks.erase ( redrawCallback_, rsc );

          delay_ = 2.0 * timer_.toDouble ();

          clamp ( delay_, MAX_DRAW_DELAY_, 4.0 );

          if ( lod_ > 0.0 && avgLod_ > 1.1 * lod_ )
          {
            lod_ = clamped ( avgLod_, MIN_DETAIL_LEVEL_, 1.0 );
          }
        }

        win.swapBuffers ();
      }

      break;

    case GfxEvent::RESIZE:

      {
        Size2i  ws = win.getWindowSize ();

        params_.set ( Params::VIEW_XPOS,   0 );
        params_.set ( Params::VIEW_YPOS,   0 );
        params_.set ( Params::VIEW_WIDTH,  (int) ws.width  );
        params_.set ( Params::VIEW_HEIGHT, (int) ws.height );

        takeAction_ ( win, scene, (int) GLActions::INIT_VIEW );
      }

      break;

    case GfxEvent::INIT:

      takeAction_ ( win, scene, (int) GLActions::INIT_GL );

      break;

    case GfxEvent::MOUSE:

      {
        MouseEvent&  mev = static_cast<MouseEvent&> ( *ev );

        win.mousePos[0] = mev.xPos;
        win.mousePos[1] = mev.yPos;

        params_.set ( Params::MOUSE_BUTTON, mev.button );
        params_.set ( Params::MOUSE_MODS,   mev.mods   );
        params_.set ( Params::MOUSE_DOWN,   mev.down   );
        params_.set ( Params::MOUSE_XPOS,   mev.xPos   );
        params_.set ( Params::MOUSE_YPOS,   mev.yPos   );

        takeAction_ ( win, scene, (int) GLActions::MOUSE_EVENT );
      }

      break;

    case GfxEvent::MOTION:

      {
        MotionEvent&  mev = static_cast<MotionEvent&> ( *ev );

        win.mousePos[0] = mev.xPos;
        win.mousePos[1] = mev.yPos;

        params_.set ( Params::MOUSE_XPOS, mev.xPos );
        params_.set ( Params::MOUSE_YPOS, mev.yPos );

        takeAction_ ( win, scene, (int) GLActions::MOTION_EVENT );
      }

      break;

    case GfxEvent::CLICK2:

      {
        Click2Event&  cl2 = static_cast<Click2Event&> ( *ev );

        win.mousePos[0] = cl2.xPos;
        win.mousePos[1] = cl2.yPos;

        params_.set ( Params::MOUSE_BUTTON, cl2.button );
        params_.set ( Params::MOUSE_MODS,   cl2.mods   );
        params_.set ( Params::MOUSE_XPOS,   cl2.xPos   );
        params_.set ( Params::MOUSE_YPOS,   cl2.yPos   );

        takeAction_ ( win, scene, (int) GLActions::CLICK2_EVENT );
      }

      break;

    case GfxEvent::KEY:

      {
        KeyEvent&  kev = static_cast<KeyEvent&> ( *ev );

        win.mousePos[0] = kev.xPos;
        win.mousePos[1] = kev.yPos;

        params_.set ( Params::KEY_CODE, kev.key  );
        params_.set ( Params::KEY_MODS, kev.mods );
        params_.set ( Params::KEY_XPOS, kev.xPos );
        params_.set ( Params::KEY_YPOS, kev.yPos );

        takeAction_ ( win, scene, (int) GLActions::KEY_EVENT );
      }

      break;

    case GfxEvent::COMMAND:

      {
        CommandEvent&  cev = static_cast<CommandEvent&> ( *ev );

        params_.set ( Params::COMMAND, cev.command );
        takeAction_ ( win, scene, (int) GLActions::EXE_COMMAND );
      }

      break;
    }
  }
}


//-----------------------------------------------------------------------
//   redrawCallback_
//-----------------------------------------------------------------------


void NativeDisplay::redrawCallback_

  ( RenderContext&      rtx,
    const Ref<Object>&  obj )

{
  rtx.postRedraw ( staticCast<GfxNode>( obj ) );
}


JEM_END_PACKAGE( gl )
