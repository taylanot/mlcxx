
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

#ifndef JEM_GL_NATIVEDISPLAY_H
#define JEM_GL_NATIVEDISPLAY_H

#include <jem/util/Timer.h>
#include <jem/util/ObjFlex.h>
#include <jem/util/Properties.h>
#include "NativeWindow.h"


namespace jem
{
  class Thread;
}


JEM_BEGIN_PACKAGE( gl )


class FrameGrabber;


//-----------------------------------------------------------------------
//   class NativeDisplay
//-----------------------------------------------------------------------


class NativeDisplay
{
 public:

  typedef NativeDisplay     Self;
  typedef NativeWindow      Window;
  typedef OutputStream      Output;
  typedef FrameGrabber      Grabber;


  void                      init

    ( int&                    argc,
      char**                  argv,
      int                     mode );

  void                      mainLoop       ();

  int                       getScreenDPI   ();

  Ref<Window>               newWindow

    ( const String&           title,
      int                     width,
      int                     height,
      Ref<ErrorHandler>       handler );

  void                      freezeWindow

    ( const Ref<Window>&      win );

  void                      resumeWindow

    ( const Ref<Window>&      win,
      const Ref<GfxNode>&     node );

  void                      closeWindow

    ( const Ref<Window>&      win );

  void                      setScene

    ( const Ref<Window>&      win,
      const Ref<GfxNode>&     scene );

  void                      grabFrame

    ( const Ref<Window>&      win,
      const Ref<Output>&      out,
      const Ref<Grabber>&     grabber,
      const Ref<GfxNode>&     scene );

  void                      flushEvents

    ( const Ref<Window>&      win,
      const Ref<GfxNode>&     node );

  bool                      hasWindows     () const;

  bool                      shouldClose

    ( const Ref<Window>&      win );

  inline int                getMode        () const;
  inline bool               initialized    () const;
  static inline Self*       getInstance    ();
  static inline Thread*     getThread      ();


 protected:

                            NativeDisplay  ();
  virtual                  ~NativeDisplay  ();

  void                      popCommand_

    ( double                  sleep );

  void                      postEvent_

    ( Window&                 win,
      GfxEvent*               ev );


 protected:

  virtual void              init_

    ( int&                    argc,
      char**                  argv )          = 0;

  virtual void              mainLoop_      () = 0;

  virtual int               getScreenDPI_  () = 0;

  virtual Ref<Window>       newWindow_

    ( const String&           title,
      int                     width,
      int                     height )        = 0;


 private:

  class                     Command_;
  class                     InitCmd_;
  class                     GetDpiCmd_;
  class                     MainLoopCmd_;
  class                     NewWinCmd_;
  class                     FreezeCmd_;
  class                     ResumeCmd_;
  class                     CloseWinCmd_;
  class                     SetSceneCmd_;
  class                     GrabFrameCmd_;
  class                     FlushCmd_;
  class                     TestCloseCmd_;
  class                     CmdQueue_;
  class                     Thread_;

  friend class              Command_;
  friend class              InitCmd_;
  friend class              GetDpiCmd_;
  friend class              MainLoopCmd_;
  friend class              NewWinCmd_;
  friend class              FreezeCmd_;
  friend class              ResumeCmd_;
  friend class              CloseWinCmd_;
  friend class              SetSceneCmd_;
  friend class              GrabFrameCmd_;
  friend class              FlushCmd_;
  friend class              TestCloseCmd_;
  friend class              CmdQueue_;
  friend class              Thread_;


 private:

  void                      redrawFast_

    ( Window&                 win,
      GfxNode&                scene );

  inline bool               takeAction_

    ( Window&                 win,
      GfxNode&                root,
      int                     action );

  void                      flushEvents_

    ( Window&                 win,
      GfxNode&                scene );

  void                      sendEvents_

    ( Window&                 win,
      GfxNode&                scene );

  static void               redrawCallback_

    ( RenderContext&          rtx,
      const Ref<Object>&      obj );


 private:

  static const double       MAX_DRAW_DELAY_;
  static const double       MIN_DETAIL_LEVEL_;

  static const int          INITIALIZED_;
  static const int          IN_MAIN_LOOP_;

  static Self*              instance_;
  static Thread*            thread_;

  int                       mode_;
  int                       state_;
  Ref<CmdQueue_>            cmds_;
  EventQueue                events_;
  util::ObjFlex             winlist_;
  Properties                params_;
  util::Timer               timer_;
  double                    lod_;
  double                    avgLod_;
  double                    delay_;
  double                    sleep_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


NativeDisplay*              newOnScreenDisplay  ();
NativeDisplay*              newOffScreenDisplay ();



//#######################################################################
//   Implementation
//#######################################################################


//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


inline int NativeDisplay::getMode () const
{
  return mode_;
}


//-----------------------------------------------------------------------
//   initialized
//-----------------------------------------------------------------------


inline bool NativeDisplay::initialized () const
{
  return ((state_ & INITIALIZED_) != 0);
}


//-----------------------------------------------------------------------
//   getInstance
//-----------------------------------------------------------------------


inline NativeDisplay* NativeDisplay::getInstance ()
{
  return instance_;
}


//-----------------------------------------------------------------------
//   getThread
//-----------------------------------------------------------------------


inline Thread* NativeDisplay::getThread ()
{
  return thread_;
}


JEM_END_PACKAGE( gl )

#endif
