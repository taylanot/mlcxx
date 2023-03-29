
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

#ifndef JEM_GL_GLFWDISPLAY_H
#define JEM_GL_GLFWDISPLAY_H

#include "NativeDisplay.h"


JEM_BEGIN_PACKAGE( gl )


class GlfwWindow;


//-----------------------------------------------------------------------
//   class GlfwDisplay
//-----------------------------------------------------------------------


class GlfwDisplay : public NativeDisplay
{
 public:

  typedef GlfwDisplay       Self;
  typedef NativeDisplay     Super;

  friend class              GlfwWindow;


                            GlfwDisplay    ();


 protected:

  virtual                  ~GlfwDisplay    ();

  virtual void              init_

    ( int&                    argc,
      char**                  argv )          override;

  virtual void              mainLoop_      () override;

  virtual int               getScreenDPI_  () override;

  virtual Ref<Window>       newWindow_

    ( const String&           title,
      int                     width,
      int                     height )        override;


 private:

  inline void               postEvent_

    ( GLFWwindow*             wptr,
      GfxEvent*               ev );

  GlfwWindow*               getWindow_

    ( GLFWwindow*             wptr );

  static void               idleFunc_      ();

  static void               resizeFunc_

    ( GLFWwindow*             wptr,
      int                     width,
      int                     height );

  static void               redrawFunc_

    ( GLFWwindow*             wptr );

  static void               mouseFunc_

    ( GLFWwindow*             wptr,
      int                     button,
      int                     action,
      int                     mods );

  static void               motionFunc_

    ( GLFWwindow*             wptr,
      double                  xpos,
      double                  ypos );

  static void               keyFunc_

    ( GLFWwindow*             wptr,
      int                     key,
      int                     code,
      int                     action,
      int                     mods );

  static void               charsFunc_

    ( GLFWwindow*             wptr,
      unsigned int            code,
      int                     mods );

  static void               closeFunc_

    ( GLFWwindow*             wptr );

  static void               toFramebuf_

    ( GLFWwindow*             wptr,
      double&                 x,
      double&                 y );

  static inline int         getKeyMods_

    ( GLFWwindow*             wptr );

  static inline int         getKeyMods_

    ( int                     mods );

  static void               errorFunc_

    ( int                     err,
      const char*             desc );

  static bool               isSpecialKey_

    ( int                     key );


 private:

  static Self*              instance_;

  util::ObjFlex             winlist_;
  int                       screenDPI_;

  Ref<GlfwWindow>           last_;

};


JEM_END_PACKAGE( gl )

#endif
