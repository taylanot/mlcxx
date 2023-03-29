
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

#ifndef JEM_GL_GLUTDISPLAY_H
#define JEM_GL_GLUTDISPLAY_H

#include "NativeDisplay.h"


JEM_BEGIN_PACKAGE( gl )


class GlutWindow;


//-----------------------------------------------------------------------
//   class GlutDisplay
//-----------------------------------------------------------------------


class GlutDisplay : public NativeDisplay
{
 public:

  typedef GlutDisplay       Self;
  typedef NativeDisplay     Super;

  friend class              GlutWindow;


                            GlutDisplay    ();


 protected:

  virtual                  ~GlutDisplay    ();

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

    ( int                     wid,
      GfxEvent*               ev );

  GlutWindow*               getWindow_

    ( int                     wid );

  static void               idleFunc_      ();

  static void               resizeFunc_

    ( int                     width,
      int                     height );

  static void               redrawFunc_    ();

  static void               mouseFunc_

    ( int                     btn,
      int                     state,
      int                     xpos,
      int                     ypos );

  static void               motionFunc_

    ( int                     xpos,
      int                     ypos );

  static void               keyFunc_

    ( unsigned char           key,
      int                     xpos,
      int                     ypos );

  static void               specialFunc_

    ( int                     key,
      int                     xpos,
      int                     ypos );

  static inline int         getKeyMods_    ();


 private:

  static Self*              instance_;

  util::ObjFlex             winlist_;

  Ref<GlutWindow>           last_;
  Ref<GlutWindow>           zombie_;

};


JEM_END_PACKAGE( gl )

#endif
