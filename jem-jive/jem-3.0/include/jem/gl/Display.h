
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

#ifndef JEM_GL_DISPLAY_H
#define JEM_GL_DISPLAY_H

#include <jem/gl/Viewer.h>


namespace jem
{
  class Thread;
}


JEM_BEGIN_PACKAGE( gl )


class ErrorHandler;


//-----------------------------------------------------------------------
//   class Display
//-----------------------------------------------------------------------


class Display
{
 public:

  static const int        NEW_THREAD;
  static const int        OFF_SCREEN;
  static const int        EXIT_ON_CLOSE;


  static int              open

    ( int                   mode = 0 );

  static int              open

    ( int&                  argc,
      char**                argv,
      int                   mode = 0 );

  static bool             isOpen        ();
  static bool             hasWindows    ();
  static int              getMode       ();
  static void             mainLoop      ();

  static Ref<Viewer>      newWindow

    ( const String&         title,
      int                   width   = -1,
      int                   height  = -1,
      Ref<ErrorHandler>     handler = nullptr );

  static Thread*          getThread     ();


 protected:

                          Display       ();
                         ~Display       ();

 private:

  static void             closedError_  ();

};


JEM_END_PACKAGE( gl )

#endif
