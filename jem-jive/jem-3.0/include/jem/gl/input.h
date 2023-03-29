
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

#ifndef JEM_GL_INPUT_H
#define JEM_GL_INPUT_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   struct KeyPress
//-----------------------------------------------------------------------


class                    KeyPress
{
 public:

  int                       code;
  int                       mods;
  int                       x;
  int                       y;
};


//-----------------------------------------------------------------------
//   struct MouseClick
//-----------------------------------------------------------------------


class                     MouseClick
{
 public:

  enum
  {
                            UP,
                            DOWN,
                            SINGLE,
                            DOUBLE
  };

  int                       what;
  int                       button;
  int                       mods;
  int                       x;
  int                       y;
};


//-----------------------------------------------------------------------
//   struct MouseDrag
//-----------------------------------------------------------------------


class                     MouseDrag
{
 public:

  static const int          THRESHOLD;

  int                       button;
  int                       mods;
  int                       seqnr;
  int                       x;
  int                       y;
  int                       dx;
  int                       dy;
};


//-----------------------------------------------------------------------
//   class Buttons
//-----------------------------------------------------------------------


class Buttons
{
 public:

  static const int        LEFT;
  static const int        MIDDLE;
  static const int        RIGHT;

};


//-----------------------------------------------------------------------
//   class Keys
//-----------------------------------------------------------------------


class Keys
{
 public:

  enum
  {
                          ESCAPE = 256,

                          F1,
                          F2,
                          F3,
                          F4,
                          F5,
                          F6,
                          F7,
                          F8,
                          F9,
                          F10,
                          F11,
                          F12,

                          LEFT,
                          RIGHT,
                          UP,
                          DOWN,
                          PAGE_UP,
                          PAGE_DOWN,
                          HOME,
                          END
  };

  static const int        FIRST_CODE = ESCAPE;
  static const int        LAST_CODE  = END;

  static const int        SHIFT;
  static const int        CTRL;
  static const int        ALT;

};


JEM_END_PACKAGE( gl )

#endif
