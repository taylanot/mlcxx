
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

#ifndef JEM_GL_ACTIONS_H
#define JEM_GL_ACTIONS_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Actions
//-----------------------------------------------------------------------


class Actions
{
 public:

  static const char*    HIDE;
  static const char*    SHOW;
  static const char*    TAKE_FOCUS;
  static const char*    GIVE_FOCUS;
  static const char*    GIVE_INPUT;
  static const char*    LIST_OPTIONS;
  static const char*    SET_FONT;

};


//-----------------------------------------------------------------------
//   class GLActions
//-----------------------------------------------------------------------


class GLActions
{
 public:

  enum
  {
                        REDRAW,
                        REDRAW_FAST,
                        INIT_GL,
                        INIT_VIEW,
                        GRAB_FRAME,
                        DRAW_CDATA,
                        KEY_EVENT,
                        MOUSE_EVENT,
                        MOTION_EVENT,
                        CLICK2_EVENT,
                        EXE_COMMAND,

                        LAST_ACTION = EXE_COMMAND
  };

};


//-----------------------------------------------------------------------
//   class ActionParams
//-----------------------------------------------------------------------


class ActionParams
{
 public:

  static const char*    FOCUS;
  static const char*    OPTIONS;

};


//-----------------------------------------------------------------------
//   class GLActionParams
//-----------------------------------------------------------------------


class GLActionParams
{
 public:

  static const char*    DETAIL_LEVEL;
  static const char*    SNAPSHOT;

  static const char*    XJITTER;
  static const char*    YJITTER;

  static const char*    VIEW_XPOS;
  static const char*    VIEW_YPOS;
  static const char*    VIEW_WIDTH;
  static const char*    VIEW_HEIGHT;

  static const char*    FRAME_GRABBER;
  static const char*    FRAME_OUTPUT;

  static const char*    KEY_CODE;
  static const char*    KEY_MODS;
  static const char*    KEY_XPOS;
  static const char*    KEY_YPOS;

  static const char*    MOUSE_BUTTON;
  static const char*    MOUSE_MODS;
  static const char*    MOUSE_DOWN;
  static const char*    MOUSE_XPOS;
  static const char*    MOUSE_YPOS;

  static const char*    COMMAND;

};


JEM_END_PACKAGE( gl )

#endif
