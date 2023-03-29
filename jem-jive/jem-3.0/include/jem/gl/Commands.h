
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

#ifndef JEM_GL_COMMANDS_H
#define JEM_GL_COMMANDS_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Commands
//-----------------------------------------------------------------------


class Commands
{
 public:

  static const char*    ZOOM;
  static const char*    ROTATE;
  static const char*    TRANSLATE;
  static const char*    CENTER;
  static const char*    ALIGN;
  static const char*    SAVE_STATE;
  static const char*    RESTORE_STATE;
  static const char*    SAVE_SNAPSHOT;
  static const char*    SHOW_OPTIONS;
  static const char*    SELECT;
  static const char*    HELP;
  static const char*    QUIT;

  static const char*    SHOW_NAMES;
  static const char*    HIDE_NAMES;
  static const char*    SHOW_AXES;
  static const char*    HIDE_AXES;
  static const char*    SHOW_LINES;
  static const char*    HIDE_LINES;
  static const char*    SHOW_FACES;
  static const char*    HIDE_FACES;
  static const char*    SHOW_PALETTE;
  static const char*    HIDE_PALETTE;
  static const char*    SHOW_MARKERS;
  static const char*    HIDE_MARKERS;
  static const char*    SHOW_KEY;
  static const char*    HIDE_KEY;
  static const char*    SHOW_XTICKS;
  static const char*    HIDE_XTICKS;
  static const char*    SHOW_YTICKS;
  static const char*    HIDE_YTICKS;
  static const char*    AUTO_SCALE;

};


JEM_END_PACKAGE( gl )

#endif
