
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_APP_COMMANDS_H
#define JIVE_APP_COMMANDS_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   class Commands
//-----------------------------------------------------------------------


class Commands
{
 public:

  static const char*      EXIT;
  static const char*      QUIT;
  static const char*      STEP;
  static const char*      STOP;
  static const char*      CONT;
  static const char*      BG;
  static const char*      FG;
  static const char*      SET;
  static const char*      SET_VAR;
  static const char*      TOUCH_VAR;
  static const char*      EVAL;
  static const char*      SHOW;
  static const char*      SHOW_DATA;
  static const char*      LIST;
  static const char*      HELP;
  static const char*      REHASH;
  static const char*      DUMP;

};


JIVE_END_PACKAGE( app )

#endif
