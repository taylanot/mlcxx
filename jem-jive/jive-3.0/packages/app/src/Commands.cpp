
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


#include <jive/app/Commands.h>


JIVE_BEGIN_PACKAGE( app )


//=======================================================================
//   class Commands
//=======================================================================


const char*  Commands::EXIT      = "exit";
const char*  Commands::QUIT      = "quit";
const char*  Commands::STEP      = "step";
const char*  Commands::STOP      = "stop";
const char*  Commands::CONT      = "cont";
const char*  Commands::BG        = "bg";
const char*  Commands::FG        = "fg";
const char*  Commands::SET       = "set";
const char*  Commands::SET_VAR   = "set-var";
const char*  Commands::TOUCH_VAR = "touch-var";
const char*  Commands::EVAL      = "eval";
const char*  Commands::SHOW      = "show";
const char*  Commands::SHOW_DATA = "show-data";
const char*  Commands::LIST      = "list";
const char*  Commands::HELP      = "help";
const char*  Commands::REHASH    = "rehash";
const char*  Commands::DUMP      = "dump";


JIVE_END_PACKAGE( app )
