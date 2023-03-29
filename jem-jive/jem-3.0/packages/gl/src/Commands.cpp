
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


#include <jem/gl/Commands.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Commands
//=======================================================================


const char*  Commands::ZOOM          = "Zoom";
const char*  Commands::ROTATE        = "Rotate";
const char*  Commands::TRANSLATE     = "Translate";
const char*  Commands::CENTER        = "Center";
const char*  Commands::ALIGN         = "Align";
const char*  Commands::SAVE_STATE    = "Save State";
const char*  Commands::RESTORE_STATE = "Restore State";
const char*  Commands::SAVE_SNAPSHOT = "Save Snapshot";
const char*  Commands::SHOW_OPTIONS  = "Options ...";
const char*  Commands::SELECT        = "Select ...";
const char*  Commands::HELP          = "Help";
const char*  Commands::QUIT          = "Quit";

const char*  Commands::SHOW_NAMES    = "Show Names";
const char*  Commands::HIDE_NAMES    = "Hide Names";
const char*  Commands::SHOW_AXES     = "Show Axes";
const char*  Commands::HIDE_AXES     = "Hide Axes";
const char*  Commands::SHOW_LINES    = "Show Lines";
const char*  Commands::HIDE_LINES    = "Hide Lines";
const char*  Commands::SHOW_FACES    = "Show Faces";
const char*  Commands::HIDE_FACES    = "Hide Faces";
const char*  Commands::SHOW_PALETTE  = "Show Palette";
const char*  Commands::HIDE_PALETTE  = "Hide Palette";
const char*  Commands::SHOW_MARKERS  = "Show Markers";
const char*  Commands::HIDE_MARKERS  = "Hide Markers";
const char*  Commands::SHOW_KEY      = "Show Key";
const char*  Commands::HIDE_KEY      = "Hide Key";
const char*  Commands::SHOW_XTICKS   = "Show X-Ticks";
const char*  Commands::HIDE_XTICKS   = "Hide X-Ticks";
const char*  Commands::SHOW_YTICKS   = "Show Y-Ticks";
const char*  Commands::HIDE_YTICKS   = "Hide Y-Ticks";
const char*  Commands::AUTO_SCALE    = "Auto Scale";


JEM_END_PACKAGE( gl )
