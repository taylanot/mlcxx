
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


#include <jem/gl/Actions.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Actions
//=======================================================================


#undef  JEM_DEFINE_ACTION
#define JEM_DEFINE_ACTION( name )               \
                                                \
  const char* Actions::name = #name


JEM_DEFINE_ACTION ( HIDE         );
JEM_DEFINE_ACTION ( SHOW         );
JEM_DEFINE_ACTION ( TAKE_FOCUS   );
JEM_DEFINE_ACTION ( GIVE_FOCUS   );
JEM_DEFINE_ACTION ( GIVE_INPUT   );
JEM_DEFINE_ACTION ( LIST_OPTIONS );
JEM_DEFINE_ACTION ( SET_FONT     );


//=======================================================================
//   ActionParams
//=======================================================================


#undef  JEM_DEFINE_PARAM
#define JEM_DEFINE_PARAM( NAME, name )          \
                                                \
  const char* ActionParams::NAME = #name

JEM_DEFINE_PARAM( FOCUS,   focus   );
JEM_DEFINE_PARAM( OPTIONS, options );


//=======================================================================
//   class GLActionParams
//=======================================================================


#undef  JEM_DEFINE_PARAM
#define JEM_DEFINE_PARAM( NAME, name )          \
                                                \
  const char* GLActionParams::NAME = #name


JEM_DEFINE_PARAM ( DETAIL_LEVEL,  lod          );
JEM_DEFINE_PARAM ( SNAPSHOT,      snapshot     );

JEM_DEFINE_PARAM ( XJITTER,       xjitter      );
JEM_DEFINE_PARAM ( YJITTER,       yjitter      );

JEM_DEFINE_PARAM ( VIEW_XPOS,     viewXPos     );
JEM_DEFINE_PARAM ( VIEW_YPOS,     viewYPos     );
JEM_DEFINE_PARAM ( VIEW_WIDTH,    viewWidth    );
JEM_DEFINE_PARAM ( VIEW_HEIGHT,   viewHeight   );

JEM_DEFINE_PARAM ( FRAME_GRABBER, frameGrabber );
JEM_DEFINE_PARAM ( FRAME_OUTPUT,  frameOutput  );

JEM_DEFINE_PARAM ( KEY_CODE,      keyCode      );
JEM_DEFINE_PARAM ( KEY_MODS,      keyMods      );
JEM_DEFINE_PARAM ( KEY_XPOS,      keyXPos      );
JEM_DEFINE_PARAM ( KEY_YPOS,      keyYPos      );

JEM_DEFINE_PARAM ( MOUSE_BUTTON,  mouseButton  );
JEM_DEFINE_PARAM ( MOUSE_MODS,    mouseMods    );
JEM_DEFINE_PARAM ( MOUSE_DOWN,    mouseDown    );
JEM_DEFINE_PARAM ( MOUSE_XPOS,    mouseXPos    );
JEM_DEFINE_PARAM ( MOUSE_YPOS,    mouseYPos    );

JEM_DEFINE_PARAM ( COMMAND,       command      );


JEM_END_PACKAGE( gl )
