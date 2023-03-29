
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

#ifndef JEM_GL_PROPERTYNAMES_H
#define JEM_GL_PROPERTYNAMES_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class PropertyNames
//-----------------------------------------------------------------------


class PropertyNames
{
 public:

  static const char*    AMBIENT;
  static const char*    ANGLE;
  static const char*    AUTO_QUALITY;
  static const char*    AXIS;
  static const char*    BASE_NUMBER;
  static const char*    BG_COLOR;
  static const char*    COLOR;
  static const char*    COLORS;
  static const char*    DIRECTION;
  static const char*    ENABLE;
  static const char*    FAR_CLIP;
  static const char*    FONT;
  static const char*    HEIGHT;
  static const char*    HIDDEN;
  static const char*    HIDE_MIN_MAX;
  static const char*    KEY;
  static const char*    KEY_POS;
  static const char*    LIGHT;
  static const char*    LINE_COLOR;
  static const char*    LINE_WIDTH;
  static const char*    MAX_COLOR;
  static const char*    MAX_POINTS;
  static const char*    MAX_VALUE;
  static const char*    MIN_COLOR;
  static const char*    MIN_VALUE;
  static const char*    NEAR_CLIP;
  static const char*    POINT_STYLE;
  static const char*    POSITION;
  static const char*    QUALITY;
  static const char*    RANGE;
  static const char*    RANGES;
  static const char*    ROTATION;
  static const char*    ROTMAT;
  static const char*    SCALE;
  static const char*    SCALE_STEP;
  static const char*    SHOW_KEY;
  static const char*    SIZE;
  static const char*    SPECULAR;
  static const char*    TITLE;
  static const char*    TRANSLATION;
  static const char*    VIEW_ANGLE;
  static const char*    WIDTH;
  static const char*    XLABEL;
  static const char*    XRANGE;
  static const char*    YLABEL;
  static const char*    YRANGE;
  static const char*    ZOOM;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef PropertyNames   PropNames;


JEM_END_PACKAGE( gl )

#endif
