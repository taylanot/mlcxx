
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

#ifndef JEM_GL_SHADERNAMES_H
#define JEM_GL_SHADERNAMES_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class ShaderNames
//-----------------------------------------------------------------------


class ShaderNames
{
 public:

  static const char*      FLAT_CONST_COLOR;
  static const char*      FLAT_MULTI_COLOR;
  static const char*      FLAT_TEX1D;
  static const char*      FLAT_TEX1D_ALPHA;
  static const char*      FLAT_TEX2D;
  static const char*      FLAT_TEX2D_ALPHA;

  static const char*      SHADE_CONST_COLOR;
  static const char*      SHADE_MULTI_COLOR;
  static const char*      SHADE_TEX1D;
  static const char*      SHADE_TEX1D_ALPHA;
  static const char*      SHADE_TEX2D;
  static const char*      SHADE_TEX2D_ALPHA;

  static const char*      ACCUMULATION;
  static const char*      ORBS_CHECKER;
  static const char*      ORBS_UNIFORM;
  static const char*      ORBS_VARYING;

  static const char*      LINES_CONST_COLOR;
  static const char*      LINES_MULTI_COLOR;
  static const char*      LINES_TEX1D;
  static const char*      LINES_STIPPLE;

};


//-----------------------------------------------------------------------
//   class ShaderUnis
//-----------------------------------------------------------------------


class ShaderUnis
{
 public:

  static const char*      ACC_FACTOR;
  static const char*      ALPHA_MIN;
  static const char*      AMBIENT;
  static const char*      COLOR;
  static const char*      COLOR2;
  static const char*      FACTOR;
  static const char*      FRAME_SIZE;
  static const char*      LIGHT_COLOR0;
  static const char*      LIGHT_COLOR1;
  static const char*      LIGHT_COLOR2;
  static const char*      LIGHT_COUNT;
  static const char*      LIGHT_DIR0;
  static const char*      LIGHT_DIR1;
  static const char*      LIGHT_DIR2;
  static const char*      MVP;
  static const char*      MV;
  static const char*      NORM_MAT;
  static const char*      PATTERN;
  static const char*      PROJ_MAT;
  static const char*      RADIUS;
  static const char*      SHININESS;
  static const char*      SPECULAR;
  static const char*      TEXTURE_1D;
  static const char*      TEXTURE_2D;
  static const char*      WIDTH;

  static const char*      LIGHT_COLOR[];
  static const char*      LIGHT_DIR  [];

};


//-----------------------------------------------------------------------
//   class ShaderBuffers
//-----------------------------------------------------------------------


class ShaderBuffers
{
 public:

  static const byte       COLORS     [1];
  static const byte       NORMALS    [1];
  static const byte       POSITIONS  [1];
  static const byte       RADII      [1];
  static const byte       TEX1_COORDS[1];
  static const byte       TEX2_COORDS[1];

};


JEM_END_PACKAGE( gl )

#endif
