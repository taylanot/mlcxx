
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

#ifndef JIVE_GL_NAMES_H
#define JIVE_GL_NAMES_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class RunvarNames
//-----------------------------------------------------------------------


class RunvarNames
{
 public:

  static const char*    UPDATE_LOAD;
  static const char*    RENDER_LOAD;

};


//-----------------------------------------------------------------------
//   class PropertyNames
//-----------------------------------------------------------------------


class PropertyNames
{
 public:

  static const char*    AUTO_SCALE;
  static const char*    CAMERA;
  static const char*    CONF_FILE;
  static const char*    DATA;
  static const char*    DATA_SETS;
  static const char*    DEFORMATION;
  static const char*    DISPLACEMENTS;
  static const char*    DOFS;
  static const char*    ELEMENTS;
  static const char*    FILE_NAME;
  static const char*    GRAPH;
  static const char*    HEIGHT;
  static const char*    HIDE_PALETTE;
  static const char*    LIGHTS;
  static const char*    MAP_FILE;
  static const char*    MARKERS;
  static const char*    MESH_NAME;
  static const char*    MODULE;
  static const char*    NEW_DATA_SETS;
  static const char*    NEW_PALETTES;
  static const char*    NEW_PLUGINS;
  static const char*    NEW_STYLES;
  static const char*    NODES;
  static const char*    NULL_VALUE;
  static const char*    OFF_SCREEN;
  static const char*    PALETTE;
  static const char*    PALETTES;
  static const char*    PLUGINS;
  static const char*    POINTS;
  static const char*    SAMPLE_COND;
  static const char*    SCALE;
  static const char*    SCALE_STEP;
  static const char*    SHAPE_TABLE;
  static const char*    SNAP_COND;
  static const char*    SNAP_FILE;
  static const char*    STYLES;
  static const char*    TABLE;
  static const char*    TABLE_TYPE;
  static const char*    TITLE;
  static const char*    UPDATE_COND;
  static const char*    VECTOR;
  static const char*    WIDTH;
  static const char*    WINDOW;
  static const char*    WORLD;
  static const char*    XDATA;
  static const char*    YDATA;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef RunvarNames     VarNames;
typedef PropertyNames   PropNames;


JIVE_END_PACKAGE( gl )

#endif
