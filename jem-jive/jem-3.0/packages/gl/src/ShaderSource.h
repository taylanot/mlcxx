
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

#ifndef JEM_GL_SHADERSOURCE_H
#define JEM_GL_SHADERSOURCE_H

#include <jem/defines.h>
#include <jem/base/String.h>


JEM_BEGIN_PACKAGE( gl )


class ShaderProg;


//-----------------------------------------------------------------------
//   class ShaderSource
//-----------------------------------------------------------------------


class ShaderSource
{
 public:

  static void             load

    ( GLContext&            gtx,
      ShaderProg&           shader,
      const String&         name );

  static ShaderProg&      getProgram

    ( GLContext&            gtx,
      const String&         name );


 private:

  // Generic shaders.

  static const char*      VERT_FLAT_CONST_COLOR_;
  static const char*      VERT_FLAT_MULTI_COLOR_;
  static const char*      VERT_FLAT_TEX1D_;
  static const char*      VERT_FLAT_TEX2D_;

  static const char*      VERT_SHADE_CONST_COLOR_;
  static const char*      VERT_SHADE_MULTI_COLOR_;
  static const char*      VERT_SHADE_TEX1D_;
  static const char*      VERT_SHADE_TEX2D_;

  static const char*      FRAG_FLAT_CONST_COLOR_;
  static const char*      FRAG_FLAT_MULTI_COLOR_;
  static const char*      FRAG_FLAT_TEX1D_;
  static const char*      FRAG_FLAT_TEX1D_ALPHA_;
  static const char*      FRAG_FLAT_TEX2D_;
  static const char*      FRAG_FLAT_TEX2D_ALPHA_;

  static const char*      FRAG_SHADE_CONST_COLOR_;
  static const char*      FRAG_SHADE_MULTI_COLOR_;
  static const char*      FRAG_SHADE_TEX1D_;
  static const char*      FRAG_SHADE_TEX1D_ALPHA_;
  static const char*      FRAG_SHADE_TEX2D_;
  static const char*      FRAG_SHADE_TEX2D_ALPHA_;

  // Special shaders.

  static const char*      VERT_CONST_ACCUMULATION_;
  static const char*      FRAG_CONST_ACCUMULATION_;

  static const char*      VERT_UNIFORM_ORB_;
  static const char*      GEOM_UNIFORM_ORB_;
  static const char*      FRAG_UNIFORM_ORB_;

  static const char*      VERT_VARYING_ORB_;
  static const char*      GEOM_VARYING_ORB_;
  static const char*      FRAG_VARYING_ORB_;

  static const char*      GEOM_CHECKER_ORB_;
  static const char*      FRAG_CHECKER_ORB_;

  static const char*      VERT_LINES_MULTI_;
  static const char*      VERT_LINES_TEX1D_;
  static const char*      GEOM_LINES_CONST_;
  static const char*      GEOM_LINES_MULTI_;
  static const char*      GEOM_LINES_TEX1D_;
  static const char*      GEOM_LINES_STIPPLE_CONST_;
  static const char*      FRAG_LINES_STIPPLE_CONST_;

};


JEM_END_PACKAGE( gl )

#endif
