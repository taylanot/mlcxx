
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

#ifndef JEM_GL_GLFUNCTIONS1_2_H
#define JEM_GL_GLFUNCTIONS1_2_H

#include <jem/gl/GLFunctions1_1.h>
#include <jem/gl/ftable/GLFuncTable1_2.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class GLFunctions1_2Core
//-----------------------------------------------------------------------


class GLFunctions1_2Core : public GLFunctions1_1Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_2_CORE;

  explicit                GLFunctions1_2Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions1_2Core  ();


 public:

  inline void             glDrawRangeElements

    ( GLenum                mode,
      GLuint                start,
      GLuint                end,
      GLsizei               count,
      GLenum                type,
      const void*           indices );

  inline void             glTexImage3D

    ( GLenum                target,
      GLint                 level,
      GLint                 internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLint                 border,
      GLenum                format,
      GLenum                type,
      const void*           pixels );

  inline void             glTexSubImage3D

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLenum                format,
      GLenum                type,
      const void*           pixels );

  inline void             glCopyTexSubImage3D

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );


 private:

  Ref<GLFuncTable1_2Core>   table_;

};


typedef GLFunctions1_2Core  GLFuncs1_2Core;


//-----------------------------------------------------------------------
//   class GLFunctions1_2
//-----------------------------------------------------------------------


class GLFunctions1_2 : public GLFunctions1_2Core,
                       public GLFunctions1Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_2;

  explicit                GLFunctions1_2

    ( GLFuncTables&         tables );

};


typedef GLFunctions1_2    GLFuncs1_2;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions1_2Core
//=======================================================================


inline void GLFunctions1_2Core::glDrawRangeElements

  ( GLenum       mode,
    GLuint       start,
    GLuint       end,
    GLsizei      count,
    GLenum       type,
    const void*  indices )

{
  table_->glDrawRangeElements ( mode, start, end, count,
                                type, indices );
}


inline void GLFunctions1_2Core::glTexImage3D

  ( GLenum       target,
    GLint        level,
    GLint        internalformat,
    GLsizei      width,
    GLsizei      height,
    GLsizei      depth,
    GLint        border,
    GLenum       format,
    GLenum       type,
    const void*  pixels )

{
  table_->glTexImage3D ( target, level, internalformat,
                         width, height, depth, border,
                         format, type, pixels );
}


inline void GLFunctions1_2Core::glTexSubImage3D

  ( GLenum       target,
    GLint        level,
    GLint        xoffset,
    GLint        yoffset,
    GLint        zoffset,
    GLsizei      width,
    GLsizei      height,
    GLsizei      depth,
    GLenum       format,
    GLenum       type,
    const void*  pixels )

{
  table_->glTexSubImage3D ( target, level,
                            xoffset, yoffset, zoffset,
                            width, height, depth, format, type,
                            pixels );
}


inline void GLFunctions1_2Core::glCopyTexSubImage3D

  ( GLenum   target,
    GLint    level,
    GLint    xoffset,
    GLint    yoffset,
    GLint    zoffset,
    GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height )

{
  table_->glCopyTexSubImage3D ( target, level,
                                xoffset, yoffset, zoffset,
                                x, y, width, height );
}


JEM_END_PACKAGE( gl )

#endif
