
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

#ifndef JEM_GL_GLFUNCTABLE1_1_H
#define JEM_GL_GLFUNCTABLE1_1_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable1_1Core
//-----------------------------------------------------------------------

// OpenGL 1.1 core functions.

class GLFuncTable1_1Core : public Object
{
 public:

  explicit                GLFuncTable1_1Core

    ( GLFuncLoader*         loader );


 public:

  GLboolean              (JEM_APIENTRYP glIsTexture)

    ( GLuint                texture );

  void                   (JEM_APIENTRYP glGenTextures)

    ( GLsizei               n,
      GLuint*               textures );

  void                   (JEM_APIENTRYP glDeleteTextures)

    ( GLsizei               n,
      const GLuint*         textures );

  void                   (JEM_APIENTRYP glBindTexture)

    ( GLenum                target,
      GLuint                texture );

  void                   (JEM_APIENTRYP glTexSubImage2D)

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      const GLvoid*         pixels );

  void                   (JEM_APIENTRYP glTexSubImage1D)

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLsizei               width,
      GLenum                format,
      GLenum                type,
      const GLvoid*         pixels );

  void                   (JEM_APIENTRYP glCopyTexSubImage2D)

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glCopyTexSubImage1D)

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width );

  void                   (JEM_APIENTRYP glCopyTexImage2D)

    ( GLenum                target,
      GLint                 level,
      GLenum                internalformat,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height,
      GLint                 border );

  void                   (JEM_APIENTRYP glCopyTexImage1D)

    ( GLenum                target,
      GLint                 level,
      GLenum                internalformat,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLint                 border );

  void                   (JEM_APIENTRYP glPolygonOffset)

    ( GLfloat               factor,
      GLfloat               units );

  void                   (JEM_APIENTRYP glGetPointerv)

    ( GLenum                pname,
      GLvoid**              params );

  void                   (JEM_APIENTRYP glDrawElements)

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const GLvoid*         indices );

  void                   (JEM_APIENTRYP glDrawArrays)

    ( GLenum                mode,
      GLint                 first,
      GLsizei               count );


 protected:

  virtual                ~GLFuncTable1_1Core  ();

};


JEM_END_PACKAGE( gl )

#endif
