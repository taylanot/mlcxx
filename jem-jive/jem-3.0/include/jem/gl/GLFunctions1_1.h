
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

#ifndef JEM_GL_GLFUNCTIONS1_1_H
#define JEM_GL_GLFUNCTIONS1_1_H

#include <jem/gl/GLFunctions1.h>
#include <jem/gl/ftable/GLFuncTable1_1.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class GLFunctions1_1Core
//-----------------------------------------------------------------------


class GLFunctions1_1Core : public GLFunctions1Core
{
 public:

  static const int          VERSION = GLVersion::VERSION_1_1_CORE;

  explicit                  GLFunctions1_1Core

    ( GLFuncTables&           tables );

  virtual                  ~GLFunctions1_1Core  ();


 public:

  inline GLboolean          glIsTexture

    ( GLuint                  texture );

  inline void               glGenTextures

    ( GLsizei                 n,
      GLuint*                 textures );

  inline void               glDeleteTextures

    ( GLsizei                 n,
      const GLuint*           textures );

  inline void               glBindTexture

    ( GLenum                  target,
      GLuint                  texture );

  inline void               glTexSubImage

    ( GLenum                  target,
      GLint                   level,
      GLint                   xoffset,
      GLint                   yoffset,
      GLsizei                 width,
      GLsizei                 height,
      GLenum                  format,
      GLenum                  type,
      const GLvoid*           pixels );

  inline void               glTexSubImage

    ( GLenum                  target,
      GLint                   level,
      GLint                   xoffset,
      GLsizei                 width,
      GLenum                  format,
      GLenum                  type,
      const GLvoid*           pixels );

  inline void               glCopyTexSubImage

    ( GLenum                  target,
      GLint                   level,
      GLint                   xoffset,
      GLint                   yoffset,
      GLint                   x,
      GLint                   y,
      GLsizei                 width,
      GLsizei                 height );

  inline void               glCopyTexSubImage

    ( GLenum                  target,
      GLint                   level,
      GLint                   xoffset,
      GLint                   x,
      GLint                   y,
      GLsizei                 width );

  inline void               glCopyTexImage

    ( GLenum                  target,
      GLint                   level,
      GLenum                  internalformat,
      GLint                   x,
      GLint                   y,
      GLsizei                 width,
      GLsizei                 height,
      GLint                   border );

  inline void               glCopyTexImage

    ( GLenum                  target,
      GLint                   level,
      GLenum                  internalformat,
      GLint                   x,
      GLint                   y,
      GLsizei                 width,
      GLint                   border );

  inline void               glPolygonOffset

    ( GLfloat                 factor,
      GLfloat                 units );

  inline void               glGetPointer

    ( GLenum                  pname,
      GLvoid**                param );

  inline void               glDrawElements

    ( GLenum                  mode,
      GLsizei                 count,
      const GLubyte*          indices );

  inline void               glDrawElements

    ( GLenum                  mode,
      GLsizei                 count,
      const GLushort*         indices );

  inline void               glDrawElements

    ( GLenum                  mode,
      GLsizei                 count,
      const GLuint*           indices );

  inline void               glDrawElements

    ( GLenum                  mode,
      GLsizei                 count,
      GLenum                  type,
      const GLvoid*           indices );

  inline void               glDrawArrays

    ( GLenum                  mode,
      GLint                   first,
      GLsizei                 count );


 private:

  Ref<GLFuncTable1_1Core>   table_;

};


typedef GLFunctions1_1Core  GLFuncs1_1Core;


//-----------------------------------------------------------------------
//   class GLFunctions1_1
//-----------------------------------------------------------------------


class GLFunctions1_1 : public GLFunctions1_1Core,
                       public GLFunctions1Compat
{
 public:

  static const int          VERSION = GLVersion::VERSION_1_1;

  explicit                  GLFunctions1_1

    ( GLFuncTables&           tables );

};


typedef GLFunctions1_1      GLFuncs1_1;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions1_1Core
//=======================================================================


inline GLboolean GLFunctions1_1Core::glIsTexture

  ( GLuint  texture )

{
  return table_->glIsTexture ( texture );
}


inline void GLFunctions1_1Core::glGenTextures

  ( GLsizei  n,
    GLuint*  textures )

{
  table_->glGenTextures ( n, textures );
}


inline void GLFunctions1_1Core::glDeleteTextures

  ( GLsizei        n,
    const GLuint*  textures )

{
  table_->glDeleteTextures ( n, textures );
}


inline void GLFunctions1_1Core::glBindTexture

  ( GLenum  target,
    GLuint  texture )

{
  table_->glBindTexture ( target, texture );
}


inline void GLFunctions1_1Core::glTexSubImage

  ( GLenum         target,
    GLint          level,
    GLint          xoffset,
    GLint          yoffset,
    GLsizei        width,
    GLsizei        height,
    GLenum         format,
    GLenum         type,
    const GLvoid*  pixels )

{
  table_->glTexSubImage2D ( target, level,  xoffset, yoffset,
                            width,  height, format,  type,
                            pixels );
}


inline void GLFunctions1_1Core::glTexSubImage

  ( GLenum         target,
    GLint          level,
    GLint          xoffset,
    GLsizei        width,
    GLenum         format,
    GLenum         type,
    const GLvoid*  pixels )

{
  table_->glTexSubImage1D ( target, level,  xoffset,
                            width,  format, type, pixels );
}


inline void GLFunctions1_1Core::glCopyTexSubImage

  ( GLenum   target,
    GLint    level,
    GLint    xoffset,
    GLint    yoffset,
    GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height )

{
  table_->glCopyTexSubImage2D ( target, level, xoffset, yoffset,
                                x, y,   width, height );
}


inline void GLFunctions1_1Core::glCopyTexSubImage

  ( GLenum   target,
    GLint    level,
    GLint    xoffset,
    GLint    x,
    GLint    y,
    GLsizei  width )

{
  table_->glCopyTexSubImage1D ( target, level, xoffset,
                                x,      y,     width );
}


inline void GLFunctions1_1Core::glCopyTexImage

  ( GLenum   target,
    GLint    level,
    GLenum   internalformat,
    GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height,
    GLint    border )

{
  table_->glCopyTexImage2D ( target, level, internalformat,
                             x, y, width, height, border );
}


inline void GLFunctions1_1Core::glCopyTexImage

  ( GLenum   target,
    GLint    level,
    GLenum   internalformat,
    GLint    x,
    GLint    y,
    GLsizei  width,
    GLint    border )

{
  table_->glCopyTexImage1D ( target, level, internalformat,
                             x, y, width, border );
}


inline void GLFunctions1_1Core::glPolygonOffset

  ( GLfloat  factor,
    GLfloat  units )

{
  table_->glPolygonOffset ( factor, units );
}


inline void GLFunctions1_1Core::glGetPointer

  ( GLenum    pname,
    GLvoid**  params )

{
  table_->glGetPointerv ( pname, params );
}


inline void GLFunctions1_1Core::glDrawElements

  ( GLenum          mode,
    GLsizei         count,
    const GLubyte*  indices )

{
  table_->glDrawElements ( mode, count, GL_UNSIGNED_BYTE, indices );
}


inline void GLFunctions1_1Core::glDrawElements

  ( GLenum           mode,
    GLsizei          count,
    const GLushort*  indices )

{
  table_->glDrawElements ( mode, count, GL_UNSIGNED_SHORT, indices );
}


inline void GLFunctions1_1Core::glDrawElements

  ( GLenum         mode,
    GLsizei        count,
    const GLuint*  indices )

{
  table_->glDrawElements ( mode, count, GL_UNSIGNED_INT, indices );
}


inline void GLFunctions1_1Core::glDrawElements

  ( GLenum         mode,
    GLsizei        count,
    GLenum         type,
    const GLvoid*  indices )

{
  table_->glDrawElements ( mode, count, type, indices );
}


inline void GLFunctions1_1Core::glDrawArrays

  ( GLenum   mode,
    GLint    first,
    GLsizei  count )

{
  table_->glDrawArrays ( mode, first, count );
}


JEM_END_PACKAGE( gl )

#endif
