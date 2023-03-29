
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

#ifndef JEM_GL_GLFUNCTIONS4_4_H
#define JEM_GL_GLFUNCTIONS4_4_H

#include <jem/gl/GLFunctions4_3.h>
#include <jem/gl/ftable/GLFuncTable4_4.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions4_4Core
//-----------------------------------------------------------------------


class GLFunctions4_4Core : public GLFunctions4_3Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_4_CORE;

  explicit                GLFunctions4_4Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions4_4Core  ();


 public:

  inline void             glBufferStorage

    ( GLenum                target,
      GLsizeiptr            size,
      const void*           data,
      GLbitfield            flags );

  inline void             glClearTexImage

    ( GLuint                texture,
      GLint                 level,
      GLenum                format,
      GLenum                type,
      const void*           data );

  inline void             glClearTexSubImage

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLenum                format,
      GLenum                type,
      const void*           data );

  inline void             glBindBuffersBase

    ( GLenum                target,
      GLuint                first,
      GLsizei               count,
      const GLuint*         buffers );

  inline void             glBindBuffersRange

    ( GLenum                target,
      GLuint                first,
      GLsizei               count,
      const GLuint*         buffers,
      const GLintptr*       offsets,
      const GLsizeiptr*     sizes );

  inline void             glBindTextures

    ( GLuint                first,
      GLsizei               count,
      const GLuint*         textures );

  inline void             glBindSamplers

    ( GLuint                first,
      GLsizei               count,
      const GLuint*         samplers );

  inline void             glBindImageTextures

    ( GLuint                first,
      GLsizei               count,
      const GLuint*         textures );

  inline void             glBindVertexBuffers

    ( GLuint                first,
      GLsizei               count,
      const GLuint*         buffers,
      const GLintptr*       offsets,
      const GLsizei*        strides );


 private:

  Ref
    <GLFuncTable4_4Core>  table_;

};


typedef
  GLFunctions4_4Core      GLFuncs4_4Core;


//-----------------------------------------------------------------------
//   class GLFunctions4_4
//-----------------------------------------------------------------------


class GLFunctions4_4 : public GLFunctions4_4Core,
                       public GLFunctions4Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_4;

  explicit                GLFunctions4_4

    ( GLFuncTables&         tables );

};


typedef GLFunctions4_4    GLFuncs4_4;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions4_4Core
//=======================================================================


inline void GLFunctions4_4Core::glBufferStorage

  ( GLenum       target,
    GLsizeiptr   size,
    const void*  data,
    GLbitfield   flags )

{
  table_->glBufferStorage ( target, size, data, flags );
}


inline void GLFunctions4_4Core::glClearTexImage

  ( GLuint       texture,
    GLint        level,
    GLenum       format,
    GLenum       type,
    const void*  data )

{
  table_->glClearTexImage ( texture, level, format, type, data );
}


inline void GLFunctions4_4Core::glClearTexSubImage

  ( GLuint       texture,
    GLint        level,
    GLint        xoffset,
    GLint        yoffset,
    GLint        zoffset,
    GLsizei      width,
    GLsizei      height,
    GLsizei      depth,
    GLenum       format,
    GLenum       type,
    const void*  data )

{
  table_->glClearTexSubImage ( texture, level,
                               xoffset, yoffset, zoffset,
                               width, height, depth,
                               format, type, data );
}


inline void GLFunctions4_4Core::glBindBuffersBase

  ( GLenum         target,
    GLuint         first,
    GLsizei        count,
    const GLuint*  buffers )

{
  table_->glBindBuffersBase ( target, first, count, buffers );
}


inline void GLFunctions4_4Core::glBindBuffersRange

  ( GLenum             target,
    GLuint             first,
    GLsizei            count,
    const GLuint*      buffers,
    const GLintptr*    offsets,
    const GLsizeiptr*  sizes )

{
  table_->glBindBuffersRange ( target, first, count, buffers, offsets,
                               sizes );
}


inline void GLFunctions4_4Core::glBindTextures

  ( GLuint         first,
    GLsizei        count,
    const GLuint*  textures )

{
  table_->glBindTextures ( first, count, textures );
}


inline void GLFunctions4_4Core::glBindSamplers

  ( GLuint         first,
    GLsizei        count,
    const GLuint*  samplers )

{
  table_->glBindSamplers ( first, count, samplers );
}


inline void GLFunctions4_4Core::glBindImageTextures

  ( GLuint         first,
    GLsizei        count,
    const GLuint*  textures )

{
  table_->glBindImageTextures ( first, count, textures );
}


inline void GLFunctions4_4Core::glBindVertexBuffers

  ( GLuint           first,
    GLsizei          count,
    const GLuint*    buffers,
    const GLintptr*  offsets,
    const GLsizei*   strides )

{
  table_->glBindVertexBuffers ( first, count, buffers, offsets,
                                strides );
}


JEM_END_PACKAGE( gl )

#endif
