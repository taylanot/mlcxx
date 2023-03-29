
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

#ifndef JEM_GL_GLFUNCTABLE4_4_H
#define JEM_GL_GLFUNCTABLE4_4_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable4_4Core
//-----------------------------------------------------------------------

// OpenGL 4.4 core functions.

class GLFuncTable4_4Core : public Object
{
 public:

  explicit                GLFuncTable4_4Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glBufferStorage)

    ( GLenum                target,
      GLsizeiptr            size,
      const void*           data,
      GLbitfield            flags );

  void                   (JEM_APIENTRYP glClearTexImage)

    ( GLuint                texture,
      GLint                 level,
      GLenum                format,
      GLenum                type,
      const void*           data );

  void                   (JEM_APIENTRYP glClearTexSubImage)

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

  void                   (JEM_APIENTRYP glBindBuffersBase)

    ( GLenum                target,
      GLuint                first,
      GLsizei               count,
      const GLuint*         buffers );

  void                   (JEM_APIENTRYP glBindBuffersRange)

    ( GLenum                target,
      GLuint                first,
      GLsizei               count,
      const GLuint*         buffers,
      const GLintptr*       offsets,
      const GLsizeiptr*     sizes );

  void                   (JEM_APIENTRYP glBindTextures)

    ( GLuint                first,
      GLsizei               count,
      const GLuint*         textures );

  void                   (JEM_APIENTRYP glBindSamplers)

    ( GLuint                first,
      GLsizei               count,
      const GLuint*         samplers );

  void                   (JEM_APIENTRYP glBindImageTextures)

    ( GLuint                first,
      GLsizei               count,
      const GLuint*         textures );

  void                   (JEM_APIENTRYP glBindVertexBuffers)

    ( GLuint                first,
      GLsizei               count,
      const GLuint*         buffers,
      const GLintptr*       offsets,
      const GLsizei*        strides );


 protected:

  virtual                ~GLFuncTable4_4Core  ();

};


JEM_END_PACKAGE( gl )

#endif
