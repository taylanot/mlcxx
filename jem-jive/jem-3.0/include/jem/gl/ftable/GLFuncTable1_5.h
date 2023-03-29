
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

#ifndef JEM_GL_GLFUNCTABLE1_5_H
#define JEM_GL_GLFUNCTABLE1_5_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable1_5Core
//-----------------------------------------------------------------------

// OpenGL 1.5 core functions.

class GLFuncTable1_5Core : public Object
{
 public:

  explicit                GLFuncTable1_5Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glGenQueries)

    ( GLsizei               n,
      GLuint*               ids );

  void                   (JEM_APIENTRYP glDeleteQueries)

    ( GLsizei               n,
      const GLuint*         ids );

  GLboolean              (JEM_APIENTRYP glIsQuery)

    ( GLuint                id );

  void                   (JEM_APIENTRYP glBeginQuery)

    ( GLenum                target,
      GLuint                id );

  void                   (JEM_APIENTRYP glEndQuery)

    ( GLenum                target );

  void                   (JEM_APIENTRYP glGetQueryiv)

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetQueryObjectiv)

    ( GLuint                id,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetQueryObjectuiv)

    ( GLuint                id,
      GLenum                pname,
      GLuint*               params );

  void                   (JEM_APIENTRYP glBindBuffer)

    ( GLenum                target,
      GLuint                buffer );

  void                   (JEM_APIENTRYP glDeleteBuffers)

    ( GLsizei               n,
      const GLuint*         buffers );

  void                   (JEM_APIENTRYP glGenBuffers)

    ( GLsizei               n,
      GLuint*               buffers );

  GLboolean              (JEM_APIENTRYP glIsBuffer)

    ( GLuint                buffer );

  void                   (JEM_APIENTRYP glBufferData)

    ( GLenum                target,
      GLsizeiptr            size,
      const void*           data,
      GLenum                usage );

  void                   (JEM_APIENTRYP glBufferSubData)

    ( GLenum                target,
      GLintptr              offset,
      GLsizeiptr            size,
      const void*           data );

  void                   (JEM_APIENTRYP glGetBufferSubData)

    ( GLenum                target,
      GLintptr              offset,
      GLsizeiptr            size,
      void*                 data );

  void                   (JEM_APIENTRYP glMapBuffer)

    ( GLenum                target,
      GLenum                access );

  GLboolean              (JEM_APIENTRYP glUnmapBuffer)

    ( GLenum                target );

  void                   (JEM_APIENTRYP glGetBufferParameteriv)

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetBufferPointerv)

    ( GLenum                target,
      GLenum                pname,
      void**                params );


 protected:

  virtual                ~GLFuncTable1_5Core  ();

};


JEM_END_PACKAGE( gl )

#endif
