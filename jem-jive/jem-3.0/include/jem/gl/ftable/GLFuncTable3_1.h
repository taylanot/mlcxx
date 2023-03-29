
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

#ifndef JEM_GL_GLFUNCTABLE3_1_H
#define JEM_GL_GLFUNCTABLE3_1_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable3_1Core
//-----------------------------------------------------------------------

// OpenGL 3.1 core functions.

class GLFuncTable3_1Core : public Object
{
 public:

  explicit                GLFuncTable3_1Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glDrawArraysInstanced)

    ( GLenum                mode,
      GLint                 first,
      GLsizei               count,
      GLsizei               instancecount );

  void                   (JEM_APIENTRYP glDrawElementsInstanced)

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLsizei               instancecount );

  void                   (JEM_APIENTRYP glTexBuffer)

    ( GLenum                target,
      GLenum                internalformat,
      GLuint                buffer );

  void                   (JEM_APIENTRYP glPrimitiveRestartIndex)

    ( GLuint                index );

  void                   (JEM_APIENTRYP glCopyBufferSubData)

    ( GLenum                readTarget,
      GLenum                writeTarget,
      GLintptr              readOffset,
      GLintptr              writeOffset,
      GLsizeiptr            size );

  void                   (JEM_APIENTRYP glGetUniformIndices)

    ( GLuint                program,
      GLsizei               uniformCount,
      const GLchar* const*  uniformNames,
      GLuint*               uniformIndices );

  void                   (JEM_APIENTRYP glGetActiveUniformsiv)

    ( GLuint                program,
      GLsizei               uniformCount,
      const GLuint*         uniformIndices,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetActiveUniformName)

    ( GLuint                program,
      GLuint                uniformIndex,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               uniformName );

  GLuint                 (JEM_APIENTRYP glGetUniformBlockIndex)

    ( GLuint                program,
      const GLchar*         uniformBlockName );

  void                   (JEM_APIENTRYP glGetActiveUniformBlockiv)

    ( GLuint                program,
      GLuint                uniformBlockIndex,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetActiveUniformBlockName)

    ( GLuint                program,
      GLuint                uniformBlockIndex,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               uniformBlockName );

  void                   (JEM_APIENTRYP glUniformBlockBinding)

    ( GLuint                program,
      GLuint                uniformBlockIndex,
      GLuint                uniformBlockBinding );


 protected:

  virtual                ~GLFuncTable3_1Core  ();

};


JEM_END_PACKAGE( gl )

#endif
