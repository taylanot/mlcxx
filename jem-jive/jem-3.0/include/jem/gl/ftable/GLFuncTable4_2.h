
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

#ifndef JEM_GL_GLFUNCTABLE4_2_H
#define JEM_GL_GLFUNCTABLE4_2_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable4_2Core
//-----------------------------------------------------------------------

// OpenGL 4.2 core functions.

class GLFuncTable4_2Core : public Object
{
 public:

  explicit                GLFuncTable4_2Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP
                          glDrawArraysInstancedBaseInstance)

    ( GLenum                mode,
      GLint                 first,
      GLsizei               count,
      GLsizei               instancecount,
      GLuint                baseinstance );

  void                   (JEM_APIENTRYP
                          glDrawElementsInstancedBaseInstance)

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLsizei               instancecount,
      GLuint                baseinstance );

  void                   (JEM_APIENTRYP
                          glDrawElementsInstancedBaseVertexBaseInstance)

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLsizei               instancecount,
      GLint                 basevertex,
      GLuint                baseinstance );

  void                   (JEM_APIENTRYP glGetInternalformativ)

    ( GLenum                target,
      GLenum                internalformat,
      GLenum                pname,
      GLsizei               bufSize,
      GLint*                params );

  void                   (JEM_APIENTRYP
                          glGetActiveAtomicCounterBufferiv)

    ( GLuint                program,
      GLuint                bufferIndex,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glBindImageTexture)

    ( GLuint                unit,
      GLuint                texture,
      GLint                 level,
      GLboolean             layered,
      GLint                 layer,
      GLenum                access,
      GLenum                format );

  void                   (JEM_APIENTRYP glMemoryBarrier)

    ( GLbitfield            barriers );

  void                   (JEM_APIENTRYP glTexStorage1D)

    ( GLenum                target,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width );

  void                   (JEM_APIENTRYP glTexStorage2D)

    ( GLenum                target,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glTexStorage3D)

    ( GLenum                target,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth );

  void                   (JEM_APIENTRYP
                          glDrawTransformFeedbackInstanced)

    ( GLenum                mode,
      GLuint                id,
      GLsizei               instancecount );

  void                   (JEM_APIENTRYP
                          glDrawTransformFeedbackStreamInstanced)

    ( GLenum                mode,
      GLuint                id,
      GLuint                stream,
      GLsizei               instancecount );


 protected:

  virtual                ~GLFuncTable4_2Core  ();

};


JEM_END_PACKAGE( gl )

#endif
