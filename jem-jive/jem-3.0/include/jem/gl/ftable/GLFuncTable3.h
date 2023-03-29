
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

#ifndef JEM_GL_GLFUNCTABLE3_H
#define JEM_GL_GLFUNCTABLE3_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable3Core
//-----------------------------------------------------------------------

// OpenGL 3.0 core functions.

class GLFuncTable3Core : public Object
{
 public:

  explicit                GLFuncTable3Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glColorMaski)

    ( GLuint                index,
      GLboolean             r,
      GLboolean             g,
      GLboolean             b,
      GLboolean             a );

  void                   (JEM_APIENTRYP glGetBooleani_v)

    ( GLenum                target,
      GLuint                index,
      GLboolean*            data );

  void                   (JEM_APIENTRYP glGetIntegeri_v)

    ( GLenum                target,
      GLuint                index,
      GLint*                data );

  void                   (JEM_APIENTRYP glEnablei)

    ( GLenum                target,
      GLuint                index );

  void                   (JEM_APIENTRYP glDisablei)

    ( GLenum                target,
      GLuint                index );

  GLboolean              (JEM_APIENTRYP glIsEnabledi)

    ( GLenum                target,
      GLuint                index );

  void                   (JEM_APIENTRYP glBeginTransformFeedback)

    ( GLenum                primitiveMode );

  void                   (JEM_APIENTRYP glEndTransformFeedback) ();

  void                   (JEM_APIENTRYP glBindBufferRange)

    ( GLenum                target,
      GLuint                index,
      GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size );

  void                   (JEM_APIENTRYP glBindBufferBase)

    ( GLenum                target,
      GLuint                index,
      GLuint                buffer );

  void                   (JEM_APIENTRYP glTransformFeedbackVaryings)

    ( GLuint                program,
      GLsizei               count,
      const GLchar* const*  varyings,
      GLenum                bufferMode );

  void                   (JEM_APIENTRYP glGetTransformFeedbackVarying)

    ( GLuint                program,
      GLuint                index,
      GLsizei               bufSize,
      GLsizei*              length,
      GLsizei*              size,
      GLenum*               type,
      GLchar*               name );

  void                   (JEM_APIENTRYP glClampColor)

    ( GLenum                target,
      GLenum                clamp );

  void                   (JEM_APIENTRYP glBeginConditionalRender)

    ( GLuint                id,
      GLenum                mode );

  void                   (JEM_APIENTRYP glEndConditionalRender) ();

  void                   (JEM_APIENTRYP glVertexAttribIPointer)

    ( GLuint                index,
      GLint                 size,
      GLenum                type,
      GLsizei               stride,
      const void*           pointer );

  void                   (JEM_APIENTRYP glGetVertexAttribIiv)

    ( GLuint                index,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetVertexAttribIuiv)

    ( GLuint                index,
      GLenum                pname,
      GLuint*               params );

  void                   (JEM_APIENTRYP glVertexAttribI1i)

    ( GLuint                index,
      GLint                 x );

  void                   (JEM_APIENTRYP glVertexAttribI2i)

    ( GLuint                index,
      GLint                 x,
      GLint                 y );

  void                   (JEM_APIENTRYP glVertexAttribI3i)

    ( GLuint                index,
      GLint                 x,
      GLint                 y,
      GLint                 z );

  void                   (JEM_APIENTRYP glVertexAttribI4i)

    ( GLuint                index,
      GLint                 x,
      GLint                 y,
      GLint                 z,
      GLint                 w );

  void                   (JEM_APIENTRYP glVertexAttribI1ui)

    ( GLuint                index,
      GLuint                x );

  void                   (JEM_APIENTRYP glVertexAttribI2ui)

    ( GLuint                index,
      GLuint                x,
      GLuint                y );

  void                   (JEM_APIENTRYP glVertexAttribI3ui)

    ( GLuint                index,
      GLuint                x,
      GLuint                y,
      GLuint                z );

  void                   (JEM_APIENTRYP glVertexAttribI4ui)

    ( GLuint                index,
      GLuint                x,
      GLuint                y,
      GLuint                z,
      GLuint                w );

  void                   (JEM_APIENTRYP glVertexAttribI1iv)

    ( GLuint                index,
      const GLint*          v );

  void                   (JEM_APIENTRYP glVertexAttribI2iv)

    ( GLuint                index,
      const GLint*          v );

  void                   (JEM_APIENTRYP glVertexAttribI3iv)

    ( GLuint                index,
      const GLint*          v );

  void                   (JEM_APIENTRYP glVertexAttribI4iv)

    ( GLuint                index,
      const GLint*          v );

  void                   (JEM_APIENTRYP glVertexAttribI1uiv)

    ( GLuint                index,
      const GLuint*         v );

  void                   (JEM_APIENTRYP glVertexAttribI2uiv)

    ( GLuint                index,
      const GLuint*         v );

  void                   (JEM_APIENTRYP glVertexAttribI3uiv)

    ( GLuint                index,
      const GLuint*         v );

  void                   (JEM_APIENTRYP glVertexAttribI4uiv)

    ( GLuint                index,
      const GLuint*         v );

  void                   (JEM_APIENTRYP glVertexAttribI4bv)

    ( GLuint                index,
      const GLbyte*         v );

  void                   (JEM_APIENTRYP glVertexAttribI4sv)

    ( GLuint                index,
      const GLshort*        v );

  void                   (JEM_APIENTRYP glVertexAttribI4ubv)

    ( GLuint                index,
      const GLubyte*        v );

  void                   (JEM_APIENTRYP glVertexAttribI4usv)

    ( GLuint                index,
      const GLushort*        v );

  void                   (JEM_APIENTRYP glGetUniformuiv)

    ( GLuint                program,
      GLint                 location,
      GLuint*               params );

  void                   (JEM_APIENTRYP glBindFragDataLocation)

    ( GLuint                program,
      GLuint                color,
      const GLchar*         name );

  GLint                  (JEM_APIENTRYP glGetFragDataLocation)

    ( GLuint                program,
      const GLchar*         name );

  void                   (JEM_APIENTRYP glUniform1ui)

    ( GLint                 location,
      GLuint                v0 );

  void                   (JEM_APIENTRYP glUniform2ui)

    ( GLint                 location,
      GLuint                v0,
      GLuint                v1 );

  void                   (JEM_APIENTRYP glUniform3ui)

    ( GLint                 location,
      GLuint                v0,
      GLuint                v1,
      GLuint                v2 );

  void                   (JEM_APIENTRYP glUniform4ui)

    ( GLint                 location,
      GLuint                v0,
      GLuint                v1,
      GLuint                v2,
      GLuint                v3 );

  void                   (JEM_APIENTRYP glUniform1uiv)

    ( GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glUniform2uiv)

    ( GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glUniform3uiv)

    ( GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glUniform4uiv)

    ( GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glTexParameterIiv)

    ( GLenum                target,
      GLenum                pname,
      const GLint*          params );

  void                   (JEM_APIENTRYP glTexParameterIuiv)

    ( GLenum                target,
      GLenum                pname,
      const GLuint*         params );

  void                   (JEM_APIENTRYP glGetTexParameterIiv)

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetTexParameterIuiv)

    ( GLenum                target,
      GLenum                pname,
      GLuint*               params );

  void                   (JEM_APIENTRYP glClearBufferiv)

    ( GLenum                buffer,
      GLint                 drawbuffer,
      const GLint*          value );

  void                   (JEM_APIENTRYP glClearBufferuiv)

    ( GLenum                buffer,
      GLint                 drawbuffer,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glClearBufferfv)

    ( GLenum                buffer,
      GLint                 drawbuffer,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glClearBufferfi)

    ( GLenum                buffer,
      GLint                 drawbuffer,
      GLfloat depth,
      GLint                 stencil );

  const GLubyte*         (JEM_APIENTRYP glGetStringi)

    ( GLenum                name,
      GLuint                index );

  GLboolean              (JEM_APIENTRYP glIsRenderbuffer)

    ( GLuint                renderbuffer );

  void                   (JEM_APIENTRYP glBindRenderbuffer)

    ( GLenum                target,
      GLuint                renderbuffer );

  void                   (JEM_APIENTRYP glDeleteRenderbuffers)

    ( GLsizei               n,
      const GLuint*         renderbuffers );

  void                   (JEM_APIENTRYP glGenRenderbuffers)

    ( GLsizei               n,
      GLuint*               renderbuffers );

  void                   (JEM_APIENTRYP glRenderbufferStorage)

    ( GLenum                target,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glGetRenderbufferParameteriv)

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  GLboolean              (JEM_APIENTRYP glIsFramebuffer)

    ( GLuint                framebuffer );

  void                   (JEM_APIENTRYP glBindFramebuffer)

    ( GLenum                target,
      GLuint                framebuffer );

  void                   (JEM_APIENTRYP glDeleteFramebuffers)

    ( GLsizei               n,
      const GLuint*         framebuffers );

  void                   (JEM_APIENTRYP glGenFramebuffers)

    ( GLsizei               n,
      GLuint*               framebuffers );

  GLenum                 (JEM_APIENTRYP glCheckFramebufferStatus)

    ( GLenum                target );

  void                   (JEM_APIENTRYP glFramebufferTexture1D)

    ( GLenum                target,
      GLenum                attachment,
      GLenum                textarget,
      GLuint                texture,
      GLint                 level );

  void                   (JEM_APIENTRYP glFramebufferTexture2D)

    ( GLenum                target,
      GLenum                attachment,
      GLenum                textarget,
      GLuint                texture,
      GLint                 level );

  void                   (JEM_APIENTRYP glFramebufferTexture3D)

    ( GLenum                target,
      GLenum                attachment,
      GLenum                textarget,
      GLuint                texture,
      GLint                 level,
      GLint                 zoffset );

  void                   (JEM_APIENTRYP glFramebufferRenderbuffer)

    ( GLenum                target,
      GLenum                attachment,
      GLenum                renderbuffertarget,
      GLuint                renderbuffer );

  void                   (JEM_APIENTRYP
                          glGetFramebufferAttachmentParameteriv)

    ( GLenum                target,
      GLenum                attachment,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGenerateMipmap)

    ( GLenum                target );

  void                   (JEM_APIENTRYP glBlitFramebuffer)

    ( GLint                 srcX0,
      GLint                 srcY0,
      GLint                 srcX1,
      GLint                 srcY1,
      GLint                 dstX0,
      GLint                 dstY0,
      GLint                 dstX1,
      GLint                 dstY1,
      GLbitfield            mask,
      GLenum                filter );

  void                   (JEM_APIENTRYP
                          glRenderbufferStorageMultisample)

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glFramebufferTextureLayer)

    ( GLenum                target,
      GLenum                attachment,
      GLuint                texture,
      GLint                 level,
      GLint                 layer );

  void*                  (JEM_APIENTRYP glMapBufferRange)

    ( GLenum                target,
      GLintptr              offset,
      GLsizeiptr            length,
      GLbitfield            access );

  void                   (JEM_APIENTRYP glFlushMappedBufferRange)

    ( GLenum                target,
      GLintptr              offset,
      GLsizeiptr            length );

  void                   (JEM_APIENTRYP glBindVertexArray)

    ( GLuint                array );

  void                   (JEM_APIENTRYP glDeleteVertexArrays)

    ( GLsizei               n,
      const GLuint*         arrays );

  void                   (JEM_APIENTRYP glGenVertexArrays)

    ( GLsizei               n,
      GLuint*               arrays );

  GLboolean              (JEM_APIENTRYP glIsVertexArray)

    ( GLuint                array );


 protected:

  virtual                ~GLFuncTable3Core  ();

};


//-----------------------------------------------------------------------
//   class GLFuncTable3Compat
//-----------------------------------------------------------------------

// OpenGL 3.0 deprecated functions.

class GLFuncTable3Compat : public Object
{
 public:

  explicit                GLFuncTable3Compat

    ( GLFuncLoader*         loader );


 public:


 protected:

  virtual                ~GLFuncTable3Compat ();

};


JEM_END_PACKAGE( gl )

#endif
