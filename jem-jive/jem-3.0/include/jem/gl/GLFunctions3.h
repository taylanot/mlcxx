
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

#ifndef JEM_GL_GLFUNCTIONS3_H
#define JEM_GL_GLFUNCTIONS3_H

#include <jem/gl/GLFunctions2_1.h>
#include <jem/gl/ftable/GLFuncTable3.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions3Core
//-----------------------------------------------------------------------


class GLFunctions3Core : public GLFunctions2_1Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_3_CORE;

  explicit                GLFunctions3Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions3Core  ();


 public:

  using GLFunctions1Core::glEnable;
  using GLFunctions1Core::glDisable;
  using GLFunctions1Core::glIsEnabled;
  using GLFunctions2Core::glUniform;
  using GLFunctions2Core::glUniform1;
  using GLFunctions2Core::glUniform2;
  using GLFunctions2Core::glUniform3;
  using GLFunctions2Core::glUniform4;
  using GLFunctions2Core::glGetUniform;

  inline void             glColorMask

    ( GLuint                index,
      GLboolean             r,
      GLboolean             g,
      GLboolean             b,
      GLboolean             a );

  inline void             glGetParameterI

    ( GLenum                target,
      GLuint                index,
      GLboolean*            data );

  inline void             glGetParameterI

    ( GLenum                target,
      GLuint                index,
      GLint*                data );

  inline void             glEnable

    ( GLenum                target,
      GLuint                index );

  inline void             glDisable

    ( GLenum                target,
      GLuint                index );

  inline GLboolean        glIsEnabled

    ( GLenum                target,
      GLuint                index );

  inline void             glBeginTransformFeedback

    ( GLenum                primitiveMode );

  inline void             glEndTransformFeedback ();

  inline void             glBindBufferRange

    ( GLenum                target,
      GLuint                index,
      GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size );

  inline void             glBindBufferBase

    ( GLenum                target,
      GLuint                index,
      GLuint                buffer );

  inline void             glTransformFeedbackVaryings

    ( GLuint                program,
      GLsizei               count,
      const GLchar* const*  varyings,
      GLenum                bufferMode );

  inline void             glGetTransformFeedbackVarying

    ( GLuint                program,
      GLuint                index,
      GLsizei               bufSize,
      GLsizei*              length,
      GLsizei*              size,
      GLenum*               type,
      GLchar*               name );

  inline void             glClampColor

    ( GLenum                target,
      GLenum                clamp );

  inline void             glBeginConditionalRender

    ( GLuint                id,
      GLenum                mode );

  inline void             glEndConditionalRender ();

  inline void             glVertexAttribIPointer

    ( GLuint                index,
      GLint                 size,
      GLenum                type,
      GLsizei               stride,
      const void*           pointer );

  inline void             glGetVertexAttribI

    ( GLuint                index,
      GLenum                pname,
      GLint*                params );

  inline void             glGetVertexAttribI

    ( GLuint                index,
      GLenum                pname,
      GLuint*               params );

  inline void             glVertexAttribI

    ( GLuint                index,
      GLint                 x );

  inline void             glVertexAttribI

    ( GLuint                index,
      GLint                 x,
      GLint                 y);

  inline void             glVertexAttribI

    ( GLuint                index,
      GLint                 x,
      GLint                 y,
      GLint                 z );

  inline void             glVertexAttribI

    ( GLuint                index,
      GLint                 x,
      GLint                 y,
      GLint                 z,
      GLint                 w );

  inline void             glVertexAttribI

    ( GLuint                index,
      GLuint                x );

  inline void             glVertexAttribI

    ( GLuint                index,
      GLuint                x,
      GLuint                y );

  inline void             glVertexAttribI

    ( GLuint                index,
      GLuint                x,
      GLuint                y,
      GLuint                z );

  inline void             glVertexAttribI

    ( GLuint                index,
      GLuint                x,
      GLuint                y,
      GLuint                z,
      GLuint                w );

  inline void             glVertexAttribI1

    ( GLuint                index,
      const GLint*          v );

  inline void             glVertexAttribI2

    ( GLuint                index,
      const GLint*          v );

  inline void             glVertexAttribI3

    ( GLuint                index,
      const GLint*          v );

  inline void             glVertexAttribI4

    ( GLuint                index,
      const GLint*          v );

  inline void             glVertexAttribI1

    ( GLuint                index,
      const GLuint*         v );

  inline void             glVertexAttribI2

    ( GLuint                index,
      const GLuint*         v );

  inline void             glVertexAttribI3

    ( GLuint                index,
      const GLuint*         v );

  inline void             glVertexAttribI4

    ( GLuint                index,
      const GLuint*         v );

  inline void             glVertexAttribI4

    ( GLuint                index,
      const GLbyte*         v );

  inline void             glVertexAttribI4

    ( GLuint                index,
      const GLshort*        v );

  inline void             glVertexAttribI4

    ( GLuint                index,
      const GLubyte*        v );

  inline void             glVertexAttribI4

    ( GLuint                index,
      const GLushort*        v );

  inline void             glGetUniform

    ( GLuint                program,
      GLint                 location,
      GLuint*               params );

  inline void             glBindFragDataLocation

    ( GLuint                program,
      GLuint                color,
      const GLchar*         name );

  inline GLint            glGetFragDataLocation

    ( GLuint                program,
      const GLchar*         name );

  inline void             glUniform

    ( GLint                 location,
      GLuint                v0 );

  inline void             glUniform

    ( GLint                 location,
      GLuint                v0,
      GLuint                v1 );

  inline void             glUniform

    ( GLint                 location,
      GLuint                v0,
      GLuint                v1,
      GLuint                v2 );

  inline void             glUniform

    ( GLint                 location,
      GLuint                v0,
      GLuint                v1,
      GLuint                v2,
      GLuint                v3 );

  inline void             glUniform1

    ( GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  inline void             glUniform2

    ( GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  inline void             glUniform3

    ( GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  inline void             glUniform4

    ( GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  inline void             glTexParameterI

    ( GLenum                target,
      GLenum                pname,
      const GLint*          params );

  inline void             glTexParameterI

    ( GLenum                target,
      GLenum                pname,
      const GLuint*         params );

  inline void             glGetTexParameterI

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  inline void             glGetTexParameterI

    ( GLenum                target,
      GLenum                pname,
      GLuint*               params );

  inline void             glClearBuffer

    ( GLenum                buffer,
      GLint                 drawbuffer,
      const GLint*          value );

  inline void             glClearBuffer

    ( GLenum                buffer,
      GLint                 drawbuffer,
      const GLuint*         value );

  inline void             glClearBuffer

    ( GLenum                buffer,
      GLint                 drawbuffer,
      const GLfloat*        value );

  inline void             glClearBuffer

    ( GLenum                buffer,
      GLint                 drawbuffer,
      GLfloat               depth,
      GLint                 stencil );

  inline const GLubyte*   glGetStringi

    ( GLenum                name,
      GLuint                index );

  inline GLboolean        glIsRenderbuffer

    ( GLuint                renderbuffer );

  inline void             glBindRenderbuffer

    ( GLenum                target,
      GLuint                renderbuffer );

  inline void             glDeleteRenderbuffers

    ( GLsizei               n,
      const GLuint*         renderbuffers );

  inline void             glGenRenderbuffers

    ( GLsizei               n,
      GLuint*               renderbuffers );

  inline void             glRenderbufferStorage

    ( GLenum                target,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  inline void             glGetRenderbufferParameter

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  inline GLboolean        glIsFramebuffer

    ( GLuint                framebuffer );

  inline void             glBindFramebuffer

    ( GLenum                target,
      GLuint                framebuffer );

  inline void             glDeleteFramebuffers

    ( GLsizei               n,
      const GLuint*         framebuffers );

  inline void             glGenFramebuffers

    ( GLsizei               n,
      GLuint*               framebuffers );

  inline GLenum           glCheckFramebufferStatus

    ( GLenum                target );

  inline void             glFramebufferTexture1D

    ( GLenum                target,
      GLenum                attachment,
      GLenum                textarget,
      GLuint                texture,
      GLint                 level );

  inline void             glFramebufferTexture2D

    ( GLenum                target,
      GLenum                attachment,
      GLenum                textarget,
      GLuint                texture,
      GLint                 level );

  inline void             glFramebufferTexture3D

    ( GLenum                target,
      GLenum                attachment,
      GLenum                textarget,
      GLuint                texture,
      GLint                 level,
      GLint                 zoffset );

  inline void             glFramebufferRenderbuffer

    ( GLenum                target,
      GLenum                attachment,
      GLenum                renderbuffertarget,
      GLuint                renderbuffer );

  inline void             glGetFramebufferAttachmentParameter

    ( GLenum                target,
      GLenum                attachment,
      GLenum                pname,
      GLint&                param );

  inline void             glGenerateMipmap

    ( GLenum                target );

  inline void             glBlitFramebuffer

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

  inline void             glRenderbufferStorageMultisample

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  inline void             glFramebufferTextureLayer

    ( GLenum                target,
      GLenum                attachment,
      GLuint                texture,
      GLint                 level,
      GLint                 layer );

  inline void*            glMapBufferRange

    ( GLenum                target,
      GLintptr              offset,
      GLsizeiptr            length,
      GLbitfield            access );

  inline void             glFlushMappedBufferRange

    ( GLenum                target,
      GLintptr              offset,
      GLsizeiptr            length );

  inline void             glBindVertexArray

    ( GLuint                array );

  inline void             glDeleteVertexArrays

    ( GLsizei               n,
      const GLuint*         arrays );

  inline void             glGenVertexArrays

    ( GLsizei               n,
      GLuint*               arrays );

  inline GLboolean        glIsVertexArray

    ( GLuint                array );


 private:

  Ref<GLFuncTable3Core>   table_;

};

typedef GLFunctions3Core  GLFuncs3Core;


//-----------------------------------------------------------------------
//   class GLFunctions3Compat
//-----------------------------------------------------------------------


class GLFunctions3Compat : public GLFunctions2Compat
{
 public:

  explicit                GLFunctions3Compat

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions3Compat  ();

 private:

  Ref
    <GLFuncTable3Compat>  table_;

};


typedef
  GLFunctions3Compat      GLFuncs3Compat;


//-----------------------------------------------------------------------
//   class GLFunctions3
//-----------------------------------------------------------------------


class GLFunctions3 : public GLFunctions3Core,
                     public GLFunctions3Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_3;

  explicit                GLFunctions3

    ( GLFuncTables&         tables );

};


typedef GLFunctions3      GLFuncs3;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions3Core
//=======================================================================


inline void GLFunctions3Core::glColorMask

  ( GLuint     index,
    GLboolean  r,
    GLboolean  g,
    GLboolean  b,
    GLboolean  a )

{
  table_->glColorMaski ( index, r, g, b, a );
}


inline void GLFunctions3Core::glGetParameterI

  ( GLenum      target,
    GLuint      index,
    GLboolean*  data )

{
  table_->glGetBooleani_v ( target, index, data );
}


inline void GLFunctions3Core::glGetParameterI

  ( GLenum  target,
    GLuint  index,
    GLint*  data )

{
  table_->glGetIntegeri_v ( target, index, data );
}


inline void GLFunctions3Core::glEnable

  ( GLenum  target,
    GLuint  index )

{
  table_->glEnablei ( target, index );
}


inline void GLFunctions3Core::glDisable

  ( GLenum  target,
    GLuint  index )

{
  table_->glDisablei ( target, index );
}


inline GLboolean GLFunctions3Core::glIsEnabled

  ( GLenum  target,
    GLuint  index )

{
  return table_->glIsEnabledi ( target, index );
}


inline void GLFunctions3Core::glBeginTransformFeedback

  ( GLenum  primitiveMode )

{
  table_->glBeginTransformFeedback ( primitiveMode );
}


inline void GLFunctions3Core::glEndTransformFeedback ()

{
  table_->glEndTransformFeedback ();
}


inline void GLFunctions3Core::glBindBufferRange

  ( GLenum      target,
    GLuint      index,
    GLuint      buffer,
    GLintptr    offset,
    GLsizeiptr  size )

{
  table_->glBindBufferRange ( target, index, buffer, offset, size );
}


inline void GLFunctions3Core::glBindBufferBase

  ( GLenum  target,
    GLuint  index,
    GLuint  buffer )

{
  table_->glBindBufferBase ( target, index, buffer );
}


inline void GLFunctions3Core::glTransformFeedbackVaryings

  ( GLuint                program,
    GLsizei               count,
    const GLchar* const*  varyings,
    GLenum                bufferMode )

{
  table_->glTransformFeedbackVaryings ( program, count, varyings,
                                        bufferMode );
}


inline void GLFunctions3Core::glGetTransformFeedbackVarying

  ( GLuint    program,
    GLuint    index,
    GLsizei   bufSize,
    GLsizei*  length,
    GLsizei*  size,
    GLenum*   type,
    GLchar*   name )

{
  table_->glGetTransformFeedbackVarying ( program, index, bufSize,
                                          length, size, type, name );
}


inline void GLFunctions3Core::glClampColor

  ( GLenum  target,
    GLenum  clamp )

{
  table_->glClampColor ( target, clamp );
}


inline void GLFunctions3Core::glBeginConditionalRender

  ( GLuint  id,
    GLenum  mode )

{
  table_->glBeginConditionalRender ( id, mode );
}


inline void GLFunctions3Core::glEndConditionalRender ()

{
  table_->glEndConditionalRender ();
}


inline void GLFunctions3Core::glVertexAttribIPointer

  ( GLuint       index,
    GLint        size,
    GLenum       type,
    GLsizei      stride,
    const void*  pointer )

{
  table_->glVertexAttribIPointer ( index, size, type, stride,
                                   pointer );
}


inline void GLFunctions3Core::glGetVertexAttribI

  ( GLuint  index,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetVertexAttribIiv ( index, pname, params );
}


inline void GLFunctions3Core::glGetVertexAttribI

  ( GLuint   index,
    GLenum   pname,
    GLuint*  params )

{
  table_->glGetVertexAttribIuiv ( index, pname, params );
}


inline void GLFunctions3Core::glVertexAttribI

  ( GLuint  index,
    GLint   x )

{
  table_->glVertexAttribI1i ( index, x );
}


inline void GLFunctions3Core::glVertexAttribI

  ( GLuint  index,
    GLint   x,
    GLint   y )

{
  table_->glVertexAttribI2i ( index, x, y );
}


inline void GLFunctions3Core::glVertexAttribI

  ( GLuint  index,
    GLint   x,
    GLint   y,
    GLint   z )

{
  table_->glVertexAttribI3i ( index, x, y, z );
}


inline void GLFunctions3Core::glVertexAttribI

  ( GLuint  index,
    GLint   x,
    GLint   y,
    GLint   z,
    GLint   w )

{
  table_->glVertexAttribI4i ( index, x, y, z, w );
}


inline void GLFunctions3Core::glVertexAttribI

  ( GLuint  index,
    GLuint  x )

{
  table_->glVertexAttribI1ui ( index, x );
}


inline void GLFunctions3Core::glVertexAttribI

  ( GLuint  index,
    GLuint  x,
    GLuint  y )

{
  table_->glVertexAttribI2ui ( index, x, y );
}


inline void GLFunctions3Core::glVertexAttribI

  ( GLuint  index,
    GLuint  x,
    GLuint  y,
    GLuint  z )

{
  table_->glVertexAttribI3ui ( index, x, y, z );
}


inline void GLFunctions3Core::glVertexAttribI

  ( GLuint  index,
    GLuint  x,
    GLuint  y,
    GLuint  z,
    GLuint  w )

{
  table_->glVertexAttribI4ui ( index, x, y, z, w );
}


inline void GLFunctions3Core::glVertexAttribI1

  ( GLuint        index,
    const GLint*  v )

{
  table_->glVertexAttribI1iv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI2

  ( GLuint        index,
    const GLint*  v )

{
  table_->glVertexAttribI2iv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI3

  ( GLuint        index,
    const GLint*  v )

{
  table_->glVertexAttribI3iv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI4

  ( GLuint        index,
    const GLint*  v )

{
  table_->glVertexAttribI4iv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI1

  ( GLuint         index,
    const GLuint*  v )

{
  table_->glVertexAttribI1uiv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI2

  ( GLuint         index,
    const GLuint*  v )

{
  table_->glVertexAttribI2uiv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI3

  ( GLuint         index,
    const GLuint*  v )

{
  table_->glVertexAttribI3uiv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI4

  ( GLuint         index,
    const GLuint*  v )

{
  table_->glVertexAttribI4uiv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI4

  ( GLuint         index,
    const GLbyte*  v )

{
  table_->glVertexAttribI4bv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI4

  ( GLuint          index,
    const GLshort*  v )

{
  table_->glVertexAttribI4sv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI4

  ( GLuint          index,
    const GLubyte*  v )

{
  table_->glVertexAttribI4ubv ( index, v );
}


inline void GLFunctions3Core::glVertexAttribI4

  ( GLuint           index,
    const GLushort*  v )

{
  table_->glVertexAttribI4usv ( index, v );
}


inline void GLFunctions3Core::glGetUniform

  ( GLuint   program,
    GLint    location,
    GLuint*  params )

{
  table_->glGetUniformuiv ( program, location, params );
}


inline void GLFunctions3Core::glBindFragDataLocation

  ( GLuint         program,
    GLuint         color,
    const GLchar*  name )

{
  table_->glBindFragDataLocation ( program, color, name );
}


inline GLint GLFunctions3Core::glGetFragDataLocation

  ( GLuint         program,
    const GLchar*  name )

{
  return table_->glGetFragDataLocation ( program, name );
}


inline void GLFunctions3Core::glUniform

  ( GLint   location,
    GLuint  v0 )

{
  table_->glUniform1ui ( location, v0 );
}


inline void GLFunctions3Core::glUniform

  ( GLint   location,
    GLuint  v0,
    GLuint  v1 )

{
  table_->glUniform2ui ( location, v0, v1 );
}


inline void GLFunctions3Core::glUniform

  ( GLint   location,
    GLuint  v0,
    GLuint  v1,
    GLuint  v2 )

{
  table_->glUniform3ui ( location, v0, v1, v2 );
}


inline void GLFunctions3Core::glUniform

  ( GLint   location,
    GLuint  v0,
    GLuint  v1,
    GLuint  v2,
    GLuint  v3 )

{
  table_->glUniform4ui ( location, v0, v1, v2, v3 );
}


inline void GLFunctions3Core::glUniform1

  ( GLint          location,
    GLsizei        count,
    const GLuint*  value )

{
  table_->glUniform1uiv ( location, count, value );
}


inline void GLFunctions3Core::glUniform2

  ( GLint          location,
    GLsizei        count,
    const GLuint*  value )

{
  table_->glUniform2uiv ( location, count, value );
}


inline void GLFunctions3Core::glUniform3

  ( GLint          location,
    GLsizei        count,
    const GLuint*  value )

{
  table_->glUniform3uiv ( location, count, value );
}


inline void GLFunctions3Core::glUniform4

  ( GLint          location,
    GLsizei        count,
    const GLuint*  value )

{
  table_->glUniform4uiv ( location, count, value );
}


inline void GLFunctions3Core::glTexParameterI

  ( GLenum        target,
    GLenum        pname,
    const GLint*  params )

{
  table_->glTexParameterIiv ( target, pname, params );
}


inline void GLFunctions3Core::glTexParameterI

  ( GLenum         target,
    GLenum         pname,
    const GLuint*  params )

{
  table_->glTexParameterIuiv ( target, pname, params );
}


inline void GLFunctions3Core::glGetTexParameterI

  ( GLenum  target,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetTexParameterIiv ( target, pname, params );
}


inline void GLFunctions3Core::glGetTexParameterI

  ( GLenum   target,
    GLenum   pname,
    GLuint*  params )

{
  table_->glGetTexParameterIuiv ( target, pname, params );
}


inline void GLFunctions3Core::glClearBuffer

  ( GLenum        buffer,
    GLint         drawbuffer,
    const GLint*  value )

{
  table_->glClearBufferiv ( buffer, drawbuffer, value );
}


inline void GLFunctions3Core::glClearBuffer

  ( GLenum         buffer,
    GLint          drawbuffer,
    const GLuint*  value )

{
  table_->glClearBufferuiv ( buffer, drawbuffer, value );
}


inline void GLFunctions3Core::glClearBuffer

  ( GLenum          buffer,
    GLint           drawbuffer,
    const GLfloat*  value )

{
  table_->glClearBufferfv ( buffer, drawbuffer, value );
}


inline void GLFunctions3Core::glClearBuffer

  ( GLenum   buffer,
    GLint    drawbuffer,
    GLfloat  depth,
    GLint    stencil )

{
  table_->glClearBufferfi ( buffer, drawbuffer, depth, stencil );
}


inline const GLubyte* GLFunctions3Core::glGetStringi

  ( GLenum  name,
    GLuint  index )

{
  return table_->glGetStringi ( name, index );
}


inline GLboolean GLFunctions3Core::glIsRenderbuffer

  ( GLuint  renderbuffer )

{
  return table_->glIsRenderbuffer ( renderbuffer );
}


inline void GLFunctions3Core::glBindRenderbuffer

  ( GLenum  target,
    GLuint  renderbuffer )

{
  table_->glBindRenderbuffer ( target, renderbuffer );
}


inline void GLFunctions3Core::glDeleteRenderbuffers

  ( GLsizei        n,
    const GLuint*  renderbuffers )

{
  table_->glDeleteRenderbuffers ( n, renderbuffers );
}


inline void GLFunctions3Core::glGenRenderbuffers

  ( GLsizei  n,
    GLuint*  renderbuffers )

{
  table_->glGenRenderbuffers ( n, renderbuffers );
}


inline void GLFunctions3Core::glRenderbufferStorage

  ( GLenum   target,
    GLenum   internalformat,
    GLsizei  width,
    GLsizei  height )

{
  table_->glRenderbufferStorage ( target, internalformat,
                                  width, height );
}


inline void GLFunctions3Core::glGetRenderbufferParameter

  ( GLenum  target,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetRenderbufferParameteriv ( target, pname, params );
}


inline GLboolean GLFunctions3Core::glIsFramebuffer

  ( GLuint  framebuffer )

{
  return table_->glIsFramebuffer ( framebuffer );
}


inline void GLFunctions3Core::glBindFramebuffer

  ( GLenum  target,
    GLuint  framebuffer )

{
        table_->glBindFramebuffer ( target, framebuffer );
}


inline void GLFunctions3Core::glDeleteFramebuffers

  ( GLsizei        n,
    const GLuint*  framebuffers )

{
  table_->glDeleteFramebuffers ( n, framebuffers );
}


inline void GLFunctions3Core::glGenFramebuffers

  ( GLsizei  n,
    GLuint*  framebuffers )

{
  table_->glGenFramebuffers ( n, framebuffers );
}


inline GLenum GLFunctions3Core::glCheckFramebufferStatus

  ( GLenum  target )

{
  return table_->glCheckFramebufferStatus ( target );
}


inline void GLFunctions3Core::glFramebufferTexture1D

  ( GLenum  target,
    GLenum  attachment,
    GLenum  textarget,
    GLuint  texture,
    GLint   level )

{
  table_->glFramebufferTexture1D ( target, attachment, textarget,
                                   texture, level );
}


inline void GLFunctions3Core::glFramebufferTexture2D

  ( GLenum  target,
    GLenum  attachment,
    GLenum  textarget,
    GLuint  texture,
    GLint   level )

{
  table_->glFramebufferTexture2D ( target, attachment, textarget,
                                   texture, level );
}


inline void GLFunctions3Core::glFramebufferTexture3D

  ( GLenum  target,
    GLenum  attachment,
    GLenum  textarget,
    GLuint  texture,
    GLint   level,
    GLint   zoffset )

{
  table_->glFramebufferTexture3D ( target, attachment, textarget,
                                   texture, level, zoffset );
}


inline void GLFunctions3Core::glFramebufferRenderbuffer

  ( GLenum  target,
    GLenum  attachment,
    GLenum  renderbuffertarget,
    GLuint  renderbuffer )

{
  table_->glFramebufferRenderbuffer ( target, attachment,
                                      renderbuffertarget,
                                      renderbuffer );
}


inline void GLFunctions3Core::glGetFramebufferAttachmentParameter

  ( GLenum  target,
    GLenum  attachment,
    GLenum  pname,
    GLint&  param )

{
  table_->glGetFramebufferAttachmentParameteriv ( target, attachment,
                                                  pname, &param );
}


inline void GLFunctions3Core::glGenerateMipmap ( GLenum target )

{
  table_->glGenerateMipmap ( target );
}


inline void GLFunctions3Core::glBlitFramebuffer

  ( GLint       srcX0,
    GLint       srcY0,
    GLint       srcX1,
    GLint       srcY1,
    GLint       dstX0,
    GLint       dstY0,
    GLint       dstX1,
    GLint       dstY1,
    GLbitfield  mask,
    GLenum      filter )

{
  table_->glBlitFramebuffer ( srcX0, srcY0, srcX1, srcY1, dstX0,
                              dstY0, dstX1, dstY1, mask, filter );
}


inline void GLFunctions3Core::glRenderbufferStorageMultisample

  ( GLenum   target,
    GLsizei  samples,
    GLenum   internalformat,
    GLsizei  width,
    GLsizei  height )

{
  table_->glRenderbufferStorageMultisample ( target, samples,
                                             internalformat,
                                             width, height );
}


inline void GLFunctions3Core::glFramebufferTextureLayer

  ( GLenum  target,
    GLenum  attachment,
    GLuint  texture,
    GLint   level,
    GLint   layer )

{
  table_->glFramebufferTextureLayer ( target, attachment, texture,
                                      level, layer );
}


inline void* GLFunctions3Core::glMapBufferRange

  ( GLenum      target,
    GLintptr    offset,
    GLsizeiptr  length,
    GLbitfield  access )

{
  return table_->glMapBufferRange ( target, offset, length, access );
}


inline void GLFunctions3Core::glFlushMappedBufferRange

  ( GLenum      target,
    GLintptr    offset,
    GLsizeiptr  length )

{
  table_->glFlushMappedBufferRange ( target, offset, length );
}


inline void GLFunctions3Core::glBindVertexArray ( GLuint array )

{
  table_->glBindVertexArray ( array );
}


inline void GLFunctions3Core::glDeleteVertexArrays

  ( GLsizei        n,
    const GLuint*  arrays )

{
  table_->glDeleteVertexArrays ( n, arrays );
}


inline void GLFunctions3Core::glGenVertexArrays

  ( GLsizei  n,
    GLuint*  arrays )

{
  table_->glGenVertexArrays ( n, arrays );
}


inline GLboolean GLFunctions3Core::glIsVertexArray ( GLuint array )

{
  return table_->glIsVertexArray ( array );
}


JEM_END_PACKAGE( gl )

#endif
