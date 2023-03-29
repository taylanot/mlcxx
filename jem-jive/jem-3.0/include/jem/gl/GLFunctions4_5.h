
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

#ifndef JEM_GL_GLFUNCTIONS4_5_H
#define JEM_GL_GLFUNCTIONS4_5_H

#include <jem/gl/GLFunctions4_4.h>
#include <jem/gl/ftable/GLFuncTable4_5.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions4_5Core
//-----------------------------------------------------------------------


class GLFunctions4_5Core : public GLFunctions4_4Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_5_CORE;

  explicit                GLFunctions4_5Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions4_5Core  ();


 public:

  inline void             glClipControl

    ( GLenum                origin,
      GLenum                depth );

  inline void             glCreateTransformFeedbacks

    ( GLsizei               n,
      GLuint*               ids );

  inline void             glTransformFeedbackBufferBase

    ( GLuint                xfb,
      GLuint                index,
      GLuint                buffer );

  inline void             glTransformFeedbackBufferRange

    ( GLuint                xfb,
      GLuint                index,
      GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size );

  inline void             glGetTransformFeedback

    ( GLuint                xfb,
      GLenum                pname,
      GLint*                param );

  inline void             glGetTransformFeedback

    ( GLuint                xfb,
      GLenum                pname,
      GLuint                index,
      GLint*                param );

  inline void             glGetTransformFeedback

    ( GLuint                xfb,
      GLenum                pname,
      GLuint                index,
      GLint64*              param );

  inline void             glCreateBuffers

    ( GLsizei               n,
      GLuint*               buffers );

  inline void             glNamedBufferStorage

    ( GLuint                buffer,
      GLsizeiptr            size,
      const void*           data,
      GLbitfield            flags );

  inline void             glNamedBufferData

    ( GLuint                buffer,
      GLsizeiptr            size,
      const void*           data,
      GLenum                usage );

  inline void             glNamedBufferSubData

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size,
      const void*           data );

  inline void             glCopyNamedBufferSubData

    ( GLuint                readBuffer,
      GLuint                writeBuffer,
      GLintptr              readOffset,
      GLintptr              writeOffset,
      GLsizeiptr            size );

  inline void             glClearNamedBufferData

    ( GLuint                buffer,
      GLenum                internalformat,
      GLenum                format,
      GLenum                type,
      const void*           data );

  inline void             glClearNamedBufferSubData

    ( GLuint                buffer,
      GLenum                internalformat,
      GLintptr              offset,
      GLsizeiptr            size,
      GLenum                format,
      GLenum                type,
      const void*           data );

  inline void*            glMapNamedBuffer

    ( GLuint                buffer,
      GLenum                access );

  inline void*            glMapNamedBufferRange

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            length,
      GLbitfield            access );

  inline GLboolean        glUnmapNamedBuffer

    ( GLuint                buffer );

  inline void             glFlushMappedNamedBufferRange

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            length );

  inline void             glGetNamedBufferParameter

    ( GLuint                buffer,
      GLenum                pname,
      GLint*                params );

  inline void             glGetNamedBufferParameter

    ( GLuint                buffer,
      GLenum                pname,
      GLint64*              params );

  inline void             glGetNamedBufferPointer

    ( GLuint                buffer,
      GLenum                pname,
      void* *               params );

  inline void             glGetNamedBufferSubData

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size,
      void*                 data );

  inline void             glCreateFramebuffers

    ( GLsizei               n,
      GLuint*               framebuffers );

  inline void             glNamedFramebufferRenderbuffer

    ( GLuint                framebuffer,
      GLenum                attachment,
      GLenum                renderbuffertarget,
      GLuint                renderbuffer );

  inline void             glNamedFramebufferParameter

    ( GLuint                framebuffer,
      GLenum                pname,
      GLint                 param );

  inline void             glNamedFramebufferTexture

    ( GLuint                framebuffer,
      GLenum                attachment,
      GLuint                texture,
      GLint                 level );

  inline void             glNamedFramebufferTextureLayer

    ( GLuint                framebuffer,
      GLenum                attachment,
      GLuint                texture,
      GLint                 level,
      GLint                 layer );

  inline void             glNamedFramebufferDrawBuffer

    ( GLuint                framebuffer,
      GLenum                buf );

  inline void             glNamedFramebufferDrawBuffers

    ( GLuint                framebuffer,
      GLsizei               n,
      const GLenum*         bufs );

  inline void             glNamedFramebufferReadBuffer

    ( GLuint                framebuffer,
      GLenum                src );

  inline void             glInvalidateNamedFramebufferData

    ( GLuint                framebuffer,
      GLsizei               numAttachments,
      const GLenum*         attachments );

  inline void             glInvalidateNamedFramebufferSubData

    ( GLuint                framebuffer,
      GLsizei               numAttachments,
      const GLenum*         attachments,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  inline void             glClearNamedFramebuffer

    ( GLuint                framebuffer,
      GLenum                buffer,
      GLint                 drawbuffer,
      const GLint*          value );

  inline void             glClearNamedFramebuffer

    ( GLuint                framebuffer,
      GLenum                buffer,
      GLint                 drawbuffer,
      const GLuint*         value );

  inline void             glClearNamedFramebuffer

    ( GLuint                framebuffer,
      GLenum                buffer,
      GLint                 drawbuffer,
      const GLfloat*        value );

  inline void             glClearNamedFramebuffer

    ( GLuint                framebuffer,
      GLenum                buffer,
      const GLfloat         depth,
      GLint                 stencil );

  inline void             glBlitNamedFramebuffer

    ( GLuint                readFramebuffer,
      GLuint                drawFramebuffer,
      GLint                 srcX0,
      GLint                 srcY0,
      GLint                 srcX1,
      GLint                 srcY1,
      GLint                 dstX0,
      GLint                 dstY0,
      GLint                 dstX1,
      GLint                 dstY1,
      GLbitfield            mask,
      GLenum                filter );

  inline GLenum           glCheckNamedFramebufferStatus

    ( GLuint                framebuffer,
      GLenum                target );

  inline void             glGetNamedFramebufferParameter

    ( GLuint                framebuffer,
      GLenum                pname,
      GLint*                param );

  inline void             glGetNamedFramebufferAttachmentParameter

    ( GLuint                framebuffer,
      GLenum                attachment,
      GLenum                pname,
      GLint*                params );

  inline void             glCreateRenderbuffers

    ( GLsizei               n,
      GLuint*               renderbuffers );

  inline void             glNamedRenderbufferStorage

    ( GLuint                renderbuffer,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  inline void             glNamedRenderbufferStorageMultisample

    ( GLuint                renderbuffer,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  inline void             glGetNamedRenderbufferParameter

    ( GLuint                renderbuffer,
      GLenum                pname,
      GLint*                params );

  inline void             glCreateTextures

    ( GLenum                target,
      GLsizei               n,
      GLuint*               textures );

  inline void             glTextureBuffer

    ( GLuint                texture,
      GLenum                internalformat,
      GLuint                buffer );

  inline void             glTextureBufferRange

    ( GLuint                texture,
      GLenum                internalformat,
      GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size );

  inline void             glTextureStorage1D

    ( GLuint                texture,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width );

  inline void             glTextureStorage2D

    ( GLuint                texture,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  inline void             glTextureStorage3D

    ( GLuint                texture,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth );

  inline void             glTextureStorage2DMultisample

    ( GLuint                texture,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLboolean             fixedsamplelocations );

  inline void             glTextureStorage3DMultisample

    ( GLuint                texture,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLboolean             fixedsamplelocations );

  inline void             glTextureSubImage1D

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLsizei               width,
      GLenum                format,
      GLenum                type,
      const void*           pixels );

  inline void             glTextureSubImage2D

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      const void*           pixels );

  inline void             glTextureSubImage3D

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
      const void*           pixels );

  inline void             glCompressedTextureSubImage1D

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLsizei               width,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  inline void             glCompressedTextureSubImage2D

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  inline void             glCompressedTextureSubImage3D

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  inline void             glCopyTextureSubImage1D

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width );

  inline void             glCopyTextureSubImage2D

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  inline void             glCopyTextureSubImage3D

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  inline void             glTextureParameter

    ( GLuint                texture,
      GLenum                pname,
      GLfloat               param );

  inline void             glTextureParameter

    ( GLuint                texture,
      GLenum                pname,
      const GLfloat*        param );

  inline void             glTextureParameter

    ( GLuint                texture,
      GLenum                pname,
      GLint                 param );

  inline void             glTextureParameterI

    ( GLuint                texture,
      GLenum                pname,
      const GLint*          params );

  inline void             glTextureParameterI

    ( GLuint                texture,
      GLenum                pname,
      const GLuint*         params );

  inline void             glTextureParameter

    ( GLuint                texture,
      GLenum                pname,
      const GLint*          param );

  inline void             glGenerateTextureMipmap

    ( GLuint                texture );

  inline void             glBindTextureUnit

    ( GLuint                unit,
      GLuint                texture );

  inline void             glGetTextureImage

    ( GLuint                texture,
      GLint                 level,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 pixels );

  inline void             glGetCompressedTextureImage

    ( GLuint                texture,
      GLint                 level,
      GLsizei               bufSize,
      void*                 pixels );

  inline void             glGetTextureLevelParameter

    ( GLuint                texture,
      GLint                 level,
      GLenum                pname,
      GLfloat*              params );

  inline void             glGetTextureLevelParameter

    ( GLuint                texture,
      GLint                 level,
      GLenum                pname,
      GLint*                params );

  inline void             glGetTextureParameter

    ( GLuint                texture,
      GLenum                pname,
      GLfloat*              params );

  inline void             glGetTextureParameterI

    ( GLuint                texture,
      GLenum                pname,
      GLint*                params );

  inline void             glGetTextureParameterI

    ( GLuint                texture,
      GLenum                pname,
      GLuint*               params );

  inline void             glGetTextureParameter

    ( GLuint                texture,
      GLenum                pname,
      GLint*                params );

  inline void             glCreateVertexArrays

    ( GLsizei               n,
      GLuint*               arrays );

  inline void             glDisableVertexArrayAttrib

    ( GLuint                vaobj,
      GLuint                index );

  inline void             glEnableVertexArrayAttrib

    ( GLuint                vaobj,
      GLuint                index );

  inline void             glVertexArrayElementBuffer

    ( GLuint                vaobj,
      GLuint                buffer );

  inline void             glVertexArrayVertexBuffer

    ( GLuint                vaobj,
      GLuint                bindingindex,
      GLuint                buffer,
      GLintptr              offset,
      GLsizei               stride );

  inline void             glVertexArrayVertexBuffers

    ( GLuint                vaobj,
      GLuint                first,
      GLsizei               count,
      const GLuint*         buffers,
      const GLintptr*       offsets,
      const GLsizei*        strides );

  inline void             glVertexArrayAttribBinding

    ( GLuint                vaobj,
      GLuint                attribindex,
      GLuint                bindingindex );

  inline void             glVertexArrayAttribFormat

    ( GLuint                vaobj,
      GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLboolean             normalized,
      GLuint                relativeoffset );

  inline void             glVertexArrayAttribIFormat

    ( GLuint                vaobj,
      GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLuint                relativeoffset );

  inline void             glVertexArrayAttribLFormat

    ( GLuint                vaobj,
      GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLuint                relativeoffset );

  inline void             glVertexArrayBindingDivisor

    ( GLuint                vaobj,
      GLuint                bindingindex,
      GLuint                divisor );

  inline void             glGetVertexArray

    ( GLuint                vaobj,
      GLenum                pname,
      GLint*                param );

  inline void             glGetVertexArrayIndexed

    ( GLuint                vaobj,
      GLuint                index,
      GLenum                pname,
      GLint*                param );

  inline void             glGetVertexArrayIndexed

    ( GLuint                vaobj,
      GLuint                index,
      GLenum                pname,
      GLint64*              param );

  inline void             glCreateSamplers

    ( GLsizei               n,
      GLuint*               samplers );

  inline void             glCreateProgramPipelines

    ( GLsizei               n,
      GLuint*               pipelines );

  inline void             glCreateQueries

    ( GLenum                target,
      GLsizei               n,
      GLuint*               ids );

  inline void             glGetQueryBufferObjecti64v

    ( GLuint                id,
      GLuint                buffer,
      GLenum                pname,
      GLintptr              offset );

  inline void             glGetQueryBufferObjectiv

    ( GLuint                id,
      GLuint                buffer,
      GLenum                pname,
      GLintptr              offset );

  inline void             glGetQueryBufferObjectui64v

    ( GLuint                id,
      GLuint                buffer,
      GLenum                pname,
      GLintptr              offset );

  inline void             glGetQueryBufferObjectuiv

    ( GLuint                id,
      GLuint                buffer,
      GLenum                pname,
      GLintptr              offset );

  inline void             glMemoryBarrierByRegion

    ( GLbitfield            barriers );

  inline void             glGetTextureSubImage

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
      GLsizei               bufSize,
      void*                 pixels );

  inline void             glGetCompressedTextureSubImage

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLsizei               bufSize,
      void*                 pixels );

  inline GLenum           glGetGraphicsResetStatus ();

  inline void             glGetnCompressedTexImage

    ( GLenum                target,
      GLint                 lod,
      GLsizei               bufSize,
      void*                 pixels );

  inline void             glGetnTexImage

    ( GLenum                target,
      GLint                 level,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 pixels );

  inline void             glGetnUniform

    ( GLuint                program,
      GLint                 location,
      GLsizei               bufSize,
      GLdouble*             params );

  inline void             glGetnUniform

    ( GLuint                program,
      GLint                 location,
      GLsizei               bufSize,
      GLfloat*              params );

  inline void             glGetnUniform

    ( GLuint                program,
      GLint                 location,
      GLsizei               bufSize,
      GLint*                params );

  inline void             glGetnUniform

    ( GLuint                program,
      GLint                 location,
      GLsizei               bufSize,
      GLuint*               params );

  inline void             glReadnPixels

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 data );

  inline void             glGetnMap

    ( GLenum                target,
      GLenum                query,
      GLsizei               bufSize,
      GLdouble*             v );

  inline void             glGetnMap

    ( GLenum                target,
      GLenum                query,
      GLsizei               bufSize,
      GLfloat*              v );

  inline void             glGetnMap

    ( GLenum                target,
      GLenum                query,
      GLsizei               bufSize,
      GLint*                v );

  inline void             glGetnPixelMap

    ( GLenum                map,
      GLsizei               bufSize,
      GLfloat*              values );

  inline void             glGetnPixelMap

    ( GLenum                map,
      GLsizei               bufSize,
      GLuint*               values );

  inline void             glGetnPixelMap

    ( GLenum                map,
      GLsizei               bufSize,
      GLushort*             values );

  inline void             glGetnPolygonStipple

    ( GLsizei               bufSize,
      GLubyte*              pattern );

  inline void             glGetnColorTable

    ( GLenum                target,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 table );

  inline void             glGetnConvolutionFilter

    ( GLenum                target,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 image );

  inline void             glGetnSeparableFilter

    ( GLenum                target,
      GLenum                format,
      GLenum                type,
      GLsizei               rowBufSize,
      void*                 row,
      GLsizei               columnBufSize,
      void*                 column,
      void*                 span );

  inline void             glGetnHistogram

    ( GLenum                target,
      GLboolean             reset,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 values );

  inline void             glGetnMinmax

    ( GLenum                target,
      GLboolean             reset,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 values );

  inline void             glTextureBarrier ();


 private:

  Ref
    <GLFuncTable4_5Core>  table_;

};


typedef
  GLFunctions4_5Core      GLFuncs4_5Core;


//-----------------------------------------------------------------------
//   class GLFunctions4_5
//-----------------------------------------------------------------------


class GLFunctions4_5 : public GLFunctions4_5Core,
                       public GLFunctions4Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_5;

  explicit                GLFunctions4_5

    ( GLFuncTables&         tables );

};


typedef GLFunctions4_5    GLFuncs4_5;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions4_5Core
//=======================================================================


inline void GLFunctions4_5Core::glClipControl

  ( GLenum  origin,
    GLenum  depth )

{
  table_->glClipControl ( origin, depth );
}


inline void GLFunctions4_5Core::glCreateTransformFeedbacks

  ( GLsizei  n,
    GLuint*  ids )

{
  table_->glCreateTransformFeedbacks ( n, ids );
}


inline void GLFunctions4_5Core::glTransformFeedbackBufferBase

  ( GLuint  xfb,
    GLuint  index,
    GLuint  buffer )

{
  table_->glTransformFeedbackBufferBase ( xfb, index, buffer );
}


inline void GLFunctions4_5Core::glTransformFeedbackBufferRange

  ( GLuint      xfb,
    GLuint      index,
    GLuint      buffer,
    GLintptr    offset,
    GLsizeiptr  size )

{
  table_->glTransformFeedbackBufferRange ( xfb, index, buffer, offset,
                                           size );
}


inline void GLFunctions4_5Core::glGetTransformFeedback

  ( GLuint  xfb,
    GLenum  pname,
    GLint*  param )

{
  table_->glGetTransformFeedbackiv ( xfb, pname, param );
}


inline void GLFunctions4_5Core::glGetTransformFeedback

  ( GLuint  xfb,
    GLenum  pname,
    GLuint  index,
    GLint*  param )

{
  table_->glGetTransformFeedbacki_v ( xfb, pname, index, param );
}


inline void GLFunctions4_5Core::glGetTransformFeedback

  ( GLuint    xfb,
    GLenum    pname,
    GLuint    index,
    GLint64*  param )

{
  table_->glGetTransformFeedbacki64_v ( xfb, pname, index, param );
}


inline void GLFunctions4_5Core::glCreateBuffers

  ( GLsizei  n,
    GLuint*  buffers )

{
  table_->glCreateBuffers ( n, buffers );
}


inline void GLFunctions4_5Core::glNamedBufferStorage

  ( GLuint       buffer,
    GLsizeiptr   size,
    const void*  data,
    GLbitfield   flags )

{
  table_->glNamedBufferStorage ( buffer, size, data, flags );
}


inline void GLFunctions4_5Core::glNamedBufferData

  ( GLuint       buffer,
    GLsizeiptr   size,
    const void*  data,
    GLenum       usage )

{
  table_->glNamedBufferData ( buffer, size, data, usage );
}


inline void GLFunctions4_5Core::glNamedBufferSubData

  ( GLuint       buffer,
    GLintptr     offset,
    GLsizeiptr   size,
    const void*  data )

{
  table_->glNamedBufferSubData ( buffer, offset, size, data );
}


inline void GLFunctions4_5Core::glCopyNamedBufferSubData

  ( GLuint      readBuffer,
    GLuint      writeBuffer,
    GLintptr    readOffset,
    GLintptr    writeOffset,
    GLsizeiptr  size )

{
  table_->glCopyNamedBufferSubData ( readBuffer, writeBuffer,
                                     readOffset, writeOffset, size );
}


inline void GLFunctions4_5Core::glClearNamedBufferData

  ( GLuint       buffer,
    GLenum       internalformat,
    GLenum       format,
    GLenum       type,
    const void*  data )

{
  table_->glClearNamedBufferData ( buffer, internalformat, format,
                                   type, data );
}


inline void GLFunctions4_5Core::glClearNamedBufferSubData

  ( GLuint       buffer,
    GLenum       internalformat,
    GLintptr     offset,
    GLsizeiptr   size,
    GLenum       format,
    GLenum       type,
    const void*  data )

{
  table_->glClearNamedBufferSubData ( buffer, internalformat, offset,
                                      size, format, type, data );
}


inline void* GLFunctions4_5Core::glMapNamedBuffer

  ( GLuint  buffer,
    GLenum  access )

{
  return table_->glMapNamedBuffer ( buffer, access );
}


inline void* GLFunctions4_5Core::glMapNamedBufferRange

  ( GLuint      buffer,
    GLintptr    offset,
    GLsizeiptr  length,
    GLbitfield  access )

{
  return table_->glMapNamedBufferRange ( buffer, offset, length,
                                         access );
}


inline GLboolean GLFunctions4_5Core::glUnmapNamedBuffer

  ( GLuint  buffer )

{
  return table_->glUnmapNamedBuffer ( buffer );
}


inline void GLFunctions4_5Core::glFlushMappedNamedBufferRange

  ( GLuint      buffer,
    GLintptr    offset,
    GLsizeiptr  length )

{
  table_->glFlushMappedNamedBufferRange ( buffer, offset, length );
}


inline void GLFunctions4_5Core::glGetNamedBufferParameter

  ( GLuint  buffer,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetNamedBufferParameteriv ( buffer, pname, params );
}


inline void GLFunctions4_5Core::glGetNamedBufferParameter

  ( GLuint    buffer,
    GLenum    pname,
    GLint64*  params )

{
  table_->glGetNamedBufferParameteri64v ( buffer, pname, params );
}


inline void GLFunctions4_5Core::glGetNamedBufferPointer

  ( GLuint   buffer,
    GLenum   pname,
    void* *  params )

{
  table_->glGetNamedBufferPointerv ( buffer, pname, params );
}


inline void GLFunctions4_5Core::glGetNamedBufferSubData

  ( GLuint      buffer,
    GLintptr    offset,
    GLsizeiptr  size,
    void*       data )

{
  table_->glGetNamedBufferSubData ( buffer, offset, size, data );
}


inline void GLFunctions4_5Core::glCreateFramebuffers

  ( GLsizei  n,
    GLuint*  framebuffers )

{
  table_->glCreateFramebuffers ( n, framebuffers );
}


inline void GLFunctions4_5Core::glNamedFramebufferRenderbuffer

  ( GLuint  framebuffer,
    GLenum  attachment,
    GLenum  renderbuffertarget,
    GLuint  renderbuffer )

{
  table_->glNamedFramebufferRenderbuffer ( framebuffer, attachment,
                                           renderbuffertarget,
                                           renderbuffer );
}


inline void GLFunctions4_5Core::glNamedFramebufferParameter

  ( GLuint  framebuffer,
    GLenum  pname,
    GLint   param )

{
  table_->glNamedFramebufferParameteri ( framebuffer, pname, param );
}


inline void GLFunctions4_5Core::glNamedFramebufferTexture

  ( GLuint  framebuffer,
    GLenum  attachment,
    GLuint  texture,
    GLint   level )

{
  table_->glNamedFramebufferTexture ( framebuffer, attachment,
                                      texture, level );
}


inline void GLFunctions4_5Core::glNamedFramebufferTextureLayer

  ( GLuint  framebuffer,
    GLenum  attachment,
    GLuint  texture,
    GLint   level,
    GLint   layer )

{
  table_->glNamedFramebufferTextureLayer ( framebuffer, attachment,
                                           texture, level, layer );
}


inline void GLFunctions4_5Core::glNamedFramebufferDrawBuffer

  ( GLuint  framebuffer,
    GLenum  buf )

{
  table_->glNamedFramebufferDrawBuffer ( framebuffer, buf );
}


inline void GLFunctions4_5Core::glNamedFramebufferDrawBuffers

  ( GLuint         framebuffer,
    GLsizei        n,
    const GLenum*  bufs )

{
  table_->glNamedFramebufferDrawBuffers ( framebuffer, n, bufs );
}


inline void GLFunctions4_5Core::glNamedFramebufferReadBuffer

  ( GLuint  framebuffer,
    GLenum  src )

{
  table_->glNamedFramebufferReadBuffer ( framebuffer, src );
}


inline void GLFunctions4_5Core::glInvalidateNamedFramebufferData

  ( GLuint         framebuffer,
    GLsizei        numAttachments,
    const GLenum*  attachments )

{
  table_->glInvalidateNamedFramebufferData ( framebuffer,
                                             numAttachments,
                                             attachments );
}


inline void GLFunctions4_5Core::glInvalidateNamedFramebufferSubData

  ( GLuint         framebuffer,
    GLsizei        numAttachments,
    const GLenum*  attachments,
    GLint          x,
    GLint          y,
    GLsizei        width,
    GLsizei        height )

{
  table_->glInvalidateNamedFramebufferSubData ( framebuffer,
                                                numAttachments,
                                                attachments,
                                                x, y, width, height );
}


inline void GLFunctions4_5Core::glClearNamedFramebuffer

  ( GLuint        framebuffer,
    GLenum        buffer,
    GLint         drawbuffer,
    const GLint*  value )

{
  table_->glClearNamedFramebufferiv ( framebuffer, buffer, drawbuffer,
                                      value );
}


inline void GLFunctions4_5Core::glClearNamedFramebuffer

  ( GLuint         framebuffer,
    GLenum         buffer,
    GLint          drawbuffer,
    const GLuint*  value )

{
  table_->glClearNamedFramebufferuiv ( framebuffer, buffer, drawbuffer,
                                       value );
}


inline void GLFunctions4_5Core::glClearNamedFramebuffer

  ( GLuint          framebuffer,
    GLenum          buffer,
    GLint           drawbuffer,
    const GLfloat*  value )

{
  table_->glClearNamedFramebufferfv ( framebuffer, buffer, drawbuffer,
                                      value );
}


inline void GLFunctions4_5Core::glClearNamedFramebuffer

  ( GLuint         framebuffer,
    GLenum         buffer,
    const GLfloat  depth,
    GLint          stencil )

{
  table_->glClearNamedFramebufferfi ( framebuffer, buffer, depth,
                                      stencil );
}


inline void GLFunctions4_5Core::glBlitNamedFramebuffer

  ( GLuint      readFramebuffer,
    GLuint      drawFramebuffer,
    GLint       srcX0,
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
  table_->glBlitNamedFramebuffer ( readFramebuffer, drawFramebuffer,
                                   srcX0, srcY0, srcX1, srcY1,
                                   dstX0, dstY0, dstX1, dstY1,
                                   mask, filter );
}


inline GLenum GLFunctions4_5Core::glCheckNamedFramebufferStatus

  ( GLuint  framebuffer,
    GLenum  target )

{
  return table_->
    glCheckNamedFramebufferStatus ( framebuffer, target );
}


inline void GLFunctions4_5Core::glGetNamedFramebufferParameter

  ( GLuint  framebuffer,
    GLenum  pname,
    GLint*  param )

{
  table_->glGetNamedFramebufferParameteriv ( framebuffer, pname,
                                             param );
}


inline void
  GLFunctions4_5Core::glGetNamedFramebufferAttachmentParameter

  ( GLuint  framebuffer,
    GLenum  attachment,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetNamedFramebufferAttachmentParameteriv (
    framebuffer,
    attachment,
    pname, params
  );
}


inline void GLFunctions4_5Core::glCreateRenderbuffers

  ( GLsizei  n,
    GLuint*  renderbuffers )

{
  table_->glCreateRenderbuffers ( n, renderbuffers );
}


inline void GLFunctions4_5Core::glNamedRenderbufferStorage

  ( GLuint   renderbuffer,
    GLenum   internalformat,
    GLsizei  width,
    GLsizei  height )

{
  table_->glNamedRenderbufferStorage ( renderbuffer, internalformat,
                                       width, height );
}


inline void GLFunctions4_5Core::glNamedRenderbufferStorageMultisample

  ( GLuint   renderbuffer,
    GLsizei  samples,
    GLenum   internalformat,
    GLsizei  width,
    GLsizei  height )

{
  table_->glNamedRenderbufferStorageMultisample ( renderbuffer,
                                                  samples,
                                                  internalformat,
                                                  width, height );
}


inline void GLFunctions4_5Core::glGetNamedRenderbufferParameter

  ( GLuint  renderbuffer,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetNamedRenderbufferParameteriv ( renderbuffer, pname,
                                              params );
}


inline void GLFunctions4_5Core::glCreateTextures

  ( GLenum   target,
    GLsizei  n,
    GLuint*  textures )

{
  table_->glCreateTextures ( target, n, textures );
}


inline void GLFunctions4_5Core::glTextureBuffer

  ( GLuint  texture,
    GLenum  internalformat,
    GLuint  buffer )

{
  table_->glTextureBuffer ( texture, internalformat, buffer );
}


inline void GLFunctions4_5Core::glTextureBufferRange

  ( GLuint      texture,
    GLenum      internalformat,
    GLuint      buffer,
    GLintptr    offset,
    GLsizeiptr  size )

{
  table_->glTextureBufferRange ( texture, internalformat, buffer,
                                 offset, size );
}


inline void GLFunctions4_5Core::glTextureStorage1D

  ( GLuint   texture,
    GLsizei  levels,
    GLenum   internalformat,
    GLsizei  width )

{
  table_->glTextureStorage1D ( texture, levels, internalformat,
                               width );
}


inline void GLFunctions4_5Core::glTextureStorage2D

  ( GLuint   texture,
    GLsizei  levels,
    GLenum   internalformat,
    GLsizei  width,
    GLsizei  height )

{
  table_->glTextureStorage2D ( texture, levels, internalformat,
                               width, height );
}


inline void GLFunctions4_5Core::glTextureStorage3D

  ( GLuint   texture,
    GLsizei  levels,
    GLenum   internalformat,
    GLsizei  width,
    GLsizei  height,
    GLsizei  depth )

{
  table_->glTextureStorage3D ( texture, levels, internalformat,
                               width, height, depth );
}


inline void GLFunctions4_5Core::glTextureStorage2DMultisample

  ( GLuint     texture,
    GLsizei    samples,
    GLenum     internalformat,
    GLsizei    width,
    GLsizei    height,
    GLboolean  fixedsamplelocations )

{
  table_->glTextureStorage2DMultisample ( texture, samples,
                                          internalformat,
                                          width, height,
                                          fixedsamplelocations );
}


inline void GLFunctions4_5Core::glTextureStorage3DMultisample

  ( GLuint     texture,
    GLsizei    samples,
    GLenum     internalformat,
    GLsizei    width,
    GLsizei    height,
    GLsizei    depth,
    GLboolean  fixedsamplelocations )

{
  table_->glTextureStorage3DMultisample ( texture, samples,
                                          internalformat,
                                          width, height, depth,
                                          fixedsamplelocations );
}


inline void GLFunctions4_5Core::glTextureSubImage1D

  ( GLuint       texture,
    GLint        level,
    GLint        xoffset,
    GLsizei      width,
    GLenum       format,
    GLenum       type,
    const void*  pixels )

{
  table_->glTextureSubImage1D ( texture, level,
                                xoffset,
                                width,
                                format, type, pixels );
}


inline void GLFunctions4_5Core::glTextureSubImage2D

  ( GLuint       texture,
    GLint        level,
    GLint        xoffset,
    GLint        yoffset,
    GLsizei      width,
    GLsizei      height,
    GLenum       format,
    GLenum       type,
    const void*  pixels )

{
  table_->glTextureSubImage2D ( texture, level,
                                xoffset, yoffset,
                                width, height,
                                format, type, pixels );
}


inline void GLFunctions4_5Core::glTextureSubImage3D

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
    const void*  pixels )

{
  table_->glTextureSubImage3D ( texture, level,
                                xoffset, yoffset, zoffset,
                                width, height, depth,
                                format, type, pixels );
}


inline void GLFunctions4_5Core::glCompressedTextureSubImage1D

  ( GLuint       texture,
    GLint        level,
    GLint        xoffset,
    GLsizei      width,
    GLenum       format,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTextureSubImage1D ( texture, level,
                                          xoffset,
                                          width,
                                          format, imageSize, data );
}


inline void GLFunctions4_5Core::glCompressedTextureSubImage2D

  ( GLuint       texture,
    GLint        level,
    GLint        xoffset,
    GLint        yoffset,
    GLsizei      width,
    GLsizei      height,
    GLenum       format,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTextureSubImage2D ( texture, level,
                                          xoffset, yoffset,
                                          width, height,
                                          format, imageSize, data );
}


inline void GLFunctions4_5Core::glCompressedTextureSubImage3D

  ( GLuint       texture,
    GLint        level,
    GLint        xoffset,
    GLint        yoffset,
    GLint        zoffset,
    GLsizei      width,
    GLsizei      height,
    GLsizei      depth,
    GLenum       format,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTextureSubImage3D ( texture, level,
                                          xoffset, yoffset, zoffset,
                                          width, height, depth,
                                          format, imageSize, data );
}


inline void GLFunctions4_5Core::glCopyTextureSubImage1D

  ( GLuint   texture,
    GLint    level,
    GLint    xoffset,
    GLint    x,
    GLint    y,
    GLsizei  width )

{
  table_->glCopyTextureSubImage1D ( texture, level,
                                    xoffset,
                                    x, y,
                                    width );
}


inline void GLFunctions4_5Core::glCopyTextureSubImage2D

  ( GLuint   texture,
    GLint    level,
    GLint    xoffset,
    GLint    yoffset,
    GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height )

{
  table_->glCopyTextureSubImage2D ( texture, level,
                                    xoffset, yoffset,
                                    x, y,
                                    width, height );
}


inline void GLFunctions4_5Core::glCopyTextureSubImage3D

  ( GLuint   texture,
    GLint    level,
    GLint    xoffset,
    GLint    yoffset,
    GLint    zoffset,
    GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height )

{
  table_->glCopyTextureSubImage3D ( texture, level,
                                    xoffset, yoffset, zoffset,
                                    x, y,
                                    width, height );
}


inline void GLFunctions4_5Core::glTextureParameter

  ( GLuint   texture,
    GLenum   pname,
    GLfloat  param )

{
  table_->glTextureParameterf ( texture, pname, param );
}


inline void GLFunctions4_5Core::glTextureParameter

  ( GLuint          texture,
    GLenum          pname,
    const GLfloat*  param )

{
  table_->glTextureParameterfv ( texture, pname, param );
}


inline void GLFunctions4_5Core::glTextureParameter

  ( GLuint  texture,
    GLenum  pname,
    GLint   param )

{
  table_->glTextureParameteri ( texture, pname, param );
}


inline void GLFunctions4_5Core::glTextureParameterI

  ( GLuint        texture,
    GLenum        pname,
    const GLint*  params )

{
  table_->glTextureParameterIiv ( texture, pname, params );
}


inline void GLFunctions4_5Core::glTextureParameterI

  ( GLuint         texture,
    GLenum         pname,
    const GLuint*  params )

{
  table_->glTextureParameterIuiv ( texture, pname, params );
}


inline void GLFunctions4_5Core::glTextureParameter

  ( GLuint        texture,
    GLenum        pname,
    const GLint*  param )

{
  table_->glTextureParameteriv ( texture, pname, param );
}


inline void GLFunctions4_5Core::glGenerateTextureMipmap

  ( GLuint  texture )

{
  table_->glGenerateTextureMipmap ( texture );
}


inline void GLFunctions4_5Core::glBindTextureUnit

  ( GLuint  unit,
    GLuint  texture )

{
  table_->glBindTextureUnit ( unit, texture );
}


inline void GLFunctions4_5Core::glGetTextureImage

  ( GLuint   texture,
    GLint    level,
    GLenum   format,
    GLenum   type,
    GLsizei  bufSize,
    void*    pixels )

{
  table_->glGetTextureImage ( texture, level, format, type, bufSize,
                              pixels );
}


inline void GLFunctions4_5Core::glGetCompressedTextureImage

  ( GLuint   texture,
    GLint    level,
    GLsizei  bufSize,
    void*    pixels )

{
  table_->glGetCompressedTextureImage ( texture, level, bufSize,
                                        pixels );
}


inline void GLFunctions4_5Core::glGetTextureLevelParameter

  ( GLuint    texture,
    GLint     level,
    GLenum    pname,
    GLfloat*  params )

{
  table_->glGetTextureLevelParameterfv ( texture, level, pname,
                                         params );
}


inline void GLFunctions4_5Core::glGetTextureLevelParameter

  ( GLuint  texture,
    GLint   level,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetTextureLevelParameteriv ( texture, level, pname,
                                         params );
}


inline void GLFunctions4_5Core::glGetTextureParameter

  ( GLuint    texture,
    GLenum    pname,
    GLfloat*  params )

{
  table_->glGetTextureParameterfv ( texture, pname, params );
}


inline void GLFunctions4_5Core::glGetTextureParameterI

  ( GLuint  texture,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetTextureParameterIiv ( texture, pname, params );
}


inline void GLFunctions4_5Core::glGetTextureParameterI

  ( GLuint   texture,
    GLenum   pname,
    GLuint*  params )

{
  table_->glGetTextureParameterIuiv ( texture, pname, params );
}


inline void GLFunctions4_5Core::glGetTextureParameter

  ( GLuint  texture,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetTextureParameteriv ( texture, pname, params );
}


inline void GLFunctions4_5Core::glCreateVertexArrays

  ( GLsizei  n,
    GLuint*  arrays )

{
  table_->glCreateVertexArrays ( n, arrays );
}


inline void GLFunctions4_5Core::glDisableVertexArrayAttrib

  ( GLuint  vaobj,
    GLuint  index )

{
  table_->glDisableVertexArrayAttrib ( vaobj, index );
}


inline void GLFunctions4_5Core::glEnableVertexArrayAttrib

  ( GLuint  vaobj,
    GLuint  index )

{
  table_->glEnableVertexArrayAttrib ( vaobj, index );
}


inline void GLFunctions4_5Core::glVertexArrayElementBuffer

  ( GLuint  vaobj,
    GLuint  buffer )

{
  table_->glVertexArrayElementBuffer ( vaobj, buffer );
}


inline void GLFunctions4_5Core::glVertexArrayVertexBuffer

  ( GLuint    vaobj,
    GLuint    bindingindex,
    GLuint    buffer,
    GLintptr  offset,
    GLsizei   stride )

{
  table_->glVertexArrayVertexBuffer ( vaobj, bindingindex, buffer,
                                      offset, stride );
}


inline void GLFunctions4_5Core::glVertexArrayVertexBuffers

  ( GLuint           vaobj,
    GLuint           first,
    GLsizei          count,
    const GLuint*    buffers,
    const GLintptr*  offsets,
    const GLsizei*   strides )

{
  table_->glVertexArrayVertexBuffers ( vaobj, first, count, buffers,
                                       offsets, strides );
}


inline void GLFunctions4_5Core::glVertexArrayAttribBinding

  ( GLuint  vaobj,
    GLuint  attribindex,
    GLuint  bindingindex )

{
  table_->glVertexArrayAttribBinding ( vaobj, attribindex,
                                       bindingindex );
}


inline void GLFunctions4_5Core::glVertexArrayAttribFormat

  ( GLuint     vaobj,
    GLuint     attribindex,
    GLint      size,
    GLenum     type,
    GLboolean  normalized,
    GLuint     relativeoffset )

{
  table_->glVertexArrayAttribFormat ( vaobj, attribindex, size, type,
                                      normalized, relativeoffset );
}


inline void GLFunctions4_5Core::glVertexArrayAttribIFormat

  ( GLuint  vaobj,
    GLuint  attribindex,
    GLint   size,
    GLenum  type,
    GLuint  relativeoffset )

{
  table_->glVertexArrayAttribIFormat ( vaobj, attribindex, size, type,
                                       relativeoffset );
}


inline void GLFunctions4_5Core::glVertexArrayAttribLFormat

  ( GLuint  vaobj,
    GLuint  attribindex,
    GLint   size,
    GLenum  type,
    GLuint  relativeoffset )

{
  table_->glVertexArrayAttribLFormat ( vaobj, attribindex, size, type,
                                       relativeoffset );
}


inline void GLFunctions4_5Core::glVertexArrayBindingDivisor

  ( GLuint  vaobj,
    GLuint  bindingindex,
    GLuint  divisor )

{
  table_->glVertexArrayBindingDivisor ( vaobj, bindingindex, divisor );
}


inline void GLFunctions4_5Core::glGetVertexArray

  ( GLuint  vaobj,
    GLenum  pname,
    GLint*  param )

{
  table_->glGetVertexArrayiv ( vaobj, pname, param );
}


inline void GLFunctions4_5Core::glGetVertexArrayIndexed

  ( GLuint  vaobj,
    GLuint  index,
    GLenum  pname,
    GLint*  param )

{
  table_->glGetVertexArrayIndexediv ( vaobj, index, pname, param );
}


inline void GLFunctions4_5Core::glGetVertexArrayIndexed

  ( GLuint    vaobj,
    GLuint    index,
    GLenum    pname,
    GLint64*  param )

{
  table_->glGetVertexArrayIndexed64iv ( vaobj, index, pname, param );
}


inline void GLFunctions4_5Core::glCreateSamplers

  ( GLsizei  n,
    GLuint*  samplers )

{
  table_->glCreateSamplers ( n, samplers );
}


inline void GLFunctions4_5Core::glCreateProgramPipelines

  ( GLsizei  n,
    GLuint*  pipelines )

{
  table_->glCreateProgramPipelines ( n, pipelines );
}


inline void GLFunctions4_5Core::glCreateQueries

  ( GLenum   target,
    GLsizei  n,
    GLuint*  ids )

{
  table_->glCreateQueries ( target, n, ids );
}


inline void GLFunctions4_5Core::glGetQueryBufferObjecti64v

  ( GLuint    id,
    GLuint    buffer,
    GLenum    pname,
    GLintptr  offset )

{
  table_->glGetQueryBufferObjecti64v ( id, buffer, pname, offset );
}


inline void GLFunctions4_5Core::glGetQueryBufferObjectiv

  ( GLuint    id,
    GLuint    buffer,
    GLenum    pname,
    GLintptr  offset )

{
  table_->glGetQueryBufferObjectiv ( id, buffer, pname, offset );
}


inline void GLFunctions4_5Core::glGetQueryBufferObjectui64v

  ( GLuint    id,
    GLuint    buffer,
    GLenum    pname,
    GLintptr  offset )

{
  table_->glGetQueryBufferObjectui64v ( id, buffer, pname, offset );
}


inline void GLFunctions4_5Core::glGetQueryBufferObjectuiv

  ( GLuint    id,
    GLuint    buffer,
    GLenum    pname,
    GLintptr  offset )

{
  table_->glGetQueryBufferObjectuiv ( id, buffer, pname, offset );
}


inline void GLFunctions4_5Core::glMemoryBarrierByRegion

  ( GLbitfield  barriers )

{
  table_->glMemoryBarrierByRegion ( barriers );
}


inline void GLFunctions4_5Core::glGetTextureSubImage

  ( GLuint   texture,
    GLint    level,
    GLint    xoffset,
    GLint    yoffset,
    GLint    zoffset,
    GLsizei  width,
    GLsizei  height,
    GLsizei  depth,
    GLenum   format,
    GLenum   type,
    GLsizei  bufSize,
    void*    pixels )

{
  table_->glGetTextureSubImage ( texture, level,
                                 xoffset, yoffset, zoffset,
                                 width, height, depth,
                                 format, type, bufSize, pixels );
}


inline void GLFunctions4_5Core::glGetCompressedTextureSubImage

  ( GLuint   texture,
    GLint    level,
    GLint    xoffset,
    GLint    yoffset,
    GLint    zoffset,
    GLsizei  width,
    GLsizei  height,
    GLsizei  depth,
    GLsizei  bufSize,
    void*    pixels )

{
  table_->glGetCompressedTextureSubImage ( texture, level,
                                           xoffset, yoffset, zoffset,
                                           width, height, depth,
                                           bufSize, pixels );
}


inline GLenum GLFunctions4_5Core::glGetGraphicsResetStatus ()
{
  return table_->glGetGraphicsResetStatus ();
}


inline void GLFunctions4_5Core::glGetnCompressedTexImage

  ( GLenum   target,
    GLint    lod,
    GLsizei  bufSize,
    void*    pixels )

{
  table_->glGetnCompressedTexImage ( target, lod, bufSize, pixels );
}


inline void GLFunctions4_5Core::glGetnTexImage

  ( GLenum   target,
    GLint    level,
    GLenum   format,
    GLenum   type,
    GLsizei  bufSize,
    void*    pixels )

{
  table_->glGetnTexImage ( target, level, format, type, bufSize,
                           pixels );
}


inline void GLFunctions4_5Core::glGetnUniform

  ( GLuint     program,
    GLint      location,
    GLsizei    bufSize,
    GLdouble*  params )

{
  table_->glGetnUniformdv ( program, location, bufSize, params );
}


inline void GLFunctions4_5Core::glGetnUniform

  ( GLuint    program,
    GLint     location,
    GLsizei   bufSize,
    GLfloat*  params )

{
  table_->glGetnUniformfv ( program, location, bufSize, params );
}


inline void GLFunctions4_5Core::glGetnUniform

  ( GLuint   program,
    GLint    location,
    GLsizei  bufSize,
    GLint*   params )

{
  table_->glGetnUniformiv ( program, location, bufSize, params );
}


inline void GLFunctions4_5Core::glGetnUniform

  ( GLuint   program,
    GLint    location,
    GLsizei  bufSize,
    GLuint*  params )

{
  table_->glGetnUniformuiv ( program, location, bufSize, params );
}


inline void GLFunctions4_5Core::glReadnPixels

  ( GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height,
    GLenum   format,
    GLenum   type,
    GLsizei  bufSize,
    void*    data )

{
  table_->glReadnPixels ( x, y, width, height, format, type, bufSize,
                          data );
}


inline void GLFunctions4_5Core::glGetnMap

  ( GLenum     target,
    GLenum     query,
    GLsizei    bufSize,
    GLdouble*  v )

{
  table_->glGetnMapdv ( target, query, bufSize, v );
}


inline void GLFunctions4_5Core::glGetnMap

  ( GLenum    target,
    GLenum    query,
    GLsizei   bufSize,
    GLfloat*  v )

{
  table_->glGetnMapfv ( target, query, bufSize, v );
}


inline void GLFunctions4_5Core::glGetnMap

  ( GLenum   target,
    GLenum   query,
    GLsizei  bufSize,
    GLint*   v )

{
  table_->glGetnMapiv ( target, query, bufSize, v );
}


inline void GLFunctions4_5Core::glGetnPixelMap

  ( GLenum    map,
    GLsizei   bufSize,
    GLfloat*  values )

{
  table_->glGetnPixelMapfv ( map, bufSize, values );
}


inline void GLFunctions4_5Core::glGetnPixelMap

  ( GLenum   map,
    GLsizei  bufSize,
    GLuint*  values )

{
  table_->glGetnPixelMapuiv ( map, bufSize, values );
}


inline void GLFunctions4_5Core::glGetnPixelMap

  ( GLenum     map,
    GLsizei    bufSize,
    GLushort*  values )

{
  table_->glGetnPixelMapusv ( map, bufSize, values );
}


inline void GLFunctions4_5Core::glGetnPolygonStipple

  ( GLsizei   bufSize,
    GLubyte*  pattern )

{
  table_->glGetnPolygonStipple ( bufSize, pattern );
}


inline void GLFunctions4_5Core::glGetnColorTable

  ( GLenum   target,
    GLenum   format,
    GLenum   type,
    GLsizei  bufSize,
    void*    table )

{
  table_->glGetnColorTable ( target, format, type, bufSize, table );
}


inline void GLFunctions4_5Core::glGetnConvolutionFilter

  ( GLenum   target,
    GLenum   format,
    GLenum   type,
    GLsizei  bufSize,
    void*    image )

{
  table_->glGetnConvolutionFilter ( target, format, type, bufSize,
                                    image );
}


inline void GLFunctions4_5Core::glGetnSeparableFilter

  ( GLenum   target,
    GLenum   format,
    GLenum   type,
    GLsizei  rowBufSize,
    void*    row,
    GLsizei  columnBufSize,
    void*    column,
    void*    span )

{
  table_->glGetnSeparableFilter ( target, format, type, rowBufSize,
                                  row, columnBufSize, column, span );
}


inline void GLFunctions4_5Core::glGetnHistogram

  ( GLenum     target,
    GLboolean  reset,
    GLenum     format,
    GLenum     type,
    GLsizei    bufSize,
    void*      values )

{
  table_->glGetnHistogram ( target, reset, format, type, bufSize,
                            values );
}


inline void GLFunctions4_5Core::glGetnMinmax

  ( GLenum     target,
    GLboolean  reset,
    GLenum     format,
    GLenum     type,
    GLsizei    bufSize,
    void*      values )

{
  table_->glGetnMinmax ( target, reset, format, type, bufSize,
                         values );
}


inline void GLFunctions4_5Core::glTextureBarrier ()
{
  table_->glTextureBarrier ();
}


JEM_END_PACKAGE( gl )

#endif
