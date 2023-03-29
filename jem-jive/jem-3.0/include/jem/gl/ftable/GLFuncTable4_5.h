
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

#ifndef JEM_GL_GLFUNCTABLE4_5_H
#define JEM_GL_GLFUNCTABLE4_5_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable4_5Core
//-----------------------------------------------------------------------

// OpenGL 4.5 core functions.

class GLFuncTable4_5Core : public Object
{
 public:

  explicit                GLFuncTable4_5Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glClipControl)

    ( GLenum                origin,
      GLenum                depth );

  void                   (JEM_APIENTRYP glCreateTransformFeedbacks)

    ( GLsizei               n,
      GLuint*               ids );

  void                   (JEM_APIENTRYP glTransformFeedbackBufferBase)

    ( GLuint                xfb,
      GLuint                index,
      GLuint                buffer );

  void                   (JEM_APIENTRYP
                          glTransformFeedbackBufferRange)

    ( GLuint                xfb,
      GLuint                index,
      GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size );

  void                   (JEM_APIENTRYP glGetTransformFeedbackiv)

    ( GLuint                xfb,
      GLenum                pname,
      GLint*                param );

  void                   (JEM_APIENTRYP glGetTransformFeedbacki_v)

    ( GLuint                xfb,
      GLenum                pname,
      GLuint                index,
      GLint*                param );

  void                   (JEM_APIENTRYP glGetTransformFeedbacki64_v)

    ( GLuint                xfb,
      GLenum                pname,
      GLuint                index,
      GLint64*              param );

  void                   (JEM_APIENTRYP glCreateBuffers)

    ( GLsizei               n,
      GLuint*               buffers );

  void                   (JEM_APIENTRYP glNamedBufferStorage)

    ( GLuint                buffer,
      GLsizeiptr            size,
      const void*           data,
      GLbitfield            flags );

  void                   (JEM_APIENTRYP glNamedBufferData)

    ( GLuint                buffer,
      GLsizeiptr            size,
      const void*           data,
      GLenum                usage );

  void                   (JEM_APIENTRYP glNamedBufferSubData)

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size,
      const void*           data );

  void                   (JEM_APIENTRYP glCopyNamedBufferSubData)

    ( GLuint                readBuffer,
      GLuint                writeBuffer,
      GLintptr              readOffset,
      GLintptr              writeOffset,
      GLsizeiptr            size );

  void                   (JEM_APIENTRYP glClearNamedBufferData)

    ( GLuint                buffer,
      GLenum                internalformat,
      GLenum                format,
      GLenum                type,
      const void*           data );

  void                   (JEM_APIENTRYP glClearNamedBufferSubData)

    ( GLuint                buffer,
      GLenum                internalformat,
      GLintptr              offset,
      GLsizeiptr            size,
      GLenum                format,
      GLenum                type,
      const void*           data );

  void*                  (JEM_APIENTRYP glMapNamedBuffer)

    ( GLuint                buffer,
      GLenum                access );

  void*                  (JEM_APIENTRYP glMapNamedBufferRange)

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            length,
      GLbitfield            access );

  GLboolean              (JEM_APIENTRYP glUnmapNamedBuffer)

    ( GLuint                buffer );

  void                   (JEM_APIENTRYP glFlushMappedNamedBufferRange)

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            length );

  void                   (JEM_APIENTRYP glGetNamedBufferParameteriv)

    ( GLuint                buffer,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetNamedBufferParameteri64v)

    ( GLuint                buffer,
      GLenum                pname,
      GLint64*              params );

  void                   (JEM_APIENTRYP glGetNamedBufferPointerv)

    ( GLuint                buffer,
      GLenum                pname,
      void* *               params );

  void                   (JEM_APIENTRYP glGetNamedBufferSubData)

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size,
      void*                 data );

  void                   (JEM_APIENTRYP glCreateFramebuffers)

    ( GLsizei               n,
      GLuint*               framebuffers );

  void                   (JEM_APIENTRYP
                          glNamedFramebufferRenderbuffer)

    ( GLuint                framebuffer,
      GLenum                attachment,
      GLenum                renderbuffertarget,
      GLuint                renderbuffer );

  void                   (JEM_APIENTRYP glNamedFramebufferParameteri)

    ( GLuint                framebuffer,
      GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glNamedFramebufferTexture)

    ( GLuint                framebuffer,
      GLenum                attachment,
      GLuint                texture,
      GLint                 level );

  void                   (JEM_APIENTRYP
                          glNamedFramebufferTextureLayer)

    ( GLuint                framebuffer,
      GLenum                attachment,
      GLuint                texture,
      GLint                 level,
      GLint                 layer );

  void                   (JEM_APIENTRYP glNamedFramebufferDrawBuffer)

    ( GLuint                framebuffer,
      GLenum                buf );

  void                   (JEM_APIENTRYP glNamedFramebufferDrawBuffers)

    ( GLuint                framebuffer,
      GLsizei               n,
      const GLenum*         bufs );

  void                   (JEM_APIENTRYP glNamedFramebufferReadBuffer)

    ( GLuint                framebuffer,
      GLenum                src );

  void                   (JEM_APIENTRYP
                          glInvalidateNamedFramebufferData)

    ( GLuint                framebuffer,
      GLsizei               numAttachments,
      const GLenum*         attachments );

  void                   (JEM_APIENTRYP
                          glInvalidateNamedFramebufferSubData)

    ( GLuint                framebuffer,
      GLsizei               numAttachments,
      const GLenum*         attachments,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glClearNamedFramebufferiv)

    ( GLuint                framebuffer,
      GLenum                buffer,
      GLint                 drawbuffer,
      const GLint*          value );

  void                   (JEM_APIENTRYP glClearNamedFramebufferuiv)

    ( GLuint                framebuffer,
      GLenum                buffer,
      GLint                 drawbuffer,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glClearNamedFramebufferfv)

    ( GLuint                framebuffer,
      GLenum                buffer,
      GLint                 drawbuffer,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glClearNamedFramebufferfi)

    ( GLuint                framebuffer,
      GLenum                buffer,
      const GLfloat         depth,
      GLint                 stencil );

  void                   (JEM_APIENTRYP glBlitNamedFramebuffer)

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

  GLenum                 (JEM_APIENTRYP glCheckNamedFramebufferStatus)

    ( GLuint                framebuffer,
      GLenum                target );

  void                   (JEM_APIENTRYP
                          glGetNamedFramebufferParameteriv)

    ( GLuint                framebuffer,
      GLenum                pname,
      GLint*                param );

  void                   (JEM_APIENTRYP
                          glGetNamedFramebufferAttachmentParameteriv)

    ( GLuint                framebuffer,
      GLenum                attachment,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glCreateRenderbuffers)

    ( GLsizei               n,
      GLuint*               renderbuffers );

  void                   (JEM_APIENTRYP glNamedRenderbufferStorage)

    ( GLuint                renderbuffer,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP
                          glNamedRenderbufferStorageMultisample)

    ( GLuint                renderbuffer,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP
                          glGetNamedRenderbufferParameteriv)

    ( GLuint                renderbuffer,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glCreateTextures)

    ( GLenum                target,
      GLsizei               n,
      GLuint*               textures );

  void                   (JEM_APIENTRYP glTextureBuffer)

    ( GLuint                texture,
      GLenum                internalformat,
      GLuint                buffer );

  void                   (JEM_APIENTRYP glTextureBufferRange)

    ( GLuint                texture,
      GLenum                internalformat,
      GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size );

  void                   (JEM_APIENTRYP glTextureStorage1D)

    ( GLuint                texture,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width );

  void                   (JEM_APIENTRYP glTextureStorage2D)

    ( GLuint                texture,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glTextureStorage3D)

    ( GLuint                texture,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth );

  void                   (JEM_APIENTRYP glTextureStorage2DMultisample)

    ( GLuint                texture,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLboolean             fixedsamplelocations );

  void                   (JEM_APIENTRYP glTextureStorage3DMultisample)

    ( GLuint                texture,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLboolean             fixedsamplelocations );

  void                   (JEM_APIENTRYP glTextureSubImage1D)

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLsizei               width,
      GLenum                format,
      GLenum                type,
      const void*           pixels );

  void                   (JEM_APIENTRYP glTextureSubImage2D)

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      const void*           pixels );

  void                   (JEM_APIENTRYP glTextureSubImage3D)

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

  void                   (JEM_APIENTRYP glCompressedTextureSubImage1D)

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLsizei               width,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  void                   (JEM_APIENTRYP glCompressedTextureSubImage2D)

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  void                   (JEM_APIENTRYP glCompressedTextureSubImage3D)

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

  void                   (JEM_APIENTRYP glCopyTextureSubImage1D)

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width );

  void                   (JEM_APIENTRYP glCopyTextureSubImage2D)

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glCopyTextureSubImage3D)

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glTextureParameterf)

    ( GLuint                texture,
      GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glTextureParameterfv)

    ( GLuint                texture,
      GLenum                pname,
      const GLfloat*        param );

  void                   (JEM_APIENTRYP glTextureParameteri)

    ( GLuint                texture,
      GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glTextureParameterIiv)

    ( GLuint                texture,
      GLenum                pname,
      const GLint*          params );

  void                   (JEM_APIENTRYP glTextureParameterIuiv)

    ( GLuint                texture,
      GLenum                pname,
      const GLuint*         params );

  void                   (JEM_APIENTRYP glTextureParameteriv)

    ( GLuint                texture,
      GLenum                pname,
      const GLint*          param );

  void                   (JEM_APIENTRYP glGenerateTextureMipmap)

    ( GLuint                texture );

  void                   (JEM_APIENTRYP glBindTextureUnit)

    ( GLuint                unit,
      GLuint                texture );

  void                   (JEM_APIENTRYP glGetTextureImage)

    ( GLuint                texture,
      GLint                 level,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 pixels );

  void                   (JEM_APIENTRYP glGetCompressedTextureImage)

    ( GLuint                texture,
      GLint                 level,
      GLsizei               bufSize,
      void*                 pixels );

  void                   (JEM_APIENTRYP glGetTextureLevelParameterfv)

    ( GLuint                texture,
      GLint                 level,
      GLenum                pname,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetTextureLevelParameteriv)

    ( GLuint                texture,
      GLint                 level,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetTextureParameterfv)

    ( GLuint                texture,
      GLenum                pname,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetTextureParameterIiv)

    ( GLuint                texture,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetTextureParameterIuiv)

    ( GLuint                texture,
      GLenum                pname,
      GLuint*               params );

  void                   (JEM_APIENTRYP glGetTextureParameteriv)

    ( GLuint                texture,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glCreateVertexArrays)

    ( GLsizei               n,
      GLuint*               arrays );

  void                   (JEM_APIENTRYP glDisableVertexArrayAttrib)

    ( GLuint                vaobj,
      GLuint                index );

  void                   (JEM_APIENTRYP glEnableVertexArrayAttrib)

    ( GLuint                vaobj,
      GLuint                index );

  void                   (JEM_APIENTRYP glVertexArrayElementBuffer)

    ( GLuint                vaobj,
      GLuint                buffer );

  void                   (JEM_APIENTRYP glVertexArrayVertexBuffer)

    ( GLuint                vaobj,
      GLuint                bindingindex,
      GLuint                buffer,
      GLintptr              offset,
      GLsizei               stride );

  void                   (JEM_APIENTRYP glVertexArrayVertexBuffers)

    ( GLuint                vaobj,
      GLuint                first,
      GLsizei               count,
      const GLuint*         buffers,
      const GLintptr*       offsets,
      const GLsizei*        strides );

  void                   (JEM_APIENTRYP glVertexArrayAttribBinding)

    ( GLuint                vaobj,
      GLuint                attribindex,
      GLuint                bindingindex );

  void                   (JEM_APIENTRYP glVertexArrayAttribFormat)

    ( GLuint                vaobj,
      GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLboolean             normalized,
      GLuint                relativeoffset );

  void                   (JEM_APIENTRYP glVertexArrayAttribIFormat)

    ( GLuint                vaobj,
      GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLuint                relativeoffset );

  void                   (JEM_APIENTRYP glVertexArrayAttribLFormat)

    ( GLuint                vaobj,
      GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLuint                relativeoffset );

  void                   (JEM_APIENTRYP glVertexArrayBindingDivisor)

    ( GLuint                vaobj,
      GLuint                bindingindex,
      GLuint                divisor );

  void                   (JEM_APIENTRYP glGetVertexArrayiv)

    ( GLuint                vaobj,
      GLenum                pname,
      GLint*                param );

  void                   (JEM_APIENTRYP glGetVertexArrayIndexediv)

    ( GLuint                vaobj,
      GLuint                index,
      GLenum                pname,
      GLint*                param );

  void                   (JEM_APIENTRYP glGetVertexArrayIndexed64iv)

    ( GLuint                vaobj,
      GLuint                index,
      GLenum                pname,
      GLint64*              param );

  void                   (JEM_APIENTRYP glCreateSamplers)

    ( GLsizei               n,
      GLuint*               samplers );

  void                   (JEM_APIENTRYP glCreateProgramPipelines)

    ( GLsizei               n,
      GLuint*               pipelines );

  void                   (JEM_APIENTRYP glCreateQueries)

    ( GLenum                target,
      GLsizei               n,
      GLuint*               ids );

  void                   (JEM_APIENTRYP glGetQueryBufferObjecti64v)

    ( GLuint                id,
      GLuint                buffer,
      GLenum                pname,
      GLintptr              offset );

  void                   (JEM_APIENTRYP glGetQueryBufferObjectiv)

    ( GLuint                id,
      GLuint                buffer,
      GLenum                pname,
      GLintptr              offset );

  void                   (JEM_APIENTRYP glGetQueryBufferObjectui64v)

    ( GLuint                id,
      GLuint                buffer,
      GLenum                pname,
      GLintptr              offset );

  void                   (JEM_APIENTRYP glGetQueryBufferObjectuiv)

    ( GLuint                id,
      GLuint                buffer,
      GLenum                pname,
      GLintptr              offset );

  void                   (JEM_APIENTRYP glMemoryBarrierByRegion)

    ( GLbitfield            barriers );

  void                   (JEM_APIENTRYP glGetTextureSubImage)

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

  void                   (JEM_APIENTRYP
                          glGetCompressedTextureSubImage)

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

  GLenum                 (JEM_APIENTRYP glGetGraphicsResetStatus) ();

  void                   (JEM_APIENTRYP glGetnCompressedTexImage)

    ( GLenum                target,
      GLint                 lod,
      GLsizei               bufSize,
      void*                 pixels );

  void                   (JEM_APIENTRYP glGetnTexImage)

    ( GLenum                target,
      GLint                 level,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 pixels );

  void                   (JEM_APIENTRYP glGetnUniformdv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               bufSize,
      GLdouble*             params );

  void                   (JEM_APIENTRYP glGetnUniformfv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               bufSize,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetnUniformiv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               bufSize,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetnUniformuiv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               bufSize,
      GLuint*               params );

  void                   (JEM_APIENTRYP glReadnPixels)

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 data );

  void                   (JEM_APIENTRYP glGetnMapdv)

    ( GLenum                target,
      GLenum                query,
      GLsizei               bufSize,
      GLdouble*             v );

  void                   (JEM_APIENTRYP glGetnMapfv)

    ( GLenum                target,
      GLenum                query,
      GLsizei               bufSize,
      GLfloat*              v );

  void                   (JEM_APIENTRYP glGetnMapiv)

    ( GLenum                target,
      GLenum                query,
      GLsizei               bufSize,
      GLint*                v );

  void                   (JEM_APIENTRYP glGetnPixelMapfv)

    ( GLenum                map,
      GLsizei               bufSize,
      GLfloat*              values );

  void                   (JEM_APIENTRYP glGetnPixelMapuiv)

    ( GLenum                map,
      GLsizei               bufSize,
      GLuint*               values );

  void                   (JEM_APIENTRYP glGetnPixelMapusv)

    ( GLenum                map,
      GLsizei               bufSize,
      GLushort*             values );

  void                   (JEM_APIENTRYP glGetnPolygonStipple)

    ( GLsizei               bufSize,
      GLubyte*              pattern );

  void                   (JEM_APIENTRYP glGetnColorTable)

    ( GLenum                target,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 table );

  void                   (JEM_APIENTRYP glGetnConvolutionFilter)

    ( GLenum                target,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 image );

  void                   (JEM_APIENTRYP glGetnSeparableFilter)

    ( GLenum                target,
      GLenum                format,
      GLenum                type,
      GLsizei               rowBufSize,
      void*                 row,
      GLsizei               columnBufSize,
      void*                 column,
      void*                 span );

  void                   (JEM_APIENTRYP glGetnHistogram)

    ( GLenum                target,
      GLboolean             reset,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 values );

  void                   (JEM_APIENTRYP glGetnMinmax)

    ( GLenum                target,
      GLboolean             reset,
      GLenum                format,
      GLenum                type,
      GLsizei               bufSize,
      void*                 values );

  void                   (JEM_APIENTRYP glTextureBarrier) ();


 protected:

  virtual                ~GLFuncTable4_5Core  ();

};


JEM_END_PACKAGE( gl )

#endif
