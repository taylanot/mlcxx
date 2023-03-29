
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

#ifndef JEM_GL_GLFUNCTABLE4_3_H
#define JEM_GL_GLFUNCTABLE4_3_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable4_3Core
//-----------------------------------------------------------------------

// OpenGL 4.3 core functions.

class GLFuncTable4_3Core : public Object
{
 public:

  explicit                GLFuncTable4_3Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glClearBufferData)

    ( GLenum                target,
      GLenum                internalformat,
      GLenum                format,
      GLenum                type,
      const void*           data );

  void                   (JEM_APIENTRYP glClearBufferSubData)

    ( GLenum                target,
      GLenum                internalformat,
      GLintptr              offset,
      GLsizeiptr            size,
      GLenum                format,
      GLenum                type,
      const void*           data );

  void                   (JEM_APIENTRYP glDispatchCompute)

    ( GLuint                num_groups_x,
      GLuint                num_groups_y,
      GLuint                num_groups_z );

  void                   (JEM_APIENTRYP glDispatchComputeIndirect)

    ( GLintptr              indirect );

  void                   (JEM_APIENTRYP glCopyImageSubData)

    ( GLuint                srcName,
      GLenum                srcTarget,
      GLint                 srcLevel,
      GLint                 srcX,
      GLint                 srcY,
      GLint                 srcZ,
      GLuint                dstName,
      GLenum                dstTarget,
      GLint                 dstLevel,
      GLint                 dstX,
      GLint                 dstY,
      GLint                 dstZ,
      GLsizei               srcWidth,
      GLsizei               srcHeight,
      GLsizei               srcDepth );

  void                   (JEM_APIENTRYP glFramebufferParameteri)

    ( GLenum                target,
      GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glGetFramebufferParameteriv)

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetInternalformati64v)

    ( GLenum                target,
      GLenum                internalformat,
      GLenum                pname,
      GLsizei               bufSize,
      GLint64*              params );

  void                   (JEM_APIENTRYP glInvalidateTexSubImage)

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth );

  void                   (JEM_APIENTRYP glInvalidateTexImage)

    ( GLuint                texture,
      GLint                 level );

  void                   (JEM_APIENTRYP glInvalidateBufferSubData)

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            length );

  void                   (JEM_APIENTRYP glInvalidateBufferData)

    ( GLuint                buffer );

  void                   (JEM_APIENTRYP glInvalidateFramebuffer)

    ( GLenum                target,
      GLsizei               numAttachments,
      const GLenum*         attachments );

  void                   (JEM_APIENTRYP glInvalidateSubFramebuffer)

    ( GLenum                target,
      GLsizei               numAttachments,
      const GLenum*         attachments,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glMultiDrawArraysIndirect)

    ( GLenum                mode,
      const void*           indirect,
      GLsizei               drawcount,
      GLsizei               stride );

  void                   (JEM_APIENTRYP glMultiDrawElementsIndirect)

    ( GLenum                mode,
      GLenum                type,
      const void*           indirect,
      GLsizei               drawcount,
      GLsizei               stride );

  void                   (JEM_APIENTRYP glGetProgramInterfaceiv)

    ( GLuint                program,
      GLenum                programInterface,
      GLenum                pname,
      GLint*                params );

  GLuint                 (JEM_APIENTRYP glGetProgramResourceIndex)

    ( GLuint                program,
      GLenum                programInterface,
      const GLchar*         name );

  void                   (JEM_APIENTRYP glGetProgramResourceName)

    ( GLuint                program,
      GLenum                programInterface,
      GLuint                index,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               name );

  void                   (JEM_APIENTRYP glGetProgramResourceiv)

    ( GLuint                program,
      GLenum                programInterface,
      GLuint                index,
      GLsizei               propCount,
      const GLenum*         props,
      GLsizei               bufSize,
      GLsizei*              length,
      GLint*                params );

  GLint                  (JEM_APIENTRYP glGetProgramResourceLocation)

    ( GLuint                program,
      GLenum                programInterface,
      const GLchar*         name );

  GLint                  (JEM_APIENTRYP
                          glGetProgramResourceLocationIndex)

    ( GLuint                program,
      GLenum                programInterface,
      const GLchar*         name );

  void                   (JEM_APIENTRYP glShaderStorageBlockBinding)

    ( GLuint                program,
      GLuint                storageBlockIndex,
      GLuint                storageBlockBinding );

  void                   (JEM_APIENTRYP glTexBufferRange)

    ( GLenum                target,
      GLenum                internalformat,
      GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size );

  void                   (JEM_APIENTRYP glTexStorage2DMultisample)

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLboolean             fixedsamplelocations );

  void                   (JEM_APIENTRYP glTexStorage3DMultisample)

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLboolean             fixedsamplelocations );

  void                   (JEM_APIENTRYP glTextureView)

    ( GLuint                texture,
      GLenum                target,
      GLuint                origtexture,
      GLenum                internalformat,
      GLuint                minlevel,
      GLuint                numlevels,
      GLuint                minlayer,
      GLuint                numlayers );

  void                   (JEM_APIENTRYP glBindVertexBuffer)

    ( GLuint                bindingindex,
      GLuint                buffer,
      GLintptr              offset,
      GLsizei               stride );

  void                   (JEM_APIENTRYP glVertexAttribFormat)

    ( GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLboolean             normalized,
      GLuint                relativeoffset );

  void                   (JEM_APIENTRYP glVertexAttribIFormat)

    ( GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLuint                relativeoffset );

  void                   (JEM_APIENTRYP glVertexAttribLFormat)

    ( GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLuint                relativeoffset );

  void                   (JEM_APIENTRYP glVertexAttribBinding)

    ( GLuint                attribindex,
      GLuint                bindingindex );

  void                   (JEM_APIENTRYP glVertexBindingDivisor)

    ( GLuint                bindingindex,
      GLuint                divisor );

  void                   (JEM_APIENTRYP glDebugMessageControl)

    ( GLenum                source,
      GLenum                type,
      GLenum                severity,
      GLsizei               count,
      const GLuint*         ids,
      GLboolean             enabled );

  void                   (JEM_APIENTRYP glDebugMessageInsert)

    ( GLenum                source,
      GLenum                type,
      GLuint                id,
      GLenum                severity,
      GLsizei               length,
      const GLchar*         buf );

  void                   (JEM_APIENTRYP glDebugMessageCallback)

    ( GLDEBUGPROC           callback,
      const void*           userParam );

  GLuint                 (JEM_APIENTRYP glGetDebugMessageLog)

    ( GLuint                count,
      GLsizei               bufSize,
      GLenum*               sources,
      GLenum*               types,
      GLuint*               ids,
      GLenum*               severities,
      GLsizei*              lengths,
      GLchar*               messageLog );

  void                   (JEM_APIENTRYP glPushDebugGroup)

    ( GLenum                source,
      GLuint                id,
      GLsizei               length,
      const GLchar*         message );

  void                   (JEM_APIENTRYP glPopDebugGroup) ();

  void                   (JEM_APIENTRYP glObjectLabel)

    ( GLenum                identifier,
      GLuint                name,
      GLsizei               length,
      const GLchar*         label );

  void                   (JEM_APIENTRYP glGetObjectLabel)

    ( GLenum                identifier,
      GLuint                name,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               label );

  void                   (JEM_APIENTRYP glObjectPtrLabel)

    ( const void*           ptr,
      GLsizei               length,
      const GLchar*         label );

  void                   (JEM_APIENTRYP glGetObjectPtrLabel)

    ( const void*           ptr,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               label );


 protected:

  virtual                ~GLFuncTable4_3Core  ();

};


JEM_END_PACKAGE( gl )

#endif
