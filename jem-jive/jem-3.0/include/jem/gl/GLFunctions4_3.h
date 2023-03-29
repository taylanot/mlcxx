
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

#ifndef JEM_GL_GLFUNCTIONS4_3_H
#define JEM_GL_GLFUNCTIONS4_3_H

#include <jem/gl/GLFunctions4_2.h>
#include <jem/gl/ftable/GLFuncTable4_3.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions4_3Core
//-----------------------------------------------------------------------


class GLFunctions4_3Core : public GLFunctions4_2Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_3_CORE;

  explicit                GLFunctions4_3Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions4_3Core  ();


 public:

  using GLFunctions4_2Core::glGetInternalformat;

  inline void             glClearBufferData

    ( GLenum                target,
      GLenum                internalformat,
      GLenum                format,
      GLenum                type,
      const void*           data );

  inline void             glClearBufferSubData

    ( GLenum                target,
      GLenum                internalformat,
      GLintptr              offset,
      GLsizeiptr            size,
      GLenum                format,
      GLenum                type,
      const void*           data );

  inline void             glDispatchCompute

    ( GLuint                num_groups_x,
      GLuint                num_groups_y,
      GLuint                num_groups_z );

  inline void             glDispatchComputeIndirect

    ( GLintptr              indirect );

  inline void             glCopyImageSubData

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

  inline void             glFramebufferParameter

    ( GLenum                target,
      GLenum                pname,
      GLint                 param );

  inline void             glGetFramebufferParameter

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  inline void             glGetInternalformat

    ( GLenum                target,
      GLenum                internalformat,
      GLenum                pname,
      GLsizei               bufSize,
      GLint64*              params );

  inline void             glInvalidateTexSubImage

    ( GLuint                texture,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth );

  inline void             glInvalidateTexImage

    ( GLuint                texture,
      GLint                 level );

  inline void             glInvalidateBufferSubData

    ( GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            length );

  inline void             glInvalidateBufferData

    ( GLuint                buffer );

  inline void             glInvalidateFramebuffer

    ( GLenum                target,
      GLsizei               numAttachments,
      const GLenum*         attachments );

  inline void             glInvalidateSubFramebuffer

    ( GLenum                target,
      GLsizei               numAttachments,
      const GLenum*         attachments,
      GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  inline void             glMultiDrawArraysIndirect

    ( GLenum                mode,
      const void*           indirect,
      GLsizei               drawcount,
      GLsizei               stride );

  inline void             glMultiDrawElementsIndirect

    ( GLenum                mode,
      GLenum                type,
      const void*           indirect,
      GLsizei               drawcount,
      GLsizei               stride );

  inline void             glGetProgramInterface

    ( GLuint                program,
      GLenum                programInterface,
      GLenum                pname,
      GLint*                params );

  inline GLuint           glGetProgramResourceIndex

    ( GLuint                program,
      GLenum                programInterface,
      const GLchar*         name );

  inline void             glGetProgramResourceName

    ( GLuint                program,
      GLenum                programInterface,
      GLuint                index,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               name );

  inline void             glGetProgramResource

    ( GLuint                program,
      GLenum                programInterface,
      GLuint                index,
      GLsizei               propCount,
      const GLenum*         props,
      GLsizei               bufSize,
      GLsizei*              length,
      GLint*                params );

  inline GLint            glGetProgramResourceLocation

    ( GLuint                program,
      GLenum                programInterface,
      const GLchar*         name );

  inline GLint            glGetProgramResourceLocationIndex

    ( GLuint                program,
      GLenum                programInterface,
      const GLchar*         name );

  inline void             glShaderStorageBlockBinding

    ( GLuint                program,
      GLuint                storageBlockIndex,
      GLuint                storageBlockBinding );

  inline void             glTexBufferRange

    ( GLenum                target,
      GLenum                internalformat,
      GLuint                buffer,
      GLintptr              offset,
      GLsizeiptr            size );

  inline void             glTexStorage2DMultisample

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLboolean             fixedsamplelocations );

  inline void             glTexStorage3DMultisample

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLboolean             fixedsamplelocations );

  inline void             glTextureView

    ( GLuint                texture,
      GLenum                target,
      GLuint                origtexture,
      GLenum                internalformat,
      GLuint                minlevel,
      GLuint                numlevels,
      GLuint                minlayer,
      GLuint                numlayers );

  inline void             glBindVertexBuffer

    ( GLuint                bindingindex,
      GLuint                buffer,
      GLintptr              offset,
      GLsizei               stride );

  inline void             glVertexAttribFormat

    ( GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLboolean             normalized,
      GLuint                relativeoffset );

  inline void             glVertexAttribIFormat

    ( GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLuint                relativeoffset );

  inline void             glVertexAttribLFormat

    ( GLuint                attribindex,
      GLint                 size,
      GLenum                type,
      GLuint                relativeoffset );

  inline void             glVertexAttribBinding

    ( GLuint                attribindex,
      GLuint                bindingindex );

  inline void             glVertexBindingDivisor

    ( GLuint                bindingindex,
      GLuint                divisor );

  inline void             glDebugMessageControl

    ( GLenum                source,
      GLenum                type,
      GLenum                severity,
      GLsizei               count,
      const GLuint*         ids,
      GLboolean             enabled );

  inline void             glDebugMessageInsert

    ( GLenum                source,
      GLenum                type,
      GLuint                id,
      GLenum                severity,
      GLsizei               length,
      const GLchar*         buf );

  inline void             glDebugMessageCallback

    ( GLDEBUGPROC           callback,
      const void*           userParam );

  inline GLuint           glGetDebugMessageLog

    ( GLuint                count,
      GLsizei               bufSize,
      GLenum*               sources,
      GLenum*               types,
      GLuint*               ids,
      GLenum*               severities,
      GLsizei*              lengths,
      GLchar*               messageLog );

  inline void             glPushDebugGroup

    ( GLenum                source,
      GLuint                id,
      GLsizei               length,
      const GLchar*         message );

  inline void             glPopDebugGroup ();

  inline void             glObjectLabel

    ( GLenum                identifier,
      GLuint                name,
      GLsizei               length,
      const GLchar*         label );

  inline void             glGetObjectLabel

    ( GLenum                identifier,
      GLuint                name,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               label );

  inline void             glObjectPtrLabel

    ( const void*           ptr,
      GLsizei               length,
      const GLchar*         label );

  inline void             glGetObjectPtrLabel

    ( const void*           ptr,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               label );


 private:

  Ref
    <GLFuncTable4_3Core>  table_;

};


typedef
  GLFunctions4_3Core       GLFuncs4_3Core;


//-----------------------------------------------------------------------
//   class GLFunctions4_3
//-----------------------------------------------------------------------


class GLFunctions4_3 : public GLFunctions4_3Core,
                       public GLFunctions4Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_3;

  explicit                GLFunctions4_3

    ( GLFuncTables&         tables );

};


typedef GLFunctions4_3    GLFuncs4_3;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions4_3Core
//=======================================================================


inline void GLFunctions4_3Core::glClearBufferData

  ( GLenum       target,
    GLenum       internalformat,
    GLenum       format,
    GLenum       type,
    const void*  data )

{
  table_->glClearBufferData ( target, internalformat, format, type,
                              data );
}


inline void GLFunctions4_3Core::glClearBufferSubData

  ( GLenum       target,
    GLenum       internalformat,
    GLintptr     offset,
    GLsizeiptr   size,
    GLenum       format,
    GLenum       type,
    const void*  data )

{
  table_->glClearBufferSubData ( target, internalformat, offset, size,
                                 format, type, data );
}


inline void GLFunctions4_3Core::glDispatchCompute

  ( GLuint  num_groups_x,
    GLuint  num_groups_y,
    GLuint  num_groups_z )

{
  table_->glDispatchCompute ( num_groups_x, num_groups_y,
                              num_groups_z );
}


inline void GLFunctions4_3Core::glDispatchComputeIndirect

  ( GLintptr  indirect )

{
  table_->glDispatchComputeIndirect ( indirect );
}


inline void GLFunctions4_3Core::glCopyImageSubData

  ( GLuint   srcName,
    GLenum   srcTarget,
    GLint    srcLevel,
    GLint    srcX,
    GLint    srcY,
    GLint    srcZ,
    GLuint   dstName,
    GLenum   dstTarget,
    GLint    dstLevel,
    GLint    dstX,
    GLint    dstY,
    GLint    dstZ,
    GLsizei  srcWidth,
    GLsizei  srcHeight,
    GLsizei  srcDepth )

{
  table_->glCopyImageSubData ( srcName, srcTarget, srcLevel,
                               srcX, srcY, srcZ,
                               dstName, dstTarget, dstLevel,
                               dstX, dstY, dstZ,
                               srcWidth, srcHeight, srcDepth );
}


inline void GLFunctions4_3Core::glFramebufferParameter

  ( GLenum  target,
    GLenum  pname,
    GLint   param )

{
  table_->glFramebufferParameteri ( target, pname, param );
}


inline void GLFunctions4_3Core::glGetFramebufferParameter

  ( GLenum  target,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetFramebufferParameteriv ( target, pname, params );
}


inline void GLFunctions4_3Core::glGetInternalformat

  ( GLenum    target,
    GLenum    internalformat,
    GLenum    pname,
    GLsizei   bufSize,
    GLint64*  params )

{
  table_->glGetInternalformati64v ( target, internalformat, pname,
                                    bufSize, params );
}


inline void GLFunctions4_3Core::glInvalidateTexSubImage

  ( GLuint   texture,
    GLint    level,
    GLint    xoffset,
    GLint    yoffset,
    GLint    zoffset,
    GLsizei  width,
    GLsizei  height,
    GLsizei  depth )

{
  table_->glInvalidateTexSubImage ( texture, level,
                                    xoffset, yoffset, zoffset,
                                    width, height, depth );
}


inline void GLFunctions4_3Core::glInvalidateTexImage

  ( GLuint  texture,
    GLint   level )

{
  table_->glInvalidateTexImage ( texture, level );
}


inline void GLFunctions4_3Core::glInvalidateBufferSubData

  ( GLuint      buffer,
    GLintptr    offset,
    GLsizeiptr  length )

{
  table_->glInvalidateBufferSubData ( buffer, offset, length );
}


inline void GLFunctions4_3Core::glInvalidateBufferData

  ( GLuint  buffer )

{
  table_->glInvalidateBufferData ( buffer );
}


inline void GLFunctions4_3Core::glInvalidateFramebuffer

  ( GLenum         target,
    GLsizei        numAttachments,
    const GLenum*  attachments )

{
  table_->glInvalidateFramebuffer ( target, numAttachments,
                                    attachments );
}


inline void GLFunctions4_3Core::glInvalidateSubFramebuffer

  ( GLenum         target,
    GLsizei        numAttachments,
    const GLenum*  attachments,
    GLint          x,
    GLint          y,
    GLsizei        width,
    GLsizei        height )

{
  table_->glInvalidateSubFramebuffer ( target, numAttachments,
                                       attachments, x, y, width,
                                       height );
}


inline void GLFunctions4_3Core::glMultiDrawArraysIndirect

  ( GLenum       mode,
    const void*  indirect,
    GLsizei      drawcount,
    GLsizei      stride )

{
  table_->glMultiDrawArraysIndirect ( mode, indirect, drawcount,
                                      stride );
}


inline void GLFunctions4_3Core::glMultiDrawElementsIndirect

  ( GLenum       mode,
    GLenum       type,
    const void*  indirect,
    GLsizei      drawcount,
    GLsizei      stride )

{
  table_->glMultiDrawElementsIndirect ( mode, type, indirect,
                                        drawcount, stride );
}


inline void GLFunctions4_3Core::glGetProgramInterface

  ( GLuint  program,
    GLenum  programInterface,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetProgramInterfaceiv ( program, programInterface, pname,
                                    params );
}


inline GLuint GLFunctions4_3Core::glGetProgramResourceIndex

  ( GLuint         program,
    GLenum         programInterface,
    const GLchar*  name )

{
  return table_->glGetProgramResourceIndex ( program, programInterface,
                                             name );
}


inline void GLFunctions4_3Core::glGetProgramResourceName

  ( GLuint    program,
    GLenum    programInterface,
    GLuint    index,
    GLsizei   bufSize,
    GLsizei*  length,
    GLchar*   name )

{
  table_->glGetProgramResourceName ( program, programInterface, index,
                                     bufSize, length, name );
}


inline void GLFunctions4_3Core::glGetProgramResource

  ( GLuint         program,
    GLenum         programInterface,
    GLuint         index,
    GLsizei        propCount,
    const GLenum*  props,
    GLsizei        bufSize,
    GLsizei*       length,
    GLint*         params )

{
  table_->glGetProgramResourceiv ( program, programInterface, index,
                                   propCount, props, bufSize, length,
                                   params );
}


inline GLint GLFunctions4_3Core::glGetProgramResourceLocation

  ( GLuint         program,
    GLenum         programInterface,
    const GLchar*  name )

{
  return table_->

    glGetProgramResourceLocation ( program, programInterface, name );
}


inline GLint GLFunctions4_3Core::glGetProgramResourceLocationIndex

  ( GLuint         program,
    GLenum         programInterface,
    const GLchar*  name )

{
  return table_->glGetProgramResourceLocationIndex

    ( program, programInterface, name );
}


inline void GLFunctions4_3Core::glShaderStorageBlockBinding

  ( GLuint  program,
    GLuint  storageBlockIndex,
    GLuint  storageBlockBinding )

{
  table_->glShaderStorageBlockBinding ( program, storageBlockIndex,
                                        storageBlockBinding );
}


inline void GLFunctions4_3Core::glTexBufferRange

  ( GLenum      target,
    GLenum      internalformat,
    GLuint      buffer,
    GLintptr    offset,
    GLsizeiptr  size )

{
  table_->glTexBufferRange ( target, internalformat, buffer, offset,
                             size );
}


inline void GLFunctions4_3Core::glTexStorage2DMultisample

  ( GLenum     target,
    GLsizei    samples,
    GLenum     internalformat,
    GLsizei    width,
    GLsizei    height,
    GLboolean  fixedsamplelocations )

{
  table_->glTexStorage2DMultisample ( target, samples, internalformat,
                                      width, height,
                                      fixedsamplelocations );
}


inline void GLFunctions4_3Core::glTexStorage3DMultisample

  ( GLenum     target,
    GLsizei    samples,
    GLenum     internalformat,
    GLsizei    width,
    GLsizei    height,
    GLsizei    depth,
    GLboolean  fixedsamplelocations )

{
  table_->glTexStorage3DMultisample ( target, samples, internalformat,
                                      width, height, depth,
                                      fixedsamplelocations );
}


inline void GLFunctions4_3Core::glTextureView

  ( GLuint  texture,
    GLenum  target,
    GLuint  origtexture,
    GLenum  internalformat,
    GLuint  minlevel,
    GLuint  numlevels,
    GLuint  minlayer,
    GLuint  numlayers )

{
  table_->glTextureView ( texture, target, origtexture, internalformat,
                          minlevel, numlevels, minlayer, numlayers );
}


inline void GLFunctions4_3Core::glBindVertexBuffer

  ( GLuint    bindingindex,
    GLuint    buffer,
    GLintptr  offset,
    GLsizei   stride )

{
  table_->glBindVertexBuffer ( bindingindex, buffer, offset, stride );
}


inline void GLFunctions4_3Core::glVertexAttribFormat

  ( GLuint     attribindex,
    GLint      size,
    GLenum     type,
    GLboolean  normalized,
    GLuint     relativeoffset )

{
  table_->glVertexAttribFormat ( attribindex, size, type, normalized,
                                 relativeoffset );
}


inline void GLFunctions4_3Core::glVertexAttribIFormat

  ( GLuint  attribindex,
    GLint   size,
    GLenum  type,
    GLuint  relativeoffset )

{
  table_->glVertexAttribIFormat ( attribindex, size, type,
                                  relativeoffset );
}


inline void GLFunctions4_3Core::glVertexAttribLFormat

  ( GLuint  attribindex,
    GLint   size,
    GLenum  type,
    GLuint  relativeoffset )

{
  table_->glVertexAttribLFormat ( attribindex, size, type,
                                  relativeoffset );
}


inline void GLFunctions4_3Core::glVertexAttribBinding

  ( GLuint  attribindex,
    GLuint  bindingindex )

{
  table_->glVertexAttribBinding ( attribindex, bindingindex );
}


inline void GLFunctions4_3Core::glVertexBindingDivisor

  ( GLuint  bindingindex,
    GLuint  divisor )

{
  table_->glVertexBindingDivisor ( bindingindex, divisor );
}


inline void GLFunctions4_3Core::glDebugMessageControl

  ( GLenum         source,
    GLenum         type,
    GLenum         severity,
    GLsizei        count,
    const GLuint*  ids,
    GLboolean      enabled )

{
  table_->glDebugMessageControl ( source, type, severity, count, ids,
                                  enabled );
}


inline void GLFunctions4_3Core::glDebugMessageInsert

  ( GLenum         source,
    GLenum         type,
    GLuint         id,
    GLenum         severity,
    GLsizei        length,
    const GLchar*  buf )

{
  table_->glDebugMessageInsert ( source, type, id, severity, length,
                                 buf );
}


inline void GLFunctions4_3Core::glDebugMessageCallback

  ( GLDEBUGPROC  callback,
    const void*  userParam )

{
  table_->glDebugMessageCallback ( callback, userParam );
}


inline GLuint GLFunctions4_3Core::glGetDebugMessageLog

  ( GLuint    count,
    GLsizei   bufSize,
    GLenum*   sources,
    GLenum*   types,
    GLuint*   ids,
    GLenum*   severities,
    GLsizei*  lengths,
    GLchar*   messageLog )

{
  return table_->glGetDebugMessageLog ( count, bufSize, sources, types,
                                        ids, severities, lengths,
                                        messageLog );
}


inline void GLFunctions4_3Core::glPushDebugGroup

  ( GLenum         source,
    GLuint         id,
    GLsizei        length,
    const GLchar*  message )

{
  table_->glPushDebugGroup ( source, id, length, message );
}


inline void GLFunctions4_3Core::glPopDebugGroup ()
{
  table_->glPopDebugGroup ();
}


inline void GLFunctions4_3Core::glObjectLabel

  ( GLenum         identifier,
    GLuint         name,
    GLsizei        length,
    const GLchar*  label )

{
  table_->glObjectLabel ( identifier, name, length, label );
}


inline void GLFunctions4_3Core::glGetObjectLabel

  ( GLenum    identifier,
    GLuint    name,
    GLsizei   bufSize,
    GLsizei*  length,
    GLchar*   label )

{
  table_->glGetObjectLabel ( identifier, name, bufSize, length,
                             label );
}


inline void GLFunctions4_3Core::glObjectPtrLabel

  ( const void*    ptr,
    GLsizei        length,
    const GLchar*  label )

{
  table_->glObjectPtrLabel ( ptr, length, label );
}


inline void GLFunctions4_3Core::glGetObjectPtrLabel

  ( const void*  ptr,
    GLsizei      bufSize,
    GLsizei*     length,
    GLchar*      label )

{
  table_->glGetObjectPtrLabel ( ptr, bufSize, length, label );
}


JEM_END_PACKAGE( gl )

#endif
