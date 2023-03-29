
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

#ifndef JEM_GL_GLFUNCTIONS3_2_H
#define JEM_GL_GLFUNCTIONS3_2_H

#include <jem/gl/GLFunctions3_1.h>
#include <jem/gl/ftable/GLFuncTable3_2.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions3_2Core
//-----------------------------------------------------------------------


class GLFunctions3_2Core : public GLFunctions3_1Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_3_2_CORE;

  explicit                GLFunctions3_2Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions3_2Core  ();


 public:

  using GLFunctions1Core  ::glGetParameter;
  using GLFunctions3Core  ::glGetParameterI;
  using GLFunctions1_5Core::glGetBufferParameter;

  inline void             glDrawElementsBaseVertex

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLint                 basevertex );

  inline void             glDrawRangeElementsBaseVertex

    ( GLenum                mode,
      GLuint                start,
      GLuint                end,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLint                 basevertex );

  inline void             glDrawElementsInstancedBaseVertex

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLsizei               instancecount,
      GLint                 basevertex );

  inline void             glMultiDrawElementsBaseVertex

    ( GLenum                mode,
      const GLsizei*        count,
      GLenum                type,
      const void* const*    indices,
      GLsizei               drawcount,
      const GLint*          basevertex );

  inline void             glProvokingVertex

    ( GLenum                mode );

  inline GLsync           glFenceSync

    ( GLenum                condition,
      GLbitfield            flags );

  inline GLboolean        glIsSync

    ( GLsync                sync );

  inline void             glDeleteSync

    ( GLsync                sync );

  inline GLenum           glClientWaitSync

    ( GLsync                sync,
      GLbitfield            flags,
      GLuint64              timeout );

  inline void             glWaitSync

    ( GLsync                sync,
      GLbitfield            flags,
      GLuint64              timeout );

  inline void             glGetParameter

    ( GLenum                pname,
      GLint64*              data );

  inline void             glGetSync

    ( GLsync                sync,
      GLenum                pname,
      GLsizei               bufSize,
      GLsizei*              length,
      GLint*                values );

  inline void             glGetParameterI

    ( GLenum                target,
      GLuint                index,
      GLint64*              data );

  inline void             glGetBufferParameter

    ( GLenum                target,
      GLenum                pname,
      GLint64*              params );

  inline void             glFramebufferTexture

    ( GLenum                target,
      GLenum                attachment,
      GLuint                texture,
      GLint                 level );

  inline void             glTexImage2DMultisample

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLboolean             fixedsamplelocations );

  inline void             glTexImage3DMultisample

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLboolean             fixedsamplelocations );

  inline void             glGetMultisample

    ( GLenum                pname,
      GLuint                index,
      GLfloat*              val );

  inline void             glSampleMask

    ( GLuint                maskNumber,
      GLbitfield            mask );


 private:

  Ref
    <GLFuncTable3_2Core>  table_;

};


typedef
  GLFunctions3_2Core      GLFuncs3_2Core;


//-----------------------------------------------------------------------
//   class GLFunctions3_2
//-----------------------------------------------------------------------


class GLFunctions3_2 : public GLFunctions3_2Core,
                       public GLFunctions3Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_3_2;

  explicit                GLFunctions3_2

    ( GLFuncTables&         tables );

};


typedef GLFunctions3_2    GLFuncs3_2;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions3_2Core
//=======================================================================


inline void GLFunctions3_2Core::glDrawElementsBaseVertex

  ( GLenum       mode,
    GLsizei      count,
    GLenum       type,
    const void*  indices,
    GLint        basevertex )

{
  table_->glDrawElementsBaseVertex ( mode, count, type, indices,
                                     basevertex );
}


inline void GLFunctions3_2Core::glDrawRangeElementsBaseVertex

  ( GLenum       mode,
    GLuint       start,
    GLuint       end,
    GLsizei      count,
    GLenum       type,
    const void*  indices,
    GLint        basevertex )

{
  table_->glDrawRangeElementsBaseVertex ( mode, start, end, count,
                                          type, indices, basevertex );
}


inline void GLFunctions3_2Core::glDrawElementsInstancedBaseVertex

  ( GLenum       mode,
    GLsizei      count,
    GLenum       type,
    const void*  indices,
    GLsizei      instancecount,
    GLint        basevertex )

{
  table_->glDrawElementsInstancedBaseVertex ( mode, count, type,
                                              indices, instancecount,
                                              basevertex );
}


inline void GLFunctions3_2Core::glMultiDrawElementsBaseVertex

  ( GLenum              mode,
    const GLsizei*      count,
    GLenum              type,
    const void* const*  indices,
    GLsizei             drawcount,
    const GLint*        basevertex )

{
  table_->glMultiDrawElementsBaseVertex ( mode, count, type, indices,
                                          drawcount, basevertex );
}


inline void GLFunctions3_2Core::glProvokingVertex

  ( GLenum  mode )

{
  table_->glProvokingVertex ( mode );
}


inline GLsync GLFunctions3_2Core::glFenceSync

  ( GLenum      condition,
    GLbitfield  flags )

{
  return table_->glFenceSync ( condition, flags );
}


inline GLboolean GLFunctions3_2Core::glIsSync

  ( GLsync  sync )

{
  return table_->glIsSync ( sync );
}


inline void GLFunctions3_2Core::glDeleteSync

  ( GLsync  sync )

{
  table_->glDeleteSync ( sync );
}


inline GLenum GLFunctions3_2Core::glClientWaitSync

  ( GLsync      sync,
    GLbitfield  flags,
    GLuint64    timeout )

{
  return table_->glClientWaitSync ( sync, flags, timeout );
}


inline void GLFunctions3_2Core::glWaitSync

  ( GLsync      sync,
    GLbitfield  flags,
    GLuint64    timeout )

{
  table_->glWaitSync ( sync, flags, timeout );
}


inline void GLFunctions3_2Core::glGetParameter

  ( GLenum    pname,
    GLint64*  data )

{
  table_->glGetInteger64v ( pname, data );
}


inline void GLFunctions3_2Core::glGetSync

  ( GLsync    sync,
    GLenum    pname,
    GLsizei   bufSize,
    GLsizei*  length,
    GLint*    values )

{
  table_->glGetSynciv ( sync, pname, bufSize, length, values );
}


inline void GLFunctions3_2Core::glGetParameterI

  ( GLenum    target,
    GLuint    index,
    GLint64*  data )

{
  table_->glGetInteger64i_v ( target, index, data );
}


inline void GLFunctions3_2Core::glGetBufferParameter

  ( GLenum    target,
    GLenum    pname,
    GLint64*  params )

{
  table_->glGetBufferParameteri64v ( target, pname, params );
}


inline void GLFunctions3_2Core::glFramebufferTexture

  ( GLenum  target,
    GLenum  attachment,
    GLuint  texture,
    GLint   level )

{
  table_->glFramebufferTexture ( target, attachment, texture, level );
}


inline void GLFunctions3_2Core::glTexImage2DMultisample

  ( GLenum     target,
    GLsizei    samples,
    GLenum     internalformat,
    GLsizei    width,
    GLsizei    height,
    GLboolean  fixedsamplelocations )

{
  table_->glTexImage2DMultisample ( target, samples, internalformat,
                                    width, height,
                                    fixedsamplelocations );
}


inline void GLFunctions3_2Core::glTexImage3DMultisample

  ( GLenum     target,
    GLsizei    samples,
    GLenum     internalformat,
    GLsizei    width,
    GLsizei    height,
    GLsizei    depth,
    GLboolean  fixedsamplelocations )

{
  table_->glTexImage3DMultisample ( target, samples, internalformat,
                                    width, height, depth,
                                    fixedsamplelocations );
}


inline void GLFunctions3_2Core::glGetMultisample

  ( GLenum    pname,
    GLuint    index,
    GLfloat*  val )

{
  table_->glGetMultisamplefv ( pname, index, val );
}


inline void GLFunctions3_2Core::glSampleMask

  ( GLuint      maskNumber,
    GLbitfield  mask )

{
  table_->glSampleMaski ( maskNumber, mask );
}


JEM_END_PACKAGE( gl )

#endif
