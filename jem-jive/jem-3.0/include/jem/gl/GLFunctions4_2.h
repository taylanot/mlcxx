
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

#ifndef JEM_GL_GLFUNCTIONS4_2_H
#define JEM_GL_GLFUNCTIONS4_2_H

#include <jem/gl/GLFunctions4_1.h>
#include <jem/gl/ftable/GLFuncTable4_2.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions4_2Core
//-----------------------------------------------------------------------


class GLFunctions4_2Core : public GLFunctions4_1Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_2_CORE;

  explicit                GLFunctions4_2Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions4_2Core  ();


 public:

  inline void             glDrawArraysInstancedBaseInstance

    ( GLenum                mode,
      GLint                 first,
      GLsizei               count,
      GLsizei               instancecount,
      GLuint                baseinstance );

  inline void             glDrawElementsInstancedBaseInstance

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLsizei               instancecount,
      GLuint                baseinstance );

  inline void             glDrawElementsInstancedBaseVertexBaseInstance

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLsizei               instancecount,
      GLint                 basevertex,
      GLuint                baseinstance );

  inline void             glGetInternalformat

    ( GLenum                target,
      GLenum                internalformat,
      GLenum                pname,
      GLsizei               bufSize,
      GLint*                params );

  inline void             glGetActiveAtomicCounterBuffer

    ( GLuint                program,
      GLuint                bufferIndex,
      GLenum                pname,
      GLint*                params );

  inline void             glBindImageTexture

    ( GLuint                unit,
      GLuint                texture,
      GLint                 level,
      GLboolean             layered,
      GLint                 layer,
      GLenum                access,
      GLenum                format );

  inline void             glMemoryBarrier

    ( GLbitfield            barriers );

  inline void             glTexStorage1D

    ( GLenum                target,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width );

  inline void             glTexStorage2D

    ( GLenum                target,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height );

  inline void             glTexStorage3D

    ( GLenum                target,
      GLsizei               levels,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth );

  inline void             glDrawTransformFeedbackInstanced

    ( GLenum                mode,
      GLuint                id,
      GLsizei               instancecount );

  inline void             glDrawTransformFeedbackStreamInstanced

    ( GLenum                mode,
      GLuint                id,
      GLuint                stream,
      GLsizei               instancecount );


 private:

  Ref
    <GLFuncTable4_2Core>  table_;

};


typedef
  GLFunctions4_2Core      GLFuncs4_2Core;


//-----------------------------------------------------------------------
//   class GLFunctions4_2
//-----------------------------------------------------------------------


class GLFunctions4_2 : public GLFunctions4_2Core,
                       public GLFunctions4Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_2;

  explicit                GLFunctions4_2

    ( GLFuncTables&         tables );

};


typedef GLFunctions4_2    GLFuncs4_2;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions4_2Core
//=======================================================================


inline void GLFunctions4_2Core::glDrawArraysInstancedBaseInstance

  ( GLenum   mode,
    GLint    first,
    GLsizei  count,
    GLsizei  instancecount,
    GLuint   baseinstance )

{
  table_->glDrawArraysInstancedBaseInstance ( mode, first, count,
                                              instancecount,
                                              baseinstance );
}


inline void GLFunctions4_2Core::glDrawElementsInstancedBaseInstance

  ( GLenum       mode,
    GLsizei      count,
    GLenum       type,
    const void*  indices,
    GLsizei      instancecount,
    GLuint       baseinstance )

{
  table_->glDrawElementsInstancedBaseInstance ( mode, count, type,
                                                indices, instancecount,
                                                baseinstance );
}


inline void GLFunctions4_2Core::
  glDrawElementsInstancedBaseVertexBaseInstance

  ( GLenum       mode,
    GLsizei      count,
    GLenum       type,
    const void*  indices,
    GLsizei      instancecount,
    GLint        basevertex,
    GLuint       baseinstance )

{
  table_->glDrawElementsInstancedBaseVertexBaseInstance

    ( mode, count, type, indices, instancecount,
      basevertex, baseinstance );
}


inline void GLFunctions4_2Core::glGetInternalformat

  ( GLenum   target,
    GLenum   internalformat,
    GLenum   pname,
    GLsizei  bufSize,
    GLint*   params )

{
  table_->glGetInternalformativ ( target, internalformat, pname,
                                  bufSize, params );
}


inline void GLFunctions4_2Core::glGetActiveAtomicCounterBuffer

  ( GLuint  program,
    GLuint  bufferIndex,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetActiveAtomicCounterBufferiv ( program, bufferIndex,
                                             pname, params );
}


inline void GLFunctions4_2Core::glBindImageTexture

  ( GLuint     unit,
    GLuint     texture,
    GLint      level,
    GLboolean  layered,
    GLint      layer,
    GLenum     access,
    GLenum     format )

{
  table_->glBindImageTexture ( unit, texture, level, layered, layer,
                               access, format );
}


inline void GLFunctions4_2Core::glMemoryBarrier

  ( GLbitfield  barriers )

{
  table_->glMemoryBarrier ( barriers );
}


inline void GLFunctions4_2Core::glTexStorage1D

  ( GLenum   target,
    GLsizei  levels,
    GLenum   internalformat,
    GLsizei  width )

{
  table_->glTexStorage1D ( target, levels, internalformat, width );
}


inline void GLFunctions4_2Core::glTexStorage2D

  ( GLenum   target,
    GLsizei  levels,
    GLenum   internalformat,
    GLsizei  width,
    GLsizei  height )

{
  table_->glTexStorage2D ( target, levels, internalformat, width,
                           height );
}


inline void GLFunctions4_2Core::glTexStorage3D

  ( GLenum   target,
    GLsizei  levels,
    GLenum   internalformat,
    GLsizei  width,
    GLsizei  height,
    GLsizei  depth )

{
  table_->glTexStorage3D ( target, levels, internalformat, width,
                           height, depth );
}


inline void GLFunctions4_2Core::glDrawTransformFeedbackInstanced

  ( GLenum   mode,
    GLuint   id,
    GLsizei  instancecount )

{
  table_->glDrawTransformFeedbackInstanced ( mode, id, instancecount );
}


inline void GLFunctions4_2Core::glDrawTransformFeedbackStreamInstanced

  ( GLenum   mode,
    GLuint   id,
    GLuint   stream,
    GLsizei  instancecount )

{
  table_->glDrawTransformFeedbackStreamInstanced ( mode, id, stream,
                                                   instancecount );
}


JEM_END_PACKAGE( gl )

#endif
