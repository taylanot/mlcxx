
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

#ifndef JEM_GL_GLFUNCTIONS3_1_H
#define JEM_GL_GLFUNCTIONS3_1_H

#include <jem/gl/GLFunctions3.h>
#include <jem/gl/ftable/GLFuncTable3_1.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions3_1Core
//-----------------------------------------------------------------------


class GLFunctions3_1Core : public GLFunctions3Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_3_1_CORE;

  explicit                GLFunctions3_1Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions3_1Core  ();


 public:

  inline void             glDrawArraysInstanced

    ( GLenum                mode,
      GLint                 first,
      GLsizei               count,
      GLsizei               instancecount );

  inline void             glDrawElementsInstanced

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLsizei               instancecount );

  inline void             glTexBuffer

    ( GLenum                target,
      GLenum                internalformat,
      GLuint                buffer );

  inline void             glPrimitiveRestartIndex

    ( GLuint                index );

  inline void             glCopyBufferSubData

    ( GLenum                readTarget,
      GLenum                writeTarget,
      GLintptr              readOffset,
      GLintptr              writeOffset,
      GLsizeiptr            size );

  inline void             glGetUniformIndices

    ( GLuint                program,
      GLsizei               uniformCount,
      const GLchar* const*  uniformNames,
      GLuint*               uniformIndices );

  inline void             glGetActiveUniforms

    ( GLuint                program,
      GLsizei               uniformCount,
      const GLuint*         uniformIndices,
      GLenum                pname,
      GLint*                params );

  inline void             glGetActiveUniformName

    ( GLuint                program,
      GLuint                uniformIndex,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               uniformName );

  inline GLuint           glGetUniformBlockIndex

    ( GLuint                program,
      const GLchar*         uniformBlockName );

  inline void             glGetActiveUniformBlock

    ( GLuint                program,
      GLuint                uniformBlockIndex,
      GLenum                pname,
      GLint*                params );

  inline void             glGetActiveUniformBlockName

    ( GLuint                program,
      GLuint                uniformBlockIndex,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               uniformBlockName );

  inline void             glUniformBlockBinding

    ( GLuint                program,
      GLuint                uniformBlockIndex,
      GLuint                uniformBlockBinding );


 private:

  Ref
    <GLFuncTable3_1Core>  table_;

};


typedef
  GLFunctions3_1Core      GLFuncs3_1Core;


//-----------------------------------------------------------------------
//   class GLFunctions3_1
//-----------------------------------------------------------------------


class GLFunctions3_1 : public GLFunctions3_1Core,
                       public GLFunctions3Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_3_1;

  explicit                GLFunctions3_1

    ( GLFuncTables&         tables );

};


typedef GLFunctions3_1    GLFuncs3_1;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions3_1Core
//=======================================================================


inline void GLFunctions3_1Core::glDrawArraysInstanced

  ( GLenum   mode,
    GLint    first,
    GLsizei  count,
    GLsizei  instancecount )

{
  table_->glDrawArraysInstanced ( mode, first, count,
                                  instancecount );
}


inline void GLFunctions3_1Core::glDrawElementsInstanced

  ( GLenum       mode,
    GLsizei      count,
    GLenum       type,
    const void*  indices,
    GLsizei      instancecount )

{
  table_->glDrawElementsInstanced ( mode, count, type, indices,
                                    instancecount );
}


inline void GLFunctions3_1Core::glTexBuffer

  ( GLenum  target,
    GLenum  internalformat,
    GLuint  buffer )

{
  table_->glTexBuffer ( target, internalformat, buffer );
}


inline void GLFunctions3_1Core::glPrimitiveRestartIndex

  ( GLuint index )

{
  table_->glPrimitiveRestartIndex ( index );
}


inline void GLFunctions3_1Core::glCopyBufferSubData

  ( GLenum      readTarget,
    GLenum      writeTarget,
    GLintptr    readOffset,
    GLintptr    writeOffset,
    GLsizeiptr  size )

{
  table_->glCopyBufferSubData ( readTarget, writeTarget, readOffset,
                                writeOffset, size );
}


inline void GLFunctions3_1Core::glGetUniformIndices

  ( GLuint                program,
    GLsizei               uniformCount,
    const GLchar* const*  uniformNames,
    GLuint*               uniformIndices )

{
  table_->glGetUniformIndices ( program, uniformCount,
                                uniformNames, uniformIndices );
}


inline void GLFunctions3_1Core::glGetActiveUniforms

  ( GLuint         program,
    GLsizei        uniformCount,
    const GLuint*  uniformIndices,
    GLenum         pname,
    GLint*         params )

{
  table_->glGetActiveUniformsiv ( program, uniformCount,
                                  uniformIndices, pname,
                                  params );
}


inline void GLFunctions3_1Core::glGetActiveUniformName

  ( GLuint    program,
    GLuint    uniformIndex,
    GLsizei   bufSize,
    GLsizei*  length,
    GLchar*   uniformName )

{
  table_->glGetActiveUniformName ( program, uniformIndex, bufSize,
                                   length, uniformName );
}


inline GLuint GLFunctions3_1Core::glGetUniformBlockIndex

  ( GLuint         program,
    const GLchar*  uniformBlockName )

{
  return table_->glGetUniformBlockIndex ( program, uniformBlockName );
}


inline void GLFunctions3_1Core::glGetActiveUniformBlock

  ( GLuint  program,
    GLuint  uniformBlockIndex,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetActiveUniformBlockiv ( program, uniformBlockIndex,
                                      pname, params );
}


inline void GLFunctions3_1Core::glGetActiveUniformBlockName

  ( GLuint    program,
    GLuint    uniformBlockIndex,
    GLsizei   bufSize,
    GLsizei*  length,
    GLchar*   uniformBlockName )

{
  table_->glGetActiveUniformBlockName ( program, uniformBlockIndex,
                                        bufSize, length,
                                        uniformBlockName );
}


inline void GLFunctions3_1Core::glUniformBlockBinding

  ( GLuint  program,
    GLuint  uniformBlockIndex,
    GLuint  uniformBlockBinding )

{
  table_->glUniformBlockBinding ( program, uniformBlockIndex,
                                  uniformBlockBinding );
}


JEM_END_PACKAGE( gl )

#endif
