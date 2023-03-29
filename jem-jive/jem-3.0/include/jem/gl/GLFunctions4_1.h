
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

#ifndef JEM_GL_GLFUNCTIONS4_1_H
#define JEM_GL_GLFUNCTIONS4_1_H

#include <jem/gl/GLFunctions4.h>
#include <jem/gl/ftable/GLFuncTable4_1.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions4_1Core
//-----------------------------------------------------------------------


class GLFunctions4_1Core : public GLFunctions4Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_1_CORE;

  explicit                GLFunctions4_1Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions4_1Core  ();


 public:

  using GLFunctions3_2Core::glGetParameterI;
  using GLFunctions1Core  ::glDepthRange;
  using GLFunctions1Core  ::glClearDepth;

  inline void             glReleaseShaderCompiler ();

  inline void             glShaderBinary

    ( GLsizei               count,
      const GLuint*         shaders,
      GLenum                binaryformat,
      const void*           binary,
      GLsizei               length );

  inline void             glGetShaderPrecisionFormat

    ( GLenum                shadertype,
      GLenum                precisiontype,
      GLint*                range,
      GLint*                precision );

  inline void             glDepthRange

    ( GLfloat               n,
      GLfloat               f );

  inline void             glClearDepth

    ( GLfloat               d );

  inline void             glGetProgramBinary

    ( GLuint                program,
      GLsizei               bufSize,
      GLsizei*              length,
      GLenum*               binaryFormat,
      void*                 binary );

  inline void             glProgramBinary

    ( GLuint                program,
      GLenum                binaryFormat,
      const void*           binary,
      GLsizei               length );

  inline void             glProgramParameter

    ( GLuint                program,
      GLenum                pname,
      GLint                 value );

  inline void             glUseProgramStages

    ( GLuint                pipeline,
      GLbitfield            stages,
      GLuint                program );

  inline void             glActiveShaderProgram

    ( GLuint                pipeline,
      GLuint                program );

  inline GLuint           glCreateShaderProgram

    ( GLenum                type,
      GLsizei               count,
      const GLchar* const*  strings );

  inline void             glBindProgramPipeline

    ( GLuint                pipeline );

  inline void             glDeleteProgramPipelines

    ( GLsizei               n,
      const GLuint*         pipelines );

  inline void             glGenProgramPipelines

    ( GLsizei               n,
      GLuint*               pipelines );

  inline GLboolean        glIsProgramPipeline

    ( GLuint                pipeline );

  inline void             glGetProgramPipeline

    ( GLuint                pipeline,
      GLenum                pname,
      GLint*                params );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLint                 v0 );

  inline void             glProgramUniform1

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLint*          value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLfloat               v0 );

  inline void             glProgramUniform1

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLdouble              v0 );

  inline void             glProgramUniform1

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLuint                v0 );

  inline void             glProgramUniform1

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLint                 v0,
      GLint                 v1 );

  inline void             glProgramUniform2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLint*          value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLfloat               v0,
      GLfloat               v1 );

  inline void             glProgramUniform2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLdouble              v0,
      GLdouble              v1 );

  inline void             glProgramUniform2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLuint                v0,
      GLuint                v1 );

  inline void             glProgramUniform2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLint                 v0,
      GLint                 v1,
      GLint                 v2 );

  inline void             glProgramUniform3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLint*          value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLfloat               v0,
      GLfloat               v1,
      GLfloat               v2 );

  inline void             glProgramUniform3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLdouble              v0,
      GLdouble              v1,
      GLdouble              v2 );

  inline void             glProgramUniform3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLuint                v0,
      GLuint                v1,
      GLuint                v2 );

  inline void             glProgramUniform3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLint                 v0,
      GLint                 v1,
      GLint                 v2,
      GLint                 v3 );

  inline void             glProgramUniform4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLint*          value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLfloat               v0,
      GLfloat               v1,
      GLfloat               v2,
      GLfloat               v3 );

  inline void             glProgramUniform4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLdouble              v0,
      GLdouble              v1,
      GLdouble              v2,
      GLdouble              v3 );

  inline void             glProgramUniform4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  inline void             glProgramUniform

    ( GLuint                program,
      GLint                 location,
      GLuint                v0,
      GLuint                v1,
      GLuint                v2,
      GLuint                v3 );

  inline void             glProgramUniform4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  inline void             glProgramUniformMatrix2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glProgramUniformMatrix3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glProgramUniformMatrix4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glProgramUniformMatrix2x3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix3x2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix2x4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix4x2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix3x4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix4x3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glProgramUniformMatrix2x3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glProgramUniformMatrix3x2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glProgramUniformMatrix2x4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glProgramUniformMatrix4x2

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glProgramUniformMatrix3x4

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glProgramUniformMatrix4x3

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glValidateProgramPipeline

    ( GLuint                pipeline );

  inline void             glGetProgramPipelineInfoLog

    ( GLuint                pipeline,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               infoLog );

  inline void             glVertexAttribL

    ( GLuint                index,
      GLdouble              x );

  inline void             glVertexAttribL

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y );

  inline void             glVertexAttribL

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  inline void             glVertexAttribL

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  inline void             glVertexAttribL1

    ( GLuint                index,
      const GLdouble*       v );

  inline void             glVertexAttribL2

    ( GLuint                index,
      const GLdouble*       v );

  inline void             glVertexAttribL3

    ( GLuint                index,
      const GLdouble*       v );

  inline void             glVertexAttribL4

    ( GLuint                index,
      const GLdouble*       v );

  inline void             glVertexAttribLPointer

    ( GLuint                index,
      GLint                 size,
      GLenum                type,
      GLsizei               stride,
      const void*           pointer );

  inline void             glGetVertexAttribL

    ( GLuint                index,
      GLenum                pname,
      GLdouble*             params );

  inline void             glViewportArray

    ( GLuint                first,
      GLsizei               count,
      const GLfloat*        v );

  inline void             glViewportIndexed

    ( GLuint                index,
      GLfloat               x,
      GLfloat               y,
      GLfloat               w,
      GLfloat               h );

  inline void             glViewportIndexed

    ( GLuint                index,
      const GLfloat*        v );

  inline void             glScissorArray

    ( GLuint                first,
      GLsizei               count,
      const GLint*          v );

  inline void             glScissorIndexed

    ( GLuint                index,
      GLint                 left,
      GLint                 bottom,
      GLsizei               width,
      GLsizei               height );

  inline void             glScissorIndexed

    ( GLuint                index,
      const GLint*          v );

  inline void             glDepthRangeArray

    ( GLuint                first,
      GLsizei               count,
      const GLdouble*       v );

  inline void             glDepthRangeIndexed

    ( GLuint                index,
      GLdouble              n,
      GLdouble              f );

  inline void             glGetParameterI

    ( GLenum                target,
      GLuint                index,
      GLfloat*              data );

  inline void             glGetParameterI

    ( GLenum                target,
      GLuint                index,
      GLdouble*             data );


 private:

  Ref
    <GLFuncTable4_1Core>  table_;

};


typedef
  GLFunctions4_1Core      GLFuncs4_1Core;


//-----------------------------------------------------------------------
//   class GLFunctions4_1
//-----------------------------------------------------------------------


class GLFunctions4_1 : public GLFunctions4_1Core,
                       public GLFunctions4Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_1;

  explicit                GLFunctions4_1

    ( GLFuncTables&         tables );

};


typedef GLFunctions4_1    GLFuncs4_1;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions4_1Core
//=======================================================================


inline void GLFunctions4_1Core::glReleaseShaderCompiler ()
{
  table_->glReleaseShaderCompiler ();
}


inline void GLFunctions4_1Core::glShaderBinary

  ( GLsizei        count,
    const GLuint*  shaders,
    GLenum         binaryformat,
    const void*    binary,
    GLsizei        length )

{
  table_->glShaderBinary ( count, shaders, binaryformat, binary,
                           length );
}


inline void GLFunctions4_1Core::glGetShaderPrecisionFormat

  ( GLenum  shadertype,
    GLenum  precisiontype,
    GLint*  range,
    GLint*  precision )

{
  table_->glGetShaderPrecisionFormat ( shadertype, precisiontype,
                                       range, precision );
}


inline void GLFunctions4_1Core::glDepthRange

  ( GLfloat  n,
    GLfloat  f )

{
  table_->glDepthRangef ( n, f );
}


inline void GLFunctions4_1Core::glClearDepth

  ( GLfloat  d )

{
  table_->glClearDepthf ( d );
}


inline void GLFunctions4_1Core::glGetProgramBinary

  ( GLuint    program,
    GLsizei   bufSize,
    GLsizei*  length,
    GLenum*   binaryFormat,
    void*     binary )

{
  table_->glGetProgramBinary ( program, bufSize, length, binaryFormat,
                               binary );
}


inline void GLFunctions4_1Core::glProgramBinary

  ( GLuint       program,
    GLenum       binaryFormat,
    const void*  binary,
    GLsizei      length )

{
  table_->glProgramBinary ( program, binaryFormat, binary, length );
}


inline void GLFunctions4_1Core::glProgramParameter

  ( GLuint  program,
    GLenum  pname,
    GLint   value )

{
  table_->glProgramParameteri ( program, pname, value );
}


inline void GLFunctions4_1Core::glUseProgramStages

  ( GLuint      pipeline,
    GLbitfield  stages,
    GLuint      program )

{
  table_->glUseProgramStages ( pipeline, stages, program );
}


inline void GLFunctions4_1Core::glActiveShaderProgram

  ( GLuint  pipeline,
    GLuint  program )

{
  table_->glActiveShaderProgram ( pipeline, program );
}


inline GLuint GLFunctions4_1Core::glCreateShaderProgram

  ( GLenum                type,
    GLsizei               count,
    const GLchar* const*  strings )

{
  return table_->glCreateShaderProgramv ( type, count, strings );
}


inline void GLFunctions4_1Core::glBindProgramPipeline

  ( GLuint  pipeline )

{
  table_->glBindProgramPipeline ( pipeline );
}


inline void GLFunctions4_1Core::glDeleteProgramPipelines

  ( GLsizei        n,
    const GLuint*  pipelines )

{
  table_->glDeleteProgramPipelines ( n, pipelines );
}


inline void GLFunctions4_1Core::glGenProgramPipelines

  ( GLsizei  n,
    GLuint*  pipelines )

{
  table_->glGenProgramPipelines ( n, pipelines );
}


inline GLboolean GLFunctions4_1Core::glIsProgramPipeline

  ( GLuint  pipeline )

{
  return table_->glIsProgramPipeline ( pipeline );
}


inline void GLFunctions4_1Core::glGetProgramPipeline

  ( GLuint  pipeline,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetProgramPipelineiv ( pipeline, pname, params );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint  program,
    GLint   location,
    GLint   v0 )

{
  table_->glProgramUniform1i ( program, location, v0 );
}


inline void GLFunctions4_1Core::glProgramUniform1

  ( GLuint        program,
    GLint         location,
    GLsizei       count,
    const GLint*  value )

{
  table_->glProgramUniform1iv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint   program,
    GLint    location,
    GLfloat  v0 )

{
  table_->glProgramUniform1f ( program, location, v0 );
}


inline void GLFunctions4_1Core::glProgramUniform1

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    const GLfloat*  value )

{
  table_->glProgramUniform1fv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint    program,
    GLint     location,
    GLdouble  v0 )

{
  table_->glProgramUniform1d ( program, location, v0 );
}


inline void GLFunctions4_1Core::glProgramUniform1

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    const GLdouble*  value )

{
  table_->glProgramUniform1dv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint  program,
    GLint   location,
    GLuint  v0 )

{
  table_->glProgramUniform1ui ( program, location, v0 );
}


inline void GLFunctions4_1Core::glProgramUniform1

  ( GLuint         program,
    GLint          location,
    GLsizei        count,
    const GLuint*  value )

{
  table_->glProgramUniform1uiv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint  program,
    GLint   location,
    GLint   v0,
    GLint   v1 )

{
  table_->glProgramUniform2i ( program, location, v0, v1 );
}


inline void GLFunctions4_1Core::glProgramUniform2

  ( GLuint        program,
    GLint         location,
    GLsizei       count,
    const GLint*  value )

{
  table_->glProgramUniform2iv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint   program,
    GLint    location,
    GLfloat  v0,
    GLfloat  v1 )

{
  table_->glProgramUniform2f ( program, location, v0, v1 );
}


inline void GLFunctions4_1Core::glProgramUniform2

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    const GLfloat*  value )

{
  table_->glProgramUniform2fv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint    program,
    GLint     location,
    GLdouble  v0,
    GLdouble  v1 )

{
  table_->glProgramUniform2d ( program, location, v0, v1 );
}


inline void GLFunctions4_1Core::glProgramUniform2

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    const GLdouble*  value )

{
  table_->glProgramUniform2dv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint  program,
    GLint   location,
    GLuint  v0,
    GLuint  v1 )

{
  table_->glProgramUniform2ui ( program, location, v0, v1 );
}


inline void GLFunctions4_1Core::glProgramUniform2

  ( GLuint         program,
    GLint          location,
    GLsizei        count,
    const GLuint*  value )

{
  table_->glProgramUniform2uiv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint  program,
    GLint   location,
    GLint   v0,
    GLint   v1,
    GLint   v2 )

{
  table_->glProgramUniform3i ( program, location, v0, v1, v2 );
}


inline void GLFunctions4_1Core::glProgramUniform3

  ( GLuint        program,
    GLint         location,
    GLsizei       count,
    const GLint*  value )

{
  table_->glProgramUniform3iv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint   program,
    GLint    location,
    GLfloat  v0,
    GLfloat  v1,
    GLfloat  v2 )

{
  table_->glProgramUniform3f ( program, location, v0, v1, v2 );
}


inline void GLFunctions4_1Core::glProgramUniform3

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    const GLfloat*  value )

{
  table_->glProgramUniform3fv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint    program,
    GLint     location,
    GLdouble  v0,
    GLdouble  v1,
    GLdouble  v2 )

{
  table_->glProgramUniform3d ( program, location, v0, v1, v2 );
}


inline void GLFunctions4_1Core::glProgramUniform3

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    const GLdouble*  value )

{
  table_->glProgramUniform3dv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint  program,
    GLint   location,
    GLuint  v0,
    GLuint  v1,
    GLuint  v2 )

{
  table_->glProgramUniform3ui ( program, location, v0, v1, v2 );
}


inline void GLFunctions4_1Core::glProgramUniform3

  ( GLuint         program,
    GLint          location,
    GLsizei        count,
    const GLuint*  value )

{
  table_->glProgramUniform3uiv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint  program,
    GLint   location,
    GLint   v0,
    GLint   v1,
    GLint   v2,
    GLint   v3 )

{
  table_->glProgramUniform4i ( program, location, v0, v1, v2, v3 );
}


inline void GLFunctions4_1Core::glProgramUniform4

  ( GLuint        program,
    GLint         location,
    GLsizei       count,
    const GLint*  value )

{
  table_->glProgramUniform4iv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint   program,
    GLint    location,
    GLfloat  v0,
    GLfloat  v1,
    GLfloat  v2,
    GLfloat  v3 )

{
  table_->glProgramUniform4f ( program, location, v0, v1, v2, v3 );
}


inline void GLFunctions4_1Core::glProgramUniform4

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    const GLfloat*  value )

{
  table_->glProgramUniform4fv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint    program,
    GLint     location,
    GLdouble  v0,
    GLdouble  v1,
    GLdouble  v2,
    GLdouble  v3 )

{
  table_->glProgramUniform4d ( program, location, v0, v1, v2, v3 );
}


inline void GLFunctions4_1Core::glProgramUniform4

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    const GLdouble*  value )

{
  table_->glProgramUniform4dv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniform

  ( GLuint  program,
    GLint   location,
    GLuint  v0,
    GLuint  v1,
    GLuint  v2,
    GLuint  v3 )

{
  table_->glProgramUniform4ui ( program, location, v0, v1, v2, v3 );
}


inline void GLFunctions4_1Core::glProgramUniform4

  ( GLuint         program,
    GLint          location,
    GLsizei        count,
    const GLuint*  value )

{
  table_->glProgramUniform4uiv ( program, location, count, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix2

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix2fv ( program, location, count,
                                      transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix3

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix3fv ( program, location, count,
                                      transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix4

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix4fv ( program, location, count,
                                      transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix2

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix2dv ( program, location, count,
                                      transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix3

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix3dv ( program, location, count,
                                      transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix4

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix4dv ( program, location, count,
                                      transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix2x3

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix2x3fv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix3x2

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix3x2fv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix2x4

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix2x4fv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix4x2

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix4x2fv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix3x4

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix3x4fv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix4x3

  ( GLuint          program,
    GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glProgramUniformMatrix4x3fv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix2x3

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix2x3dv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix3x2

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix3x2dv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix2x4

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix2x4dv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix4x2

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix4x2dv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix3x4

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix3x4dv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glProgramUniformMatrix4x3

  ( GLuint           program,
    GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glProgramUniformMatrix4x3dv ( program, location, count,
                                        transpose, value );
}


inline void GLFunctions4_1Core::glValidateProgramPipeline

  ( GLuint  pipeline )

{
  table_->glValidateProgramPipeline ( pipeline );
}


inline void GLFunctions4_1Core::glGetProgramPipelineInfoLog

  ( GLuint    pipeline,
    GLsizei   bufSize,
    GLsizei*  length,
    GLchar*   infoLog )

{
  table_->glGetProgramPipelineInfoLog ( pipeline, bufSize, length,
                                        infoLog );
}


inline void GLFunctions4_1Core::glVertexAttribL

  ( GLuint    index,
    GLdouble  x )

{
  table_->glVertexAttribL1d ( index, x );
}


inline void GLFunctions4_1Core::glVertexAttribL

  ( GLuint    index,
    GLdouble  x,
    GLdouble  y )

{
  table_->glVertexAttribL2d ( index, x, y );
}


inline void GLFunctions4_1Core::glVertexAttribL

  ( GLuint    index,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  table_->glVertexAttribL3d ( index, x, y, z );
}


inline void GLFunctions4_1Core::glVertexAttribL

  ( GLuint    index,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z,
    GLdouble  w )

{
  table_->glVertexAttribL4d ( index, x, y, z, w );
}


inline void GLFunctions4_1Core::glVertexAttribL1

  ( GLuint           index,
    const GLdouble*  v )

{
  table_->glVertexAttribL1dv ( index, v );
}


inline void GLFunctions4_1Core::glVertexAttribL2

  ( GLuint           index,
    const GLdouble*  v )

{
  table_->glVertexAttribL2dv ( index, v );
}


inline void GLFunctions4_1Core::glVertexAttribL3

  ( GLuint           index,
    const GLdouble*  v )

{
  table_->glVertexAttribL3dv ( index, v );
}


inline void GLFunctions4_1Core::glVertexAttribL4

  ( GLuint           index,
    const GLdouble*  v )

{
  table_->glVertexAttribL4dv ( index, v );
}


inline void GLFunctions4_1Core::glVertexAttribLPointer

  ( GLuint       index,
    GLint        size,
    GLenum       type,
    GLsizei      stride,
    const void*  pointer )

{
  table_->glVertexAttribLPointer ( index, size, type, stride,
                                   pointer );
}


inline void GLFunctions4_1Core::glGetVertexAttribL

  ( GLuint     index,
    GLenum     pname,
    GLdouble*  params )

{
  table_->glGetVertexAttribLdv ( index, pname, params );
}


inline void GLFunctions4_1Core::glViewportArray

  ( GLuint          first,
    GLsizei         count,
    const GLfloat*  v )

{
  table_->glViewportArrayv ( first, count, v );
}


inline void GLFunctions4_1Core::glViewportIndexed

  ( GLuint   index,
    GLfloat  x,
    GLfloat  y,
    GLfloat  w,
    GLfloat  h )

{
  table_->glViewportIndexedf ( index, x, y, w, h );
}


inline void GLFunctions4_1Core::glViewportIndexed

  ( GLuint          index,
    const GLfloat*  v )

{
  table_->glViewportIndexedfv ( index, v );
}


inline void GLFunctions4_1Core::glScissorArray

  ( GLuint        first,
    GLsizei       count,
    const GLint*  v )

{
  table_->glScissorArrayv ( first, count, v );
}


inline void GLFunctions4_1Core::glScissorIndexed

  ( GLuint   index,
    GLint    left,
    GLint    bottom,
    GLsizei  width,
    GLsizei  height )

{
  table_->glScissorIndexed ( index, left, bottom, width, height );
}


inline void GLFunctions4_1Core::glScissorIndexed

  ( GLuint        index,
    const GLint*  v )

{
  table_->glScissorIndexedv ( index, v );
}


inline void GLFunctions4_1Core::glDepthRangeArray

  ( GLuint           first,
    GLsizei          count,
    const GLdouble*  v )

{
  table_->glDepthRangeArrayv ( first, count, v );
}


inline void GLFunctions4_1Core::glDepthRangeIndexed

  ( GLuint    index,
    GLdouble  n,
    GLdouble  f )

{
  table_->glDepthRangeIndexed ( index, n, f );
}


inline void GLFunctions4_1Core::glGetParameterI

  ( GLenum    target,
    GLuint    index,
    GLfloat*  data )

{
  table_->glGetFloati_v ( target, index, data );
}


inline void GLFunctions4_1Core::glGetParameterI

  ( GLenum     target,
    GLuint     index,
    GLdouble*  data )

{
  table_->glGetDoublei_v ( target, index, data );
}


JEM_END_PACKAGE( gl )

#endif
