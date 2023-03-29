
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

#ifndef JEM_GL_GLFUNCTABLE4_1_H
#define JEM_GL_GLFUNCTABLE4_1_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable4_1Core
//-----------------------------------------------------------------------

// OpenGL 4.1 core functions.

class GLFuncTable4_1Core : public Object
{
 public:

  explicit                GLFuncTable4_1Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glReleaseShaderCompiler) ();

  void                   (JEM_APIENTRYP glShaderBinary)

    ( GLsizei               count,
      const GLuint*         shaders,
      GLenum                binaryformat,
      const void*           binary,
      GLsizei               length );

  void                   (JEM_APIENTRYP glGetShaderPrecisionFormat)

    ( GLenum                shadertype,
      GLenum                precisiontype,
      GLint*                range,
      GLint*                precision );

  void                   (JEM_APIENTRYP glDepthRangef)

    ( GLfloat               n,
      GLfloat               f );

  void                   (JEM_APIENTRYP glClearDepthf)

    ( GLfloat               d );

  void                   (JEM_APIENTRYP glGetProgramBinary)

    ( GLuint                program,
      GLsizei               bufSize,
      GLsizei*              length,
      GLenum*               binaryFormat,
      void*                 binary );

  void                   (JEM_APIENTRYP glProgramBinary)

    ( GLuint                program,
      GLenum                binaryFormat,
      const void*           binary,
      GLsizei               length );

  void                   (JEM_APIENTRYP glProgramParameteri)

    ( GLuint                program,
      GLenum                pname,
      GLint                 value );

  void                   (JEM_APIENTRYP glUseProgramStages)

    ( GLuint                pipeline,
      GLbitfield            stages,
      GLuint                program );

  void                   (JEM_APIENTRYP glActiveShaderProgram)

    ( GLuint                pipeline,
      GLuint                program );

  GLuint                 (JEM_APIENTRYP glCreateShaderProgramv)

    ( GLenum                type,
      GLsizei               count,
      const GLchar* const*  strings );

  void                   (JEM_APIENTRYP glBindProgramPipeline)

    ( GLuint                pipeline );

  void                   (JEM_APIENTRYP glDeleteProgramPipelines)

    ( GLsizei               n,
      const GLuint*         pipelines );

  void                   (JEM_APIENTRYP glGenProgramPipelines)

    ( GLsizei               n,
      GLuint*               pipelines );

  GLboolean              (JEM_APIENTRYP glIsProgramPipeline)

    ( GLuint                pipeline );

  void                   (JEM_APIENTRYP glGetProgramPipelineiv)

    ( GLuint                pipeline,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glProgramUniform1i)

    ( GLuint                program,
      GLint                 location,
      GLint                 v0 );

  void                   (JEM_APIENTRYP glProgramUniform1iv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLint*          value );

  void                   (JEM_APIENTRYP glProgramUniform1f)

    ( GLuint                program,
      GLint                 location,
      GLfloat               v0 );

  void                   (JEM_APIENTRYP glProgramUniform1fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniform1d)

    ( GLuint                program,
      GLint                 location,
      GLdouble              v0 );

  void                   (JEM_APIENTRYP glProgramUniform1dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniform1ui)

    ( GLuint                program,
      GLint                 location,
      GLuint                v0 );

  void                   (JEM_APIENTRYP glProgramUniform1uiv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glProgramUniform2i)

    ( GLuint                program,
      GLint                 location,
      GLint                 v0,
      GLint                 v1 );

  void                   (JEM_APIENTRYP glProgramUniform2iv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLint*          value );

  void                   (JEM_APIENTRYP glProgramUniform2f)

    ( GLuint                program,
      GLint                 location,
      GLfloat               v0,
      GLfloat               v1 );

  void                   (JEM_APIENTRYP glProgramUniform2fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniform2d)

    ( GLuint                program,
      GLint                 location,
      GLdouble              v0,
      GLdouble              v1 );

  void                   (JEM_APIENTRYP glProgramUniform2dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniform2ui)

    ( GLuint                program,
      GLint                 location,
      GLuint                v0,
      GLuint                v1 );

  void                   (JEM_APIENTRYP glProgramUniform2uiv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glProgramUniform3i)

    ( GLuint                program,
      GLint                 location,
      GLint                 v0,
      GLint                 v1,
      GLint                 v2 );

  void                   (JEM_APIENTRYP glProgramUniform3iv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLint*          value );

  void                   (JEM_APIENTRYP glProgramUniform3f)

    ( GLuint                program,
      GLint                 location,
      GLfloat               v0,
      GLfloat               v1,
      GLfloat               v2 );

  void                   (JEM_APIENTRYP glProgramUniform3fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniform3d)

    ( GLuint                program,
      GLint                 location,
      GLdouble              v0,
      GLdouble              v1,
      GLdouble              v2 );

  void                   (JEM_APIENTRYP glProgramUniform3dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniform3ui)

    ( GLuint                program,
      GLint                 location,
      GLuint                v0,
      GLuint                v1,
      GLuint                v2 );

  void                   (JEM_APIENTRYP glProgramUniform3uiv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glProgramUniform4i)

    ( GLuint                program,
      GLint                 location,
      GLint                 v0,
      GLint                 v1,
      GLint                 v2,
      GLint                 v3 );

  void                   (JEM_APIENTRYP glProgramUniform4iv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLint*          value );

  void                   (JEM_APIENTRYP glProgramUniform4f)

    ( GLuint                program,
      GLint                 location,
      GLfloat               v0,
      GLfloat               v1,
      GLfloat               v2,
      GLfloat               v3 );

  void                   (JEM_APIENTRYP glProgramUniform4fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniform4d)

    ( GLuint                program,
      GLint                 location,
      GLdouble              v0,
      GLdouble              v1,
      GLdouble              v2,
      GLdouble              v3 );

  void                   (JEM_APIENTRYP glProgramUniform4dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniform4ui)

    ( GLuint                program,
      GLint                 location,
      GLuint                v0,
      GLuint                v1,
      GLuint                v2,
      GLuint                v3 );

  void                   (JEM_APIENTRYP glProgramUniform4uiv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix2fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix3fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix4fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix2dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix3dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix4dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix2x3fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix3x2fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix2x4fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix4x2fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix3x4fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix4x3fv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix2x3dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix3x2dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix2x4dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix4x2dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix3x4dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glProgramUniformMatrix4x3dv)

    ( GLuint                program,
      GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glValidateProgramPipeline)

    ( GLuint                pipeline );

  void                   (JEM_APIENTRYP glGetProgramPipelineInfoLog)

    ( GLuint                pipeline,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               infoLog );

  void                   (JEM_APIENTRYP glVertexAttribL1d)

    ( GLuint                index,
      GLdouble              x );

  void                   (JEM_APIENTRYP glVertexAttribL2d)

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y );

  void                   (JEM_APIENTRYP glVertexAttribL3d)

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                   (JEM_APIENTRYP glVertexAttribL4d)

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  void                   (JEM_APIENTRYP glVertexAttribL1dv)

    ( GLuint                index,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glVertexAttribL2dv)

    ( GLuint                index,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glVertexAttribL3dv)

    ( GLuint                index,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glVertexAttribL4dv)

    ( GLuint                index,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glVertexAttribLPointer)

    ( GLuint                index,
      GLint                 size,
      GLenum                type,
      GLsizei               stride,
      const void*           pointer );

  void                   (JEM_APIENTRYP glGetVertexAttribLdv)

    ( GLuint                index,
      GLenum                pname,
      GLdouble*             params );

  void                   (JEM_APIENTRYP glViewportArrayv)

    ( GLuint                first,
      GLsizei               count,
      const GLfloat*        v );

  void                   (JEM_APIENTRYP glViewportIndexedf)

    ( GLuint                index,
      GLfloat               x,
      GLfloat               y,
      GLfloat               w,
      GLfloat               h );

  void                   (JEM_APIENTRYP glViewportIndexedfv)

    ( GLuint                index,
      const GLfloat*        v );

  void                   (JEM_APIENTRYP glScissorArrayv)

    ( GLuint                first,
      GLsizei               count,
      const GLint*          v );

  void                   (JEM_APIENTRYP glScissorIndexed)

    ( GLuint                index,
      GLint                 left,
      GLint                 bottom,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glScissorIndexedv)

    ( GLuint                index,
      const GLint*          v );

  void                   (JEM_APIENTRYP glDepthRangeArrayv)

    ( GLuint                first,
      GLsizei               count,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glDepthRangeIndexed)

    ( GLuint                index,
      GLdouble              n,
      GLdouble              f );

  void                   (JEM_APIENTRYP glGetFloati_v)

    ( GLenum                target,
      GLuint                index,
      GLfloat*              data );

  void                   (JEM_APIENTRYP glGetDoublei_v)

    ( GLenum                target,
      GLuint                index,
      GLdouble*             data );


 protected:

  virtual                ~GLFuncTable4_1Core  ();

};


JEM_END_PACKAGE( gl )

#endif
