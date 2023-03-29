
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

#ifndef JEM_GL_GLFUNCTABLE2_H
#define JEM_GL_GLFUNCTABLE2_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable2Core
//-----------------------------------------------------------------------

// OpenGL 2.0 core functions.

class GLFuncTable2Core : public Object
{
 public:

  explicit                GLFuncTable2Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glAttachShader)

    ( GLuint                program,
      GLuint                shader );

  void                   (JEM_APIENTRYP glBindAttribLocation)

    ( GLuint                program,
      GLuint                index,
      const GLchar*         name );

  void                   (JEM_APIENTRYP glBlendEquationSeparate)

    ( GLenum                modeRGB,
      GLenum                modeAlpha );

  void                   (JEM_APIENTRYP glCompileShader)

    ( GLuint                shader );

  GLuint                 (JEM_APIENTRYP glCreateProgram) ();

  GLuint                 (JEM_APIENTRYP glCreateShader)

    ( GLenum                type );

  void                   (JEM_APIENTRYP glDeleteProgram)

    ( GLuint                program );

  void                   (JEM_APIENTRYP glDeleteShader)

    ( GLuint                shader );

  void                   (JEM_APIENTRYP glDetachShader)

    ( GLuint                program,
      GLuint                shader );

  void                   (JEM_APIENTRYP glDisableVertexAttribArray)

    ( GLuint                index );

  void                   (JEM_APIENTRYP glDrawBuffers)

    ( GLsizei               n,
      const GLenum*         bufs );

  void                   (JEM_APIENTRYP glEnableVertexAttribArray)

    ( GLuint                index );

  void                   (JEM_APIENTRYP glGetActiveAttrib)

    ( GLuint                program,
      GLuint                index,
      GLsizei               bufSize,
      GLsizei*              length,
      GLint*                size,
      GLenum*               type,
      GLchar*               name );

  void                   (JEM_APIENTRYP glGetActiveUniform)

    ( GLuint                program,
      GLuint                index,
      GLsizei               bufSize,
      GLsizei*              length,
      GLint*                size,
      GLenum*               type,
      GLchar*               name );

  void                   (JEM_APIENTRYP glGetAttachedShaders)

    ( GLuint                program,
      GLsizei               maxCount,
      GLsizei*              count,
      GLuint*               shaders );

  GLint                  (JEM_APIENTRYP glGetAttribLocation)

    ( GLuint                program,
      const GLchar*         name );

  void                   (JEM_APIENTRYP glGetProgramInfoLog)

    ( GLuint                program,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               infoLog );

  void                   (JEM_APIENTRYP glGetProgramiv)

    ( GLuint                program,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetShaderiv)

    ( GLuint                shader,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetShaderInfoLog)

    ( GLuint                shader,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               infoLog );

  void                   (JEM_APIENTRYP glGetShaderSource)

    ( GLuint                shader,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               source );

  GLint                  (JEM_APIENTRYP glGetUniformLocation)

    ( GLuint                program,
      const GLchar*         name );

  void                   (JEM_APIENTRYP glGetUniformfv)

    ( GLuint                program,
      GLint                 location,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetUniformiv)

    ( GLuint                program,
      GLint                 location,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetVertexAttribdv)

    ( GLuint                index,
      GLenum                pname,
      GLdouble*             params );

  void                   (JEM_APIENTRYP glGetVertexAttribfv)

    ( GLuint                index,
      GLenum                pname,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetVertexAttribiv)

    ( GLuint                index,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetVertexAttribPointerv)

    ( GLuint                index,
      GLenum                pname,
      void**                pointer );

  GLboolean              (JEM_APIENTRYP glIsProgram)

    ( GLuint                program );

  GLboolean              (JEM_APIENTRYP glIsShader)

    ( GLuint                shader );

  void                   (JEM_APIENTRYP glLinkProgram)

    ( GLuint                program );

  void                   (JEM_APIENTRYP glShaderSource)

    ( GLuint                shader,
      GLsizei               count,
      const GLchar* const*  string,
      const GLint*          length );

  void                   (JEM_APIENTRYP glStencilFuncSeparate)

    ( GLenum                face,
      GLenum                func,
      GLint                 ref,
      GLuint                mask );

  void                   (JEM_APIENTRYP glStencilMaskSeparate)

    ( GLenum                face,
      GLint                 mask );

  void                   (JEM_APIENTRYP glStencilOpSeparate)

    ( GLenum                face,
      GLenum                sfail,
      GLenum                dpfail,
      GLuint                mask );

  void                   (JEM_APIENTRYP glUniform1f)

    ( GLint                 location,
      GLfloat               v0 );

  void                   (JEM_APIENTRYP glUniform2f)

    ( GLint                 location,
      GLfloat               v0,
      GLfloat               v1 );

  void                   (JEM_APIENTRYP glUniform3f)

    ( GLint                 location,
      GLfloat               v0,
      GLfloat               v1,
      GLfloat               v2 );

  void                   (JEM_APIENTRYP glUniform4f)

    ( GLint                 location,
      GLfloat               v0,
      GLfloat               v1,
      GLfloat               v2,
      GLfloat               v3 );

  void                   (JEM_APIENTRYP glUniform1i)

    ( GLint                 location,
      GLint                 v0 );

  void                   (JEM_APIENTRYP glUniform2i)

    ( GLint                 location,
      GLint                 v0,
      GLint                 v1 );

  void                   (JEM_APIENTRYP glUniform3i)

    ( GLint                 location,
      GLint                 v0,
      GLint                 v1,
      GLint                 v2 );

  void                   (JEM_APIENTRYP glUniform4i)

    ( GLint                 location,
      GLint                 v0,
      GLint                 v1,
      GLint                 v2,
      GLint                 v3 );

  void                   (JEM_APIENTRYP glUniform1fv)

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniform2fv)

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniform3fv)

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniform4fv)

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniform1iv)

    ( GLint                 location,
      GLsizei               count,
      const GLint*          value );

  void                   (JEM_APIENTRYP glUniform2iv)

    ( GLint                 location,
      GLsizei               count,
      const GLint*          value );

  void                   (JEM_APIENTRYP glUniform3iv)

    ( GLint                 location,
      GLsizei               count,
      const GLint*          value );

  void                   (JEM_APIENTRYP glUniform4iv)

    ( GLint                 location,
      GLsizei               count,
      const GLint*          value );

  void                   (JEM_APIENTRYP glUniformMatrix2fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniformMatrix3fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniformMatrix4fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUseProgram)

    ( GLuint                program );

  void                   (JEM_APIENTRYP glValidateProgram)

    ( GLuint                program );

  void                   (JEM_APIENTRYP glVertexAttrib1d)

    ( GLuint                index,
      GLdouble              x );

  void                   (JEM_APIENTRYP glVertexAttrib1dv)

    ( GLuint                index,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glVertexAttrib1f)

    ( GLuint                index,
      GLfloat               x );

  void                   (JEM_APIENTRYP glVertexAttrib1fv)

    ( GLuint                index,
      const GLfloat*        v );

  void                   (JEM_APIENTRYP glVertexAttrib1s)

    ( GLuint                index,
      GLshort               x );

  void                   (JEM_APIENTRYP glVertexAttrib1sv)

    ( GLuint                index,
      const GLshort*        v );

  void                   (JEM_APIENTRYP glVertexAttrib2d)

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y );

  void                   (JEM_APIENTRYP glVertexAttrib2dv)

    ( GLuint                index,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glVertexAttrib2f)

    ( GLuint                index,
      GLfloat               x,
      GLfloat               y );

  void                   (JEM_APIENTRYP glVertexAttrib2fv)

    ( GLuint                index,
      const GLfloat*        v );

  void                   (JEM_APIENTRYP glVertexAttrib2s)

    ( GLuint                index,
      GLshort               x,
      GLshort               y );

  void                   (JEM_APIENTRYP glVertexAttrib2sv)

    ( GLuint                index,
      const GLshort*        v );

  void                   (JEM_APIENTRYP glVertexAttrib3d)

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                   (JEM_APIENTRYP glVertexAttrib3dv)

    ( GLuint                index,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glVertexAttrib3f)

    ( GLuint                index,
      GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  void                   (JEM_APIENTRYP glVertexAttrib3fv)

    ( GLuint                index,
      const GLfloat*        v );

  void                   (JEM_APIENTRYP glVertexAttrib3s)

    ( GLuint                index,
      GLshort               x,
      GLshort               y,
      GLshort               z );

  void                   (JEM_APIENTRYP glVertexAttrib3sv)

    ( GLuint                index,
      const GLshort*        v );

  void                   (JEM_APIENTRYP glVertexAttrib4Nbv)

    ( GLuint                index,
      const GLbyte*         v );

  void                   (JEM_APIENTRYP glVertexAttrib4Niv)

    ( GLuint                index,
      const GLint*          v );

  void                   (JEM_APIENTRYP glVertexAttrib4Nsv)

    ( GLuint                index,
      const GLshort*        v );

  void                   (JEM_APIENTRYP glVertexAttrib4Nub)

    ( GLuint                index,
      GLubyte               x,
      GLubyte               y,
      GLubyte               z,
      GLubyte               w );

  void                   (JEM_APIENTRYP glVertexAttrib4Nubv)

    ( GLuint                index,
      const GLubyte*        v );

  void                   (JEM_APIENTRYP glVertexAttrib4Nuiv)

    ( GLuint                index,
      const GLuint*         v );

  void                   (JEM_APIENTRYP glVertexAttrib4Nusv)

    ( GLuint                index,
      const GLushort*       v );

  void                   (JEM_APIENTRYP glVertexAttrib4bv)

    ( GLuint                index,
      const GLbyte*         v );

  void                   (JEM_APIENTRYP glVertexAttrib4d)

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  void                   (JEM_APIENTRYP glVertexAttrib4dv)

    ( GLuint                index,
      const GLdouble*       v );

  void                   (JEM_APIENTRYP glVertexAttrib4f)

    ( GLuint                index,
      GLfloat               x,
      GLfloat               y,
      GLfloat               z,
      GLfloat               w );

  void                   (JEM_APIENTRYP glVertexAttrib4fv)

    ( GLuint                index,
      const GLfloat*        v );

  void                   (JEM_APIENTRYP glVertexAttrib4iv)

    ( GLuint                index,
      const GLint*          v );

  void                   (JEM_APIENTRYP glVertexAttrib4s)

    ( GLuint                index,
      GLshort               x,
      GLshort               y,
      GLshort               z,
      GLshort               w );

  void                   (JEM_APIENTRYP glVertexAttrib4sv)

    ( GLuint                index,
      const GLshort*        v );

  void                   (JEM_APIENTRYP glVertexAttrib4ubv)

    ( GLuint                index,
      const GLubyte*        v );

  void                   (JEM_APIENTRYP glVertexAttrib4uiv)

    ( GLuint                index,
      const GLuint*         v );

  void                   (JEM_APIENTRYP glVertexAttrib4usv)

    ( GLuint                index,
      const GLushort*       v );

  void                   (JEM_APIENTRYP glVertexAttribPointer)

    ( GLuint                index,
      GLint                 size,
      GLenum                type,
      GLboolean             normalized,
      GLsizei               stride,
      const void*           pointer );


 protected:

  virtual                ~GLFuncTable2Core  ();

};


//-----------------------------------------------------------------------
//   class GLFuncTable2Compat
//-----------------------------------------------------------------------

// OpenGL 2.0 deprecated functions.

class GLFuncTable2Compat : public Object
{
 public:

  explicit                GLFuncTable2Compat

    ( GLFuncLoader*         loader );


 public:


 protected:

  virtual                ~GLFuncTable2Compat ();

};


JEM_END_PACKAGE( gl )

#endif
