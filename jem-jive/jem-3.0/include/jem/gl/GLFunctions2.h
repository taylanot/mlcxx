
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

#ifndef JEM_GL_GLFUNCTIONS2_H
#define JEM_GL_GLFUNCTIONS2_H

#include <jem/gl/GLFunctions1_5.h>
#include <jem/gl/ftable/GLFuncTable2.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions2Core
//-----------------------------------------------------------------------


class GLFunctions2Core : public GLFunctions1_5Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_2_CORE;

  explicit                GLFunctions2Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions2Core  ();


 public:

  inline void             glAttachShader

    ( GLuint                program,
      GLuint                shader );

  inline void             glBindAttribLocation

    ( GLuint                program,
      GLuint                index,
      const GLchar*         name );

  inline void             glBlendEquationSeparate

    ( GLenum                modeRGB,
      GLenum                modeAlpha );

  inline void             glCompileShader

    ( GLuint                shader );

  inline GLuint           glCreateProgram      ();

  inline GLuint           glCreateShader

    ( GLenum                type );

  inline void             glDeleteProgram

    ( GLuint                program );

  inline void             glDeleteShader

    ( GLuint                shader );

  inline void             glDetachShader

    ( GLuint                program,
      GLuint                shader );

  inline void             glDisableVertexAttribArray

    ( GLuint                index );

  inline void             glDrawBuffers

    ( GLsizei               n,
      const GLenum*         bufs );

  inline void             glEnableVertexAttribArray

    ( GLuint                index );

  inline void             glGetActiveAttrib

    ( GLuint                program,
      GLuint                index,
      GLsizei               bufSize,
      GLsizei*              length,
      GLint*                size,
      GLenum*               type,
      GLchar*               name );

  inline void             glGetActiveUniform

    ( GLuint                program,
      GLuint                index,
      GLsizei               bufSize,
      GLsizei*              length,
      GLint*                size,
      GLenum*               type,
      GLchar*               name );

  inline void             glGetAttachedShaders

    ( GLuint                program,
      GLsizei               maxCount,
      GLsizei*              count,
      GLuint*               shaders );

  inline GLint            glGetAttribLocation

    ( GLuint                program,
      const GLchar*         name );

  inline void             glGetProgramInfoLog

    ( GLuint                program,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               infoLog );

  inline void             glGetProgram

    ( GLuint                program,
      GLenum                pname,
      GLint*                params );

  inline void             glGetShader

    ( GLuint                shader,
      GLenum                pname,
      GLint*                params );

  inline void             glGetShaderInfoLog

    ( GLuint                shader,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               infoLog );

  inline void             glGetShaderSource

    ( GLuint                shader,
      GLsizei               bufSize,
      GLsizei*              length,
      GLchar*               source );

  inline GLint             glGetUniformLocation

    ( GLuint                program,
      const GLchar*         name );

  inline void             glGetUniform

    ( GLuint                program,
      GLint                 location,
      GLfloat*              params );

  inline void             glGetUniform

    ( GLuint                program,
      GLint                 location,
      GLint*                params );

  inline void             glGetVertexAttrib

    ( GLuint                index,
      GLenum                pname,
      GLdouble*             params );

  inline void             glGetVertexAttrib

    ( GLuint                index,
      GLenum                pname,
      GLfloat*              params );

  inline void             glGetVertexAttrib

    ( GLuint                index,
      GLenum                pname,
      GLint*                params );

  inline void             glGetVertexAttribPointer

    ( GLuint                index,
      GLenum                pname,
      void**                pointer );

  GLboolean               glIsProgram

    ( GLuint                program );

  GLboolean               glIsShader

    ( GLuint                shader );

  inline void             glLinkProgram

    ( GLuint                program );

  inline void             glShaderSource

    ( GLuint                shader,
      GLsizei               count,
      const GLchar* const*  string,
      const GLint*          length );

  inline void             glStencilFuncSeparate

    ( GLenum                face,
      GLenum                func,
      GLint                 ref,
      GLuint                mask );

  inline void             glStencilMaskSeparate

    ( GLenum                face,
      GLint                 mask );

  inline void             glStencilOpSeparate

    ( GLenum                face,
      GLenum                sfail,
      GLenum                dpfail,
      GLuint                mask );

  inline void             glUniform

    ( GLint                 location,
      GLfloat               v0 );

  inline void             glUniform

    ( GLint                 location,
      GLfloat               v0,
      GLfloat               v1 );

  inline void             glUniform

    ( GLint                 location,
      GLfloat               v0,
      GLfloat               v1,
      GLfloat               v2 );

  inline void             glUniform

    ( GLint                 location,
      GLfloat               v0,
      GLfloat               v1,
      GLfloat               v2,
      GLfloat               v3 );

  inline void             glUniform

    ( GLint                 location,
      GLint                 v0 );

  inline void             glUniform

    ( GLint                 location,
      GLint                 v0,
      GLint                 v1 );

  inline void             glUniform

    ( GLint                 location,
      GLint                 v0,
      GLint                 v1,
      GLint                 v2 );

  inline void             glUniform

    ( GLint                 location,
      GLint                 v0,
      GLint                 v1,
      GLint                 v2,
      GLint                 v3 );

  inline void             glUniform1

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  inline void             glUniform2

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  inline void             glUniform3

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  inline void             glUniform4

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value );

  inline void             glUniform1

    ( GLint                 location,
      GLsizei               count,
      const GLint*          value );

  inline void             glUniform2

    ( GLint                 location,
      GLsizei               count,
      const GLint*          value );

  inline void             glUniform3

    ( GLint                 location,
      GLsizei               count,
      const GLint*          value );

  inline void             glUniform4

    ( GLint                 location,
      GLsizei               count,
      const GLint*          value );

  inline void             glUniformMatrix2

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glUniformMatrix3

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glUniformMatrix4

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glUseProgram

    ( GLuint                program );

  inline void             glValidateProgram

    ( GLuint                program );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLdouble              x );

  inline void             glVertexAttrib1

    ( GLuint                index,
      const GLdouble*       v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLfloat               x );

  inline void             glVertexAttrib1

    ( GLuint                index,
      const GLfloat*        v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLshort               x );

  inline void             glVertexAttrib1

    ( GLuint                index,
      const GLshort*        v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y );

  inline void             glVertexAttrib2

    ( GLuint                index,
      const GLdouble*       v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLfloat               x,
      GLfloat               y );

  inline void             glVertexAttrib2

    ( GLuint                index,
      const GLfloat*        v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLshort               x,
      GLshort               y );

  inline void             glVertexAttrib2

    ( GLuint                index,
      const GLshort*        v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  inline void             glVertexAttrib3

    ( GLuint                index,
      const GLdouble*       v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  inline void             glVertexAttrib3

    ( GLuint                index,
      const GLfloat*        v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLshort               x,
      GLshort               y,
      GLshort               z );

  inline void             glVertexAttrib3

    ( GLuint                index,
      const GLshort*        v );

  inline void             glVertexAttrib4N

    ( GLuint                index,
      const GLbyte*         v );

  inline void             glVertexAttrib4N

    ( GLuint                index,
      const GLint*          v );

  inline void             glVertexAttrib4N

    ( GLuint                index,
      const GLshort*        v );

  inline void             glVertexAttrib4N

    ( GLuint                index,
      GLubyte               x,
      GLubyte               y,
      GLubyte               z,
      GLubyte               w );

  inline void             glVertexAttrib4N

    ( GLuint                index,
      const GLubyte*        v );

  inline void             glVertexAttrib4N

    ( GLuint                index,
      const GLuint*         v );

  inline void             glVertexAttrib4N

    ( GLuint                index,
      const GLushort*       v );

  inline void             glVertexAttrib4

    ( GLuint                index,
      const GLbyte*         v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  inline void             glVertexAttrib4

    ( GLuint                index,
      const GLdouble*       v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLfloat               x,
      GLfloat               y,
      GLfloat               z,
      GLfloat               w );

  inline void             glVertexAttrib4

    ( GLuint                index,
      const GLfloat*        v );

  inline void             glVertexAttrib4

    ( GLuint                index,
      const GLint*          v );

  inline void             glVertexAttrib

    ( GLuint                index,
      GLshort               x,
      GLshort               y,
      GLshort               z,
      GLshort               w );

  inline void             glVertexAttrib4

    ( GLuint                index,
      const GLshort*        v );

  inline void             glVertexAttrib4

    ( GLuint                index,
      const GLubyte*        v );

  inline void             glVertexAttrib4

    ( GLuint                index,
      const GLuint*         v );

  inline void             glVertexAttrib4

    ( GLuint                index,
      const GLushort*       v );

  inline void             glVertexAttribPointer

    ( GLuint                index,
      GLint                 size,
      GLenum                type,
      GLboolean             normalized,
      GLsizei               stride,
      const void*           pointer );


 private:

  Ref<GLFuncTable2Core>   table_;

};


typedef
  GLFunctions2Core        GLFuncs2Core;


//-----------------------------------------------------------------------
//   class GLFunctions2Compat
//-----------------------------------------------------------------------


class GLFunctions2Compat : public GLFunctions1Compat
{
 public:

  explicit                GLFunctions2Compat

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions2Compat  ();

 private:

  Ref
    <GLFuncTable2Compat>  table_;

};


typedef
  GLFunctions2Compat      GLFuncs2Compat;


//-----------------------------------------------------------------------
//   class GLFunctions2
//-----------------------------------------------------------------------


class GLFunctions2 : public GLFunctions2Core,
                     public GLFunctions2Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_2;

  explicit                GLFunctions2

    ( GLFuncTables&         tables );

};


typedef GLFunctions2      GLFuncs2;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions2Core
//=======================================================================


inline void GLFunctions2Core::glAttachShader

  ( GLuint  program,
    GLuint  shader )

{
  table_->glAttachShader ( program, shader );
}


inline void GLFunctions2Core::glBindAttribLocation

  ( GLuint         program,
    GLuint         index,
    const GLchar*  name )

{
  table_->glBindAttribLocation ( program, index, name );
}


inline void GLFunctions2Core::glBlendEquationSeparate

    ( GLenum  modeRGB,
      GLenum  modeAlpha )

{
  table_->glBlendEquationSeparate ( modeRGB, modeAlpha );
}


inline void GLFunctions2Core::glCompileShader ( GLuint shader )
{
  table_->glCompileShader ( shader );
}


inline GLuint GLFunctions2Core::glCreateProgram ()
{
  return table_->glCreateProgram ();
}


inline GLuint GLFunctions2Core::glCreateShader ( GLenum type )
{
  return table_->glCreateShader ( type );
}


inline void GLFunctions2Core::glDeleteProgram ( GLuint program )
{
  table_->glDeleteProgram ( program );
}


inline void GLFunctions2Core::glDeleteShader ( GLuint  shader )
{
  table_->glDeleteShader ( shader );
}


inline void GLFunctions2Core::glDetachShader

  ( GLuint  program,
    GLuint  shader )

{
  table_->glDetachShader ( program, shader );
}


inline void GLFunctions2Core::glDisableVertexAttribArray

  ( GLuint  index )

{
  table_->glDisableVertexAttribArray ( index );
}


inline void GLFunctions2Core::glDrawBuffers

  ( GLsizei        n,
    const GLenum*  bufs )
{
  table_->glDrawBuffers ( n, bufs );
}


inline void GLFunctions2Core::glEnableVertexAttribArray

  ( GLuint  index )

{
  table_->glEnableVertexAttribArray ( index );
}


inline void GLFunctions2Core::glGetActiveAttrib

  ( GLuint    program,
    GLuint    index,
    GLsizei   bufSize,
    GLsizei*  length,
    GLint*    size,
    GLenum*   type,
    GLchar*   name )

{
  table_->glGetActiveAttrib ( program, index, bufSize, length,
                              size, type, name );
}


inline void GLFunctions2Core::glGetActiveUniform

  ( GLuint    program,
    GLuint    index,
    GLsizei   bufSize,
    GLsizei*  length,
    GLint*    size,
    GLenum*   type,
    GLchar*   name )

{
  table_->glGetActiveUniform ( program, index, bufSize, length,
                               size, type, name );
}


inline void GLFunctions2Core::glGetAttachedShaders

  ( GLuint    program,
    GLsizei   maxCount,
    GLsizei*  count,
    GLuint*   shaders )

{
  table_->glGetAttachedShaders ( program, maxCount, count, shaders );
}


inline GLint GLFunctions2Core::glGetAttribLocation

  ( GLuint         program,
    const GLchar*  name )

{
  return table_->glGetAttribLocation ( program, name );
}


inline void GLFunctions2Core::glGetProgramInfoLog

  ( GLuint    program,
    GLsizei   bufSize,
    GLsizei*  length,
    GLchar*   infoLog )

{
  table_->glGetProgramInfoLog ( program, bufSize, length, infoLog );
}


inline void GLFunctions2Core::glGetProgram

  ( GLuint  program,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetProgramiv ( program, pname, params);
}


inline void GLFunctions2Core::glGetShader

  ( GLuint  shader,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetShaderiv ( shader, pname, params );
}


inline void GLFunctions2Core::glGetShaderInfoLog

  ( GLuint    shader,
    GLsizei   bufSize,
    GLsizei*  length,
    GLchar*   infoLog )

{
  table_->glGetShaderInfoLog ( shader, bufSize, length, infoLog );
}


inline void GLFunctions2Core::glGetShaderSource

  ( GLuint    shader,
    GLsizei   bufSize,
    GLsizei*  length,
    GLchar*   source )

{
  table_->glGetShaderSource ( shader, bufSize, length, source );
}


inline GLint GLFunctions2Core::glGetUniformLocation

  ( GLuint         program,
    const GLchar*  name )

{
  return table_->glGetUniformLocation ( program, name );
}


inline void GLFunctions2Core::glGetUniform

  ( GLuint    program,
    GLint     location,
    GLfloat*  params )

{
  table_->glGetUniformfv ( program, location, params );
}


inline void GLFunctions2Core::glGetUniform

  ( GLuint  program,
    GLint   location,
    GLint*  params )

{
  table_->glGetUniformiv ( program, location, params );
}


inline void GLFunctions2Core::glGetVertexAttrib

  ( GLuint     index,
    GLenum     pname,
    GLdouble*  params )

{
  table_->glGetVertexAttribdv ( index, pname, params );
}


inline void GLFunctions2Core::glGetVertexAttrib

  ( GLuint     index,
    GLenum     pname,
    GLfloat*   params )

{
  table_->glGetVertexAttribfv ( index, pname, params );
}


inline void GLFunctions2Core::glGetVertexAttrib

  ( GLuint  index,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetVertexAttribiv ( index, pname, params );
}


inline void GLFunctions2Core::glGetVertexAttribPointer

  ( GLuint  index,
    GLenum  pname,
    void**  pointer )

{
  table_->glGetVertexAttribPointerv ( index, pname, pointer );
}

inline GLboolean GLFunctions2Core::glIsProgram ( GLuint program )
{
  return table_->glIsProgram ( program );
}


inline GLboolean GLFunctions2Core::glIsShader ( GLuint shader )
{
  return table_->glIsShader ( shader );
}


inline void GLFunctions2Core::glLinkProgram ( GLuint program )
{
  table_->glLinkProgram ( program );
}


inline void GLFunctions2Core::glShaderSource

  ( GLuint                shader,
    GLsizei               count,
    const GLchar* const*  string,
    const GLint*          length )

{
  table_->glShaderSource ( shader, count, string, length );
}


inline void GLFunctions2Core::glStencilFuncSeparate

  ( GLenum  face,
    GLenum  func,
    GLint   ref,
    GLuint  mask )

{
  table_->glStencilFuncSeparate ( face, func, ref, mask );
}


inline void GLFunctions2Core::glStencilMaskSeparate

  ( GLenum  face,
    GLint   mask )

{
  table_->glStencilMaskSeparate ( face, mask );
}


inline void GLFunctions2Core::glStencilOpSeparate

  ( GLenum  face,
    GLenum  sfail,
    GLenum  dpfail,
    GLuint  mask )

{
  table_->glStencilOpSeparate ( face, sfail, dpfail, mask );
}


inline void GLFunctions2Core::glUniform

  ( GLint    location,
    GLfloat  v0 )

{
  table_->glUniform1f ( location, v0 );
}


inline void GLFunctions2Core::glUniform

  ( GLint    location,
    GLfloat  v0,
    GLfloat  v1 )

{
  table_->glUniform2f ( location, v0, v1 );
}


inline void GLFunctions2Core::glUniform

  ( GLint    location,
    GLfloat  v0,
    GLfloat  v1,
    GLfloat  v2 )

{
  table_->glUniform3f ( location, v0, v1, v2 );
}


inline void GLFunctions2Core::glUniform

  ( GLint    location,
    GLfloat  v0,
    GLfloat  v1,
    GLfloat  v2,
    GLfloat  v3 )

{
  table_->glUniform4f ( location, v0, v1, v2, v3 );
}


inline void GLFunctions2Core::glUniform

  ( GLint  location,
    GLint  v0 )

{
  table_->glUniform1i ( location, v0 );
}


inline void GLFunctions2Core::glUniform

  ( GLint  location,
    GLint  v0,
    GLint  v1 )

{
  table_->glUniform2i ( location, v0, v1 );
}


inline void GLFunctions2Core::glUniform

  ( GLint  location,
    GLint  v0,
    GLint  v1,
    GLint  v2 )

{
  table_->glUniform3i ( location, v0, v1, v2 );
}


inline void GLFunctions2Core::glUniform

  ( GLint  location,
    GLint  v0,
    GLint  v1,
    GLint  v2,
    GLint  v3 )

{
  table_->glUniform4i ( location, v0, v1, v2,v3 );
}


inline void GLFunctions2Core::glUniform1

  ( GLint           location,
    GLsizei         count,
    const GLfloat*  value )

{
  table_->glUniform1fv ( location, count, value );
}


inline void GLFunctions2Core::glUniform2

  ( GLint           location,
    GLsizei         count,
    const GLfloat*  value )

{
  table_->glUniform2fv ( location, count, value );
}


inline void GLFunctions2Core::glUniform3

  ( GLint           location,
    GLsizei         count,
    const GLfloat*  value )

{
  table_->glUniform3fv ( location, count, value );
}


inline void GLFunctions2Core::glUniform4

    ( GLint                 location,
      GLsizei               count,
      const GLfloat*        value )

{
  table_->glUniform4fv ( location, count, value );
}


inline void GLFunctions2Core::glUniform1

  ( GLint         location,
    GLsizei       count,
    const GLint*  value )

{
  table_->glUniform1iv ( location, count, value );
}


inline void GLFunctions2Core::glUniform2

  ( GLint         location,
    GLsizei       count,
    const GLint*  value )

{
  table_->glUniform2iv ( location, count, value );
}


inline void GLFunctions2Core::glUniform3

  ( GLint         location,
    GLsizei       count,
    const GLint*  value )

{
  table_->glUniform3iv ( location, count, value );
}


inline void GLFunctions2Core::glUniform4

  ( GLint         location,
    GLsizei       count,
    const GLint*  value )

{
  table_->glUniform4iv ( location, count, value );
}


inline void GLFunctions2Core::glUniformMatrix2

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix2fv ( location, count, transpose, value );
}


inline void GLFunctions2Core::glUniformMatrix3

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix3fv ( location, count, transpose, value );
}


inline void GLFunctions2Core::glUniformMatrix4

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix4fv ( location, count, transpose, value );
}


inline void GLFunctions2Core::glUseProgram ( GLuint program )
{
  table_->glUseProgram ( program );
}


inline void GLFunctions2Core::glValidateProgram ( GLuint program )
{
  table_->glValidateProgram ( program );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint    index,
    GLdouble  x )

{
  table_->glVertexAttrib1d ( index, x );
}


inline void GLFunctions2Core::glVertexAttrib1

  ( GLuint           index,
    const GLdouble*  v )

{
  table_->glVertexAttrib1dv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint   index,
    GLfloat  x )

{
  table_->glVertexAttrib1f ( index, x );
}


inline void GLFunctions2Core::glVertexAttrib1

  ( GLuint          index,
    const GLfloat*  v )

{
  table_->glVertexAttrib1fv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint   index,
    GLshort  x )

{
  table_->glVertexAttrib1s ( index, x );
}


inline void GLFunctions2Core::glVertexAttrib1

  ( GLuint          index,
    const GLshort*  v )

{
  table_->glVertexAttrib1sv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint    index,
    GLdouble  x,
    GLdouble  y )

{
  table_->glVertexAttrib2d ( index, x, y );
}


inline void GLFunctions2Core::glVertexAttrib2

  ( GLuint           index,
    const GLdouble*  v )

{
  table_->glVertexAttrib2dv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint   index,
    GLfloat  x,
    GLfloat  y )

{
  table_->glVertexAttrib2f ( index, x, y );
}


inline void GLFunctions2Core::glVertexAttrib2

  ( GLuint          index,
    const GLfloat*  v )

{
  table_->glVertexAttrib2fv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint   index,
    GLshort  x,
    GLshort  y )

{
  table_->glVertexAttrib2s ( index, x, y );
}


inline void GLFunctions2Core::glVertexAttrib2

  ( GLuint          index,
    const GLshort*  v )

{
  table_->glVertexAttrib2sv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint    index,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  table_->glVertexAttrib3d ( index, x, y, z );
}


inline void GLFunctions2Core::glVertexAttrib3

  ( GLuint           index,
    const GLdouble*  v )

{
  table_->glVertexAttrib3dv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint   index,
    GLfloat  x,
    GLfloat  y,
    GLfloat  z )

{
  table_->glVertexAttrib3f ( index, x, y, z );
}


inline void GLFunctions2Core::glVertexAttrib3

  ( GLuint          index,
    const GLfloat*  v )

{
  table_->glVertexAttrib3fv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint   index,
    GLshort  x,
    GLshort  y,
    GLshort  z )

{
  table_->glVertexAttrib3s ( index, x, y, z );
}


inline void GLFunctions2Core::glVertexAttrib3

  ( GLuint          index,
    const GLshort*  v )

{
  table_->glVertexAttrib3sv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4N

  ( GLuint         index,
    const GLbyte*  v )

{
  table_->glVertexAttrib4Nbv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4N

  ( GLuint        index,
    const GLint*  v )

{
  table_->glVertexAttrib4Niv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4N

  ( GLuint          index,
    const GLshort*  v )

{
  table_->glVertexAttrib4Nsv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4N

  ( GLuint   index,
    GLubyte  x,
    GLubyte  y,
    GLubyte  z,
    GLubyte  w )

{
  table_->glVertexAttrib4Nub ( index, x, y, z, w );
}


inline void GLFunctions2Core::glVertexAttrib4N

  ( GLuint          index,
    const GLubyte*  v )

{
  table_->glVertexAttrib4Nubv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4N

  ( GLuint         index,
    const GLuint*  v )

{
  table_->glVertexAttrib4Nuiv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4N

  ( GLuint           index,
    const GLushort*  v )

{
  table_->glVertexAttrib4Nusv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4

  ( GLuint         index,
    const GLbyte*  v )

{
  table_->glVertexAttrib4bv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint    index,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z,
    GLdouble  w )

{
  table_->glVertexAttrib4d ( index, x, y, z , w );
}


inline void GLFunctions2Core::glVertexAttrib4

  ( GLuint           index,
    const GLdouble*  v )

{
  table_->glVertexAttrib4dv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint   index,
    GLfloat  x,
    GLfloat  y,
    GLfloat  z,
    GLfloat  w )

{
  table_->glVertexAttrib4f ( index, x, y, z, w );
}


inline void GLFunctions2Core::glVertexAttrib4

  ( GLuint          index,
    const GLfloat*  v )

{
  table_->glVertexAttrib4fv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4

  ( GLuint        index,
    const GLint*  v )

{
  table_->glVertexAttrib4iv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib

  ( GLuint   index,
    GLshort  x,
    GLshort  y,
    GLshort  z,
    GLshort  w )

{
  table_->glVertexAttrib4s ( index, x, y, z, w );
}


inline void GLFunctions2Core::glVertexAttrib4

  ( GLuint          index,
    const GLshort*  v )

{
  table_->glVertexAttrib4sv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4

  ( GLuint          index,
    const GLubyte*  v )

{
  table_->glVertexAttrib4ubv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4

  ( GLuint         index,
    const GLuint*  v )

{
  table_->glVertexAttrib4uiv ( index, v );
}


inline void GLFunctions2Core::glVertexAttrib4

  ( GLuint           index,
    const GLushort*  v )

{
  table_->glVertexAttrib4usv ( index, v );
}


inline void GLFunctions2Core::glVertexAttribPointer

  ( GLuint       index,
    GLint        size,
    GLenum       type,
    GLboolean    normalized,
    GLsizei      stride,
    const void*  pointer )

{
  table_->glVertexAttribPointer ( index, size, type, normalized,
                                  stride, pointer );
}


JEM_END_PACKAGE( gl )

#endif
