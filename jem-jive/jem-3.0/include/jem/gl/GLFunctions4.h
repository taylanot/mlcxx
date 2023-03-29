
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

#ifndef JEM_GL_GLFUNCTIONS4_H
#define JEM_GL_GLFUNCTIONS4_H

#include <jem/gl/GLFunctions3_3.h>
#include <jem/gl/ftable/GLFuncTable4.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions4Core
//-----------------------------------------------------------------------


class GLFunctions4Core : public GLFunctions3_3Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_4_CORE;

  explicit                GLFunctions4Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions4Core  ();


 public:

  using GLFunctions1Core  ::glBlendFunc;
  using GLFunctions1_4Core::glBlendFuncSeparate;
  using GLFunctions1_4Core::glBlendEquation;
  using GLFunctions2Core  ::glBlendEquationSeparate;
  using GLFunctions3Core  ::glGetUniform;
  using GLFunctions3Core  ::glUniform;
  using GLFunctions3Core  ::glUniform1;
  using GLFunctions3Core  ::glUniform2;
  using GLFunctions3Core  ::glUniform3;
  using GLFunctions3Core  ::glUniform4;
  using GLFunctions2Core  ::glUniformMatrix2;
  using GLFunctions2Core  ::glUniformMatrix3;
  using GLFunctions2Core  ::glUniformMatrix4;
  using GLFunctions2_1Core::glUniformMatrix2x3;
  using GLFunctions2_1Core::glUniformMatrix2x4;
  using GLFunctions2_1Core::glUniformMatrix3x2;
  using GLFunctions2_1Core::glUniformMatrix3x4;
  using GLFunctions2_1Core::glUniformMatrix4x2;
  using GLFunctions2_1Core::glUniformMatrix4x3;

  inline void             glMinSampleShading

    ( GLfloat               value );

  inline void             glBlendEquation

    ( GLuint                buf,
      GLenum                mode );

  inline void             glBlendEquationSeparate

    ( GLuint                buf,
      GLenum                modeRGB,
      GLenum                modeAlpha );

  inline void             glBlendFunc

    ( GLuint                buf,
      GLenum                src,
      GLenum                dst );

  inline void             glBlendFuncSeparate

    ( GLuint                buf,
      GLenum                srcRGB,
      GLenum                dstRGB,
      GLenum                srcAlpha,
      GLenum                dstAlpha );

  inline void             glDrawArraysIndirect

    ( GLenum                mode,
      const void*           indirect );

  inline void             glDrawElementsIndirect

    ( GLenum                mode,
      GLenum                type,
      const void*           indirect );

  inline void             glUniform

    ( GLint                 location,
      GLdouble              x );

  inline void             glUniform

    ( GLint                 location,
      GLdouble              x,
      GLdouble              y );

  inline void             glUniform

    ( GLint                 location,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  inline void             glUniform

    ( GLint                 location,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  inline void             glUniform1

    ( GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  inline void             glUniform2

    ( GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  inline void             glUniform3

    ( GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  inline void             glUniform4

    ( GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  inline void             glUniformMatrix2

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glUniformMatrix3

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glUniformMatrix4

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glUniformMatrix2x3

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glUniformMatrix2x4

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glUniformMatrix3x2

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glUniformMatrix3x4

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glUniformMatrix4x2

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glUniformMatrix4x3

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  inline void             glGetUniform

    ( GLuint                program,
      GLint                 location,
      GLdouble*             params );

  inline GLint            glGetSubroutineUniformLocation

    ( GLuint                program,
      GLenum                shadertype,
      const GLchar*         name );

  inline GLuint           glGetSubroutineIndex

    ( GLuint                program,
      GLenum                shadertype,
      const GLchar*         name );

  inline void             glGetActiveSubroutineUniform

    ( GLuint                program,
      GLenum                shadertype,
      GLuint                index,
      GLenum                pname,
      GLint*                values );

  inline void             glGetActiveSubroutineUniformName

    ( GLuint                program,
      GLenum                shadertype,
      GLuint                index,
      GLsizei               bufsize,
      GLsizei*              length,
      GLchar*               name );

  inline void             glGetActiveSubroutineName

    ( GLuint                program,
      GLenum                shadertype,
      GLuint                index,
      GLsizei               bufsize,
      GLsizei*              length,
      GLchar*               name );

  inline void             glUniformSubroutines

    ( GLenum                shadertype,
      GLsizei               count,
      const GLuint*         indices );

  inline void             glGetUniformSubroutine

    ( GLenum                shadertype,
      GLint                 location,
      GLuint*               params );

  inline void             glGetProgramStage

    ( GLuint                program,
      GLenum                shadertype,
      GLenum                pname,
      GLint*                values );

  inline void             glPatchParameter

    ( GLenum                pname,
      GLint                 value );

  inline void             glPatchParameter

    ( GLenum                pname,
      const GLfloat*        values );

  inline void             glBindTransformFeedback

    ( GLenum                target,
      GLuint                id );

  inline void             glDeleteTransformFeedbacks

    ( GLsizei               n,
      const GLuint*         ids );

  inline void             glGenTransformFeedbacks

    ( GLsizei               n,
      GLuint*               ids );

  inline GLboolean        glIsTransformFeedback

    ( GLuint                id );

  inline void             glPauseTransformFeedback ();

  inline void             glResumeTransformFeedback ();

  inline void             glDrawTransformFeedback

    ( GLenum                mode,
      GLuint                id );

  inline void             glDrawTransformFeedbackStream

    ( GLenum                mode,
      GLuint                id,
      GLuint                stream );

  inline void             glBeginQueryIndexed

    ( GLenum                target,
      GLuint                index,
      GLuint                id );

  inline void             glEndQueryIndexed

    ( GLenum                target,
      GLuint                index );

  inline void             glGetQueryIndexed

    ( GLenum                target,
      GLuint                index,
      GLenum                pname,
      GLint*                params );


 private:

  Ref<GLFuncTable4Core>   table_;

};

typedef GLFunctions4Core  GLFuncs4Core;


//-----------------------------------------------------------------------
//   class GLFunctions4Compat
//-----------------------------------------------------------------------


class GLFunctions4Compat : public GLFunctions3Compat
{
 public:

  explicit                GLFunctions4Compat

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions4Compat  ();

 private:

  Ref
    <GLFuncTable4Compat>  table_;

};


typedef
  GLFunctions4Compat      GLFuncs4Compat;


//-----------------------------------------------------------------------
//   class GLFunctions4
//-----------------------------------------------------------------------


class GLFunctions4 : public GLFunctions4Core,
                     public GLFunctions4Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_4;

  explicit                GLFunctions4

    ( GLFuncTables&         tables );

};


typedef GLFunctions4      GLFuncs4;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions4Core
//=======================================================================


inline void GLFunctions4Core::glMinSampleShading

  ( GLfloat  value )

{
  table_->glMinSampleShading ( value );
}


inline void GLFunctions4Core::glBlendEquation

  ( GLuint  buf,
    GLenum  mode )

{
  table_->glBlendEquationi ( buf, mode );
}


inline void GLFunctions4Core::glBlendEquationSeparate

  ( GLuint  buf,
    GLenum  modeRGB,
    GLenum  modeAlpha )

{
  table_->glBlendEquationSeparatei ( buf, modeRGB, modeAlpha );
}


inline void GLFunctions4Core::glBlendFunc

  ( GLuint  buf,
    GLenum  src,
    GLenum  dst )

{
  table_->glBlendFunci ( buf, src, dst );
}


inline void GLFunctions4Core::glBlendFuncSeparate

  ( GLuint  buf,
    GLenum  srcRGB,
    GLenum  dstRGB,
    GLenum  srcAlpha,
    GLenum  dstAlpha )

{
  table_->glBlendFuncSeparatei ( buf, srcRGB, dstRGB, srcAlpha,
                                 dstAlpha );
}


inline void GLFunctions4Core::glDrawArraysIndirect

  ( GLenum       mode,
    const void*  indirect )

{
  table_->glDrawArraysIndirect ( mode, indirect );
}


inline void GLFunctions4Core::glDrawElementsIndirect

  ( GLenum       mode,
    GLenum       type,
    const void*  indirect )

{
  table_->glDrawElementsIndirect ( mode, type, indirect );
}


inline void GLFunctions4Core::glUniform

  ( GLint     location,
    GLdouble  x )

{
  table_->glUniform1d ( location, x );
}


inline void GLFunctions4Core::glUniform

  ( GLint     location,
    GLdouble  x,
    GLdouble  y )

{
  table_->glUniform2d ( location, x, y );
}


inline void GLFunctions4Core::glUniform

  ( GLint     location,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  table_->glUniform3d ( location, x, y, z );
}


inline void GLFunctions4Core::glUniform

  ( GLint     location,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z,
    GLdouble  w )

{
  table_->glUniform4d ( location, x, y, z, w );
}


inline void GLFunctions4Core::glUniform1

  ( GLint            location,
    GLsizei          count,
    const GLdouble*  value )

{
  table_->glUniform1dv ( location, count, value );
}


inline void GLFunctions4Core::glUniform2

  ( GLint            location,
    GLsizei          count,
    const GLdouble*  value )

{
  table_->glUniform2dv ( location, count, value );
}


inline void GLFunctions4Core::glUniform3

  ( GLint            location,
    GLsizei          count,
    const GLdouble*  value )

{
  table_->glUniform3dv ( location, count, value );
}


inline void GLFunctions4Core::glUniform4

  ( GLint            location,
    GLsizei          count,
    const GLdouble*  value )

{
  table_->glUniform4dv ( location, count, value );
}


inline void GLFunctions4Core::glUniformMatrix2

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix2dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glUniformMatrix3

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix3dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glUniformMatrix4

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix4dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glUniformMatrix2x3

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix2x3dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glUniformMatrix2x4

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix2x4dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glUniformMatrix3x2

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix3x2dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glUniformMatrix3x4

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix3x4dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glUniformMatrix4x2

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix4x2dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glUniformMatrix4x3

  ( GLint            location,
    GLsizei          count,
    GLboolean        transpose,
    const GLdouble*  value )

{
  table_->glUniformMatrix4x3dv ( location, count, transpose, value );
}


inline void GLFunctions4Core::glGetUniform

  ( GLuint     program,
    GLint      location,
    GLdouble*  params )

{
  table_->glGetUniformdv ( program, location, params );
}


inline GLint GLFunctions4Core::glGetSubroutineUniformLocation

  ( GLuint         program,
    GLenum         shadertype,
    const GLchar*  name )

{
  return table_->glGetSubroutineUniformLocation ( program, shadertype,
                                                  name );
}


inline GLuint GLFunctions4Core::glGetSubroutineIndex

  ( GLuint         program,
    GLenum         shadertype,
    const GLchar*  name )

{
  return table_->glGetSubroutineIndex ( program, shadertype, name );
}


inline void GLFunctions4Core::glGetActiveSubroutineUniform

  ( GLuint  program,
    GLenum  shadertype,
    GLuint  index,
    GLenum  pname,
    GLint*  values )

{
  table_->glGetActiveSubroutineUniformiv ( program, shadertype, index,
                                           pname, values );
}


inline void GLFunctions4Core::glGetActiveSubroutineUniformName

  ( GLuint    program,
    GLenum    shadertype,
    GLuint    index,
    GLsizei   bufsize,
    GLsizei*  length,
    GLchar*   name )

{
  table_->glGetActiveSubroutineUniformName ( program, shadertype,
                                             index, bufsize, length,
                                             name );
}


inline void GLFunctions4Core::glGetActiveSubroutineName

  ( GLuint    program,
    GLenum    shadertype,
    GLuint    index,
    GLsizei   bufsize,
    GLsizei*  length,
    GLchar*   name )

{
  table_->glGetActiveSubroutineName ( program, shadertype, index,
                                      bufsize, length, name );
}


inline void GLFunctions4Core::glUniformSubroutines

  ( GLenum         shadertype,
    GLsizei        count,
    const GLuint*  indices )

{
  table_->glUniformSubroutinesuiv ( shadertype, count, indices );
}


inline void GLFunctions4Core::glGetUniformSubroutine

  ( GLenum   shadertype,
    GLint    location,
    GLuint*  params )

{
  table_->glGetUniformSubroutineuiv ( shadertype, location, params );
}


inline void GLFunctions4Core::glGetProgramStage

  ( GLuint  program,
    GLenum  shadertype,
    GLenum  pname,
    GLint*  values )

{
  table_->glGetProgramStageiv ( program, shadertype, pname, values );
}


inline void GLFunctions4Core::glPatchParameter

  ( GLenum  pname,
    GLint   value )

{
  table_->glPatchParameteri ( pname, value );
}


inline void GLFunctions4Core::glPatchParameter

  ( GLenum          pname,
    const GLfloat*  values )

{
  table_->glPatchParameterfv ( pname, values );
}


inline void GLFunctions4Core::glBindTransformFeedback

  ( GLenum  target,
    GLuint  id )

{
  table_->glBindTransformFeedback ( target, id );
}


inline void GLFunctions4Core::glDeleteTransformFeedbacks

  ( GLsizei        n,
    const GLuint*  ids )

{
  table_->glDeleteTransformFeedbacks ( n, ids );
}


inline void GLFunctions4Core::glGenTransformFeedbacks

  ( GLsizei  n,
    GLuint*  ids )

{
  table_->glGenTransformFeedbacks ( n, ids );
}


inline GLboolean GLFunctions4Core::glIsTransformFeedback

  ( GLuint  id )

{
  return table_->glIsTransformFeedback ( id );
}


inline void GLFunctions4Core::glPauseTransformFeedback ()
{
  table_->glPauseTransformFeedback ();
}


inline void GLFunctions4Core::glResumeTransformFeedback ()
{
  table_->glResumeTransformFeedback ();
}


inline void GLFunctions4Core::glDrawTransformFeedback

  ( GLenum  mode,
    GLuint  id )

{
  table_->glDrawTransformFeedback ( mode, id );
}


inline void GLFunctions4Core::glDrawTransformFeedbackStream

  ( GLenum  mode,
    GLuint  id,
    GLuint  stream )

{
  table_->glDrawTransformFeedbackStream ( mode, id, stream );
}


inline void GLFunctions4Core::glBeginQueryIndexed

  ( GLenum  target,
    GLuint  index,
    GLuint  id )

{
  table_->glBeginQueryIndexed ( target, index, id );
}


inline void GLFunctions4Core::glEndQueryIndexed

  ( GLenum  target,
    GLuint  index )

{
  table_->glEndQueryIndexed ( target, index );
}


inline void GLFunctions4Core::glGetQueryIndexed

  ( GLenum  target,
    GLuint  index,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetQueryIndexediv ( target, index, pname, params );
}


JEM_END_PACKAGE( gl )

#endif
