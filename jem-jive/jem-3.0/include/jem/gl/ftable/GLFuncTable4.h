
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

#ifndef JEM_GL_GLFUNCTABLE4_H
#define JEM_GL_GLFUNCTABLE4_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable4Core
//-----------------------------------------------------------------------

// OpenGL 4.0 core functions.

class GLFuncTable4Core : public Object
{
 public:

  explicit                GLFuncTable4Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glMinSampleShading)

    ( GLfloat               value );

  void                   (JEM_APIENTRYP glBlendEquationi)

    ( GLuint                buf,
      GLenum                mode );

  void                   (JEM_APIENTRYP glBlendEquationSeparatei)

    ( GLuint                buf,
      GLenum                modeRGB,
      GLenum                modeAlpha );

  void                   (JEM_APIENTRYP glBlendFunci)

    ( GLuint                buf,
      GLenum                src,
      GLenum                dst );

  void                   (JEM_APIENTRYP glBlendFuncSeparatei)

    ( GLuint                buf,
      GLenum                srcRGB,
      GLenum                dstRGB,
      GLenum                srcAlpha,
      GLenum                dstAlpha );

  void                   (JEM_APIENTRYP glDrawArraysIndirect)

    ( GLenum                mode,
      const void*           indirect );

  void                   (JEM_APIENTRYP glDrawElementsIndirect)

    ( GLenum                mode,
      GLenum                type,
      const void*           indirect );

  void                   (JEM_APIENTRYP glUniform1d)

    ( GLint                 location,
      GLdouble              x );

  void                   (JEM_APIENTRYP glUniform2d)

    ( GLint                 location,
      GLdouble              x,
      GLdouble              y );

  void                   (JEM_APIENTRYP glUniform3d)

    ( GLint                 location,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                   (JEM_APIENTRYP glUniform4d)

    ( GLint                 location,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  void                   (JEM_APIENTRYP glUniform1dv)

    ( GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniform2dv)

    ( GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniform3dv)

    ( GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniform4dv)

    ( GLint                 location,
      GLsizei               count,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix2dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix3dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix4dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix2x3dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix2x4dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix3x2dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix3x4dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix4x2dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glUniformMatrix4x3dv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLdouble*       value );

  void                   (JEM_APIENTRYP glGetUniformdv)

    ( GLuint                program,
      GLint                 location,
      GLdouble*             params );

  GLint                  (JEM_APIENTRYP
                          glGetSubroutineUniformLocation)

    ( GLuint                program,
      GLenum                shadertype,
      const GLchar*         name );

  GLuint                 (JEM_APIENTRYP glGetSubroutineIndex)

    ( GLuint                program,
      GLenum                shadertype,
      const GLchar*         name );

  void                   (JEM_APIENTRYP
                          glGetActiveSubroutineUniformiv)

    ( GLuint                program,
      GLenum                shadertype,
      GLuint                index,
      GLenum                pname,
      GLint*                values );

  void                   (JEM_APIENTRYP
                          glGetActiveSubroutineUniformName)

    ( GLuint                program,
      GLenum                shadertype,
      GLuint                index,
      GLsizei               bufsize,
      GLsizei*              length,
      GLchar*               name );

  void                   (JEM_APIENTRYP glGetActiveSubroutineName)

    ( GLuint                program,
      GLenum                shadertype,
      GLuint                index,
      GLsizei               bufsize,
      GLsizei*              length,
      GLchar*               name );

  void                   (JEM_APIENTRYP glUniformSubroutinesuiv)

    ( GLenum                shadertype,
      GLsizei               count,
      const GLuint*         indices );

  void                   (JEM_APIENTRYP glGetUniformSubroutineuiv)

    ( GLenum                shadertype,
      GLint                 location,
      GLuint*               params );

  void                   (JEM_APIENTRYP glGetProgramStageiv)

    ( GLuint                program,
      GLenum                shadertype,
      GLenum                pname,
      GLint*                values );

  void                   (JEM_APIENTRYP glPatchParameteri)

    ( GLenum                pname,
      GLint                 value );

  void                   (JEM_APIENTRYP glPatchParameterfv)

    ( GLenum                pname,
      const GLfloat*        values );

  void                   (JEM_APIENTRYP glBindTransformFeedback)

    ( GLenum                target,
      GLuint                id );

  void                   (JEM_APIENTRYP glDeleteTransformFeedbacks)

    ( GLsizei               n,
      const GLuint*         ids );

  void                   (JEM_APIENTRYP glGenTransformFeedbacks)

    ( GLsizei               n,
      GLuint*               ids );

  GLboolean              (JEM_APIENTRYP glIsTransformFeedback)

    ( GLuint                id );

  void                   (JEM_APIENTRYP glPauseTransformFeedback) ();

  void                   (JEM_APIENTRYP glResumeTransformFeedback) ();

  void                   (JEM_APIENTRYP glDrawTransformFeedback)

    ( GLenum                mode,
      GLuint                id );

  void                   (JEM_APIENTRYP glDrawTransformFeedbackStream)

    ( GLenum                mode,
      GLuint                id,
      GLuint                stream );

  void                   (JEM_APIENTRYP glBeginQueryIndexed)

    ( GLenum                target,
      GLuint                index,
      GLuint                id );

  void                   (JEM_APIENTRYP glEndQueryIndexed)

    ( GLenum                target,
      GLuint                index );

  void                   (JEM_APIENTRYP glGetQueryIndexediv)

    ( GLenum                target,
      GLuint                index,
      GLenum                pname,
      GLint*                params );


 protected:

  virtual                ~GLFuncTable4Core  ();

};


//-----------------------------------------------------------------------
//   class GLFuncTable4Compat
//-----------------------------------------------------------------------

// OpenGL 4.0 deprecated functions.

class GLFuncTable4Compat : public Object
{
 public:

  explicit                GLFuncTable4Compat

    ( GLFuncLoader*         loader );


 public:


 protected:

  virtual                ~GLFuncTable4Compat ();

};


JEM_END_PACKAGE( gl )

#endif
