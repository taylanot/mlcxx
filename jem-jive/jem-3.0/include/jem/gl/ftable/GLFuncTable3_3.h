
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

#ifndef JEM_GL_GLFUNCTABLE3_3_H
#define JEM_GL_GLFUNCTABLE3_3_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable3_3Core
//-----------------------------------------------------------------------

// OpenGL 3.3 core functions.

class GLFuncTable3_3Core : public Object
{
 public:

  explicit                GLFuncTable3_3Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glBindFragDataLocationIndexed)

    ( GLuint                program,
      GLuint                colorNumber,
      GLuint                index,
      const GLchar*         name );

  GLint                  (JEM_APIENTRYP glGetFragDataIndex)

    ( GLuint                program,
      const GLchar*         name );

  void                   (JEM_APIENTRYP glGenSamplers)

    ( GLsizei               count,
      GLuint*               samplers );

  void                   (JEM_APIENTRYP glDeleteSamplers)

    ( GLsizei               count,
      const GLuint*         samplers );

  GLboolean              (JEM_APIENTRYP glIsSampler)

    ( GLuint                sampler );

  void                   (JEM_APIENTRYP glBindSampler)

    ( GLuint                unit,
      GLuint                sampler );

  void                   (JEM_APIENTRYP glSamplerParameteri)

    ( GLuint                sampler,
      GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glSamplerParameteriv)

    ( GLuint                sampler,
      GLenum                pname,
      const GLint*          param );

  void                   (JEM_APIENTRYP glSamplerParameterf)

    ( GLuint                sampler,
      GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glSamplerParameterfv)

    ( GLuint                sampler,
      GLenum                pname,
      const GLfloat*        param );

  void                   (JEM_APIENTRYP glSamplerParameterIiv)

    ( GLuint                sampler,
      GLenum                pname,
      const GLint*          param );

  void                   (JEM_APIENTRYP glSamplerParameterIuiv)

    ( GLuint                sampler,
      GLenum                pname,
      const GLuint*         param );

  void                   (JEM_APIENTRYP glGetSamplerParameteriv)

    ( GLuint                sampler,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetSamplerParameterIiv)

    ( GLuint                sampler,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetSamplerParameterfv)

    ( GLuint                sampler,
      GLenum                pname,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetSamplerParameterIuiv)

    ( GLuint                sampler,
      GLenum                pname,
      GLuint*               params );

  void                   (JEM_APIENTRYP glQueryCounter)

    ( GLuint                id,
      GLenum                target );

  void                   (JEM_APIENTRYP glGetQueryObjecti64v)

    ( GLuint                id,
      GLenum                pname,
      GLint64*              params );

  void                   (JEM_APIENTRYP glGetQueryObjectui64v)

    ( GLuint                id,
      GLenum                pname,
      GLuint64*             params );

  void                   (JEM_APIENTRYP glVertexAttribDivisor)

    ( GLuint                index,
      GLuint                divisor );

  void                   (JEM_APIENTRYP glVertexAttribP1ui)

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      GLuint                value );

  void                   (JEM_APIENTRYP glVertexAttribP1uiv)

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glVertexAttribP2ui)

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      GLuint                value );

  void                   (JEM_APIENTRYP glVertexAttribP2uiv)

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glVertexAttribP3ui)

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      GLuint                value );

  void                   (JEM_APIENTRYP glVertexAttribP3uiv)

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      const GLuint*         value );

  void                   (JEM_APIENTRYP glVertexAttribP4ui)

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      GLuint                value );

  void                   (JEM_APIENTRYP glVertexAttribP4uiv)

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      const GLuint*         value );


 protected:

  virtual                ~GLFuncTable3_3Core  ();

};


JEM_END_PACKAGE( gl )

#endif
