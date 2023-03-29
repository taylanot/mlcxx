
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

#ifndef JEM_GL_GLFUNCTIONS3_3_H
#define JEM_GL_GLFUNCTIONS3_3_H

#include <jem/gl/GLFunctions3_2.h>
#include <jem/gl/ftable/GLFuncTable3_3.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions3_3Core
//-----------------------------------------------------------------------


class GLFunctions3_3Core : public GLFunctions3_2Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_3_3_CORE;

  explicit                GLFunctions3_3Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions3_3Core  ();


 public:

  using GLFunctions3_2Core::glGetQueryObject;

  inline void             glBindFragDataLocationIndexed

    ( GLuint                program,
      GLuint                colorNumber,
      GLuint                index,
      const GLchar*         name );

  inline GLint            glGetFragDataIndex

    ( GLuint                program,
      const GLchar*         name );

  inline void             glGenSamplers

    ( GLsizei               count,
      GLuint*               samplers );

  inline void             glDeleteSamplers

    ( GLsizei               count,
      const GLuint*         samplers );

  inline GLboolean        glIsSampler

    ( GLuint                sampler );

  inline void             glBindSampler

    ( GLuint                unit,
      GLuint                sampler );

  inline void             glSamplerParameter

    ( GLuint                sampler,
      GLenum                pname,
      GLint                 param );

  inline void             glSamplerParameter

    ( GLuint                sampler,
      GLenum                pname,
      const GLint*          param );

  inline void             glSamplerParameter

    ( GLuint                sampler,
      GLenum                pname,
      GLfloat               param );

  inline void             glSamplerParameter

    ( GLuint                sampler,
      GLenum                pname,
      const GLfloat*        param );

  inline void             glSamplerParameterI

    ( GLuint                sampler,
      GLenum                pname,
      const GLint*          param );

  inline void             glSamplerParameterI

    ( GLuint                sampler,
      GLenum                pname,
      const GLuint*         param );

  inline void             glGetSamplerParameter

    ( GLuint                sampler,
      GLenum                pname,
      GLint*                params );

  inline void             glGetSamplerParameterI

    ( GLuint                sampler,
      GLenum                pname,
      GLint*                params );

  inline void             glGetSamplerParameter

    ( GLuint                sampler,
      GLenum                pname,
      GLfloat*              params );

  inline void             glGetSamplerParameterI

    ( GLuint                sampler,
      GLenum                pname,
      GLuint*               params );

  inline void             glQueryCounter

    ( GLuint                id,
      GLenum                target );

  inline void             glGetQueryObject

    ( GLuint                id,
      GLenum                pname,
      GLint64*              params );

  inline void             glGetQueryObject

    ( GLuint                id,
      GLenum                pname,
      GLuint64*             params );

  inline void             glVertexAttribDivisor

    ( GLuint                index,
      GLuint                divisor );

  inline void             glVertexAttribP1

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      GLuint                value );

  inline void             glVertexAttribP1

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      const GLuint*         value );

  inline void             glVertexAttribP2

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      GLuint                value );

  inline void             glVertexAttribP2

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      const GLuint*         value );

  inline void             glVertexAttribP3

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      GLuint                value );

  inline void             glVertexAttribP3

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      const GLuint*         value );

  inline void             glVertexAttribP4

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      GLuint                value );

  inline void             glVertexAttribP4

    ( GLuint                index,
      GLenum                type,
      GLboolean             normalized,
      const GLuint*         value );


 private:

  Ref
    <GLFuncTable3_3Core>  table_;

};


typedef
  GLFunctions3_3Core      GLFuncs3_3Core;


//-----------------------------------------------------------------------
//   class GLFunctions3_3
//-----------------------------------------------------------------------


class GLFunctions3_3 : public GLFunctions3_3Core,
                       public GLFunctions3Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_3_3;

  explicit                GLFunctions3_3

    ( GLFuncTables&         tables );

};


typedef GLFunctions3_3    GLFuncs3_3;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions3_3Core
//=======================================================================


inline void GLFunctions3_3Core::glBindFragDataLocationIndexed

  ( GLuint         program,
    GLuint         colorNumber,
    GLuint         index,
    const GLchar*  name )

{
  table_->glBindFragDataLocationIndexed ( program, colorNumber,
                                          index,   name );
}


inline GLint GLFunctions3_3Core::glGetFragDataIndex

  ( GLuint         program,
    const GLchar*  name )

{
  return table_->glGetFragDataIndex ( program, name );
}


inline void GLFunctions3_3Core::glGenSamplers

  ( GLsizei  count,
    GLuint*  samplers )

{
  table_->glGenSamplers ( count, samplers );
}


inline void GLFunctions3_3Core::glDeleteSamplers

  ( GLsizei        count,
    const GLuint*  samplers )

{
  table_->glDeleteSamplers ( count, samplers );
}


inline GLboolean GLFunctions3_3Core::glIsSampler

  ( GLuint  sampler )

{
  return table_->glIsSampler ( sampler );
}


inline void GLFunctions3_3Core::glBindSampler

  ( GLuint  unit,
    GLuint  sampler )

{
  table_->glBindSampler ( unit, sampler );
}


inline void GLFunctions3_3Core::glSamplerParameter

  ( GLuint  sampler,
    GLenum  pname,
    GLint   param )

{
  table_->glSamplerParameteri ( sampler, pname, param );
}


inline void GLFunctions3_3Core::glSamplerParameter

  ( GLuint        sampler,
    GLenum        pname,
    const GLint*  param )

{
  table_->glSamplerParameteriv ( sampler, pname, param );
}


inline void GLFunctions3_3Core::glSamplerParameter

  ( GLuint   sampler,
    GLenum   pname,
    GLfloat  param )

{
  table_->glSamplerParameterf ( sampler, pname, param );
}


inline void GLFunctions3_3Core::glSamplerParameter

  ( GLuint          sampler,
    GLenum          pname,
    const GLfloat*  param )

{
  table_->glSamplerParameterfv ( sampler, pname, param );
}


inline void GLFunctions3_3Core::glSamplerParameterI

  ( GLuint        sampler,
    GLenum        pname,
    const GLint*  param )

{
  table_->glSamplerParameterIiv ( sampler, pname, param );
}


inline void GLFunctions3_3Core::glSamplerParameterI

  ( GLuint         sampler,
    GLenum         pname,
    const GLuint*  param )

{
  table_->glSamplerParameterIuiv ( sampler, pname, param );
}


inline void GLFunctions3_3Core::glGetSamplerParameter

  ( GLuint  sampler,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetSamplerParameteriv ( sampler, pname, params );
}


inline void GLFunctions3_3Core::glGetSamplerParameterI

  ( GLuint  sampler,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetSamplerParameterIiv ( sampler, pname, params );
}


inline void GLFunctions3_3Core::glGetSamplerParameter

  ( GLuint    sampler,
    GLenum    pname,
    GLfloat*  params )

{
  table_->glGetSamplerParameterfv ( sampler, pname, params );
}


inline void GLFunctions3_3Core::glGetSamplerParameterI

  ( GLuint   sampler,
    GLenum   pname,
    GLuint*  params )

{
  table_->glGetSamplerParameterIuiv ( sampler, pname, params );
}


inline void GLFunctions3_3Core::glQueryCounter

  ( GLuint  id,
    GLenum  target )

{
  table_->glQueryCounter ( id, target );
}


inline void GLFunctions3_3Core::glGetQueryObject

  ( GLuint    id,
    GLenum    pname,
    GLint64*  params )

{
  table_->glGetQueryObjecti64v ( id, pname, params );
}


inline void GLFunctions3_3Core::glGetQueryObject

  ( GLuint     id,
    GLenum     pname,
    GLuint64*  params )

{
  table_->glGetQueryObjectui64v ( id, pname, params );
}


inline void GLFunctions3_3Core::glVertexAttribDivisor

  ( GLuint  index,
    GLuint  divisor )

{
  table_->glVertexAttribDivisor ( index, divisor );
}


inline void GLFunctions3_3Core::glVertexAttribP1

  ( GLuint     index,
    GLenum     type,
    GLboolean  normalized,
    GLuint     value )

{
  table_->glVertexAttribP1ui ( index, type, normalized, value );
}


inline void GLFunctions3_3Core::glVertexAttribP1

  ( GLuint         index,
    GLenum         type,
    GLboolean      normalized,
    const GLuint*  value )

{
  table_->glVertexAttribP1uiv ( index, type, normalized, value );
}


inline void GLFunctions3_3Core::glVertexAttribP2

  ( GLuint     index,
    GLenum     type,
    GLboolean  normalized,
    GLuint     value )

{
  table_->glVertexAttribP2ui ( index, type, normalized, value );
}


inline void GLFunctions3_3Core::glVertexAttribP2

  ( GLuint         index,
    GLenum         type,
    GLboolean      normalized,
    const GLuint*  value )

{
  table_->glVertexAttribP2uiv ( index, type, normalized, value );
}


inline void GLFunctions3_3Core::glVertexAttribP3

  ( GLuint     index,
    GLenum     type,
    GLboolean  normalized,
    GLuint     value )

{
  table_->glVertexAttribP3ui ( index, type, normalized, value );
}


inline void GLFunctions3_3Core::glVertexAttribP3

  ( GLuint         index,
    GLenum         type,
    GLboolean      normalized,
    const GLuint*  value )

{
  table_->glVertexAttribP3uiv ( index, type, normalized, value );
}


inline void GLFunctions3_3Core::glVertexAttribP4

  ( GLuint     index,
    GLenum     type,
    GLboolean  normalized,
    GLuint     value )

{
  table_->glVertexAttribP4ui ( index, type, normalized, value );
}


inline void GLFunctions3_3Core::glVertexAttribP4

  ( GLuint         index,
    GLenum         type,
    GLboolean      normalized,
    const GLuint*  value )

{
  table_->glVertexAttribP4uiv ( index, type, normalized, value );
}


JEM_END_PACKAGE( gl )

#endif
