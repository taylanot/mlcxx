
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

#ifndef JEM_GL_GLFUNCTIONS1_5_H
#define JEM_GL_GLFUNCTIONS1_5_H

#include <jem/gl/GLFunctions1_4.h>
#include <jem/gl/ftable/GLFuncTable1_5.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class GLFunctions1_5Core
//-----------------------------------------------------------------------


class GLFunctions1_5Core : public GLFunctions1_4Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_5_CORE;

  explicit                GLFunctions1_5Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions1_5Core  ();


 public:

  inline void             glGenQueries

    ( GLsizei               n,
      GLuint*               ids );

  inline void             glDeleteQueries

    ( GLsizei               n,
      const GLuint*         ids );

  inline GLboolean        glIsQuery

    ( GLuint                id );

  inline void             glBeginQuery

    ( GLenum                target,
      GLuint                id );

  inline void             glEndQuery

    ( GLenum                target );

  inline void             glGetQuery

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  inline void             glGetQueryObject

    ( GLuint                id,
      GLenum                pname,
      GLint*                params );

  inline void             glGetQueryObject

    ( GLuint                id,
      GLenum                pname,
      GLuint*               params );

  inline void             glBindBuffer

    ( GLenum                target,
      GLuint                buffer );

  inline void             glDeleteBuffers

    ( GLsizei               n,
      const GLuint*         buffers );

  inline void             glGenBuffers

    ( GLsizei               n,
      GLuint*               buffers );

  inline GLboolean        glIsBuffer

    ( GLuint                buffer );

  inline void             glBufferData

    ( GLenum                target,
      GLsizeiptr            size,
      const void*           data,
      GLenum                usage );

  inline void             glBufferSubData

    ( GLenum                target,
      GLintptr              offset,
      GLsizeiptr            size,
      const void*           data );

  inline void             glGetBufferSubData

    ( GLenum                target,
      GLintptr              offset,
      GLsizeiptr            size,
      void*                 data );

  inline void             glMapBuffer

    ( GLenum                target,
      GLenum                access );

  inline GLboolean        glUnmapBuffer

    ( GLenum                target );

  inline void             glGetBufferParameter

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  inline void             glGetBufferPointer

    ( GLenum                target,
      GLenum                pname,
      void**                params );


 private:

  Ref
    <GLFuncTable1_5Core>  table_;

};


typedef
  GLFunctions1_5Core      GLFuncs1_5Core;


//-----------------------------------------------------------------------
//   class GLFunctions1_4
//-----------------------------------------------------------------------


class GLFunctions1_5 : public GLFunctions1_5Core,
                       public GLFunctions1Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_5;

  explicit                GLFunctions1_5

    ( GLFuncTables&         tables );

};


typedef GLFunctions1_5    GLFuncs1_5;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions1_5Core
//=======================================================================


inline void GLFunctions1_5Core::glGenQueries

  ( GLsizei  n,
    GLuint*  ids )

{
  table_->glGenQueries ( n, ids );
}


inline void GLFunctions1_5Core::glDeleteQueries

  ( GLsizei        n,
    const GLuint*  ids )

{
  table_->glDeleteQueries ( n, ids );
}


inline GLboolean GLFunctions1_5Core::glIsQuery ( GLuint id )
{
  return table_->glIsQuery ( id );
}


inline void GLFunctions1_5Core::glBeginQuery

  ( GLenum  target,
    GLuint  id )

{
  table_->glBeginQuery ( target, id );
}

inline void GLFunctions1_5Core::glEndQuery ( GLenum target )
{
  table_->glEndQuery ( target );
}


inline void GLFunctions1_5Core::glGetQuery

  ( GLenum  target,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetQueryiv ( target, pname, params );
}


inline void GLFunctions1_5Core::glGetQueryObject

  ( GLuint  id,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetQueryObjectiv ( id, pname, params );
}


inline void GLFunctions1_5Core::glGetQueryObject

  ( GLuint   id,
    GLenum   pname,
    GLuint*  params )

{
  table_->glGetQueryObjectuiv ( id, pname, params );
}


inline void GLFunctions1_5Core::glBindBuffer

  ( GLenum  target,
    GLuint  buffer )

{
  table_->glBindBuffer ( target, buffer );
}


inline void GLFunctions1_5Core::glDeleteBuffers

  ( GLsizei        n,
    const GLuint*  buffers )

{
  table_->glDeleteBuffers ( n, buffers );
}


inline void GLFunctions1_5Core::glGenBuffers

  ( GLsizei  n,
    GLuint*  buffers )

{
  table_->glGenBuffers ( n, buffers );
}


inline GLboolean GLFunctions1_5Core::glIsBuffer ( GLuint buffer )
{
  return table_->glIsBuffer ( buffer );
}


inline void GLFunctions1_5Core::glBufferData

  ( GLenum       target,
    GLsizeiptr   size,
    const void*  data,
    GLenum       usage )

{
  table_->glBufferData ( target, size, data, usage );
}


inline void GLFunctions1_5Core::glBufferSubData

  ( GLenum       target,
    GLintptr     offset,
    GLsizeiptr   size,
    const void*  data )

{
  table_->glBufferSubData ( target, offset, size, data );
}


inline void GLFunctions1_5Core::glGetBufferSubData

  ( GLenum      target,
    GLintptr    offset,
    GLsizeiptr  size,
    void*       data )

{
  table_->glGetBufferSubData ( target, offset, size, data );
}


inline void GLFunctions1_5Core::glMapBuffer

  ( GLenum  target,
    GLenum  access )

{
  table_->glMapBuffer ( target, access );
}


inline GLboolean GLFunctions1_5Core::glUnmapBuffer ( GLenum target )
{
  return table_->glUnmapBuffer ( target );
}


inline void GLFunctions1_5Core::glGetBufferParameter

  ( GLenum  target,
    GLenum  pname,
    GLint*  params )

{
  table_->glGetBufferParameteriv ( target, pname, params );
}


inline void GLFunctions1_5Core::glGetBufferPointer

  ( GLenum  target,
    GLenum  pname,
    void**  params )

{
  table_->glGetBufferPointerv ( target, pname, params );
}


JEM_END_PACKAGE( gl )

#endif
