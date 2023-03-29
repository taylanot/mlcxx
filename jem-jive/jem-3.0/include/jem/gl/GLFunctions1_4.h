
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

#ifndef JEM_GL_GLFUNCTIONS1_4_H
#define JEM_GL_GLFUNCTIONS1_4_H

#include <jem/gl/GLFunctions1_3.h>
#include <jem/gl/ftable/GLFuncTable1_4.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class GLFunctions1_4Core
//-----------------------------------------------------------------------


class GLFunctions1_4Core : public GLFunctions1_3Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_4_CORE;

  explicit                GLFunctions1_4Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions1_4Core  ();


 public:

  inline void             glBlendFuncSeparate

    ( GLenum                sfactorRGB,
      GLenum                dfactorRGB,
      GLenum                sfactorAlpha,
      GLenum                dfactorAlpha );

  inline void             glMultiDrawArrays

    ( GLenum                mode,
      const GLint*          first,
      const GLsizei*        count,
      GLsizei               drawcount );

  inline void             glMultiDrawElements

    ( GLenum                mode,
      const GLsizei*        count,
      GLenum                type,
      const void* const*    indices,
      GLsizei               drawcount );

  inline void             glPointParameter

    ( GLenum                pname,
      GLfloat               param );

  inline void             glPointParameter

    ( GLenum                pname,
      const GLfloat*        params );

  inline void             glPointParameter

    ( GLenum                pname,
      GLint                 param );

  inline void             glPointParameter

    ( GLenum                pname,
      const GLint*          params );

  inline void             glBlendColor

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue,
      GLfloat               alpha );

  inline void             glBlendEquation

    ( GLenum                mode );


 private:

  Ref
    <GLFuncTable1_4Core>  table_;

};


typedef
  GLFunctions1_4Core      GLFuncs1_4Core;


//-----------------------------------------------------------------------
//   class GLFunctions1_4
//-----------------------------------------------------------------------


class GLFunctions1_4 : public GLFunctions1_4Core,
                       public GLFunctions1Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_4;

  explicit                GLFunctions1_4

    ( GLFuncTables&         tables );

};


typedef GLFunctions1_4    GLFuncs1_4;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions1_4Core
//=======================================================================


inline void GLFunctions1_4Core::glBlendFuncSeparate

  ( GLenum  sfactorRGB,
    GLenum  dfactorRGB,
    GLenum  sfactorAlpha,
    GLenum  dfactorAlpha )

{
  table_->glBlendFuncSeparate ( sfactorRGB,   dfactorRGB,
                                sfactorAlpha, dfactorAlpha );
}


inline void GLFunctions1_4Core::glMultiDrawArrays

  ( GLenum          mode,
    const GLint*    first,
    const GLsizei*  count,
    GLsizei         drawcount )

{
  table_->glMultiDrawArrays ( mode, first, count, drawcount );
}


inline void GLFunctions1_4Core::glMultiDrawElements

  ( GLenum              mode,
    const GLsizei*      count,
    GLenum              type,
    const void* const*  indices,
    GLsizei             drawcount )

{
  table_->glMultiDrawElements ( mode, count, type,
                                indices, drawcount );
}


inline void GLFunctions1_4Core::glPointParameter

  ( GLenum   pname,
    GLfloat  param )

{
  table_->glPointParameterf ( pname, param );
}


inline void GLFunctions1_4Core::glPointParameter

  ( GLenum          pname,
    const GLfloat*  params )

{
  table_->glPointParameterfv ( pname, params );
}


inline void GLFunctions1_4Core::glPointParameter

  ( GLenum  pname,
    GLint   param )

{
  table_->glPointParameteri ( pname, param );
}


inline void GLFunctions1_4Core::glPointParameter

  ( GLenum        pname,
    const GLint*  params )

{
  table_->glPointParameteriv ( pname, params );
}


inline void GLFunctions1_4Core::glBlendColor

  ( GLfloat  red,
    GLfloat  green,
    GLfloat  blue,
    GLfloat  alpha )

{
  table_->glBlendColor ( red, green, blue, alpha );
}


inline void GLFunctions1_4Core::glBlendEquation ( GLenum mode )
{
  table_->glBlendEquation ( mode );
}


JEM_END_PACKAGE( gl )

#endif
