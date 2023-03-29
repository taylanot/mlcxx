
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

#ifndef JEM_GL_GLFUNCTABLE1_4_H
#define JEM_GL_GLFUNCTABLE1_4_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable1_4Core
//-----------------------------------------------------------------------

// OpenGL 1.4 core functions.

class GLFuncTable1_4Core : public Object
{
 public:

  explicit                GLFuncTable1_4Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glBlendFuncSeparate)

    ( GLenum                sfactorRGB,
      GLenum                dfactorRGB,
      GLenum                sfactorAlpha,
      GLenum                dfactorAlpha );

  void                   (JEM_APIENTRYP glMultiDrawArrays)

    ( GLenum                mode,
      const GLint*          first,
      const GLsizei*        count,
      GLsizei               drawcount );

  void                   (JEM_APIENTRYP glMultiDrawElements)

    ( GLenum                mode,
      const GLsizei*        count,
      GLenum                type,
      const void* const*    indices,
      GLsizei               drawcount );

  void                   (JEM_APIENTRYP glPointParameterf)

    ( GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glPointParameterfv)

    ( GLenum                pname,
      const GLfloat*        params );

  void                   (JEM_APIENTRYP glPointParameteri)

    ( GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glPointParameteriv)

    ( GLenum                pname,
      const GLint*          params );

  void                   (JEM_APIENTRYP glBlendColor)

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue,
      GLfloat               alpha );

  void                   (JEM_APIENTRYP glBlendEquation)

    ( GLenum                mode );


 protected:

  virtual                ~GLFuncTable1_4Core  ();

};


JEM_END_PACKAGE( gl )

#endif
