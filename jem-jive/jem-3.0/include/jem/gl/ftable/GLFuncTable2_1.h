
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

#ifndef JEM_GL_GLFUNCTABLE2_1_H
#define JEM_GL_GLFUNCTABLE2_1_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable2_1Core
//-----------------------------------------------------------------------

// OpenGL 2.1 core functions.

class GLFuncTable2_1Core : public Object
{
 public:

  explicit                GLFuncTable2_1Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glUniformMatrix2x3fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniformMatrix3x2fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniformMatrix2x4fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniformMatrix4x2fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniformMatrix3x4fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  void                   (JEM_APIENTRYP glUniformMatrix4x3fv)

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );


 protected:

  virtual                ~GLFuncTable2_1Core  ();

};


JEM_END_PACKAGE( gl )

#endif
