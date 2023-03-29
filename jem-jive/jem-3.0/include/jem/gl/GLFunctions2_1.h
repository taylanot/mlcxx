
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

#ifndef JEM_GL_GLFUNCTIONS2_1_H
#define JEM_GL_GLFUNCTIONS2_1_H

#include <jem/gl/GLFunctions2.h>
#include <jem/gl/ftable/GLFuncTable2_1.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions2_1Core
//-----------------------------------------------------------------------


class GLFunctions2_1Core : public GLFunctions2Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_2_1_CORE;

  explicit                GLFunctions2_1Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions2_1Core  ();


 public:

  inline void             glUniformMatrix2x3

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glUniformMatrix3x2

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glUniformMatrix2x4

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glUniformMatrix4x2

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glUniformMatrix3x4

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );

  inline void             glUniformMatrix4x3

    ( GLint                 location,
      GLsizei               count,
      GLboolean             transpose,
      const GLfloat*        value );


 private:

  Ref
    <GLFuncTable2_1Core>  table_;

};


typedef
  GLFunctions2_1Core      GLFuncs2_1Core;


//-----------------------------------------------------------------------
//   class GLFunctions2_1
//-----------------------------------------------------------------------


class GLFunctions2_1 : public GLFunctions2_1Core,
                       public GLFunctions2Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_2_1;

  explicit                GLFunctions2_1

    ( GLFuncTables&         tables );

};


typedef GLFunctions2_1    GLFuncs2_1;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions2_1Core
//=======================================================================



inline void GLFunctions2_1Core::glUniformMatrix2x3

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix2x3fv ( location, count, transpose, value );
}


inline void GLFunctions2_1Core::glUniformMatrix3x2

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix3x2fv ( location, count, transpose, value );
}


inline void GLFunctions2_1Core::glUniformMatrix2x4

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix2x4fv ( location, count, transpose, value );
}


inline void GLFunctions2_1Core::glUniformMatrix4x2

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix4x2fv ( location, count, transpose, value );
}


inline void GLFunctions2_1Core::glUniformMatrix3x4

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix3x4fv ( location, count, transpose, value );
}


inline void GLFunctions2_1Core::glUniformMatrix4x3

  ( GLint           location,
    GLsizei         count,
    GLboolean       transpose,
    const GLfloat*  value )

{
  table_->glUniformMatrix4x3fv ( location, count, transpose, value );
}


JEM_END_PACKAGE( gl )

#endif
