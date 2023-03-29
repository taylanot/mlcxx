
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

#ifndef JEM_GL_GLFUNCTABLE1_3_H
#define JEM_GL_GLFUNCTABLE1_3_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable1_3Core
//-----------------------------------------------------------------------

// OpenGL 1.3 core functions.

class GLFuncTable1_3Core : public Object
{
 public:

  explicit                GLFuncTable1_3Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glActiveTexture)

    ( GLenum                texture );

  void                   (JEM_APIENTRYP glSampleCoverage)

    ( GLfloat               value,
      GLboolean             invert );

  void                   (JEM_APIENTRYP glCompressedTexImage3D)

    ( GLenum                target,
      GLint                 level,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLint                 border,
      GLsizei               imageSize,
      const void*           data );

  void                   (JEM_APIENTRYP glCompressedTexImage2D)

    ( GLenum                target,
      GLint                 level,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLint                 border,
      GLsizei               imageSize,
      const void*           data );

  void                   (JEM_APIENTRYP glCompressedTexImage1D)

    ( GLenum                target,
      GLint                 level,
      GLenum                internalformat,
      GLsizei               width,
      GLint                 border,
      GLsizei               imageSize,
      const void*           data );

  void                   (JEM_APIENTRYP glCompressedTexSubImage3D)

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLint                 zoffset,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  void                   (JEM_APIENTRYP glCompressedTexSubImage2D)

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  void                   (JEM_APIENTRYP glCompressedTexSubImage1D)

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLsizei               width,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  void                   (JEM_APIENTRYP glGetCompressedTexImage)

    ( GLenum                target,
      GLint                 level,
      void*                 img );


 protected:

  virtual                ~GLFuncTable1_3Core  ();

};


JEM_END_PACKAGE( gl )

#endif
