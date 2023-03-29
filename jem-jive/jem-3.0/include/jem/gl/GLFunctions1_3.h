
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

#ifndef JEM_GL_GLFUNCTIONS1_3_H
#define JEM_GL_GLFUNCTIONS1_3_H

#include <jem/gl/GLFunctions1_2.h>
#include <jem/gl/ftable/GLFuncTable1_3.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class GLFunctions1_3Core
//-----------------------------------------------------------------------


class GLFunctions1_3Core : public GLFunctions1_2Core
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_3_CORE;

  explicit                GLFunctions1_3Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions1_3Core  ();


 public:

  inline void             glActiveTexture

    ( GLenum                texture );

  inline void             glSampleCoverage

    ( GLfloat               value,
      GLboolean             invert );

  inline void             glCompressedTexImage3D

    ( GLenum                target,
      GLint                 level,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLint                 border,
      GLsizei               imageSize,
      const void*           data );

  inline void             glCompressedTexImage2D

    ( GLenum                target,
      GLint                 level,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLint                 border,
      GLsizei               imageSize,
      const void*           data );

  inline void             glCompressedTexImage1D

    ( GLenum                target,
      GLint                 level,
      GLenum                internalformat,
      GLsizei               width,
      GLint                 border,
      GLsizei               imageSize,
      const void*           data );

  inline void             glCompressedTexSubImage3D

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

  inline void             glCompressedTexSubImage2D

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLint                 yoffset,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  inline void             glCompressedTexSubImage1D

    ( GLenum                target,
      GLint                 level,
      GLint                 xoffset,
      GLsizei               width,
      GLenum                format,
      GLsizei               imageSize,
      const void*           data );

  inline void             glGetCompressedTexImage

    ( GLenum                target,
      GLint                 level,
      void*                 img );


 private:

  Ref
    <GLFuncTable1_3Core>  table_;

};


typedef
  GLFunctions1_3Core      GLFuncs1_3Core;


//-----------------------------------------------------------------------
//   class GLFunctions1_3
//-----------------------------------------------------------------------


class GLFunctions1_3 : public GLFunctions1_3Core,
                       public GLFunctions1Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_3;

  explicit                GLFunctions1_3

    ( GLFuncTables&         tables );

};


typedef GLFunctions1_3    GLFuncs1_3;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions1_3Core
//=======================================================================


inline void GLFunctions1_3Core::glActiveTexture ( GLenum texture )
{
  table_->glActiveTexture ( texture );
}


inline void GLFunctions1_3Core::glSampleCoverage

  ( GLfloat    value,
    GLboolean  invert )

{
  table_->glSampleCoverage ( value, invert );
}


inline void GLFunctions1_3Core::glCompressedTexImage3D

  ( GLenum       target,
    GLint        level,
    GLenum       internalformat,
    GLsizei      width,
    GLsizei      height,
    GLsizei      depth,
    GLint        border,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTexImage3D ( target, level, internalformat,
                                   width, height, depth, border,
                                   imageSize, data );
}


inline void GLFunctions1_3Core::glCompressedTexImage2D

  ( GLenum       target,
    GLint        level,
    GLenum       internalformat,
    GLsizei      width,
    GLsizei      height,
    GLint        border,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTexImage2D ( target, level, internalformat,
                                   width, height, border,
                                   imageSize, data );
}


inline void GLFunctions1_3Core::glCompressedTexImage1D

  ( GLenum       target,
    GLint        level,
    GLenum       internalformat,
    GLsizei      width,
    GLint        border,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTexImage1D ( target, level, internalformat,
                                   width, border,
                                   imageSize, data );
}


inline void GLFunctions1_3Core::glCompressedTexSubImage3D

  ( GLenum       target,
    GLint        level,
    GLint        xoffset,
    GLint        yoffset,
    GLint        zoffset,
    GLsizei      width,
    GLsizei      height,
    GLsizei      depth,
    GLenum       format,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTexSubImage3D ( target, level,
                                      xoffset, yoffset, zoffset,
                                      width, height, depth,
                                      format, imageSize, data );
}


inline void GLFunctions1_3Core::glCompressedTexSubImage2D

  ( GLenum       target,
    GLint        level,
    GLint        xoffset,
    GLint        yoffset,
    GLsizei      width,
    GLsizei      height,
    GLenum       format,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTexSubImage2D ( target, level,
                                      xoffset, yoffset,
                                      width, height,
                                      format, imageSize, data );
}


inline void GLFunctions1_3Core::glCompressedTexSubImage1D

  ( GLenum       target,
    GLint        level,
    GLint        xoffset,
    GLsizei      width,
    GLenum       format,
    GLsizei      imageSize,
    const void*  data )

{
  table_->glCompressedTexSubImage1D ( target, level,
                                      xoffset,
                                      width,
                                      format, imageSize, data );
}


inline void GLFunctions1_3Core::glGetCompressedTexImage

  ( GLenum  target,
    GLint   level,
    void*   img )

{
  table_->glGetCompressedTexImage ( target, level, img );
}


JEM_END_PACKAGE( gl )

#endif
