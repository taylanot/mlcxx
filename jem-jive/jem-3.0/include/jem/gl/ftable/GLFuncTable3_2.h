
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

#ifndef JEM_GL_GLFUNCTABLE3_2_H
#define JEM_GL_GLFUNCTABLE3_2_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable3_2Core
//-----------------------------------------------------------------------

// OpenGL 3.2 core functions.

class GLFuncTable3_2Core : public Object
{
 public:

  explicit                GLFuncTable3_2Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glDrawElementsBaseVertex)

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLint                 basevertex );

  void                   (JEM_APIENTRYP glDrawRangeElementsBaseVertex)

    ( GLenum                mode,
      GLuint                start,
      GLuint                end,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLint                 basevertex );

  void                   (JEM_APIENTRYP
                          glDrawElementsInstancedBaseVertex)

    ( GLenum                mode,
      GLsizei               count,
      GLenum                type,
      const void*           indices,
      GLsizei               instancecount,
      GLint                 basevertex );

  void                   (JEM_APIENTRYP glMultiDrawElementsBaseVertex)

    ( GLenum                mode,
      const GLsizei*        count,
      GLenum                type,
      const void* const*    indices,
      GLsizei               drawcount,
      const GLint*          basevertex );

  void                   (JEM_APIENTRYP glProvokingVertex)

    ( GLenum                mode );

  GLsync                 (JEM_APIENTRYP glFenceSync)

    ( GLenum                condition,
      GLbitfield            flags );

  GLboolean              (JEM_APIENTRYP glIsSync)

    ( GLsync                sync );

  void                   (JEM_APIENTRYP glDeleteSync)

    ( GLsync                sync );

  GLenum                 (JEM_APIENTRYP glClientWaitSync)

    ( GLsync                sync,
      GLbitfield            flags,
      GLuint64              timeout );

  void                   (JEM_APIENTRYP glWaitSync)

    ( GLsync                sync,
      GLbitfield            flags,
      GLuint64              timeout );

  void                   (JEM_APIENTRYP glGetInteger64v)

    ( GLenum                pname,
      GLint64*              data );

  void                   (JEM_APIENTRYP glGetSynciv)

    ( GLsync                sync,
      GLenum                pname,
      GLsizei               bufSize,
      GLsizei*              length,
      GLint*                values );

  void                   (JEM_APIENTRYP glGetInteger64i_v)

    ( GLenum                target,
      GLuint                index,
      GLint64*              data );

  void                   (JEM_APIENTRYP glGetBufferParameteri64v)

    ( GLenum                target,
      GLenum                pname,
      GLint64*              params );

  void                   (JEM_APIENTRYP glFramebufferTexture)

    ( GLenum                target,
      GLenum                attachment,
      GLuint                texture,
      GLint                 level );

  void                   (JEM_APIENTRYP glTexImage2DMultisample)

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLboolean             fixedsamplelocations );

  void                   (JEM_APIENTRYP glTexImage3DMultisample)

    ( GLenum                target,
      GLsizei               samples,
      GLenum                internalformat,
      GLsizei               width,
      GLsizei               height,
      GLsizei               depth,
      GLboolean             fixedsamplelocations );

  void                   (JEM_APIENTRYP glGetMultisamplefv)

    ( GLenum                pname,
      GLuint                index,
      GLfloat*              val );

  void                   (JEM_APIENTRYP glSampleMaski)

    ( GLuint                maskNumber,
      GLbitfield            mask );


 protected:

  virtual                ~GLFuncTable3_2Core  ();

};


JEM_END_PACKAGE( gl )

#endif
