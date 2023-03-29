
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

#ifndef JEM_GL_GLFUNCTABLE1_H
#define JEM_GL_GLFUNCTABLE1_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncLoader;


//-----------------------------------------------------------------------
//   class GLFuncTable1Core
//-----------------------------------------------------------------------

// OpenGL 1.0 core functions.

class GLFuncTable1Core : public Object
{
 public:

  explicit                GLFuncTable1Core

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glViewport)

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glDepthRange)

    ( GLdouble              near,
      GLdouble              far );

  GLboolean              (JEM_APIENTRYP glIsEnabled)

    ( GLenum                cap );

  void                   (JEM_APIENTRYP glGetTexImage)

    ( GLenum                target,
      GLint                 level,
      GLenum                format,
      GLenum                type,
      GLvoid*               pixels );

  const GLubyte*         (JEM_APIENTRYP glGetString)

    ( GLenum                name );

  void                   (JEM_APIENTRYP glGetIntegerv)

    ( GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetFloatv)

    ( GLenum                pname,
      GLfloat*              params );

  GLenum                 (JEM_APIENTRYP glGetError)

    ( void );

  void                   (JEM_APIENTRYP glGetDoublev)

    ( GLenum                pname,
      GLdouble*             params );

  void                   (JEM_APIENTRYP glGetBooleanv)

    ( GLenum                pname,
      GLboolean*            params );

  void                   (JEM_APIENTRYP glGetTexParameteriv)

    ( GLenum                target,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetTexParameterfv)

    ( GLenum                target,
      GLenum                pname,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetTexLevelParameteriv)

    ( GLenum                target,
      GLint                 level,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetTexLevelParameterfv)

    ( GLenum                target,
      GLint                 level,
      GLenum                pname,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glReadPixels)

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      GLvoid*               pixels );

  void                   (JEM_APIENTRYP glReadBuffer)

    ( GLenum                mode );

  void                   (JEM_APIENTRYP glPixelStorei)

    ( GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glPixelStoref)

    ( GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glDepthFunc)

    ( GLenum                func );

  void                   (JEM_APIENTRYP glLogicOp)

    ( GLenum                opcode );

  void                   (JEM_APIENTRYP glBlendFunc)

    ( GLenum                sfactor,
      GLenum                dfactor );

  void                   (JEM_APIENTRYP glFlush)

    ( void );

  void                   (JEM_APIENTRYP glFinish)

    ( void );

  void                   (JEM_APIENTRYP glEnable)

    ( GLenum                cap );

  void                   (JEM_APIENTRYP glDisable)

    ( GLenum                cap );

  void                   (JEM_APIENTRYP glDepthMask)

    ( GLboolean             flag );

  void                   (JEM_APIENTRYP glColorMask)

    ( GLboolean             red,
      GLboolean             green,
      GLboolean             blue,
      GLboolean             alpha );

  void                   (JEM_APIENTRYP glClearDepth)

    ( GLdouble              depth );

  void                   (JEM_APIENTRYP glClearColor)

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue,
      GLfloat               alpha );

  void                   (JEM_APIENTRYP glClear)

    ( GLbitfield            mask );

  void                   (JEM_APIENTRYP glDrawBuffer)

    ( GLenum                mode );

  void                   (JEM_APIENTRYP glTexImage2D)

    ( GLenum                target,
      GLint                 level,
      GLint                 internalformat,
      GLsizei               width,
      GLsizei               height,
      GLint                 border,
      GLenum                format,
      GLenum                type,
      const GLvoid*         pixels );

  void                   (JEM_APIENTRYP glTexImage1D)

    ( GLenum                target,
      GLint                 level,
      GLint                 internalformat,
      GLsizei               width,
      GLint                 border,
      GLenum                format,
      GLenum                type,
      const GLvoid*         pixels );

  void                   (JEM_APIENTRYP glTexParameteriv)

    ( GLenum                target,
      GLenum                pname,
      const GLint*          params );

  void                   (JEM_APIENTRYP glTexParameteri)

    ( GLenum                target,
      GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glTexParameterfv)

    ( GLenum                target,
      GLenum                pname,
      const GLfloat*        params );

  void                   (JEM_APIENTRYP glTexParameterf)

    ( GLenum                target,
      GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glScissor)

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  void                   (JEM_APIENTRYP glPolygonMode)

    ( GLenum                face,
      GLenum                mode );

  void                   (JEM_APIENTRYP glPointSize)

    ( GLfloat               size );

  void                   (JEM_APIENTRYP glLineWidth)

    ( GLfloat               width );

  void                   (JEM_APIENTRYP glHint)

    ( GLenum                target,
      GLenum                mode );

  void                   (JEM_APIENTRYP glFrontFace)

    ( GLenum                mode );

  void                   (JEM_APIENTRYP glCullFace)

    ( GLenum                mode );


 protected:

  virtual                ~GLFuncTable1Core  ();

};


//-----------------------------------------------------------------------
//   class GLFuncTable1Compat
//-----------------------------------------------------------------------

  // OpenGL 1.0 deprecated functions.

class GLFuncTable1Compat : public Object
{
 public:

  explicit                GLFuncTable1Compat

    ( GLFuncLoader*         loader );


 public:

  void                   (JEM_APIENTRYP glTranslatef)

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  void                   (JEM_APIENTRYP glTranslated)

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                   (JEM_APIENTRYP glScalef)

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  void                   (JEM_APIENTRYP glScaled)

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                   (JEM_APIENTRYP glRotatef)

    ( GLfloat               angle,
      GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  void                   (JEM_APIENTRYP glRotated)

    ( GLdouble              angle,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                   (JEM_APIENTRYP glPushMatrix)

    ( void );

  void                   (JEM_APIENTRYP glPopMatrix)

    ( void );

  void                   (JEM_APIENTRYP glOrtho)

    ( GLdouble              left,
      GLdouble              right,
      GLdouble              bottom,
      GLdouble              top,
      GLdouble              zNear,
      GLdouble              zFar );

  void                   (JEM_APIENTRYP glMatrixMode)

    ( GLenum                mode );

  void                   (JEM_APIENTRYP glLoadMatrixd)

    ( const GLdouble*       m );

  void                   (JEM_APIENTRYP glLoadMatrixf)

    ( const GLfloat*        m );

  void                   (JEM_APIENTRYP glLoadIdentity)

    ( void );

  void                   (JEM_APIENTRYP glTexEnviv)

    ( GLenum                target,
      GLenum                pname,
      const GLint*          params );

  void                   (JEM_APIENTRYP glTexEnvi)

    ( GLenum                target,
      GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glTexEnvfv)

    ( GLenum                target,
      GLenum                pname,
      const GLfloat*        params );

  void                   (JEM_APIENTRYP glTexEnvf)

    ( GLenum                target,
      GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glGetMaterialiv)

    ( GLenum                face,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetMaterialfv)

    ( GLenum                face,
      GLenum                pname,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetLightiv)

    ( GLenum                light,
      GLenum                pname,
      GLint*                params );

  void                   (JEM_APIENTRYP glGetLightfv)

    ( GLenum                light,
      GLenum                pname,
      GLfloat*              params );

  void                   (JEM_APIENTRYP glGetClipPlane)

    ( GLenum                plane,
      GLdouble*             equation );

  void                   (JEM_APIENTRYP glDrawPixels)

    ( GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      const GLvoid*         pixels );

  void                   (JEM_APIENTRYP glCopyPixels)

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height,
      GLenum                type );

  void                   (JEM_APIENTRYP glPixelTransferi)

    ( GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glPixelTransferf)

    ( GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glAlphaFunc)

    ( GLenum                func,
      GLfloat               ref );

  void                   (JEM_APIENTRYP glPushAttrib)

    ( GLbitfield            mask );

  void                   (JEM_APIENTRYP glPopAttrib)

    ( void );

  void                   (JEM_APIENTRYP glAccum)

    ( GLenum                op,
      GLfloat               value );

  void                   (JEM_APIENTRYP glClearAccum)

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue,
      GLfloat               alpha );

  void                   (JEM_APIENTRYP glSelectBuffer)

    ( GLsizei               size,
      GLuint*               buffer );

  void                   (JEM_APIENTRYP glShadeModel)

    ( GLenum                mode );

  void                   (JEM_APIENTRYP glMaterialiv)

    ( GLenum                face,
      GLenum                pname,
      const GLint*          params );

  void                   (JEM_APIENTRYP glMateriali)

    ( GLenum                face,
      GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glMaterialfv)

    ( GLenum                face,
      GLenum                pname,
      const GLfloat*        params );

  void                   (JEM_APIENTRYP glMaterialf)

    ( GLenum                face,
      GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glLineStipple)

    ( GLint                 factor,
      GLushort              pattern );

  void                   (JEM_APIENTRYP glLightModeliv)

    ( GLenum                pname,
      const GLint*          params );

  void                   (JEM_APIENTRYP glLightModeli)

    ( GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glLightModelfv)

    ( GLenum                pname,
      const GLfloat*        params );

  void                   (JEM_APIENTRYP glLightModelf)

    ( GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glLightiv)

    ( GLenum                light,
      GLenum                pname,
      const GLint*          params );

  void                   (JEM_APIENTRYP glLighti)

    ( GLenum                light,
      GLenum                pname,
      GLint                 param );

  void                   (JEM_APIENTRYP glLightfv)

    ( GLenum                light,
      GLenum                pname,
      const GLfloat*        params );

  void                   (JEM_APIENTRYP glLightf)

    ( GLenum                light,
      GLenum                pname,
      GLfloat               param );

  void                   (JEM_APIENTRYP glColorMaterial)

    ( GLenum                face,
      GLenum                mode );

  void                   (JEM_APIENTRYP glClipPlane)

    ( GLenum                plane,
      const GLdouble*       equation );

  void                   (JEM_APIENTRYP glVertex4s)

    ( GLshort               x,
      GLshort               y,
      GLshort               z,
      GLshort               w );

  void                   (JEM_APIENTRYP glVertex4i)

    ( GLint                 x,
      GLint                 y,
      GLint                 z,
      GLint                 w );

  void                   (JEM_APIENTRYP glVertex4f)

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z,
      GLfloat               w );

  void                   (JEM_APIENTRYP glVertex4d)

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  void                   (JEM_APIENTRYP glVertex3s)

    ( GLshort               x,
      GLshort               y,
      GLshort               z );

  void                   (JEM_APIENTRYP glVertex3i)

    ( GLint                 x,
      GLint                 y,
      GLint                 z );

  void                   (JEM_APIENTRYP glVertex3f)

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  void                   (JEM_APIENTRYP glVertex3d)

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                   (JEM_APIENTRYP glVertex2s)

    ( GLshort               x,
      GLshort               y );

  void                   (JEM_APIENTRYP glVertex2i)

    ( GLint                 x,
      GLint                 y );

  void                   (JEM_APIENTRYP glVertex2f)

    ( GLfloat               x,
      GLfloat               y );

  void                   (JEM_APIENTRYP glVertex2d)

    ( GLdouble              x,
      GLdouble              y );

  void                   (JEM_APIENTRYP glTexCoord2s)

    ( GLshort               s,
      GLshort               t );

  void                   (JEM_APIENTRYP glTexCoord2i)

    ( GLint                 s,
      GLint                 t );

  void                   (JEM_APIENTRYP glTexCoord2f)

    ( GLfloat               s,
      GLfloat               t );

  void                   (JEM_APIENTRYP glTexCoord2d)

    ( GLdouble              s,
      GLdouble              t );

  void                   (JEM_APIENTRYP glTexCoord1s)

    ( GLshort               s );

  void                   (JEM_APIENTRYP glTexCoord1i)

    ( GLint                 s );

  void                   (JEM_APIENTRYP glTexCoord1f)

    ( GLfloat               s );

  void                   (JEM_APIENTRYP glTexCoord1d)

    ( GLdouble              s );

  void                   (JEM_APIENTRYP glRects)

    ( GLshort               x1,
      GLshort               y1,
      GLshort               x2,
      GLshort               y2 );

  void                   (JEM_APIENTRYP glRecti)

    ( GLint                 x1,
      GLint                 y1,
      GLint                 x2,
      GLint                 y2 );

  void                   (JEM_APIENTRYP glRectf)

    ( GLfloat               x1,
      GLfloat               y1,
      GLfloat               x2,
      GLfloat               y2 );

  void                   (JEM_APIENTRYP glRectd)

    ( GLdouble              x1,
      GLdouble              y1,
      GLdouble              x2,
      GLdouble              y2 );

  void                   (JEM_APIENTRYP glRasterPos4s)

    ( GLshort               x,
      GLshort               y,
      GLshort               z,
      GLshort               w );

  void                   (JEM_APIENTRYP glRasterPos4i)

    ( GLint                 x,
      GLint                 y,
      GLint                 z,
      GLint                 w );

  void                   (JEM_APIENTRYP glRasterPos4f)

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z,
      GLfloat               w );

  void                   (JEM_APIENTRYP glRasterPos4d)

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  void                   (JEM_APIENTRYP glRasterPos3s)

    ( GLshort               x,
      GLshort               y,
      GLshort               z );

  void                   (JEM_APIENTRYP glRasterPos3i)

    ( GLint                 x,
      GLint                 y,
      GLint                 z );

  void                   (JEM_APIENTRYP glRasterPos3f)

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  void                   (JEM_APIENTRYP glRasterPos3d)

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  void                   (JEM_APIENTRYP glRasterPos2s)

    ( GLshort               x,
      GLshort               y );

  void                   (JEM_APIENTRYP glRasterPos2i)

    ( GLint                 x,
      GLint                 y );

  void                   (JEM_APIENTRYP glRasterPos2f)

    ( GLfloat               x,
      GLfloat               y );

  void                   (JEM_APIENTRYP glRasterPos2d)

    ( GLdouble              x,
      GLdouble              y );

  void                   (JEM_APIENTRYP glNormal3s)

    ( GLshort               nx,
      GLshort               ny,
      GLshort               nz );

  void                   (JEM_APIENTRYP glNormal3i)

    ( GLint                 nx,
      GLint                 ny,
      GLint                 nz );

  void                   (JEM_APIENTRYP glNormal3f)

    ( GLfloat               nx,
      GLfloat               ny,
      GLfloat               nz );

  void                   (JEM_APIENTRYP glNormal3d)

    ( GLdouble              nx,
      GLdouble              ny,
      GLdouble              nz );

  void                   (JEM_APIENTRYP glNormal3b)

    ( GLbyte                nx,
      GLbyte                ny,
      GLbyte                nz );

  void                   (JEM_APIENTRYP glEnd)

    ( void );

  void                   (JEM_APIENTRYP glColor4us)

    ( GLushort              red,
      GLushort              green,
      GLushort              blue,
      GLushort              alpha );

  void                   (JEM_APIENTRYP glColor4ui)

    ( GLuint                red,
      GLuint                green,
      GLuint                blue,
      GLuint                alpha );

  void                   (JEM_APIENTRYP glColor4ub)

    ( GLubyte               red,
      GLubyte               green,
      GLubyte               blue,
      GLubyte               alpha );

  void                   (JEM_APIENTRYP glColor4s)

    ( GLshort               red,
      GLshort               green,
      GLshort               blue,
      GLshort               alpha );

  void                   (JEM_APIENTRYP glColor4i)

    ( GLint                 red,
      GLint                 green,
      GLint                 blue,
      GLint                 alpha );

  void                   (JEM_APIENTRYP glColor4f)

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue,
      GLfloat               alpha );

  void                   (JEM_APIENTRYP glColor4d)

    ( GLdouble              red,
      GLdouble              green,
      GLdouble              blue,
      GLdouble              alpha );

  void                   (JEM_APIENTRYP glColor4b)

    ( GLbyte                red,
      GLbyte                green,
      GLbyte                blue,
      GLbyte                alpha );

  void                   (JEM_APIENTRYP glColor3us)

    ( GLushort              red,
      GLushort              green,
      GLushort              blue );

  void                   (JEM_APIENTRYP glColor3ui)

    ( GLuint                red,
      GLuint                green,
      GLuint                blue );

  void                   (JEM_APIENTRYP glColor3ub)

    ( GLubyte               red,
      GLubyte               green,
      GLubyte               blue );

  void                   (JEM_APIENTRYP glColor3s)

    ( GLshort               red,
      GLshort               green,
      GLshort               blue );

  void                   (JEM_APIENTRYP glColor3i)

    ( GLint                 red,
      GLint                 green,
      GLint                 blue );

  void                   (JEM_APIENTRYP glColor3f)

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue );

  void                   (JEM_APIENTRYP glColor3d)

    ( GLdouble              red,
      GLdouble              green,
      GLdouble              blue );

  void                   (JEM_APIENTRYP glColor3b)

    ( GLbyte                red,
      GLbyte                green,
      GLbyte                blue );

  void                   (JEM_APIENTRYP glBitmap)

    ( GLsizei               width,
      GLsizei               height,
      GLfloat               xorig,
      GLfloat               yorig,
      GLfloat               xmove,
      GLfloat               ymove,
      const GLubyte*        bitmap );

  void                   (JEM_APIENTRYP glBegin)

    ( GLenum                mode );


 protected:

  virtual                ~GLFuncTable1Compat ();

};


JEM_END_PACKAGE( gl )

#endif
