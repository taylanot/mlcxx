
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

#ifndef JEM_GL_GLFUNCTIONS1_H
#define JEM_GL_GLFUNCTIONS1_H

#include <jem/gl/gl.h>
#include <jem/gl/GLVersion.h>
#include <jem/gl/ftable/GLFuncTable1.h>
#include <jem/gl/AbstractGLFunctions.h>


JEM_BEGIN_PACKAGE( gl )


class GLFuncTables;


//-----------------------------------------------------------------------
//   class GLFunctions1Core
//-----------------------------------------------------------------------


class GLFunctions1Core : public AbstractGLFunctions
{
 public:

  static const int        VERSION = GLVersion::VERSION_1_CORE;

  explicit                GLFunctions1Core

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions1Core  ();


 public:

  inline void             glViewport

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  inline void             glDepthRange

    ( GLdouble              near,
      GLdouble              far );

  inline GLboolean        glIsEnabled

    ( GLenum                cap );

  inline void             glGetTexImage

    ( GLenum                target,
      GLint                 level,
      GLenum                format,
      GLenum                type,
      GLvoid*               pixels );

  inline void             glGetParameter

    ( GLenum                pname,
      GLint&                param );

  inline void             glGetParameter

    ( GLenum                pname,
      GLfloat&              param );

  inline void             glGetParameter

    ( GLenum                pname,
      GLdouble&             param );

  inline void             glGetParameter

    ( GLenum                pname,
      GLboolean&            param );

  inline void             glGetParameter

    ( GLenum                pname,
      GLint*                param );

  inline void             glGetParameter

    ( GLenum                pname,
      GLfloat*              param );

  inline void             glGetParameter

    ( GLenum                pname,
      GLdouble*             param );

  inline void             glGetParameter

    ( GLenum                pname,
      GLboolean*            param );

  inline void             glGetTexParameter

    ( GLenum                target,
      GLenum                pname,
      GLint&                param );

  inline void             glGetTexParameter

    ( GLenum                target,
      GLenum                pname,
      GLfloat&              param );

  inline void             glGetTexParameter

    ( GLenum                target,
      GLenum                pname,
      GLint*                param );

  inline void             glGetTexParameter

    ( GLenum                target,
      GLenum                pname,
      GLfloat*              param );

  inline void             glGetTexLevelParameter

    ( GLenum                target,
      GLint                 level,
      GLenum                pname,
      GLint&                param );

  inline void             glGetTexLevelParameter

    ( GLenum                target,
      GLint                 level,
      GLenum                pname,
      GLfloat&              param );

  inline void             glGetTexLevelParameter

    ( GLenum                target,
      GLint                 level,
      GLenum                pname,
      GLint*                param );

  inline void             glGetTexLevelParameter

    ( GLenum                target,
      GLint                 level,
      GLenum                pname,
      GLfloat*              param );

  String                  glGetString

    ( GLenum                name );

  inline GLenum           glGetError

    ( void );

  inline void             glReadPixels

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      GLvoid*               pixels );

  inline void             glReadBuffer

    ( GLenum                mode );

  inline void             glPixelStore

    ( GLenum                pname,
      GLint                 param );

  inline void             glPixelStore

    ( GLenum                pname,
      GLfloat               param );

  inline void             glDepthFunc

    ( GLenum                func );

  inline void             glLogicOp

    ( GLenum                opcode );

  inline void             glBlendFunc

    ( GLenum                sfactor,
      GLenum                dfactor );

  inline void             glFlush

    ( void );

  inline void             glFinish

    ( void );

  inline void             glEnable

    ( GLenum                cap );

  inline void             glDisable

    ( GLenum                cap );

  inline void             glDepthMask

    ( GLboolean             flag );

  inline void             glColorMask

    ( GLboolean             red,
      GLboolean             green,
      GLboolean             blue,
      GLboolean             alpha );

  inline void             glClearDepth

    ( GLdouble              depth );

  inline void             glClearColor

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue,
      GLfloat               alpha );

  inline void             glClear

    ( GLbitfield            mask );

  inline void             glDrawBuffer

    ( GLenum                mode );

  inline void             glTexImage

    ( GLenum                target,
      GLint                 level,
      GLint                 internalformat,
      GLsizei               width,
      GLint                 border,
      GLenum                format,
      GLenum                type,
      const GLvoid*         pixels );

  inline void             glTexImage

    ( GLenum                target,
      GLint                 level,
      GLint                 internalformat,
      GLsizei               width,
      GLsizei               height,
      GLint                 border,
      GLenum                format,
      GLenum                type,
      const GLvoid*         pixels );

  inline void             glTexParameter

    ( GLenum                target,
      GLenum                pname,
      const GLint*          params );

  inline void             glTexParameter

    ( GLenum                target,
      GLenum                pname,
      GLint                 param );

  inline void             glTexParameter

    ( GLenum                target,
      GLenum                pname,
      const GLfloat*        params );

  inline void             glTexParameter

    ( GLenum                target,
      GLenum                pname,
      GLfloat               param );

  inline void             glScissor

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height );

  inline void             glPolygonMode

    ( GLenum                face,
      GLenum                mode );

  inline void             glPointSize

    ( GLfloat               size );

  inline void             glLineWidth

    ( GLfloat               width );

  inline void             glHint

    ( GLenum                target,
      GLenum                mode );

  inline void             glFrontFace

    ( GLenum                mode );

  inline void             glCullFace

    ( GLenum                mode );


 private:

  Ref<GLFuncTable1Core>   table_;

};


typedef GLFunctions1Core  GLFuncs1Core;


//-----------------------------------------------------------------------
//   class GLFunctions1Compat
//-----------------------------------------------------------------------


class GLFunctions1Compat
{
 public:

  explicit                GLFunctions1Compat

    ( GLFuncTables&         tables );

  virtual                ~GLFunctions1Compat  ();


 public:

  inline void             glTranslate

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z = 0.0F );

  inline void             glTranslate

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z = 0.0 );

  inline void             glScale

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  inline void             glScale

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  inline void             glRotate

    ( GLfloat               angle,
      GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  inline void             glRotate

    ( GLdouble              angle,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  inline void             glPushMatrix

    ( void );

  inline void             glPopMatrix

    ( void );

  inline void             glOrtho

    ( GLdouble              left,
      GLdouble              right,
      GLdouble              bottom,
      GLdouble              top,
      GLdouble              zNear,
      GLdouble              zFar );

  inline void             glMatrixMode

    ( GLenum                mode );

  inline void             glLoadMatrix

    ( const GLdouble*       m );

  inline void             glLoadMatrix

    ( const GLfloat*        m );

  inline void             glLoadIdentity

    ( void );

  inline void             glTexEnv

    ( GLenum                target,
      GLenum                pname,
      const GLint*          param );

  inline void             glTexEnv

    ( GLenum                target,
      GLenum                pname,
      GLint                 param );

  inline void             glTexEnv

    ( GLenum                target,
      GLenum                pname,
      const GLfloat*        param );

  inline void             glTexEnv

    ( GLenum                target,
      GLenum                pname,
      GLfloat               param );

  inline void             glGetMaterial

    ( GLenum                face,
      GLenum                pname,
      GLint&                param );

  inline void             glGetMaterial

    ( GLenum                face,
      GLenum                pname,
      GLfloat&              param );

  inline void             glGetMaterial

    ( GLenum                face,
      GLenum                pname,
      GLint*                param );

  inline void             glGetMaterial

    ( GLenum                face,
      GLenum                pname,
      GLfloat*              param );

  inline void             glGetLight

    ( GLenum                light,
      GLenum                pname,
      GLint&                param );

  inline void             glGetLight

    ( GLenum                light,
      GLenum                pname,
      GLfloat&              param );

  inline void             glGetLight

    ( GLenum                light,
      GLenum                pname,
      GLint*                param );

  inline void             glGetLight

    ( GLenum                light,
      GLenum                pname,
      GLfloat*              param );

  inline void             glGetClipPlane

    ( GLenum                plane,
      GLdouble*             equation );

  inline void             glDrawPixels

    ( GLsizei               width,
      GLsizei               height,
      GLenum                format,
      GLenum                type,
      const GLvoid*         pixels );

  inline void             glCopyPixels

    ( GLint                 x,
      GLint                 y,
      GLsizei               width,
      GLsizei               height,
      GLenum                type );

  inline void             glPixelTransfer

    ( GLenum                pname,
      GLint                 param );

  inline void             glPixelTransfer

    ( GLenum                pname,
      GLfloat               param );

  inline void             glAlphaFunc

    ( GLenum                func,
      GLfloat               ref );

  inline void             glPushAttrib

    ( GLbitfield            mask );

  inline void             glPopAttrib

    ( void );

  inline void             glAccum

    ( GLenum                op,
      GLfloat               value );

  inline void             glClearAccum

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue,
      GLfloat               alpha );

  inline void             glSelectBuffer

    ( GLsizei               size,
      GLuint*               buffer );

  inline void             glShadeModel

    ( GLenum                mode );

  inline void             glMaterial

    ( GLenum                face,
      GLenum                pname,
      const GLint*          param );

  inline void             glMaterial

    ( GLenum                face,
      GLenum                pname,
      GLint                 param );

  inline void             glMaterial

    ( GLenum                face,
      GLenum                pname,
      const GLfloat*        param );

  inline void             glMaterial

    ( GLenum                face,
      GLenum                pname,
      GLfloat               param );

  inline void             glLineStipple

    ( GLint                 factor,
      GLushort              pattern );

  inline void             glLightModel

    ( GLenum                pname,
      const GLint*          param );

  inline void             glLightModel

    ( GLenum                pname,
      GLint                 param );

  inline void             glLightModel

    ( GLenum                pname,
      const GLfloat*        params );

  inline void             glLightModel

    ( GLenum                pname,
      GLfloat               param );

  inline void             glLight

    ( GLenum                light,
      GLenum                pname,
      const GLint*          param );

  inline void             glLight

    ( GLenum                light,
      GLenum                pname,
      GLint                 param );

  inline void             glLight

    ( GLenum                light,
      GLenum                pname,
      const GLfloat*        param );

  inline void             glLight

    ( GLenum                light,
      GLenum                pname,
      GLfloat               param );

  inline void             glColorMaterial

    ( GLenum                face,
      GLenum                mode );

  inline void             glClipPlane

    ( GLenum                plane,
      const GLdouble*       equation );

  inline void             glVertex

    ( GLshort               x,
      GLshort               y,
      GLshort               z,
      GLshort               w );

  inline void             glVertex

    ( GLint                 x,
      GLint                 y,
      GLint                 z,
      GLint                 w );

  inline void             glVertex

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z,
      GLfloat               w );

  inline void             glVertex

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  inline void             glVertex

    ( GLshort               x,
      GLshort               y,
      GLshort               z );

  inline void             glVertex

    ( GLint                 x,
      GLint                 y,
      GLint                 z );

  inline void             glVertex

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  inline void             glVertex

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  inline void             glVertex

    ( GLshort               x,
      GLshort               y );

  inline void             glVertex

    ( GLint                 x,
      GLint                 y );

  inline void             glVertex

    ( GLfloat               x,
      GLfloat               y );

  inline void             glVertex

    ( GLdouble              x,
      GLdouble              y );

  template <class T>
    inline void           glVertex4

    ( const T&              v );

  template <class T>
    inline void           glVertex3

    ( const T&              v );

  template <class T>
    inline void           glVertex2

    ( const T&              v );

  inline void             glTexCoord

    ( GLshort               s,
      GLshort               t );

  inline void             glTexCoord

    ( GLint                 s,
      GLint                 t );

  inline void             glTexCoord

    ( GLfloat               s,
      GLfloat               t );

  inline void             glTexCoord

    ( GLdouble              s,
      GLdouble              t );

  inline void             glTexCoord

    ( GLshort               s );

  inline void             glTexCoord

    ( GLint                 s );

  inline void             glTexCoord

    ( GLfloat               s );

  inline void             glTexCoord

    ( GLdouble              s );

  template <class T>
    inline void           glTexCoord2

    ( const T&              t );

  inline void             glRect

    ( GLshort               x1,
      GLshort               y1,
      GLshort               x2,
      GLshort               y2 );

  inline void             glRect

    ( GLint                 x1,
      GLint                 y1,
      GLint                 x2,
      GLint                 y2 );

  inline void             glRect

    ( GLfloat               x1,
      GLfloat               y1,
      GLfloat               x2,
      GLfloat               y2 );

  inline void             glRect

    ( GLdouble              x1,
      GLdouble              y1,
      GLdouble              x2,
      GLdouble              y2 );

  inline void             glRasterPos

    ( GLshort               x,
      GLshort               y,
      GLshort               z,
      GLshort               w );

  inline void             glRasterPos

    ( GLint                 x,
      GLint                 y,
      GLint                 z,
      GLint                 w );

  inline void             glRasterPos

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z,
      GLfloat               w );

  inline void             glRasterPos

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      GLdouble              w );

  inline void             glRasterPos

    ( GLshort               x,
      GLshort               y,
      GLshort               z );

  inline void             glRasterPos

    ( GLint                 x,
      GLint                 y,
      GLint                 z );

  inline void             glRasterPos

    ( GLfloat               x,
      GLfloat               y,
      GLfloat               z );

  inline void             glRasterPos

    ( GLdouble              x,
      GLdouble              y,
      GLdouble              z );

  inline void             glRasterPos

    ( GLshort               x,
      GLshort               y );

  inline void             glRasterPos

    ( GLint                 x,
      GLint                 y );

  inline void             glRasterPos

    ( GLfloat               x,
      GLfloat               y );

  inline void             glRasterPos

    ( GLdouble              x,
      GLdouble              y );

  template <class T>
    inline void           glRasterPos4

    ( const T&              p );

  template <class T>
    inline void           glRasterPos3

    ( const T&              p );

  template <class T>
    inline void           glRasterPos2

    ( const T&              p );

  inline void             glNormal

    ( GLshort               nx,
      GLshort               ny,
      GLshort               nz );

  inline void             glNormal

    ( GLint                 nx,
      GLint                 ny,
      GLint                 nz );

  inline void             glNormal

    ( GLfloat               nx,
      GLfloat               ny,
      GLfloat               nz );

  inline void             glNormal

    ( GLdouble              nx,
      GLdouble              ny,
      GLdouble              nz );

  inline void             glNormal

    ( GLbyte                nx,
      GLbyte                ny,
      GLbyte                nz );

  template <class T>
    inline void           glNormal3

    ( const T&              n );

  inline void             glEnd

    ( void );

  inline void             glColor

    ( GLushort              red,
      GLushort              green,
      GLushort              blue,
      GLushort              alpha );

  inline void             glColor

    ( GLuint                red,
      GLuint                green,
      GLuint                blue,
      GLuint                alpha );

  inline void             glColor

    ( GLubyte               red,
      GLubyte               green,
      GLubyte               blue,
      GLubyte               alpha );

  inline void             glColor

    ( GLshort               red,
      GLshort               green,
      GLshort               blue,
      GLshort               alpha );

  inline void             glColor

    ( GLint                 red,
      GLint                 green,
      GLint                 blue,
      GLint                 alpha );

  inline void             glColor

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue,
      GLfloat               alpha );

  inline void             glColor

    ( GLdouble              red,
      GLdouble              green,
      GLdouble              blue,
      GLdouble              alpha );

  inline void             glColor

    ( GLbyte                red,
      GLbyte                green,
      GLbyte                blue,
      GLbyte                alpha );

  inline void             glColor

    ( GLushort              red,
      GLushort              green,
      GLushort              blue );

  inline void             glColor

    ( GLuint                red,
      GLuint                green,
      GLuint                blue );

  inline void             glColor

    ( GLubyte               red,
      GLubyte               green,
      GLubyte               blue );

  inline void             glColor

    ( GLshort               red,
      GLshort               green,
      GLshort               blue );

  inline void             glColor

    ( GLint                 red,
      GLint                 green,
      GLint                 blue );

  inline void             glColor

    ( GLfloat               red,
      GLfloat               green,
      GLfloat               blue );

  inline void             glColor

    ( GLdouble              red,
      GLdouble              green,
      GLdouble              blue );

  inline void             glColor

    ( GLbyte                red,
      GLbyte                green,
      GLbyte                blue );

  template <class T>
    inline void           glColor4

    ( const T&              c );

  template <class T>
    inline void           glColor3

    ( const T&              c );

  inline void             glBitmap

    ( GLsizei               width,
      GLsizei               height,
      GLfloat               xorig,
      GLfloat               yorig,
      GLfloat               xmove,
      GLfloat               ymove,
      const GLubyte*        bitmap );

  inline void             glBegin

    ( GLenum                mode );


 private:

  Ref
    <GLFuncTable1Compat>  table_;

};


typedef
  GLFunctions1Compat      GLFuncs1Compat;


//-----------------------------------------------------------------------
//   class GLFunctions1
//-----------------------------------------------------------------------


class GLFunctions1 : public GLFunctions1Core,
                     public GLFunctions1Compat
{
 public:

  static const int        VERSION = GLVersion::VERSION_1;

  explicit                GLFunctions1

    ( GLFuncTables&         tables );

};


typedef GLFunctions1      GLFuncs1;



//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class GLFunctions1Core
//=======================================================================


inline void GLFunctions1Core::glViewport

  ( GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height )

{
  table_->glViewport ( x, y, width, height );
}


inline void GLFunctions1Core::glDepthRange

  ( GLdouble  near,
    GLdouble  far )

{
  table_->glDepthRange ( near, far );
}


inline GLboolean GLFunctions1Core::glIsEnabled

  ( GLenum  cap )

{
  return table_->glIsEnabled ( cap );
}


inline void GLFunctions1Core::glGetTexImage

  ( GLenum   target,
    GLint    level,
    GLenum   format,
    GLenum   type,
    GLvoid*  pixels )

{
  table_->glGetTexImage ( target, level, format, type, pixels );
}


inline void GLFunctions1Core::glGetParameter

  ( GLenum  pname,
    GLint&  param )

{
  table_->glGetIntegerv ( pname, &param );
}


inline void GLFunctions1Core::glGetParameter

  ( GLenum    pname,
    GLfloat&  param )

{
  table_->glGetFloatv ( pname, &param );
}


inline void GLFunctions1Core::glGetParameter

  ( GLenum     pname,
    GLdouble&  param )

{
  table_->glGetDoublev ( pname, &param );
}


inline void GLFunctions1Core::glGetParameter

  ( GLenum      pname,
    GLboolean&  param )

{
  table_->glGetBooleanv ( pname, &param );
}


inline void GLFunctions1Core::glGetParameter

  ( GLenum  pname,
    GLint*  param )

{
  table_->glGetIntegerv ( pname, param );
}


inline void GLFunctions1Core::glGetParameter

  ( GLenum    pname,
    GLfloat*  param )

{
  table_->glGetFloatv ( pname, param );
}


inline void GLFunctions1Core::glGetParameter

  ( GLenum     pname,
    GLdouble*  param )

{
  table_->glGetDoublev ( pname, param );
}


inline void GLFunctions1Core::glGetParameter

  ( GLenum      pname,
    GLboolean*  param )

{
  table_->glGetBooleanv ( pname, param );
}


inline void GLFunctions1Core::glGetTexParameter

  ( GLenum  target,
    GLenum  pname,
    GLint&  param )

{
  table_->glGetTexParameteriv ( target, pname, &param );
}


inline void GLFunctions1Core::glGetTexParameter

  ( GLenum    target,
    GLenum    pname,
    GLfloat&  param )

{
  table_->glGetTexParameterfv ( target, pname, &param );
}


inline void GLFunctions1Core::glGetTexParameter

  ( GLenum  target,
    GLenum  pname,
    GLint*  param )

{
  table_->glGetTexParameteriv ( target, pname, param );
}


inline void GLFunctions1Core::glGetTexParameter

  ( GLenum    target,
    GLenum    pname,
    GLfloat*  param )

{
  table_->glGetTexParameterfv ( target, pname, param );
}


inline void GLFunctions1Core::glGetTexLevelParameter

  ( GLenum  target,
    GLint   level,
    GLenum  pname,
    GLint&  param )

{
  table_->glGetTexLevelParameteriv ( target, level, pname, &param );
}


inline void GLFunctions1Core::glGetTexLevelParameter

  ( GLenum    target,
    GLint     level,
    GLenum    pname,
    GLfloat&  param )

{
  table_->glGetTexLevelParameterfv ( target, level, pname, &param );
}


inline void GLFunctions1Core::glGetTexLevelParameter

  ( GLenum  target,
    GLint   level,
    GLenum  pname,
    GLint*  param )

{
  table_->glGetTexLevelParameteriv ( target, level, pname, param );
}


inline void GLFunctions1Core::glGetTexLevelParameter

  ( GLenum    target,
    GLint     level,
    GLenum    pname,
    GLfloat*  param )

{
  table_->glGetTexLevelParameterfv ( target, level, pname, param );
}


inline GLenum GLFunctions1Core::glGetError ()
{
  return table_->glGetError ();
}


inline void GLFunctions1Core::glReadPixels

  ( GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height,
    GLenum   format,
    GLenum   type,
    GLvoid*  pixels )

{
  table_->glReadPixels ( x, y, width, height, format, type, pixels );
}


inline void GLFunctions1Core::glReadBuffer

  ( GLenum  mode )

{
  table_->glReadBuffer ( mode );
}


inline void GLFunctions1Core::glPixelStore

  ( GLenum  pname,
    GLint   param )

{
  table_->glPixelStorei ( pname, param );
}


inline void GLFunctions1Core::glPixelStore

  ( GLenum   pname,
    GLfloat  param )

{
  table_->glPixelStoref ( pname, param );
}


inline void GLFunctions1Core::glDepthFunc

  ( GLenum  func )

{
  table_->glDepthFunc ( func );
}


inline void GLFunctions1Core::glLogicOp

  ( GLenum  opcode )

{
  table_->glLogicOp ( opcode );
}


inline void GLFunctions1Core::glBlendFunc

  ( GLenum  sfactor,
    GLenum  dfactor )

{
  table_->glBlendFunc ( sfactor, dfactor );
}


inline void GLFunctions1Core::glFlush ()
{
  table_->glFlush ();
}


inline void GLFunctions1Core::glFinish ()
{
  table_->glFinish ();
}


inline void GLFunctions1Core::glEnable

  ( GLenum  cap )

{
  table_->glEnable ( cap );
}


inline void GLFunctions1Core::glDisable

  ( GLenum  cap )

{
  table_->glDisable ( cap );
}


inline void GLFunctions1Core::glDepthMask

  ( GLboolean  flag )

{
  table_->glDepthMask ( flag );
}


inline void GLFunctions1Core::glColorMask

  ( GLboolean  red,
    GLboolean  green,
    GLboolean  blue,
    GLboolean  alpha )

{
  table_->glColorMask ( red, green, blue, alpha );
}


inline void GLFunctions1Core::glClearDepth

  ( GLdouble  depth )

{
  table_->glClearDepth ( depth );
}


inline void GLFunctions1Core::glClearColor

  ( GLfloat  red,
    GLfloat  green,
    GLfloat  blue,
    GLfloat  alpha )

{
  table_->glClearColor ( red, green, blue, alpha );
}


inline void GLFunctions1Core::glClear

  ( GLbitfield  mask )

{
  table_->glClear ( mask );
}


inline void GLFunctions1Core::glDrawBuffer

  ( GLenum  mode )

{
  table_->glDrawBuffer ( mode );
}


inline void GLFunctions1Core::glTexImage

  ( GLenum         target,
    GLint          level,
    GLint          internalformat,
    GLsizei        width,
    GLint          border,
    GLenum         format,
    GLenum         type,
    const GLvoid*  pixels )

{
  table_->glTexImage1D ( target, level, internalformat,
                         width, border, format, type, pixels );
}


inline void GLFunctions1Core::glTexImage

  ( GLenum         target,
    GLint          level,
    GLint          internalformat,
    GLsizei        width,
    GLsizei        height,
    GLint          border,
    GLenum         format,
    GLenum         type,
    const GLvoid*  pixels )

{
  table_->glTexImage2D ( target, level, internalformat,
                         width, height, border, format,
                         type, pixels );
}


inline void GLFunctions1Core::glTexParameter

  ( GLenum        target,
    GLenum        pname,
    const GLint*  param )

{
  table_->glTexParameteriv ( target, pname, param );
}


inline void GLFunctions1Core::glTexParameter

  ( GLenum  target,
    GLenum  pname,
    GLint   param )

{
  table_->glTexParameteri ( target, pname, param );
}


inline void GLFunctions1Core::glTexParameter

  ( GLenum          target,
    GLenum          pname,
    const GLfloat*  param )

{
  table_->glTexParameterfv ( target, pname, param );
}


inline void GLFunctions1Core::glTexParameter

  ( GLenum   target,
    GLenum   pname,
    GLfloat  param )

{
  table_->glTexParameterf ( target, pname, param );
}


inline void GLFunctions1Core::glScissor

  ( GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height )

{
  table_->glScissor ( x, y, width, height );
}


inline void GLFunctions1Core::glPolygonMode

  ( GLenum  face,
    GLenum  mode )

{
  table_->glPolygonMode ( face, mode );
}


inline void GLFunctions1Core::glPointSize

  ( GLfloat  size )

{
  table_->glPointSize ( size );
}


inline void GLFunctions1Core::glLineWidth

  ( GLfloat   width )

{
  table_->glLineWidth ( width );
}


inline void GLFunctions1Core::glHint

  ( GLenum  target,
    GLenum  mode )

{
  table_->glHint ( target, mode );
}


inline void GLFunctions1Core::glFrontFace

  ( GLenum  mode )

{
  table_->glFrontFace ( mode );
}


inline void GLFunctions1Core::glCullFace

  ( GLenum  mode )

{
  table_->glCullFace ( mode );
}


//=======================================================================
//   class GLFunctions1Compat
//=======================================================================


inline void GLFunctions1Compat::glTranslate

  ( GLfloat  x,
    GLfloat  y,
    GLfloat  z )

{
  table_->glTranslatef ( x, y, z );
}


inline void GLFunctions1Compat::glTranslate

  ( GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  table_->glTranslated ( x, y, z );
}


inline void GLFunctions1Compat::glScale

  ( GLfloat  x,
    GLfloat  y,
    GLfloat  z )

{
  table_->glScalef ( x, y, z );
}


inline void GLFunctions1Compat::glScale

  ( GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  table_->glScaled ( x, y, z );
}


inline void GLFunctions1Compat::glRotate

  ( GLfloat  angle,
    GLfloat  x,
    GLfloat  y,
    GLfloat  z )

{
  table_->glRotatef ( angle, x, y, z );
}


inline void GLFunctions1Compat::glRotate

  ( GLdouble  angle,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  table_->glRotated ( angle, x, y, z );
}


inline void GLFunctions1Compat::glPushMatrix

  ( void )

{
  table_->glPushMatrix ();
}


inline void GLFunctions1Compat::glPopMatrix

  ( void )

{
  table_->glPopMatrix ();
}


inline void GLFunctions1Compat::glOrtho

  ( GLdouble  left,
    GLdouble  right,
    GLdouble  bottom,
    GLdouble  top,
    GLdouble  zNear,
    GLdouble  zFar )

{
  table_->glOrtho ( left, right, bottom, top, zNear, zFar );
}


inline void GLFunctions1Compat::glMatrixMode

  ( GLenum  mode )

{
  table_->glMatrixMode ( mode );
}


inline void GLFunctions1Compat::glLoadMatrix

  ( const GLdouble*  m )

{
  table_->glLoadMatrixd ( m );
}


inline void GLFunctions1Compat::glLoadMatrix

  ( const GLfloat*  m )

{
  table_->glLoadMatrixf ( m );
}


inline void GLFunctions1Compat::glLoadIdentity

  ( void )

{
  table_->glLoadIdentity ();
}


inline void GLFunctions1Compat::glTexEnv

  ( GLenum        target,
    GLenum        pname,
    const GLint*  param )

{
  table_->glTexEnviv ( target, pname, param );
}


inline void GLFunctions1Compat::glTexEnv

  ( GLenum  target,
    GLenum  pname,
    GLint   param )

{
  table_->glTexEnvi ( target, pname, param );
}


inline void GLFunctions1Compat::glTexEnv

  ( GLenum          target,
    GLenum          pname,
    const GLfloat*  param )

{
  table_->glTexEnvfv ( target, pname, param );
}


inline void GLFunctions1Compat::glTexEnv

  ( GLenum   target,
    GLenum   pname,
    GLfloat  param )

{
  table_->glTexEnvf ( target, pname, param );
}


inline void GLFunctions1Compat::glGetMaterial

  ( GLenum  face,
    GLenum  pname,
    GLint&  param )

{
  table_->glGetMaterialiv ( face, pname, &param );
}


inline void GLFunctions1Compat::glGetMaterial

  ( GLenum    face,
    GLenum    pname,
    GLfloat&  param )

{
  table_->glGetMaterialfv ( face, pname, &param );
}


inline void GLFunctions1Compat::glGetMaterial

  ( GLenum  face,
    GLenum  pname,
    GLint*  param )

{
  table_->glGetMaterialiv ( face, pname, param );
}


inline void GLFunctions1Compat::glGetMaterial

  ( GLenum    face,
    GLenum    pname,
    GLfloat*  param )

{
  table_->glGetMaterialfv ( face, pname, param );
}


inline void GLFunctions1Compat::glGetLight

  ( GLenum  light,
    GLenum  pname,
    GLint&  param )

{
  table_->glGetLightiv ( light, pname, &param );
}


inline void GLFunctions1Compat::glGetLight

  ( GLenum    light,
    GLenum    pname,
    GLfloat&  param )

{
  table_->glGetLightfv ( light, pname, &param );
}


inline void GLFunctions1Compat::glGetLight

  ( GLenum  light,
    GLenum  pname,
    GLint*  param )

{
  table_->glGetLightiv ( light, pname, param );
}


inline void GLFunctions1Compat::glGetLight

  ( GLenum    light,
    GLenum    pname,
    GLfloat*  param )

{
  table_->glGetLightfv ( light, pname, param );
}


inline void GLFunctions1Compat::glGetClipPlane

  ( GLenum     plane,
    GLdouble*  equation )

{
  table_->glGetClipPlane ( plane, equation );
}


inline void GLFunctions1Compat::glDrawPixels

  ( GLsizei        width,
    GLsizei        height,
    GLenum         format,
    GLenum         type,
    const GLvoid*  pixels )

{
  table_->glDrawPixels ( width, height, format, type, pixels );
}


inline void GLFunctions1Compat::glCopyPixels

  ( GLint    x,
    GLint    y,
    GLsizei  width,
    GLsizei  height,
    GLenum   type )

{
  table_->glCopyPixels ( x, y, width, height, type );
}


inline void GLFunctions1Compat::glPixelTransfer

  ( GLenum  pname,
    GLint   param )

{
  table_->glPixelTransferi ( pname, param );
}


inline void GLFunctions1Compat::glPixelTransfer

  ( GLenum   pname,
    GLfloat  param )

{
  table_->glPixelTransferf ( pname, param );
}


inline void GLFunctions1Compat::glAlphaFunc

  ( GLenum   func,
    GLfloat  ref )

{
  table_->glAlphaFunc ( func, ref );
}


inline void GLFunctions1Compat::glPushAttrib

  ( GLbitfield  mask )

{
  table_->glPushAttrib ( mask );
}


inline void GLFunctions1Compat::glPopAttrib

  ( void )

{
  table_->glPopAttrib ();
}


inline void GLFunctions1Compat::glAccum

  ( GLenum   op,
    GLfloat  value )

{
  table_->glAccum ( op, value );
}


inline void GLFunctions1Compat::glClearAccum

  ( GLfloat  red,
    GLfloat  green,
    GLfloat  blue,
    GLfloat  alpha )

{
  table_->glClearAccum ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glSelectBuffer

  ( GLsizei  size,
    GLuint*  buffer )

{
  table_->glSelectBuffer ( size, buffer );
}


inline void GLFunctions1Compat::glShadeModel

  ( GLenum  mode )

{
  table_->glShadeModel ( mode );
}


inline void GLFunctions1Compat::glMaterial

  ( GLenum        face,
    GLenum        pname,
    const GLint*  param )

{
  table_->glMaterialiv ( face, pname, param );
}


inline void GLFunctions1Compat::glMaterial

  ( GLenum  face,
    GLenum  pname,
    GLint   param )

{
  table_->glMateriali ( face, pname, param );
}


inline void GLFunctions1Compat::glMaterial

  ( GLenum          face,
    GLenum          pname,
    const GLfloat*  param )

{
  table_->glMaterialfv ( face, pname, param );
}


inline void GLFunctions1Compat::glMaterial

  ( GLenum   face,
    GLenum   pname,
    GLfloat  param )

{
  table_->glMaterialf ( face, pname, param );
}


inline void GLFunctions1Compat::glLineStipple

  ( GLint     factor,
    GLushort  pattern )

{
  table_->glLineStipple ( factor, pattern );
}


inline void GLFunctions1Compat::glLightModel

  ( GLenum        pname,
    const GLint*  param )

{
  table_->glLightModeliv ( pname, param );
}


inline void GLFunctions1Compat::glLightModel

  ( GLenum  pname,
    GLint   param )

{
  table_->glLightModeli ( pname, param );
}


inline void GLFunctions1Compat::glLightModel

  ( GLenum          pname,
    const GLfloat*  params )

{
  table_->glLightModelfv ( pname, params );
}


inline void GLFunctions1Compat::glLightModel

  ( GLenum   pname,
    GLfloat  param )

{
  table_->glLightModelf ( pname, param );
}


inline void GLFunctions1Compat::glLight

  ( GLenum        light,
    GLenum        pname,
    const GLint*  param )

{
  table_->glLightiv ( light, pname, param );
}


inline void GLFunctions1Compat::glLight

  ( GLenum  light,
    GLenum  pname,
    GLint   param )

{
  table_->glLighti ( light, pname, param );
}


inline void GLFunctions1Compat::glLight

  ( GLenum          light,
    GLenum          pname,
    const GLfloat*  param )

{
  table_->glLightfv ( light, pname, param );
}


inline void GLFunctions1Compat::glLight

  ( GLenum   light,
    GLenum   pname,
    GLfloat  param )

{
  table_->glLightf ( light, pname, param );
}


inline void GLFunctions1Compat::glColorMaterial

  ( GLenum  face,
    GLenum  mode )

{
  table_->glColorMaterial ( face, mode );
}


inline void GLFunctions1Compat::glClipPlane

  ( GLenum           plane,
    const GLdouble*  equation )

{
  table_->glClipPlane ( plane, equation );
}


inline void GLFunctions1Compat::glVertex

  ( GLshort  x,
    GLshort  y,
    GLshort  z,
    GLshort  w )

{
  table_->glVertex4s ( x, y, z, w );
}


inline void GLFunctions1Compat::glVertex

  ( GLint  x,
    GLint  y,
    GLint  z,
    GLint  w )

{
  table_->glVertex4i ( x, y, z, w );
}


inline void GLFunctions1Compat::glVertex

  ( GLfloat  x,
    GLfloat  y,
    GLfloat  z,
    GLfloat  w )

{
  table_->glVertex4f ( x, y, z, w );
}


inline void GLFunctions1Compat::glVertex

  ( GLdouble  x,
    GLdouble  y,
    GLdouble  z,
    GLdouble  w )

{
  table_->glVertex4d ( x, y, z, w );
}


inline void GLFunctions1Compat::glVertex

  ( GLshort  x,
    GLshort  y,
    GLshort  z )

{
  table_->glVertex3s ( x, y, z );
}


inline void GLFunctions1Compat::glVertex

  ( GLint  x,
    GLint  y,
    GLint  z )

{
  table_->glVertex3i ( x, y, z );
}


inline void GLFunctions1Compat::glVertex

  ( GLfloat  x,
    GLfloat  y,
    GLfloat  z )

{
  table_->glVertex3f ( x, y, z );
}


inline void GLFunctions1Compat::glVertex

  ( GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  table_->glVertex3d ( x, y, z );
}


inline void GLFunctions1Compat::glVertex

  ( GLshort  x,
    GLshort  y )

{
  table_->glVertex2s ( x, y );
}


inline void GLFunctions1Compat::glVertex

  ( GLint  x,
    GLint  y )

{
  table_->glVertex2i ( x, y );
}


inline void GLFunctions1Compat::glVertex

  ( GLfloat  x,
    GLfloat  y )

{
  table_->glVertex2f ( x, y );
}


inline void GLFunctions1Compat::glVertex

  ( GLdouble  x,
    GLdouble  y )

{
  table_->glVertex2d ( x, y );
}


template <class T>
  inline void GLFunctions1Compat::glVertex4

  ( const T&  v )

{
  glVertex ( v[0], v[1], v[2], v[3] );
}


template <class T>
  inline void GLFunctions1Compat::glVertex3

  ( const T&  v )

{
  glVertex ( v[0], v[1], v[2] );
}


template <class T>
  inline void GLFunctions1Compat::glVertex2

  ( const T&  v )

{
  glVertex ( v[0], v[1] );
}


inline void GLFunctions1Compat::glTexCoord

  ( GLshort  s,
    GLshort  t )

{
  table_->glTexCoord2s ( s, t );
}


inline void GLFunctions1Compat::glTexCoord

  ( GLint  s,
    GLint  t )

{
  table_->glTexCoord2i ( s, t );
}


inline void GLFunctions1Compat::glTexCoord

  ( GLfloat  s,
    GLfloat  t )

{
  table_->glTexCoord2f ( s, t );
}


inline void GLFunctions1Compat::glTexCoord

  ( GLdouble  s,
    GLdouble  t )

{
  table_->glTexCoord2d ( s, t );
}


inline void GLFunctions1Compat::glTexCoord

  ( GLshort  s )

{
  table_->glTexCoord1s ( s );
}


inline void GLFunctions1Compat::glTexCoord

  ( GLint  s )

{
  table_->glTexCoord1i ( s );
}


inline void GLFunctions1Compat::glTexCoord

  ( GLfloat  s )

{
  table_->glTexCoord1f ( s );
}


inline void GLFunctions1Compat::glTexCoord

  ( GLdouble  s )

{
  table_->glTexCoord1d ( s );
}


template <class T>
  inline void GLFunctions1Compat::glTexCoord2

  ( const T&  t )

{
  glTexCoord ( t[0], t[1] );
}


inline void GLFunctions1Compat::glRect

  ( GLshort  x1,
    GLshort  y1,
    GLshort  x2,
    GLshort  y2 )

{
  table_->glRects ( x1, y1, x2, y2 );
}


inline void GLFunctions1Compat::glRect

  ( GLint  x1,
    GLint  y1,
    GLint  x2,
    GLint  y2 )

{
  table_->glRecti ( x1, y1, x2, y2 );
}


inline void GLFunctions1Compat::glRect

  ( GLfloat  x1,
    GLfloat  y1,
    GLfloat  x2,
    GLfloat  y2 )

{
  table_->glRectf ( x1, y1, x2, y2 );
}


inline void GLFunctions1Compat::glRect

  ( GLdouble  x1,
    GLdouble  y1,
    GLdouble  x2,
    GLdouble  y2 )

{
  table_->glRectd ( x1, y1, x2, y2 );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLshort  x,
    GLshort  y,
    GLshort  z,
    GLshort  w )

{
  table_->glRasterPos4s ( x, y, z, w );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLint  x,
    GLint  y,
    GLint  z,
    GLint  w )

{
  table_->glRasterPos4i ( x, y, z, w );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLfloat  x,
    GLfloat  y,
    GLfloat  z,
    GLfloat  w )

{
  table_->glRasterPos4f ( x, y, z, w );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLdouble  x,
    GLdouble  y,
    GLdouble  z,
    GLdouble  w )

{
  table_->glRasterPos4d ( x, y, z, w );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLshort  x,
    GLshort  y,
    GLshort  z )

{
  table_->glRasterPos3s ( x, y, z );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLint  x,
    GLint  y,
    GLint  z )

{
  table_->glRasterPos3i ( x, y, z );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLfloat  x,
    GLfloat  y,
    GLfloat  z )

{
  table_->glRasterPos3f ( x, y, z );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  table_->glRasterPos3d ( x, y, z );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLshort  x,
    GLshort  y )

{
  table_->glRasterPos2s ( x, y );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLint  x,
    GLint  y )

{
  table_->glRasterPos2i ( x, y );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLfloat  x,
    GLfloat  y )

{
  table_->glRasterPos2f ( x, y );
}


inline void GLFunctions1Compat::glRasterPos

  ( GLdouble  x,
    GLdouble  y )

{
  table_->glRasterPos2d ( x, y );
}


template <class T>
  inline void GLFunctions1Compat::glRasterPos4

  ( const T&  p )

{
  glRasterPos ( p[0], p[1], p[2], p[3] );
}


template <class T>
  inline void GLFunctions1Compat::glRasterPos3

  ( const T&  p )

{
  glRasterPos ( p[0], p[1], p[2] );
}


template <class T>
  inline void GLFunctions1Compat::glRasterPos2

  ( const T&  p )

{
  glRasterPos ( p[0], p[1] );
}


inline void GLFunctions1Compat::glNormal

  ( GLshort  nx,
    GLshort  ny,
    GLshort  nz )

{
  table_->glNormal3s ( nx, ny, nz );
}


inline void GLFunctions1Compat::glNormal

  ( GLint  nx,
    GLint  ny,
    GLint  nz )

{
  table_->glNormal3i ( nx, ny, nz );
}


inline void GLFunctions1Compat::glNormal

  ( GLfloat  nx,
    GLfloat  ny,
    GLfloat  nz )

{
  table_->glNormal3f ( nx, ny, nz );
}


inline void GLFunctions1Compat::glNormal

  ( GLdouble  nx,
    GLdouble  ny,
    GLdouble  nz )

{
  table_->glNormal3d ( nx, ny, nz );
}


inline void GLFunctions1Compat::glNormal

  ( GLbyte  nx,
    GLbyte  ny,
    GLbyte  nz )

{
  table_->glNormal3b ( nx, ny, nz );
}


template <class T>
  inline void GLFunctions1Compat::glNormal3

  ( const T&  n )

{
  glNormal ( n[0], n[1], n[2] );
}


inline void GLFunctions1Compat::glEnd

  ( void )

{
  table_->glEnd ();
}


inline void GLFunctions1Compat::glColor

  ( GLushort  red,
    GLushort  green,
    GLushort  blue,
    GLushort  alpha )

{
  table_->glColor4us ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glColor

  ( GLuint  red,
    GLuint  green,
    GLuint  blue,
    GLuint  alpha )

{
  table_->glColor4ui ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glColor

  ( GLubyte  red,
    GLubyte  green,
    GLubyte  blue,
    GLubyte  alpha )

{
  table_->glColor4ub ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glColor

  ( GLshort  red,
    GLshort  green,
    GLshort  blue,
    GLshort  alpha )

{
  table_->glColor4s ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glColor

  ( GLint  red,
    GLint  green,
    GLint  blue,
    GLint  alpha )

{
  table_->glColor4i ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glColor

  ( GLfloat  red,
    GLfloat  green,
    GLfloat  blue,
    GLfloat  alpha )

{
  table_->glColor4f ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glColor

  ( GLdouble  red,
    GLdouble  green,
    GLdouble  blue,
    GLdouble  alpha )

{
  table_->glColor4d ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glColor

  ( GLbyte  red,
    GLbyte  green,
    GLbyte  blue,
    GLbyte  alpha )

{
  table_->glColor4b ( red, green, blue, alpha );
}


inline void GLFunctions1Compat::glColor

  ( GLushort  red,
    GLushort  green,
    GLushort  blue )

{
  table_->glColor3us ( red, green, blue );
}


inline void GLFunctions1Compat::glColor

  ( GLuint  red,
    GLuint  green,
    GLuint  blue )

{
  table_->glColor3ui ( red, green, blue );
}


inline void GLFunctions1Compat::glColor

  ( GLubyte  red,
    GLubyte  green,
    GLubyte  blue )

{
  table_->glColor3ub ( red, green, blue );
}


inline void GLFunctions1Compat::glColor

  ( GLshort  red,
    GLshort  green,
    GLshort  blue )

{
  table_->glColor3s ( red, green, blue );
}


inline void GLFunctions1Compat::glColor

  ( GLint  red,
    GLint  green,
    GLint  blue )

{
  table_->glColor3i ( red, green, blue );
}


inline void GLFunctions1Compat::glColor

  ( GLfloat  red,
    GLfloat  green,
    GLfloat  blue )

{
  table_->glColor3f ( red, green, blue );
}


inline void GLFunctions1Compat::glColor

  ( GLdouble  red,
    GLdouble  green,
    GLdouble  blue )

{
  table_->glColor3d ( red, green, blue );
}


inline void GLFunctions1Compat::glColor

  ( GLbyte  red,
    GLbyte  green,
    GLbyte  blue )

{
  table_->glColor3b ( red, green, blue );
}


template <class T>
  inline void GLFunctions1Compat::glColor4

  ( const T&  c )

{
  glColor ( c[0], c[1], c[2], c[3] );
}


template <class T>
  inline void GLFunctions1Compat::glColor3

  ( const T&  c )

{
  glColor ( c[0], c[1], c[2] );
}


inline void GLFunctions1Compat::glBitmap

  ( GLsizei         width,
    GLsizei         height,
    GLfloat         xorig,
    GLfloat         yorig,
    GLfloat         xmove,
    GLfloat         ymove,
    const GLubyte*  bitmap )

{
  table_->glBitmap ( width, height, xorig, yorig,
                     xmove, ymove,  bitmap );
}


inline void GLFunctions1Compat::glBegin

  ( GLenum  mode )

{
  table_->glBegin ( mode );
}


JEM_END_PACKAGE( gl )

#endif
