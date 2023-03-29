
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

#ifndef JEM_GL_GLCONTEXT_H
#define JEM_GL_GLCONTEXT_H

#include <jem/base/Object.h>
#include <jem/gl/gl.h>
#include <jem/gl/GLObject.h>
#include <jem/gl/GLVersion.h>


JEM_BEGIN_PACKAGE( gl )


class Rect;
class Color;
class Lights;
class Transform;
class AbstractGLFunctions;


//-----------------------------------------------------------------------
//   class GLContext
//-----------------------------------------------------------------------


class GLContext
{
 public:

  typedef GLContext       Self;

  static const int        COLOR;
  static const int        SPECULAR;
  static const int        LIGHTING;
  static const int        LINE_WIDTH;
  static const int        POINT_SIZE;
  static const int        ALPHA_TEST;

  enum                    MatrixID
  {
                            MODEL_VIEW,
                            PROJECTION
  };

                          GLContext       ();
                         ~GLContext       ();

  void                    clear           ()       noexcept;

  inline int              getID           () const noexcept;

  inline bool             hasID

    ( int                   id )             const noexcept;

  void                    rotate

    ( GLdouble              a,
      GLdouble              x,
      GLdouble              y,
      GLdouble              z,
      int                   mxid = MODEL_VIEW );

  void                    scale

    ( GLdouble              s,
      int                   mxid = MODEL_VIEW );

  void                    scale

    ( GLdouble              sx,
      GLdouble              sy,
      int                   mxid = MODEL_VIEW );

  void                    scale

    ( GLdouble              sx,
      GLdouble              sy,
      GLdouble              sz,
      int                   mxid = MODEL_VIEW );

  void                    translate

    ( GLdouble              dx,
      GLdouble              dy,
      int                   mxid = MODEL_VIEW );

  void                    translate

    ( GLdouble              dx,
      GLdouble              dy,
      GLdouble              dz,
      int                   mxid = MODEL_VIEW );

  void                    transform

    ( const GLdouble*       tr,
      int                   mxid = MODEL_VIEW );

  void                    transform

    ( const Transform&      tr,
      int                   mxid = MODEL_VIEW );

  void                    getTransform

    ( GLfloat*              tr,
      int                   mxid = MODEL_VIEW ) const;

  void                    getTransform

    ( GLdouble*             tr,
      int                   mxid = MODEL_VIEW ) const;

  Transform               getTransform

    ( int                   mxid = MODEL_VIEW ) const;

  void                    setTransform

    ( const GLdouble*       tr,
      int                   mxid = MODEL_VIEW );

  void                    setTransform

    ( const Transform&      tr,
      int                   mxid = MODEL_VIEW );

  void                    setIdentity

    ( int                   mxid = MODEL_VIEW );

  void                    setOrtho

    ( GLdouble              left,
      GLdouble              right,
      GLdouble              bot,
      GLdouble              top,
      GLdouble              near,
      GLdouble              far,
      int                   mxid = MODEL_VIEW );

  void                    setOrtho

    ( const Rect&           rect,
      GLdouble              near,
      GLdouble              far,
      int                   mxid = MODEL_VIEW );

  void                    popTransform

    ( int                   mxid = MODEL_VIEW );

  void                    pushTransform

    ( int                   mxid = MODEL_VIEW );

  void                    resetTransform

    ( int                   mxid = MODEL_VIEW );

  void                    resetTransforms    ();

  void                    getModelViewProj

    ( GLfloat*              mvp )               const;

  void                    getModelViewProj

    ( GLdouble*             mvp )               const;

  Transform               getModelViewProj   () const;

  void                    getNormalMatrix

    ( GLfloat*              norm )              const;

  void                    getNormalMatrix

    ( GLdouble*             norm )              const;

  Ref<Object>             findProgram

    ( const String&         name )              const;

  void                    storeProgram

    ( const String&         name,
      const Ref<Object>&    prog )              const;

  GLuint                  newTexture

    ( const void*           addr,
      const Ref<Object>&    owner );

  inline GLuint           newGLObject

    ( int                   type,
      const Ref<GLObject>&  obj,
      const Ref<Object>&    owner );

  GLuint                  newGLObject

    ( int                   type,
      const void*           addr,
      const Ref<Object>&    owner );

  inline GLuint           getGLObject

    ( const Ref<GLObject>&  obj )               const;

  GLuint                  getGLObject

    ( const void*           addr )              const;

  inline bool             findGLObject

    ( GLuint&               oid,
      const Ref<GLObject>&  obj )               const;

  bool                    findGLObject

    ( GLuint&               oid,
      const void*           addr )              const;

  inline void             freeGLObject

    ( const Ref<GLObject>&  obj );

  void                    freeGLObject

    ( const void*           addr );

  GLVersion               getVersion      ()    const;

  GLVersion               setVersion

    ( const GLVersion&      vers );

  bool                    checkVersion

    ( int                   major,
      int                   minor )             const;

  bool                    checkVersion

    ( GLVersion::VersionID  vid )               const;

  String                  errorString

    ( GLenum                err )               const;

  Ref<Lights>             getLights       ()    const;

  void                    setLights

    ( const Ref<Lights>&    lights );

  void                    setLighting

    ( bool                  choice );

  bool                    getLighting     ()    const;

  const Color&            getColor        ()    const;

  void                    setColor

    ( const Color&          color );

  const Color&            getSpecular     ()    const;

  void                    setSpecular

    ( const Color&          specular );

  GLfloat                 getShininess    ()    const;

  void                    setShininess

    ( GLfloat               shininess );

  GLfloat                 getLineWidth    ()    const;

  void                    setLineWidth

    ( GLfloat               width );

  GLfloat                 getPointSize    ()    const;

  void                    setPointSize

    ( GLfloat               psize );

  Rect                    getViewport     ()    const;

  void                    setViewport

    ( const Rect&           rect );

  GLfloat                 getAlphaTest    ()    const;

  void                    setAlphaTest

    ( GLfloat               threshold );

  void                    enableAlphaTest

    ( bool                  choice );

  void                    pushAttribs

    ( int                   mask );

  void                    popAttribs      ();

  template <class T>
    inline T&             getFunctions    ()    const;

  template <class T>
    inline T*             findFunctions   ()    const;


 private:

                          GLContext

    ( const GLContext&      rhs );

  GLContext&              operator =

    ( const GLContext&      rhs );


 private:

  class                   Data_;

  inline Data_&           getData_        ()    const;

  void                    initData_       ();

  AbstractGLFunctions*    getFunctions_

    ( int                   vid )               const;

  AbstractGLFunctions*    findFunctions_

    ( int                   vid )               const;


 private:

  struct                  Cache_
  {
    int                     versi1;
    int                     versi2;
    AbstractGLFunctions*    funcs1;
    AbstractGLFunctions*    funcs2;
  };

  int                     id_;
  void*                   data_;
  Cache_                  cache_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


void                      clearGLError

  ( GLContext&              gtx );

void                      checkGLError

  ( GLContext&              gtx,
    const String&           where = "" );

GLbitfield                getClearMask

  ( GLContext&              gtx );

GLint                     getDepthBits

  ( GLContext&              gtx );

void                      finishGLOps

  ( GLContext&              gtx );



//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getID
//-----------------------------------------------------------------------


inline int GLContext::getID () const noexcept
{
  return id_;
}


//-----------------------------------------------------------------------
//   hasID
//-----------------------------------------------------------------------


inline bool GLContext::hasID ( int id ) const noexcept
{
  return (id == id_);
}


//-----------------------------------------------------------------------
//   newGLObject
//-----------------------------------------------------------------------


inline GLuint GLContext::newGLObject

  ( int                   type,
    const Ref<GLObject>&  obj,
    const Ref<Object>&    owner )

{
  return newGLObject ( type, obj.get(), owner );
}


inline GLuint GLContext::getGLObject

  ( const Ref<GLObject>&  obj ) const

{
  return getGLObject ( obj.get() );
}


//-----------------------------------------------------------------------
//   findGLObject
//-----------------------------------------------------------------------


inline bool GLContext::findGLObject

  ( GLuint&               oid,
    const Ref<GLObject>&  obj ) const

{
  return findGLObject ( oid, obj.get() );
}


//-----------------------------------------------------------------------
//   freeGLObject
//-----------------------------------------------------------------------


inline void GLContext::freeGLObject ( const Ref<GLObject>&  obj )
{
  freeGLObject ( obj.get() );
}


//-----------------------------------------------------------------------
//   getFunctions
//-----------------------------------------------------------------------


template <class T>

  inline T& GLContext::getFunctions () const

{
  if ( cache_.versi1 == (int) T::VERSION )
  {
    return *(static_cast<T*>( cache_.funcs1 ));
  }

  return *(static_cast<T*>( getFunctions_( T::VERSION ) ));
}


//-----------------------------------------------------------------------
//   findFunctions
//-----------------------------------------------------------------------


template <class T>

  inline T* GLContext::findFunctions () const

{
  if ( cache_.versi1 == (int) T::VERSION )
  {
    return static_cast<T*> ( cache_.funcs1 );
  }

  return static_cast<T*> ( findFunctions_( T::VERSION ) );
}


JEM_END_PACKAGE( gl )

#endif
