
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


#include <cstring>
#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/SerialSection.h>
#include <jem/util/Properties.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Rect.h>
#include <jem/gl/Color.h>
#include <jem/gl/Lights.h>
#include <jem/gl/Transform.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLException.h>
#include <jem/gl/GLFunctions4_5.h>
#include "AttribStack.h"
#include "MatrixStack.h"
#include "GLObjectManager.h"
#include "ftable/GLFuncTables.h"
#include "ftable/NativeLoader.h"


JEM_BEGIN_PACKAGE( gl )


using jem::util::Properties;


//=======================================================================
//   class GLContext::Data_
//=======================================================================


class GLContext::Data_
{
 public:

   explicit                 Data_

     ( const GLVersion&       vers );

                           ~Data_      ();

  template <class T>
    inline T*               findFuncs  ();

  inline AbstractGLFuncs*   findFuncs

    ( int                     vid );

  AbstractGLFuncs*          makeFuncs

    ( int                     vid );

  MatrixStack&              getStack

    ( int                     mxid );

  inline void               loadMatrix

    ( int                     mxid );

  void                      setVersion

    ( const GLVersion&        vers );

  static int                nextID     ();


 public:

  struct                    Funcs
  {
    AbstractGLFuncs*          functions;
    bool                      supported;
  };

  const GLVersion           version0;
  GLVersion                 version;
  MatrixStack               mstack;
  MatrixStack               pstack;
  AttribStack               attribs;
  GLObjectManager           objects;
  GLFuncTables              ftables;
  Funcs                     functions[GLVersion::LAST_VERSION + 1];
  Properties                programs;
  Ref<Lights>               lights;
  Color                     color;
  Color                     specular;
  GLfloat                   shininess;
  GLfloat                   lineWidth;
  GLfloat                   pointSize;

  bool                      lighting;
  bool                      alphaTest;
  GLfloat                   alphaThreshold;

  Rect                      viewport;


 private:

  void                      loadMatrix_

    ( int                     mxid );

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLContext::Data_::Data_ ( const GLVersion& vers ) :

  version0 ( vers ),
  version  ( vers )

{
  using jem::newInstance;

  lights         =  newInstance<Lights> ();
  color          =  Color::white        ();
  specular       =  Color::black        ();
  shininess      =  1.0F;
  lineWidth      =  1.0F;
  pointSize      =  1.0F;
  lighting       =  true;
  alphaTest      =  false;
  alphaThreshold =  0.0F;

  std::memset ( functions, 0x0, sizeof(functions) );

  setVersion ( vers );
}


GLContext::Data_::~Data_ ()
{
  for ( int i = 0; i <= GLVersion::LAST_VERSION; i++ )
  {
    delete functions[i].functions;
  }
}


//-----------------------------------------------------------------------
//   findFuncs
//-----------------------------------------------------------------------


template <class T>

  inline T* GLContext::Data_::findFuncs ()

{
  return static_cast<T*> ( findFuncs( T::VERSION ) );
}


inline AbstractGLFuncs* GLContext::Data_::findFuncs ( int vid )
{
  if ( vid < 0 || vid > GLVersion::LAST_VERSION )
  {
    return nullptr;
  }

  if ( functions[vid].functions )
  {
    return functions[vid].functions;
  }

  if ( functions[vid].supported )
  {
    functions[vid].functions = makeFuncs ( vid );
  }

  return functions[vid].functions;
}


//-----------------------------------------------------------------------
//   makeFuncs
//-----------------------------------------------------------------------


AbstractGLFuncs* GLContext::Data_::makeFuncs ( int vid )
{
  switch ( vid )
  {
  // OpenGL version 1.*

  case GLVersion::VERSION_1:

    return new GLFunctions1       ( ftables );

  case GLVersion::VERSION_1_CORE:

    return new GLFunctions1Core   ( ftables );

  case GLVersion::VERSION_1_1:

    return new GLFunctions1_1     ( ftables );

  case GLVersion::VERSION_1_1_CORE:

    return new GLFunctions1_1Core ( ftables );

  case GLVersion::VERSION_1_2:

    return new GLFunctions1_2     ( ftables );

  case GLVersion::VERSION_1_2_CORE:

    return new GLFunctions1_2Core ( ftables );

  case GLVersion::VERSION_1_3:

    return new GLFunctions1_3     ( ftables );

  case GLVersion::VERSION_1_3_CORE:

    return new GLFunctions1_3Core ( ftables );

  case GLVersion::VERSION_1_4:

    return new GLFunctions1_4     ( ftables );

  case GLVersion::VERSION_1_4_CORE:

    return new GLFunctions1_4Core ( ftables );

  case GLVersion::VERSION_1_5:

    return new GLFunctions1_5     ( ftables );

  case GLVersion::VERSION_1_5_CORE:

    return new GLFunctions1_5Core ( ftables );

  // OpenGL version 2.*

  case GLVersion::VERSION_2:

    return new GLFunctions2       ( ftables );

  case GLVersion::VERSION_2_CORE:

    return new GLFunctions2Core   ( ftables );

  case GLVersion::VERSION_2_1:

    return new GLFunctions2_1     ( ftables );

  case GLVersion::VERSION_2_1_CORE:

    return new GLFunctions2_1Core ( ftables );

  // OpenGL version 3.*

  case GLVersion::VERSION_3:

    return new GLFunctions3       ( ftables );

  case GLVersion::VERSION_3_CORE:

    return new GLFunctions3Core   ( ftables );

  case GLVersion::VERSION_3_1:

    return new GLFunctions3_1     ( ftables );

  case GLVersion::VERSION_3_1_CORE:

    return new GLFunctions3_1Core ( ftables );

  case GLVersion::VERSION_3_2:

    return new GLFunctions3_2     ( ftables );

  case GLVersion::VERSION_3_2_CORE:

    return new GLFunctions3_2Core ( ftables );

  case GLVersion::VERSION_3_3:

    return new GLFunctions3_3     ( ftables );

  case GLVersion::VERSION_3_3_CORE:

    return new GLFunctions3_3Core ( ftables );

  // OpenGL version 4.*

  case GLVersion::VERSION_4:

    return new GLFunctions4       ( ftables );

  case GLVersion::VERSION_4_CORE:

    return new GLFunctions4Core   ( ftables );

  case GLVersion::VERSION_4_1:

    return new GLFunctions4_1     ( ftables );

  case GLVersion::VERSION_4_1_CORE:

    return new GLFunctions4_1Core ( ftables );

  case GLVersion::VERSION_4_2:

    return new GLFunctions4_2     ( ftables );

  case GLVersion::VERSION_4_2_CORE:

    return new GLFunctions4_2Core ( ftables );

  case GLVersion::VERSION_4_3:

    return new GLFunctions4_3     ( ftables );

  case GLVersion::VERSION_4_3_CORE:

    return new GLFunctions4_3Core ( ftables );

  case GLVersion::VERSION_4_4:

    return new GLFunctions4_4     ( ftables );

  case GLVersion::VERSION_4_4_CORE:

    return new GLFunctions4_4Core ( ftables );

  case GLVersion::VERSION_4_5:

    return new GLFunctions4_5     ( ftables );

  case GLVersion::VERSION_4_5_CORE:

    return new GLFunctions4_5Core ( ftables );
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   getStack
//-----------------------------------------------------------------------


MatrixStack& GLContext::Data_::getStack ( int mxid )
{
  if      ( mxid == GLContext::MODEL_VIEW )
  {
    return mstack;
  }
  else if ( mxid == GLContext::PROJECTION )
  {
    return pstack;
  }
  else
  {
    throw GLException (
      JEM_FUNC,
      String::format (
        "invalid matrix stack identifier: %d",
        mxid
      )
    );

    return mstack;
  }
}


//-----------------------------------------------------------------------
//   loadMatrix
//-----------------------------------------------------------------------


inline void GLContext::Data_::loadMatrix ( int mxid )
{
  if ( version.getVersionID() < GLVersion::VERSION_3_3 )
  {
    loadMatrix_ ( mxid );
  }
}


//-----------------------------------------------------------------------
//   setVersion
//-----------------------------------------------------------------------


void GLContext::Data_::setVersion ( const GLVersion& vers )
{
  bool  core  = (version.getProfile() == GLVersion::CORE_PROFILE);
  int   major =  version.getMajorVersion ();
  int   minor =  version.getMinorVersion ();


  for ( int i = 0; i <= GLVersion::LAST_VERSION; i++ )
  {
    delete functions[i].functions;
  }

  std::memset ( functions, 0x0, sizeof(functions) );

  if ( (major < 1) && (minor < 0) )
  {
    throw GLException (
      JEM_FUNC,
      String::format (
        "invalid OpenGL version %i.%i", major, minor
      )
    );
  }

  version = vers;

  switch ( major )
  {
  default:

    // Include all following minor versions.

    minor = -1;

  case 4:

    switch ( minor )
    {
    default:
    case 5:

      functions[GLVersion::VERSION_4_5_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_4_5].supported = true;
      }

    case 4:

      functions[GLVersion::VERSION_4_4_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_4_4].supported = true;
      }

    case 3:

      functions[GLVersion::VERSION_4_3_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_4_3].supported = true;
      }

    case 2:

      functions[GLVersion::VERSION_4_2_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_4_2].supported = true;
      }

    case 1:

      functions[GLVersion::VERSION_4_1_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_4_1].supported = true;
      }

    case 0:

      functions[GLVersion::VERSION_4_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_4].supported = true;
      }
    }

    minor = -1;

  case 3:

    switch ( minor )
    {
    default:
    case 3:

      functions[GLVersion::VERSION_3_3_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_3_3].supported = true;
      }

    case 2:

      functions[GLVersion::VERSION_3_2_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_3_2].supported = true;
      }

    case 1:

      functions[GLVersion::VERSION_3_1_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_3_1].supported = true;
      }

    case 0:

      functions[GLVersion::VERSION_3_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_3].supported = true;
      }
    }

    minor = -1;

  case 2:

    switch ( minor )
    {
    default:
    case 1:

      functions[GLVersion::VERSION_2_1_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_2_1].supported = true;
      }

    case 0:

      functions[GLVersion::VERSION_2_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_2].supported = true;
      }
    }

    minor = -1;

  case 1:

    switch ( minor )
    {
    default:
    case 5:

      functions[GLVersion::VERSION_1_5_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_1_5].supported = true;
      }

    case 4:

      functions[GLVersion::VERSION_1_4_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_1_4].supported = true;
      }

    case 3:

      functions[GLVersion::VERSION_1_3_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_1_3].supported = true;
      }

    case 2:

      functions[GLVersion::VERSION_1_2_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_1_2].supported = true;
      }

    case 1:

      functions[GLVersion::VERSION_1_1_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_1_1].supported = true;
      }

    case 0:

      functions[GLVersion::VERSION_1_CORE].supported = true;

      if ( ! core )
      {
        functions[GLVersion::VERSION_1].supported = true;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   nextID
//-----------------------------------------------------------------------


int GLContext::Data_::nextID ()
{
  static int     id = 0;

  SerialSection  section;

  if ( id == maxOf( id ) )
  {
    id = 1;
  }
  else
  {
    id++;
  }

  return id;
}


//-----------------------------------------------------------------------
//   loadMatrix_
//-----------------------------------------------------------------------


void GLContext::Data_::loadMatrix_ ( int mxid )
{
  GLFuncs1*  funcs = findFuncs<GLFuncs1> ();

  if ( funcs )
  {
    switch ( mxid )
    {
    case GLContext::MODEL_VIEW:

      funcs->glMatrixMode ( GL_MODELVIEW );
      funcs->glLoadMatrix ( mstack.getCurrent() );
      break;

    case GLContext::PROJECTION:

      funcs->glMatrixMode ( GL_PROJECTION );
      funcs->glLoadMatrix ( pstack.getCurrent() );
      break;

    default:

      throw GLException (
        JEM_FUNC,
        String::format (
          "invalid matrix stack identifier: %d",
          mxid
        )
      );
    }
  }
}


//=======================================================================
//   class GLContext
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  GLContext::COLOR      = 1 << 0;
const int  GLContext::SPECULAR   = 1 << 1;
const int  GLContext::LIGHTING   = 1 << 2;
const int  GLContext::LINE_WIDTH = 1 << 3;
const int  GLContext::POINT_SIZE = 1 << 4;
const int  GLContext::ALPHA_TEST = 1 << 5;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLContext::GLContext ()
{
  id_   = Data_::nextID ();
  data_ = nullptr;

  std::memset ( &cache_, 0x0, sizeof(cache_) );
}


GLContext::~GLContext ()
{
  clear ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void GLContext::clear () noexcept
{
  try
  {
    // Clear the OpenGL objects before deleting data_.

    if ( data_ )
    {
      getData_().objects .freeAllObjects ( *this );
      getData_().programs.clear          ();
    }

    // All clear to delete data_.

    if ( data_ )
    {
      delete (Data_*) data_;

      data_ = nullptr;
    }

    std::memset ( &cache_, 0x0, sizeof(cache_) );
  }
  catch ( ... )
  {}
}


//-----------------------------------------------------------------------
//   rotate
//-----------------------------------------------------------------------


void GLContext::rotate

  ( GLdouble  a,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z,
    int       mxid )

{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.rotate     ( a, x, y, z );
  data .loadMatrix ( mxid );
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void GLContext::scale

  ( GLdouble  s,
    int       mxid )

{
  scale ( s, s, s, mxid );
}


void GLContext::scale

  ( GLdouble  sx,
    GLdouble  sy,
    int       mxid )

{
  scale ( sx, sy, 1.0, mxid );
}


void GLContext::scale

  ( GLdouble  sx,
    GLdouble  sy,
    GLdouble  sz,
    int       mxid )

{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.scale      ( sx, sy, sz );
  data .loadMatrix ( mxid );
}


//-----------------------------------------------------------------------
//   translate
//-----------------------------------------------------------------------


void GLContext::translate

  ( GLdouble  dx,
    GLdouble  dy,
    int       mxid )

{
  translate ( dx, dy, 0.0, mxid );
}


void GLContext::translate

  ( GLdouble  dx,
    GLdouble  dy,
    GLdouble  dz,
    int       mxid )

{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.translate  ( dx, dy, dz );
  data .loadMatrix ( mxid );
}


//-----------------------------------------------------------------------
//   transform
//-----------------------------------------------------------------------


void GLContext::transform

  ( const GLdouble*  tr,
    int              mxid )

{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.transform  ( tr );
  data .loadMatrix ( mxid );
}


void GLContext::transform

  ( const Transform&  tr,
    int               mxid )

{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = getData_().getStack ( mxid );

  stack.transform  ( tr.addr() );
  data .loadMatrix ( mxid );
}


//-----------------------------------------------------------------------
//   getTransform
//-----------------------------------------------------------------------


void GLContext::getTransform

  ( GLfloat*  tr,
    int       mxid ) const

{
  MatrixStack&     stack = getData_().getStack ( mxid );
  const GLdouble*  src   = stack.getCurrent    ();

  for ( int i = 0; i < 16; i++ )
  {
    tr[i] = (GLfloat) src[i];
  }
}


void GLContext::getTransform

  ( GLdouble*  tr,
    int        mxid ) const

{
  MatrixStack&     stack = getData_().getStack ( mxid );
  const GLdouble*  src   = stack.getCurrent    ();

  std::memcpy ( tr, src, 16 * sizeof(*src) );
}


Transform GLContext::getTransform ( int mxid ) const
{
  MatrixStack&     stack = getData_().getStack ( mxid );
  const GLdouble*  src   = stack.getCurrent    ();

  Transform        tr;

  std::memcpy ( tr.addr(), src, 16 * sizeof(*src) );

  return tr;
}


//-----------------------------------------------------------------------
//   setTransform
//-----------------------------------------------------------------------


void GLContext::setTransform

  ( const GLdouble*  tr,
    int              mxid )

{
  Data_&        data  = getData_         ();
  MatrixStack&  stack = data .getStack   ( mxid );
  GLdouble*     dest  = stack.getCurrent ();

  std::memcpy     ( dest, tr, 16 * sizeof(*dest) );
  data.loadMatrix ( mxid );
}


void GLContext::setTransform

  ( const Transform&  tr,
    int               mxid )

{
  Data_&        data  = getData_         ();
  MatrixStack&  stack = data .getStack   ( mxid );
  GLdouble*     dest  = stack.getCurrent ();

  std::memcpy     ( dest, tr.addr(), 16 * sizeof(*dest) );
  data.loadMatrix ( mxid );
}


//-----------------------------------------------------------------------
//   setIdentity
//-----------------------------------------------------------------------


void GLContext::setIdentity ( int mxid )
{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.setIdentity ();
  data .loadMatrix  ( mxid );
}


//-----------------------------------------------------------------------
//   setOrtho
//-----------------------------------------------------------------------


void GLContext::setOrtho

  ( GLdouble  left,
    GLdouble  right,
    GLdouble  bot,
    GLdouble  top,
    GLdouble  near,
    GLdouble  far,
    int       mxid )

{
  JEM_PRECHECK ( (right > left) && (top > bot) );

  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.setOrtho   ( left, right, bot, top, near, far );
  data .loadMatrix ( mxid );
}


void GLContext::setOrtho

  ( const Rect&  rect,
    GLdouble     near,
    GLdouble     far,
    int          mxid )

{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.setOrtho   ( (GLdouble)  rect.x,
                     (GLdouble) (rect.x + rect.width),
                     (GLdouble)  rect.y,
                     (GLdouble) (rect.y + rect.height),
                     near, far );
  data .loadMatrix ( mxid );
}


//-----------------------------------------------------------------------
//   popTransform
//-----------------------------------------------------------------------


void GLContext::popTransform ( int mxid )
{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.popMatrix  ();
  data .loadMatrix ( mxid );
}


//-----------------------------------------------------------------------
//   pushTransform
//-----------------------------------------------------------------------


void GLContext::pushTransform ( int mxid )
{
  MatrixStack&  stack = getData_().getStack ( mxid );

  stack.pushMatrix ();
}


//-----------------------------------------------------------------------
//   resetTransform
//-----------------------------------------------------------------------


void GLContext::resetTransform ( int mxid )
{
  Data_&        data  = getData_      ();
  MatrixStack&  stack = data.getStack ( mxid );

  stack.clear      ();
  data .loadMatrix ( mxid );
}


//-----------------------------------------------------------------------
//   resetTransforms
//-----------------------------------------------------------------------


void GLContext::resetTransforms ()
{
  Data_&        data   = getData_            ();
  MatrixStack&  mstack = getData_().getStack ( MODEL_VIEW );
  MatrixStack&  pstack = getData_().getStack ( PROJECTION );

  mstack.clear ();
  pstack.clear ();

  if ( data.version.getVersionID() < GLVersion::VERSION_3_3 )
  {
    GLFuncs1*  funcs = findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      funcs->glMatrixMode ( GL_MODELVIEW );
      funcs->glLoadMatrix ( mstack.getCurrent() );
      funcs->glMatrixMode ( GL_PROJECTION );
      funcs->glLoadMatrix ( pstack.getCurrent() );
    }
  }
}


//-----------------------------------------------------------------------
//   getModelViewProj
//-----------------------------------------------------------------------


void GLContext::getModelViewProj ( GLfloat* mvp ) const
{
  GLdouble  tmp[16];

  getModelViewProj ( tmp );

  for ( int i = 0; i < 16; i++ )
  {
    mvp[i] = (GLfloat) tmp[i];
  }
}


void GLContext::getModelViewProj ( GLdouble* mvp ) const
{
  MatrixStack&     mstack = getData_().getStack ( MODEL_VIEW );
  MatrixStack&     pstack = getData_().getStack ( PROJECTION );

  const GLdouble*  m      = mstack.getCurrent ();
  const GLdouble*  p      = pstack.getCurrent ();


  mvp[ 0] = (p[ 0] * m[ 0] + p[ 4] * m[ 1] +
             p[ 8] * m[ 2] + p[12] * m[ 3]);
  mvp[ 1] = (p[ 1] * m[ 0] + p[ 5] * m[ 1] +
             p[ 9] * m[ 2] + p[13] * m[ 3]);
  mvp[ 2] = (p[ 2] * m[ 0] + p[ 6] * m[ 1] +
             p[10] * m[ 2] + p[14] * m[ 3]);
  mvp[ 3] = (p[ 3] * m[ 0] + p[ 7] * m[ 1] +
             p[11] * m[ 2] + p[15] * m[ 3]);

  mvp[ 4] = (p[ 0] * m[ 4] + p[ 4] * m[ 5] +
             p[ 8] * m[ 6] + p[12] * m[ 7]);
  mvp[ 5] = (p[ 1] * m[ 4] + p[ 5] * m[ 5] +
             p[ 9] * m[ 6] + p[13] * m[ 7]);
  mvp[ 6] = (p[ 2] * m[ 4] + p[ 6] * m[ 5] +
             p[10] * m[ 6] + p[14] * m[ 7]);
  mvp[ 7] = (p[ 3] * m[ 4] + p[ 7] * m[ 5] +
             p[11] * m[ 6] + p[15] * m[ 7]);

  mvp[ 8] = (p[ 0] * m[ 8] + p[ 4] * m[ 9] +
             p[ 8] * m[10] + p[12] * m[11]);
  mvp[ 9] = (p[ 1] * m[ 8] + p[ 5] * m[ 9] +
             p[ 9] * m[10] + p[13] * m[11]);
  mvp[10] = (p[ 2] * m[ 8] + p[ 6] * m[ 9] +
             p[10] * m[10] + p[14] * m[11]);
  mvp[11] = (p[ 3] * m[ 8] + p[ 7] * m[ 9] +
             p[11] * m[10] + p[15] * m[11]);

  mvp[12] = (p[ 0] * m[12] + p[ 4] * m[13] +
             p[ 8] * m[14] + p[12] * m[15]);
  mvp[13] = (p[ 1] * m[12] + p[ 5] * m[13] +
             p[ 9] * m[14] + p[13] * m[15]);
  mvp[14] = (p[ 2] * m[12] + p[ 6] * m[13] +
             p[10] * m[14] + p[14] * m[15]);
  mvp[15] = (p[ 3] * m[12] + p[ 7] * m[13] +
             p[11] * m[14] + p[15] * m[15]);
}


Transform GLContext::getModelViewProj () const
{
  Transform  tr;

  getModelViewProj ( tr.addr() );

  return tr;
}


//-----------------------------------------------------------------------
//   getNormalMatrix
//-----------------------------------------------------------------------


void GLContext::getNormalMatrix ( GLfloat* norm ) const
{
  Transform  tr;

  getTransform ( tr.addr(), MODEL_VIEW );
  invert       ( tr );

  // Return the transpose of tr

  norm[0] = (GLfloat) tr(0,0);
  norm[1] = (GLfloat) tr(0,1);
  norm[2] = (GLfloat) tr(0,2);

  norm[3] = (GLfloat) tr(1,0);
  norm[4] = (GLfloat) tr(1,1);
  norm[5] = (GLfloat) tr(1,2);

  norm[6] = (GLfloat) tr(2,0);
  norm[7] = (GLfloat) tr(2,1);
  norm[8] = (GLfloat) tr(2,2);
}


void GLContext::getNormalMatrix ( GLdouble* norm ) const
{
  Transform  tr;

  getTransform ( tr.addr(), MODEL_VIEW );
  invert       ( tr );

  // Return the transpose of tr

  norm[0] = tr(0,0);
  norm[1] = tr(0,1);
  norm[2] = tr(0,2);

  norm[3] = tr(1,0);
  norm[4] = tr(1,1);
  norm[5] = tr(1,2);

  norm[6] = tr(2,0);
  norm[7] = tr(2,1);
  norm[8] = tr(2,2);
}


//-----------------------------------------------------------------------
//   newTexture
//-----------------------------------------------------------------------


GLuint GLContext::newTexture

  ( const void*         addr,
    const Ref<Object>&  owner )

{
  return getData_().objects.newObject (
    *this, GLObject::TEXTURE, addr, owner
  );
}


//-----------------------------------------------------------------------
//   newGLObject
//-----------------------------------------------------------------------


GLuint GLContext::newGLObject

  ( int                 type,
    const void*         addr,
    const Ref<Object>&  owner )

{
  return getData_().objects.newObject (
    *this, type, addr, owner
  );
}


//-----------------------------------------------------------------------
//   getGLObject
//-----------------------------------------------------------------------


GLuint GLContext::getGLObject ( const void* addr ) const
{
  GLuint  oid = 0;

  if ( ! getData_().objects.findObject( oid, addr ) )
  {
    throw GLException ( JEM_FUNC, "OpenGL object not found" );
  }

  return oid;
}


//-----------------------------------------------------------------------
//   findGLObject
//-----------------------------------------------------------------------


bool GLContext::findGLObject

  ( GLuint&      oid,
    const void*  addr ) const

{
  return getData_().objects.findObject ( oid, addr );
}


//-----------------------------------------------------------------------
//   freeGLObject
//-----------------------------------------------------------------------


void GLContext::freeGLObject ( const void* addr )
{
  getData_().objects.freeObject ( *this, addr );
}


//-----------------------------------------------------------------------
//   getVersion
//-----------------------------------------------------------------------


GLVersion GLContext::getVersion () const
{
  return getData_().version;
}


//-----------------------------------------------------------------------
//   setVersion
//-----------------------------------------------------------------------


GLVersion GLContext::setVersion ( const GLVersion& vers )
{
  Data_&  data = getData_ ();

  // Note that the set version can not be later than the initial
  // version (the actual version associated with the OpenGL
  // context).

  if ( vers.getVersionID() <= data.version0.getVersionID() )
  {
    std::memset ( &cache_, 0x0, sizeof(cache_) );

    data.setVersion ( vers );
  }

  return data.version;
}


//-----------------------------------------------------------------------
//   checkVersion
//-----------------------------------------------------------------------


bool GLContext::checkVersion

  ( int  major,
    int  minor ) const

{
  GLVersion  ver ( major, minor );

  return (ver.getVersionID() <= getData_().version.getVersionID());
}


bool GLContext::checkVersion ( GLVersion::VersionID vid ) const
{
  return (vid <= getData_().version.getVersionID());
}


//-----------------------------------------------------------------------
//   errorString
//-----------------------------------------------------------------------


String GLContext::errorString ( GLenum err ) const
{
  String  str = "unknown OpenGL error";

  switch ( err )
  {
  case GL_NO_ERROR:

    str = "no error";
    break;

#ifdef GL_INVALID_ENUM

  case GL_INVALID_ENUM:

    str = "invalid enumeration value";
    break;

#endif

#ifdef GL_INVALID_VALUE

  case GL_INVALID_VALUE:

    str = "invalid parameter value";
    break;

#endif

#ifdef GL_INVALID_OPERATION

  case GL_INVALID_OPERATION:

    str = "invalid operation";
    break;

#endif

#ifdef GL_STACK_OVERFLOW

  case GL_STACK_OVERFLOW:

    str = "stack overflow";
    break;

#endif

#ifdef GL_STACK_UNDERFLOW

  case GL_STACK_UNDERFLOW:

    str = "stack underflow";
    break;

#endif

#ifdef GL_OUT_OF_MEMORY

  case GL_OUT_OF_MEMORY:

    str = "out of memory";
    break;

#endif

  default:

    str = String::format ( "error code %d", (int) err );
  }

  return str;
}


//-----------------------------------------------------------------------
//   findProgram
//-----------------------------------------------------------------------


Ref<Object> GLContext::findProgram ( const String& name ) const
{
  Ref<Object>  prog;

  getData_().programs.find ( prog, name );

  return prog;
}


//-----------------------------------------------------------------------
//   storeProgram
//-----------------------------------------------------------------------


void GLContext::storeProgram

  ( const String&       name,
    const Ref<Object>&  prog ) const

{
  getData_().programs.set ( name, prog );
}


//-----------------------------------------------------------------------
//   getLights
//-----------------------------------------------------------------------


Ref<Lights> GLContext::getLights () const
{
  return getData_().lights;
}


//-----------------------------------------------------------------------
//   setLights
//-----------------------------------------------------------------------


void GLContext::setLights ( const Ref<Lights>& lights )
{
  getData_().lights = lights;
}


//-----------------------------------------------------------------------
//   getLighting
//-----------------------------------------------------------------------


bool GLContext::getLighting () const
{
  return getData_().lighting;
}


//-----------------------------------------------------------------------
//   setLighting
//-----------------------------------------------------------------------


void GLContext::setLighting ( bool choice )
{
  Data_&  data = getData_ ();

  if ( choice != data.lighting )
  {
    if ( data.version.getVersionID() < GLVersion::VERSION_3_3 )
    {
      GLFuncs1*  funcs = findFunctions<GLFuncs1> ();

      if ( funcs )
      {
        if ( choice )
        {
          funcs->glEnable  ( GL_LIGHTING );
        }
        else
        {
          funcs->glDisable ( GL_LIGHTING );
        }
      }
    }

    data.lighting = choice;
  }
}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


const Color& GLContext::getColor () const
{
  return getData_().color;
}


//-----------------------------------------------------------------------
//   setColor
//-----------------------------------------------------------------------


void GLContext::setColor ( const Color& color )
{
  Data_&  data = getData_ ();

  data.color = color;

  if ( ! checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLFuncs1*  funcs = findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      const GLfloat*  rgba = color.addr ();

      if ( rgba[3] > Color::OPAQUE_ALPHA )
      {
        funcs->glColor3 ( rgba );
      }
      else
      {
        funcs->glColor4 ( rgba );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getSpecular
//-----------------------------------------------------------------------


const Color& GLContext::getSpecular () const
{
  return getData_().specular;
}


//-----------------------------------------------------------------------
//   setSpecular
//-----------------------------------------------------------------------


void GLContext::setSpecular ( const Color& specular )
{
  getData_().specular = specular;

  if ( ! checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLFuncs1*  funcs = findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      funcs->glMaterial ( GL_FRONT_AND_BACK, GL_SPECULAR,
                          specular.addr() );
    }
  }
}


//-----------------------------------------------------------------------
//   getShininess
//-----------------------------------------------------------------------


GLfloat GLContext::getShininess () const
{
  return getData_().shininess;
}


//-----------------------------------------------------------------------
//   setShininess
//-----------------------------------------------------------------------


void GLContext::setShininess ( GLfloat shininess )
{
  Data_&  data = getData_ ();

  data.shininess = clamped ( shininess, 0.0F, 128.0F );

  if ( ! checkVersion( GLVersion::VERSION_3_3 ) )
  {
    GLFuncs1*  funcs = findFunctions<GLFuncs1> ();

    if ( funcs )
    {
      funcs->glMaterial ( GL_FRONT_AND_BACK, GL_SHININESS,
                          data.shininess );
    }
  }
}


//-----------------------------------------------------------------------
//   getLineWidth
//-----------------------------------------------------------------------


GLfloat GLContext::getLineWidth () const
{
  return getData_().lineWidth;
}


//-----------------------------------------------------------------------
//   setLineWidth
//-----------------------------------------------------------------------


void GLContext::setLineWidth ( GLfloat width )
{
  JEM_PRECHECK ( width >= 0.0F );

  // The line width is limited to a maximum of 1.0 for the core
  // profile; larger values will cause an OpenGl error. Therefore,
  // glLineWidth is only used for versions lower than 3.3.
  // Shaders should be used to create lines with a width of larger
  // than 1.0.

  Data_&  data  = getData_ ();

  if ( ! checkVersion( GLVersion::VERSION_3_3 ) &&
       (std::fabs( data.lineWidth - width ) > 0.01F) )
  {
    GLFuncs1Core*  funcs = findFunctions<GLFuncs1Core> ();

    if ( funcs )
    {
      funcs->glLineWidth ( width );
    }
  }

  data.lineWidth = width;
}


//-----------------------------------------------------------------------
//   getPointSize
//-----------------------------------------------------------------------


GLfloat GLContext::getPointSize () const
{
  return getData_().pointSize;
}


//-----------------------------------------------------------------------
//   setPointSize
//-----------------------------------------------------------------------


void GLContext::setPointSize ( GLfloat size )
{
  JEM_PRECHECK ( size >= 0.0F );

  Data_&  data  = getData_ ();

  if ( std::fabs( data.pointSize - size ) > 0.01F )
  {
    GLFuncs1Core*  funcs = findFunctions<GLFuncs1Core> ();

    if ( funcs )
    {
      funcs->glPointSize ( size );
    }
  }

  data.pointSize = size;
}


//-----------------------------------------------------------------------
//   getViewport
//-----------------------------------------------------------------------


Rect GLContext::getViewport () const
{
  return getData_().viewport;
}


//-----------------------------------------------------------------------
//   setViewport
//-----------------------------------------------------------------------


void GLContext::setViewport ( const Rect& viewport )
{
  Data_&         data  = getData_                    ();
  GLFuncs1Core*  funcs = findFunctions<GLFuncs1Core> ();

  if ( funcs )
  {
    funcs->glViewport ( viewport.x,     viewport.y,
                        viewport.width, viewport.height );
  }

  data.viewport = viewport;
}


//-----------------------------------------------------------------------
//   getAlphaTest
//-----------------------------------------------------------------------


GLfloat GLContext::getAlphaTest () const
{
  Data_&  data = getData_ ();

  if ( data.alphaTest )
  {
    return data.alphaThreshold;
  }
  else
  {
    return -1.0F;
  }
}


//-----------------------------------------------------------------------
//   setAlphaTest
//-----------------------------------------------------------------------


void GLContext::setAlphaTest ( GLfloat threshold )
{
  Data_&  data = getData_ ();

  if ( std::fabs( threshold - data.alphaThreshold ) > 0.01F )
  {
    if ( ! checkVersion( GLVersion::VERSION_3_3 ) )
    {
      GLFuncs1*  funcs = findFunctions<GLFuncs1> ();

      if ( funcs )
      {
        funcs->glAlphaFunc ( GL_GREATER, (GLclampf) threshold );
      }
    }
  }

  data.alphaThreshold = threshold;
}


//-----------------------------------------------------------------------
//   enableAlphaTest
//-----------------------------------------------------------------------


void GLContext::enableAlphaTest ( bool choice )
{
  Data_&  data = getData_ ();

  if ( choice != data.alphaTest )
  {
    if ( ! checkVersion( GLVersion::VERSION_3_3 ) )
    {
      GLFuncs1Core*  funcs = findFunctions<GLFuncs1Core> ();

      if ( funcs )
      {
        if ( choice )
        {
          funcs->glEnable  ( GL_ALPHA_TEST );
        }
        else
        {
          funcs->glDisable ( GL_ALPHA_TEST );
        }
      }
    }
  }

  data.alphaTest = choice;
}


//-----------------------------------------------------------------------
//   pushAttribs
//-----------------------------------------------------------------------


void GLContext::pushAttribs ( int mask )
{
  Data_&               data  = getData_               ();
  AttribStack::Frame&  frame = data.attribs.pushFrame ( mask );

  frame.lighting  = data.lighting;
  frame.lineWidth = data.lineWidth;
  frame.pointSize = data.pointSize;

  if ( mask & (COLOR | SPECULAR | ALPHA_TEST) )
  {
    if ( mask & COLOR )
    {
      frame.color      = data.color;
    }

    if ( mask & SPECULAR )
    {
      frame.specular   = data.specular;
      frame.shininess  = data.shininess;
    }

    if ( mask & ALPHA_TEST )
    {
      frame.alphaTest  = data.alphaTest;
      frame.alphaValue = data.alphaThreshold;
    }
  }
}


//-----------------------------------------------------------------------
//   popAttribs
//-----------------------------------------------------------------------


void GLContext::popAttribs ()
{
  Data_&               data  = getData_              ();
  AttribStack::Frame&  frame = data.attribs.popFrame ();
  int                  mask  = frame.mask;

  frame.mask = 0;

  if ( mask & COLOR )
  {
    setColor     ( frame.color );
  }

  if ( mask & LINE_WIDTH )
  {
    setLineWidth ( frame.lineWidth );
  }

  if ( mask & POINT_SIZE )
  {
    setPointSize ( frame.pointSize );
  }

  // Handle the less common attributes.

  if ( mask & (SPECULAR | LIGHTING | ALPHA_TEST) )
  {
    if ( mask & SPECULAR )
    {
      setSpecular     ( frame.specular );
      setShininess    ( frame.shininess );
    }

    if ( mask & LIGHTING )
    {
      setLighting     ( frame.lighting );
    }

    if ( mask & ALPHA_TEST )
    {
      setAlphaTest    ( frame.alphaValue );
      enableAlphaTest ( frame.alphaTest );
    }
  }
}


//-----------------------------------------------------------------------
//   getData_
//-----------------------------------------------------------------------


inline GLContext::Data_& GLContext::getData_ () const
{
  if ( ! data_ )
  {
    const_cast<Self*>( this )->initData_ ();
  }

  return *((Data_*) data_);
}


//-----------------------------------------------------------------------
//   initData_
//-----------------------------------------------------------------------


void GLContext::initData_ ()
{
  JEM_PRECHECK ( ! data_ );

  GLVersion           version;
  GLVersion::Profile  profile;
  int                 major;
  int                 minor;


#ifdef GL_VERSION

  version.setVersion (
    jem::clone ( (const char*) glGetString( GL_VERSION ) )
  );

#endif

  major = version.getMajorVersion ();
  minor = version.getMinorVersion ();

  // Determine which profile is in effect.

  profile = GLVersion::NO_PROFILE;

  if ( major >= 3 )
  {
    if      ( (major == 3) && (minor == 0) )
    {
      profile = GLVersion::COMPAT_PROFILE;

#ifdef GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT

      {
        GLint  flags = 0;

        glGetIntegerv ( GL_CONTEXT_FLAGS, &flags );

        if ( flags & GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT )
        {
          profile = GLVersion::CORE_PROFILE;
        }
      }
#endif
    }
    else if ( (major == 3) && (minor == 1) )
    {
      typedef const GLubyte* (JEM_APIENTRYP GetStringiFn)

        ( GLenum  name,
          GLuint  index );

      GetStringiFn  getStringi = nullptr;
      GLint         extCount   = 0;

      NativeLoader  loader;

      glGetIntegerv ( GL_NUM_EXTENSIONS, &extCount );

      profile    = GLVersion::CORE_PROFILE;
      getStringi = (GetStringiFn)

        loader.getFuncAddr ( "glGetStringi" );

      if ( ! getStringi )
      {
        extCount = 0;
      }

      for ( GLint i = 0; i < extCount; i++ )
      {
        String  ext = (const char*) getStringi ( GL_EXTENSIONS,
                                                 (GLuint) i );

        if ( ext.equalsIgnoreCase( "GL_ARB_compatibility" ) )
        {
          profile = GLVersion::COMPAT_PROFILE;
          break;
        }
      }
    }
    else
    {
      profile = GLVersion::CORE_PROFILE;

#ifdef GL_CONTEXT_COMPATIBILITY_PROFILE_BIT

      {
        GLint  mask = 0;

        glGetIntegerv ( GL_CONTEXT_PROFILE_MASK, &mask );

        if ( mask & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT )
        {
          profile = GLVersion::COMPAT_PROFILE;
        }
      }

#endif
    }
  }

  version.setProfile ( profile );

  data_ = (void*) new Data_ ( version );
}


//-----------------------------------------------------------------------
//   getFunctions_
//-----------------------------------------------------------------------


AbstractGLFuncs* GLContext::getFunctions_ ( int vid ) const
{
  AbstractGLFuncs*  funcs = findFunctions_ ( vid );

  if ( ! funcs )
  {
    Data_&     data    = getData_ ();
    GLVersion  version ( (GLVersion::VersionID) vid );

    throw GLException (
      JEM_FUNC,
      "unsupported OpenGL functions requested",
      String::format (
        "required OpenGL version is %s, but current "
        "version is %s",
        version     .toString (),
        data.version.toString ()
      )
    );
  }

  return funcs;
}


//-----------------------------------------------------------------------
//   findFunctions_
//-----------------------------------------------------------------------


AbstractGLFuncs* GLContext::findFunctions_ ( int vid ) const
{
  if ( cache_.versi2 == vid )
  {
    Cache_&  cache = const_cast<Self*>( this )->cache_;

    jem::swap ( cache.versi1, cache.versi2 );
    jem::swap ( cache.funcs1, cache.funcs2 );

    return cache.funcs1;
  }
  else
  {
    AbstractGLFuncs*  funcs = getData_().findFuncs ( vid );
    Cache_&           cache = const_cast<Self*>( this )->cache_;

    cache.versi2 = cache.versi1;
    cache.funcs2 = cache.funcs1;
    cache.versi1 = vid;
    cache.funcs1 = funcs;

    return funcs;
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   clearGLError
//-----------------------------------------------------------------------


void clearGLError ( GLContext& gtx )
{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  if ( funcs )
  {
    for ( int i = 0; i < 10; i++ )
    {
      GLenum  err = funcs->glGetError ();

      if ( err == GL_NO_ERROR )
      {
        break;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   checkGLError
//-----------------------------------------------------------------------


void             checkGLError

  ( GLContext&     gtx,
    const String&  where )

{
  static int     count = 0;

  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();


  if ( ! funcs )
  {
    return;
  }

  for ( int i = 0; i < 10; i++ )
  {
    GLenum  err = funcs->glGetError ();

    if ( err == GL_NO_ERROR )
    {
      break;
    }

    if      ( err == GL_OUT_OF_MEMORY )
    {
      String  what = "OpenGL out of memory error";

      if ( where.size() > 0 )
      {
        throw GLException ( where, what );
      }
      else
      {
        throw GLException ( JEM_FUNC, what );
      }
    }
    else if ( count < 20 )
    {
      print ( System::err(), "OpenGL error:  ",
                             gtx.errorString( err ), '\n' );

      if ( where.size() )
      {
        print ( System::err(), "Error context: ", where, '\n' );
      }

      count++;
    }
  }
}


//-----------------------------------------------------------------------
//   getClearMask
//-----------------------------------------------------------------------


GLbitfield getClearMask ( GLContext& gtx )
{
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();
  GLbitfield     mask  = GL_COLOR_BUFFER_BIT;

  if ( funcs )
  {
    if ( funcs->glIsEnabled( GL_DEPTH_TEST ) == GL_TRUE )
    {
      mask |= GL_DEPTH_BUFFER_BIT;
    }

    if ( funcs->glIsEnabled( GL_STENCIL_TEST ) == GL_TRUE )
    {
      mask |= GL_STENCIL_BUFFER_BIT;
    }
  }

  return mask;
}


//-----------------------------------------------------------------------
//   getDepthBits
//-----------------------------------------------------------------------


GLint getDepthBits ( GLContext& gtx )
{
  GLFuncs3Core*  funcs3 = gtx.findFunctions<GLFuncs3Core> ();
  GLFuncs1*      funcs1 = nullptr;
  GLint          depth  = 0;


  checkGLError ( gtx, JEM_FUNC );

  if ( funcs3 )
  {
    funcs3->glGetFramebufferAttachmentParameter (
      GL_DRAW_FRAMEBUFFER,
      GL_DEPTH,
      GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,
      depth
    );

    if ( ! depth )
    {
      funcs3->glGetFramebufferAttachmentParameter (
        GL_DRAW_FRAMEBUFFER,
        GL_DEPTH_ATTACHMENT,
        GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,
        depth
      );
    }
  }

  if ( ! depth )
  {
    funcs1 = gtx.findFunctions<GLFuncs1> ();

    if ( funcs1 )
    {
      funcs1->glGetParameter ( GL_DEPTH_BITS, &depth );
    }
  }

  clearGLError ( gtx );

  return depth;
}


//-----------------------------------------------------------------------
//   finishGLOps
//-----------------------------------------------------------------------


void finishGLOps ( GLContext& gtx )
{
  GLFuncs2Core*  funcs = gtx.findFunctions<GLFuncs2Core> ();

  if ( funcs )
  {
    funcs->glFinish ();
  }
  else
  {
    GLFuncs1Core*  funcs1 = gtx.findFunctions<GLFuncs1Core> ();

    if ( funcs1 )
    {
      funcs1->glFlush ();
    }
  }
}


JEM_END_PACKAGE( gl )
