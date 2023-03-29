
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


#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Array.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Box.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Transformable.h>
#include <jem/gl/Camera.h>


JEM_DEFINE_CLASS( jem::gl::Camera );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Camera
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Camera::~Camera ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void Camera::reset ()
{}


//-----------------------------------------------------------------------
//   isOrtho
//-----------------------------------------------------------------------


bool Camera::isOrtho () const
{
  return false;
}


//-----------------------------------------------------------------------
//   lookAtObj
//-----------------------------------------------------------------------


void Camera::lookAtObj ( const GfxNode& obj )
{
  Box  box;

  obj.getBBox ( box );

  if ( ! box.isEmpty() )
  {
    Transform  t = getTransformOf ( obj.getParent() );

    box.transform ( t   );
    lookAtBox     ( box );
  }
}


//-----------------------------------------------------------------------
//   lookAtRect
//-----------------------------------------------------------------------


void Camera::lookAtRect

  ( RenderContext&  rtx,
    const Rect&     rect )

{
  if ( rect.width == 0 || rect.height == 0 )
  {
    return;
  }

  Transform  proj  = getProjection ();
  Rect       vport = getViewport   ();

  Vertex3d   a, b, c;
  Vertex3d   u, v;
  GLdouble   clip0, clip1;


  if ( isOrtho() )
  {
    clip0 = 0.0;
    clip1 = 1.0;
  }
  else
  {
    getZRange_ ( rtx, clip0, clip1, rect );
  }

  if ( invert( proj ) )
  {
    // Define three vertices at the surface of the selected volume:
    // lower left front, upper right front and middle back.

    a[0] = rect.x;
    a[1] = rect.y;
    a[2] = clip0;

    b[0] = rect.x + rect.width;
    b[1] = rect.y + rect.height;
    b[2] = clip0;

    c[0] = 0.5 * (a[0] + b[0]);
    c[1] = 0.5 * (a[1] + b[1]);
    c[2] = clip1;

    // Convert to viewing coordinates.

    fromWindow  ( a, vport );
    fromWindow  ( b, vport );
    fromWindow  ( c, vport );

    a = project ( proj, a );
    b = project ( proj, b );
    c = project ( proj, c );

    // Determine the bounding box.

    for ( int i = 0; i < 3; i++ )
    {
      u[i] = min ( a[i], b[i], c[i] );
      v[i] = max ( a[i], b[i], c[i] );
    }

    lookAtBox ( Box( u, v ) );

    if ( isOrtho() )
    {
      updateZRange ();
    }

    rtx.postRedraw ( this );
  }
}


//-----------------------------------------------------------------------
//   updateZRange
//-----------------------------------------------------------------------


void Camera::updateZRange ()
{}


//-----------------------------------------------------------------------
//   setZoom & getZoom
//-----------------------------------------------------------------------


GLdouble Camera::setZoom ( GLdouble factor )
{
  return 0.0;
}


GLdouble Camera::getZoom () const
{
  return 0.0;
}


//-----------------------------------------------------------------------
//   getZRange_
//-----------------------------------------------------------------------


void Camera::getZRange_

  ( RenderContext&  rtx,
    GLdouble&       clip0,
    GLdouble&       clip1,
    const Rect&     rect )

{
  const int      size  = (int) (rect.width * rect.height);

  GLContext&     gtx   = rtx.getGLContext ();
  GLFuncs1Core*  funcs = gtx.findFunctions<GLFuncs1Core> ();

  GLfloatVector  zBuf  ( size );
  GLfloat*       zData = zBuf.addr ();

  GLfloat        zMin  = 1.0F;
  GLfloat        zMax  = 0.0F;


  if ( ! funcs )
  {
    return;
  }

  funcs->glClear      ( GL_DEPTH_BUFFER_BIT );
  gtx.pushTransform   ();
  takeGLAction        ( rtx, (int) GLActions::DRAW_CDATA,
                        Properties() );

  // Must flush before reading pixel data!

  funcs->glFlush      ();
  gtx.popTransform    ();
  funcs->glReadPixels ( rect.x, rect.y, rect.width, rect.height,
                        GL_DEPTH_COMPONENT, GL_FLOAT, zData );

  // Find the first touched pixel.

  for ( int i = 0; i < size; i++ )
  {
    GLfloat  depth = zData[i];

    // Skip over untouched pixels.

    if ( depth >= 0.999F )
    {
      continue;
    }

    if ( depth < zMin )
    {
      zMin = depth;
    }

    if ( depth > zMax )
    {
      zMax = depth;
    }
  }

  if ( zMin >= zMax )
  {
    clip0 = clip1 = zMin;
  }
  else
  {
    clip0 = zMin;
    clip1  = zMax;
  }

  // make sure that the z-range is not empty.

  clip0 *= 0.99;
  clip1 *= 1.01;

  clamp ( clip1, 0.0, 1.0 );
}


JEM_END_PACKAGE( gl )
