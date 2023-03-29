
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/Time.h>
#include <jem/base/ClassTemplate.h>
#include <jem/gl/gl.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Box.h>
#include <jem/gl/Actions.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/OrthoCamera.h>


JEM_DEFINE_CLASS( jem::gl::OrthoCamera );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class OrthoCamera
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const GLdouble  OrthoCamera::MIN_ZOOM = -20.0;
const GLdouble  OrthoCamera::MAX_ZOOM =  20.0;

const GLdouble  OrthoCamera::RANGE    =  1.0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


OrthoCamera::OrthoCamera

  ( const String&        name,
    const Ref<GfxNode>&  child ) :

    Super ( name, child )

{
  range_    =  RANGE;
  nearClip_ = -RANGE;
  farClip_  =  RANGE;
}


OrthoCamera::~OrthoCamera ()
{}


//-----------------------------------------------------------------------
//   isOrtho
//-----------------------------------------------------------------------


bool OrthoCamera::isOrtho () const
{
  return true;
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void OrthoCamera::reset ()
{
  Super::reset ();

  range_    =  RANGE;
  nearClip_ = -RANGE;
  farClip_  =  RANGE;
}


//-----------------------------------------------------------------------
//   lookAtBox
//-----------------------------------------------------------------------


void OrthoCamera::lookAtBox ( const Box& box )
{
  JEM_PRECHECK ( ! box.isEmpty() );

  Vertex3d  p, q;
  GLdouble  a, w, h, d;


  box.getCorners     ( p, q );

  transfm_.translate ( -0.5 * (p[0] + q[0]),
                       -0.5 * (p[1] + q[1]),
                       -0.5 * (p[2] + q[2]) );

  w = q[0] - p[0];
  h = q[1] - p[1];
  d = q[2] - p[2];

  a = vport_.aspectRatio ();

  if ( w * a > h )
  {
    range_ = w;

    if ( a < 1.0 )
    {
      range_ *= a;
    }
  }
  else
  {
    range_ = h;

    if ( a > 1.0 )
    {
      range_ /= a;
    }
  }

  range_   *=  0.51 * ::pow ( 2.0, zoom_ );
  nearClip_ = -0.60 * d - 0.001 * range_;
  farClip_  =  0.60 * d + 0.001 * range_;
  newProj_  =  true;
}


//-----------------------------------------------------------------------
//   getVisibleBox
//-----------------------------------------------------------------------


void OrthoCamera::getVisibleBox ( Box& box ) const
{
  Vertex3d  p, q;
  GLdouble  a, d, h, w;


  d = (farClip_ - nearClip_ - 0.002 * range_) / 1.2;
  d = max ( d, 0.0 );
  w = range_ / (0.51 * ::pow( 2.0, zoom_ ));
  a = vport_.aspectRatio ();

  if ( a > 1.0 )
  {
    h = w * a;
  }
  else
  {
    h = w;
    w = w / (a + 0.001);
  }

  p[0] = -0.5 * w;
  p[1] = -0.5 * h;
  p[2] = -0.5 * d;
  q[0] =  0.5 * w;
  q[1] =  0.5 * h;
  q[2] =  0.5 * d;

  box.setCorners ( p, q );
}


//-----------------------------------------------------------------------
//   translateObj
//-----------------------------------------------------------------------


void OrthoCamera::translateObj

  ( GfxNode&  node,
    GLdouble  dx,
    GLdouble  dy,
    GLdouble  dz ) const

{
  Transformable*  tf = toTransformable ( &node );

  Transform       t, u;
  GLdouble        a, r;


  if ( ! tf )
  {
    return;
  }

  u   = getTransformOf   ( node.getParent() );
  t   = tf->getTransform ();

  t.transform ( u );

  r   = range_ * ::pow ( 2.0, -zoom_ );
  dx *= 2 * r;
  dy *= 2 * r;

  a   = vport_.aspectRatio ();

  if ( a > 1.0 )
  {
    dy *= a;
  }
  else
  {
    dx /= a;
  }

  t.translate ( dx, dy, dz );

  if ( ! invert( u ) )
  {
    return;
  }

  t.transform      ( u );
  tf->setTransform ( t );
}


//-----------------------------------------------------------------------
//   setZoom
//-----------------------------------------------------------------------


GLdouble OrthoCamera::setZoom ( GLdouble z )
{
  return Super::setZoom ( clamped( z, MIN_ZOOM, MAX_ZOOM ) );
}


//-----------------------------------------------------------------------
//   jitter
//-----------------------------------------------------------------------


void OrthoCamera::jitter

  ( Transform&  proj,
    GLdouble    dx,
    GLdouble    dy ) const

{
  proj(0,3) -= 2.0 * dx / (GLdouble) vport_.width;
  proj(1,3) -= 2.0 * dy / (GLdouble) vport_.height;
}


//-----------------------------------------------------------------------
//   setRange
//-----------------------------------------------------------------------


void OrthoCamera::setRange ( GLdouble range )
{
  JEM_PRECHECK ( range >= 0.0 );

  range_   = range;
  newProj_ = true;
}


//-----------------------------------------------------------------------
//   updateZRange
//-----------------------------------------------------------------------


void OrthoCamera::updateZRange ()
{
  Box        box;
  Vertex3d   a, b;
  GLdouble   d;
  bool       viz;


  getBBox ( box );

  if ( box.isEmpty() )
  {
    return;
  }

  box.getCorners ( a, b );

  newProj_  =  true;
  d         =  b[2] - a[2];
  nearClip_ = -b[2] - 0.1 * d - 1.0;
  farClip_  = -a[2] + 0.1 * d + 1.0;

  getVisibility ( viz, box, newProjection_() );

  if ( box.isEmpty() )
  {
    return;
  }

  box.getCorners ( a, b );

  d         =  b[2] - a[2];
  nearClip_ = -b[2] - 0.1 * d - 0.001 * range_;
  farClip_  = -a[2] + 0.1 * d + 0.001 * range_;
}


//-----------------------------------------------------------------------
//   setZRange
//-----------------------------------------------------------------------


void OrthoCamera::setZRange

  ( GLdouble  near,
    GLdouble  far )

{
  JEM_PRECHECK ( near > 0.0 && near < far );

  nearClip_ = near;
  farClip_  = far;
  newProj_  = true;
}


//-----------------------------------------------------------------------
//   newProjection_
//-----------------------------------------------------------------------


Transform OrthoCamera::newProjection_ () const
{
  Transform  p;
  GLdouble   a, w, h, r;


  r = range_ * ::pow ( 2.0, -zoom_ );
  a = vport_.aspectRatio ();

  if ( a > 1.0 )
  {
    w = r;
    h = r * a;
  }
  else
  {
    w = r / a;
    h = r;
  }

  p.setIdentity ();

  p(0,0) = 1.0 / w;
  p(1,1) = 1.0 / h;
  p(2,2) = 2.0 / (nearClip_ - farClip_);
  p(2,3) = (farClip_ + nearClip_) / (nearClip_ - farClip_);

  return p;
}


JEM_END_PACKAGE( gl )
