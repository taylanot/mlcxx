
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/tuple/operators.h>
#include <jem/util/Properties.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Box.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/Camera3D.h>


JEM_DEFINE_CLASS( jem::gl::Camera3D );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Camera3D
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const GLdouble  Camera3D::MIN_ZOOM   =  -5.0;
const GLdouble  Camera3D::MAX_ZOOM   =   5.0;

const GLdouble  Camera3D::MIN_ANGLE  =   1.0;
const GLdouble  Camera3D::MAX_ANGLE  = 120.0;
const GLdouble  Camera3D::VIEW_ANGLE =  40.0;

const GLdouble  Camera3D::MIN_FAR    =   1.0e-6;
const GLdouble  Camera3D::MAX_FAR    =   1.0e+6;
const GLdouble  Camera3D::MIN_NEAR   =   0.001;
const GLdouble  Camera3D::MAX_NEAR   =   0.999;
const GLdouble  Camera3D::FAR_CLIP   =  10.0;
const GLdouble  Camera3D::NEAR_CLIP  =   0.5;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Camera3D::Camera3D

  ( const String&        name,
    const Ref<GfxNode>&  child ) :

    Super ( name, child )

{
  fovy_     = VIEW_ANGLE;
  farClip_  = FAR_CLIP;
  nearClip_ = NEAR_CLIP;

  transfm_.translate ( 0.0, 0.0, -3.0 );
}


Camera3D::~Camera3D ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Camera3D::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    Vertex3d    pos;
    double      clip;
    double      fovy;
    bool        found;


    found = myProps.find ( clip, PropNames::FAR_CLIP,
                           (double) MIN_FAR,
                           (double) MAX_FAR );

    if ( found )
    {
      farClip_     = (GLdouble) clip;
      newProj_ = true;

      clamp ( nearClip_, (farClip_ * MIN_NEAR),
                         (farClip_ * MAX_NEAR) );
    }

    found = myProps.find ( clip, PropNames::NEAR_CLIP,
                           (double) (farClip_ * MIN_NEAR),
                           (double) (farClip_ * MAX_NEAR) );

    if ( found )
    {
      nearClip_ = (GLdouble) clip;
      newProj_  = true;
    }

    found = myProps.find ( fovy, PropNames::VIEW_ANGLE,
                           (double) MIN_ANGLE,
                           (double) MAX_ANGLE );

    if ( found )
    {
      fovy_    = (GLdouble) fovy;
      newProj_ = true;
    }

    pos = 0.0;

    if ( Vertex::configure( pos, PropNames::POSITION, myProps ) )
    {
      transfm_.setTranslation ( -pos[0], -pos[1], -pos[2] );
    }
  }

  Super::configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Camera3D::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  Vertex3d    pos;

  pos[0] = -transfm_(0,3);
  pos[1] = -transfm_(1,3);
  pos[2] = -transfm_(2,3);

  myProps.set ( PropNames::FAR_CLIP,   farClip_  );
  myProps.set ( PropNames::NEAR_CLIP,  nearClip_ );
  myProps.set ( PropNames::VIEW_ANGLE, fovy_     );

  Vertex::getConfig (  PropNames::POSITION, myProps, pos );
  Super ::getConfig ( props );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void Camera3D::listProps ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::FAR_CLIP,
                "The z-position (in eye coordinates) "
                "of the far clipping plane" );
  myProps.set ( PropNames::NEAR_CLIP,
                "The z-position (in eye coordinates) "
                "of the near clipping plane" );
  myProps.set ( PropNames::VIEW_ANGLE,
                "The view angle" );
  myProps.set ( PropNames::POSITION,
                "The position of the camera" );

  Super::listProps ( props );
}

//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void Camera3D::reset ()
{
  Super::reset ();

  fovy_     = VIEW_ANGLE;
  farClip_  = FAR_CLIP;
  nearClip_ = NEAR_CLIP;

  transfm_.translate ( 0.0, 0.0, -3.0 );
}


//-----------------------------------------------------------------------
//   lookAtBox
//-----------------------------------------------------------------------


void Camera3D::lookAtBox ( const Box& box )
{
  using jem::max;

  JEM_PRECHECK ( ! box.isEmpty() );

  GLdouble   teta = getTeta_ ();

  Vertex3d   a, b, c;
  GLdouble   w, h, d;
  GLdouble   dz;
  GLdouble   s, t;


  box.getCorners ( a, b );

  w  = b[0] - a[0];
  h  = b[1] - a[1];
  d  = b[2] - a[2];

  s  = max ( w, h );
  c  = 0.5 * (a + b);
  dz = 0.5 * (d + (1.05 * s) / (GLdouble) ::tan( teta ));

  transfm_.translate ( -c[0], -c[1], -(c[2] + dz) );

  s        = max ( s, d );
  t        = 0.5;
  farClip_ = dz + 0.5 * d + t * s;

  clamp ( farClip_, MIN_FAR, MAX_FAR );

  nearClip_ = dz - 0.5 * d - t * s;

  for ( int i = 0; i < 8; i++ )
  {
    if ( nearClip_ >= (farClip_ * MIN_NEAR) )
    {
      break;
    }

    t        *= 0.5;
    nearClip_ = dz - 0.5 * d - t * s;
  }

  clamp ( nearClip_, (farClip_ * MIN_NEAR),
                     (farClip_ * MAX_NEAR) );

  newProj_ = true;
}


//-----------------------------------------------------------------------
//   getVisibleBox
//-----------------------------------------------------------------------


void Camera3D::getVisibleBox ( Box& box ) const
{
  GLdouble  teta = getTeta_ ();

  Vertex3d  a, b;
  GLdouble  w, d, s;
  GLdouble  dz;


  s = 1.05 / ::tan( teta );
  w = 2.0 * nearClip_ / (s - 1.0);
  d = nearClip_ + farClip_ - s * w;

  if ( d > w )
  {
    d = (farClip_ - nearClip_) / 2.0;
    w = (farClip_ + nearClip_ - d) / s;
  }

  dz   =  0.5 * (d + s * w);

  a[0] = -0.5 * w;
  a[1] = -0.5 * w;
  a[2] = -0.5 * d + dz;
  b[0] =  0.5 * w;
  b[1] =  0.5 * w;
  b[2] =  0.5 * d + dz;

  box.setCorners ( a, b );
}


//-----------------------------------------------------------------------
//   translateObj
//-----------------------------------------------------------------------


void Camera3D::translateObj

  ( GfxNode&  node,
    GLdouble  dx,
    GLdouble  dy,
    GLdouble  dz ) const

{
  Transformable*  tf = toTransformable ( &node );

  Transform       t, u;
  Box             box;
  GLdouble        a, b, c;


  if ( ! tf )
  {
    return;
  }

  node.getBBox ( box );

  if ( box.isEmpty() )
  {
    return;
  }

  u = getTransformOf   ( node.getParent() );
  t = tf->getTransform ();

  t  .transform ( u );
  box.transform ( u );

  a   = (vport_.width + 0.01) / (vport_.height + 0.01);
  b   =  nearClip_ + (GLdouble) std::fabs ( box.center()[2] );
  b  +=  0.25 * box.size ();
  c   = (GLdouble) std::tan ( getTeta_() );

  dx *= (a * b * c);
  dy *= (b * c);
  dz *=  b;

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


GLdouble Camera3D::setZoom ( GLdouble z )
{
  return Super::setZoom ( clamped( z, MIN_ZOOM, MAX_ZOOM ) );
}


//-----------------------------------------------------------------------
//   jitter
//-----------------------------------------------------------------------


void Camera3D::jitter

  ( Transform&  proj,
    GLdouble    dx,
    GLdouble    dy ) const

{
  proj(0,2) += 2.0 * dx / (GLdouble) vport_.width;
  proj(1,2) += 2.0 * dy / (GLdouble) vport_.height;
}


//-----------------------------------------------------------------------
//   updateZRange
//-----------------------------------------------------------------------


void Camera3D::updateZRange ()
{
  using jem::max;

  Box        box;
  Vertex3d   a, b;
  GLdouble   w, h, d;
  GLdouble   s;
  bool       viz;


  getBBox ( box );

  if ( box.isEmpty() )
  {
    return;
  }

  box.getCorners ( a, b );

  w  = b[0] - a[0];
  h  = b[1] - a[1];
  d  = b[2] - a[2];

  s  = max ( w, h );

  newProj_  = true;
  farClip_  = -a[2] + 0.1 * d + 0.01 * s;
  nearClip_ = -b[2] - 0.1 * d - 0.01 * s;

  clamp ( farClip_, MIN_FAR, MAX_FAR );
  clamp ( nearClip_, (farClip_ * MIN_NEAR),
                     (farClip_ * MAX_NEAR) );

  getVisibility ( viz, box, newProjection_() );

  if ( box.isEmpty() )
  {
    return;
  }

  box.getCorners ( a, b );

  w  = b[0] - a[0];
  h  = b[1] - a[1];
  d  = b[2] - a[2];

  s  = max ( w, h );
  s  = max ( s, d );

  farClip_ = -a[2] + s;
  farClip_ = max ( farClip_, 0.1 * s );

  clamp ( farClip_, MIN_FAR, MAX_FAR );

  nearClip_ = -b[2] - s;

  if ( nearClip_ < (farClip_ * MIN_NEAR) )
  {
    nearClip_ = -b[2] - 0.5 * s;
  }

  if ( nearClip_ < (farClip_ * MIN_NEAR) )
  {
    nearClip_ = -b[2] - 0.05 * s;
  }

  if ( nearClip_ < (farClip_ * MIN_NEAR) )
  {
    nearClip_ = -b[2];
  }

  clamp ( nearClip_, (farClip_ * MIN_NEAR),
                     (farClip_ * MAX_NEAR) );
}


//-----------------------------------------------------------------------
//  setZRange
//-----------------------------------------------------------------------


void Camera3D::setZRange

  ( GLdouble  clip0,
    GLdouble  clip1 )

{
  JEM_PRECHECK ( clip0 > 0.0 && clip0 < clip1 );

  farClip_  = clamped ( clip1, MIN_FAR, MAX_FAR );
  nearClip_ = clamped ( clip0, (farClip_ * MIN_NEAR),
                               (farClip_ * MAX_NEAR) );
  newProj_  = true;
}


//-----------------------------------------------------------------------
//   setViewAngle
//-----------------------------------------------------------------------


void Camera3D::setViewAngle ( GLdouble fovy )
{
  JEM_PRECHECK ( fovy > 0.0 && fovy < 180.0 );

  fovy_    = clamped ( fovy, MIN_ANGLE, MAX_ANGLE );
  newProj_ = true;
}


//-----------------------------------------------------------------------
//   newProjection
//-----------------------------------------------------------------------


Transform Camera3D::newProjection_ () const
{
  Transform  p;

  GLdouble   teta = getTeta_ ();

  GLdouble   a, w, h;


  a = (vport_.width + 0.01) / (vport_.height + 0.01);
  h = nearClip_ * (GLdouble) ::tan ( teta );
  w = a * h;

  p.setIdentity ();

  p(0,0) =  nearClip_ / w;
  p(1,1) =  nearClip_ / h;
  p(2,2) = (farClip_ + nearClip_) / (nearClip_ - farClip_);
  p(3,2) = -1.0;
  p(2,3) =  2.0 * farClip_ * nearClip_ / (nearClip_ - farClip_);
  p(3,3) =  0.0;

  return p;
}


//-----------------------------------------------------------------------
//   getTeta_
//-----------------------------------------------------------------------


inline GLdouble Camera3D::getTeta_ () const
{
  GLdouble  teta = 0.5 * fovy_ * RADS_PER_DEGREE;

  return (GLdouble)

    ::atan( ::pow( 2.0, -zoom_ ) * ::tan( teta ) );
}


JEM_END_PACKAGE( gl )
