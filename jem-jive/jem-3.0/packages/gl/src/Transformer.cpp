
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
#include <jem/base/limits.h>
#include <jem/base/ArithmeticException.h>
#include <jem/gl/input.h>
#include <jem/gl/render.h>
#include <jem/gl/Box.h>
#include <jem/gl/Orb.h>
#include <jem/gl/Camera.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/Transformable.h>
#include <jem/gl/Transformer.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   Transformer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Transformer::ROT_Z       = 1 << 0;
const int  Transformer::ROT_XY      = 1 << 1;
const int  Transformer::TRANS_Z     = 1 << 2;
const int  Transformer::TRANS_XY    = 1 << 3;
const int  Transformer::SNAPPY      = 1 << 4;

const int  Transformer::ROT_MODE    = ROT_Z   | ROT_XY;
const int  Transformer::TRANS_MODE  = TRANS_Z | TRANS_XY;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Transformer::Transformer ()
{
  reset ();
}


Transformer::~Transformer ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void Transformer::reset ()
{
  orb_       =   nullptr;
  mode_      =   0;
  zeta_      =   0;
  theta_     =   0;
  origin_    = -10.0;
  objPos_    = -10.0;
  objCenter_ =   0.0;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void Transformer::init

  ( int               mode,
    const GfxNode&    obj,
    const Camera&     cam,
    const MouseDrag&  mouse )

{
  if ( isActive() )
  {
    reset ();
  }

  if      ( mode & ROT_MODE )
  {
    initRot_   ( mode, obj, cam, mouse );
  }
  else if ( mode & TRANS_MODE )
  {
    initTrans_ ( mode, obj, cam, mouse );
  }
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void Transformer::cancel

  ( GfxNode&       obj,
    const Camera&  cam )

{
  if ( ! isActive() )
  {
    return;
  }

  Transformable*  tf = dynamic_cast<Transformable*> ( &obj );

  if ( tf )
  {
    tf->setTransform ( objTransf_ );
  }

  reset ();
}


//-----------------------------------------------------------------------
//   transform
//-----------------------------------------------------------------------


void Transformer::transform

  ( GfxNode&          obj,
    Camera&           cam,
    const MouseDrag&  mouse )

{
  if ( ! isActive() )
  {
    return;
  }

  const Rect  vport = cam.getViewport ();

  Vertex3d    mpos;


  if ( vport.isEmpty() )
  {
    return;
  }

  mpos[0] = mouse.x;
  mpos[1] = mouse.y;
  mpos[2] = 0.0;

  fromWindow ( mpos, vport );

  if      ( mode_ & ROT_MODE )
  {
    rotate_    ( obj, cam, mpos );
  }
  else if ( mode_ & TRANS_MODE )
  {
    translate_ ( obj, cam, mpos );
  }
}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void Transformer::redraw

  ( const Camera&   cam,
    RenderContext&  rtx ) const

{
  if ( ! isActive() )
  {
    return;
  }

  const GLsizei  MAX_WSIZE = 500;
  const GLfloat  RADIUS    = 0.075F;
  const Color    RED       = Color::red    ();
  const Color    YELLOW    = Color::yellow ();

  GLContext&     gtx       = rtx.getGLContext ();
  GLFuncs1*      funcs     = gtx.findFunctions<GLFuncs1> ();
  const Rect     vport     = cam.getViewport  ();

  GLdouble       nearClip  = -1.0;
  GLdouble       farClip   = 20.0;
  GLdouble       scale     =  1.0;
  GLdouble       aspect    = vport.aspectRatio ();

  AttribScope    attribs   ( gtx, GLContext::COLOR    |
                                  GLContext::LIGHTING |
                                  GLContext::LINE_WIDTH );
  Transform      p;
  GLfloat        lw;
  GLfloat        x,  y;
  GLsizei        wsize;
  int            quality;


  p.setIdentity ();

  p(2,2) =  2.0 / (nearClip - farClip);
  p(2,3) = (farClip + nearClip) / (nearClip - farClip);

  if ( aspect > 1.0 )
  {
    p(1,1) = 1.0 / aspect;
    wsize  = vport.height;
  }
  else
  {
    p(0,0) = aspect;
    wsize  = vport.width;
  }

  if ( wsize > MAX_WSIZE )
  {
    scale = (GLdouble) MAX_WSIZE / (GLdouble) wsize;
  }

  quality = 3 + wsize / 400;

  p(0,0) *= scale;
  p(1,1) *= scale;

  gtx.setLighting   ( false );
  gtx.pushTransform ( GLContext::PROJECTION );
  gtx.setTransform  ( p, GLContext::PROJECTION );
  gtx.pushTransform ();
  gtx.setIdentity   ();

  x = (GLfloat) (origin_[0] / p(0,0));
  y = (GLfloat) (origin_[1] / p(1,1));

  gtx.translate ( x, y, 0.0F );
  gtx.setColor  ( RED );
  renderDisk    ( gtx, 0.3F * RADIUS, quality );

  if ( (mode_ & TRANS_XY) && (mode_ & SNAPPY) )
  {
    gtx.rotate       ( (GLfloat) theta_, 0.0F, 0.0F, 1.0F );
    gtx.setLineWidth ( 1.0F );

    if      ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
    {
      GLfloat  vertPos[9] = { 0.30F, -0.04F, 0.0F,
                              0.40F,  0.00F, 0.0F,
                              0.30F,  0.04F, 0.0F };

      ShaderDraw::draw ( gtx, 9, vertPos, RED,
                         ShaderDraw::TRIANGLES );

      vertPos[0] = 0.0F; vertPos[1] = 0.0F;
      vertPos[3] = 0.3F;

      ShaderDraw::draw ( gtx, 9, vertPos, RED,
                         ShaderDraw::LINES );
    }
    else if ( funcs )
    {
      funcs->glBegin     ( GL_TRIANGLES );
      {
        funcs->glVertex  ( 0.30F, -0.04F );
        funcs->glVertex  ( 0.40F,  0.00F );
        funcs->glVertex  ( 0.30F,  0.04F );
      }
      funcs->glEnd       ();

      funcs->glBegin     ( GL_LINES );
      {
        funcs->glVertex  ( 0.0F, 0.0F );
        funcs->glVertex  ( 0.3F, 0.0F );
      }
      funcs->glEnd       ();
    }
  }

  if ( mode_ & ROT_MODE )
  {
    x = (GLfloat) (objPos_[0] / p(0,0));
    y = (GLfloat) (objPos_[1] / p(1,1));

    gtx.setIdentity ();
    gtx.translate   ( x, y, 0.0F );

    if ( mode_ & SNAPPY )
    {
      gtx.setColor ( RED );
      renderDisk   ( gtx, 1.2F * RADIUS, quality );
    }

    gtx.rotate ( (GLfloat) theta_, 0.0F, 0.0F, 1.0F );
    gtx.rotate ( (GLfloat) -zeta_, 0.0F, 1.0F, 0.0F );

    if ( mode_ & ROT_XY )
    {
      y  = 0.4F;
      lw = (wsize > MAX_WSIZE) ? 3.0F : 2.0F;

      gtx.setLineWidth ( lw );

      if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
      {
        GLfloat  vertPos[9] = { 0.0F,    y, 0.0F,
                                0.0F, 0.0F, 0.0F,
                                0.0F,   -y, 0.0F };

        ShaderDraw::draw ( gtx, 9, vertPos, YELLOW,
                           ShaderDraw::LINE_STRIP );
      }
      else
      {
        gtx.setColor      ( YELLOW );
        funcs->glBegin    ( GL_LINE_STRIP );
        {
          funcs->glVertex ( 0.0F,    y );
          funcs->glVertex ( 0.0F, 0.0F );
          funcs->glVertex ( 0.0F,   -y );
        }
        funcs->glEnd      ();
      }
    }

    gtx.setLighting ( true );
    orb_->drawOrb   ( gtx, RADIUS, RED, YELLOW );
  }

  gtx.popTransform ( GLContext::PROJECTION );
  gtx.popTransform ();
}


//-----------------------------------------------------------------------
//   align
//-----------------------------------------------------------------------


void Transformer::align

  ( GfxNode&  obj,
    Camera&   cam )

{
  Transformable*  tf = dynamic_cast<Transformable*> ( &obj );

  if ( ! tf )
  {
    return;
  }

  Transform  t, u;
  Box        box;
  Vertex3d   q[3];
  Vertex3d   c;
  GLdouble   a, x, y, z;
  int        i, j;


  tf->getBareBBox ( box );

  if ( box.isEmpty() )
  {
    return;
  }

  t = tf->getTransform ();
  u = getTransformOf   ( obj.getParent() );

  t.transform ( u );

  u = inverse ( u );
  c = matmul  ( t, box.center() );

  t.translate ( -c[0], -c[1], -c[2] );

  // Select the axis to be aligned with the z-axis of the viewing
  // coordinate system.

  a = -1.0;
  i = -1;

  for ( j = 0; j < 3; j++ )
  {
    q[j][0] = t(0,j);
    q[j][1] = t(1,j);
    q[j][2] = t(2,j);

    normalize ( q[j] );

    z = std::fabs( q[j][2] );

    if ( z > a )
    {
      a = z;
      i = j;
    }
  }

  if ( i < 0 )
  {
    return;
  }

  // Rotate around an axis in the x-y plane to align the selected
  // axis with the z-axis.

  x =  q[i][1];
  y = -q[i][0];
  z =  q[i][2];

  if ( z < 0.0 )
  {
    x = -x;
    y = -y;
    z = -z;
  }

  a = DEGREES_PER_RAD * std::acos ( z );

  if ( std::fabs( a ) > 1.0e-3 )
  {
    t.rotate ( a, x, y );
  }

  // Now rotate around the z-axis to align one of the two other
  // axes with the x- or y-axis of the viewing coordinate system.

  j = (i + 1) % 3;

  x = t(0,j);
  y = t(1,j);
  z = std::sqrt ( x * x + y * y );

  if ( z > 0.0 )
  {
    if ( std::fabs( x ) > std::fabs( y ) )
    {
      a = std::asin ( y / z );

      if ( x > 0.0 )
      {
        a = -a;
      }
    }
    else
    {
      a = std::asin ( x / z );

      if ( y < 0.0 )
      {
        a = -a;
      }
    }

    a *= DEGREES_PER_RAD;

    if ( std::fabs( a ) > 1.0e-3 )
    {
      t.rotate ( a );
    }
  }

  t.translate ( c[0], c[1], c[2] );

  t  .transform    ( u );
  tf->setTransform ( t );
  cam.updateZRange ();
}


//-----------------------------------------------------------------------
//   alignAxis
//-----------------------------------------------------------------------


void Transformer::alignAxis

  ( int       axis,
    GfxNode&  obj,
    Camera&   cam )

{
  JEM_PRECHECK ( axis >= -3 && axis <= 3 );

  Transformable*  tf = dynamic_cast<Transformable*> ( &obj );

  if ( ! tf )
  {
    return;
  }

  Transform  t, u;
  GLdouble   s;
  int        j;


  t = tf->getTransform ();
  u = getTransformOf   ( obj.getParent() );

  t.transform ( u );

  u = inverse    ( u );
  s = t.getScale ();

  for ( j = 0; j < 3; j++ )
  {
    t(0,j) = t(1,j) = t(2,j) = 0.0;
  }

  switch ( axis )
  {
  case -3:

    t(0,0) = -s;
    t(1,1) =  s;
    t(2,2) = -s;

    break;

  case -2:

    t(0,0) =  s;
    t(2,1) = -s;
    t(1,2) =  s;

    break;

  case -1:

    t(2,0) = -s;
    t(0,1) = -s;
    t(1,2) =  s;

    break;

  case 1:

    t(2,0) =  s;
    t(0,1) =  s;
    t(1,2) =  s;

    break;

  case 2:

    t(0,0) = -s;
    t(2,1) =  s;
    t(1,2) =  s;

    break;

  case 3:

    t(0,0) =  s;
    t(1,1) =  s;
    t(2,2) =  s;

    break;

  default:

    return;
  }

  t  .transform    ( u );
  tf->setTransform ( t );

  cam.lookAtObj    ( obj );
  cam.updateZRange ();
}


//-----------------------------------------------------------------------
//   initRot_
//-----------------------------------------------------------------------


void Transformer::initRot_

  ( int               mode,
    const GfxNode&    obj,
    const Camera&     cam,
    const MouseDrag&  mouse )

{
  const Transformable*  tf =

    dynamic_cast<const Transformable*> ( &obj );

  if ( ! tf )
  {
    return;
  }

  const Rect  vport = cam.getViewport ();

  Box         box;
  Vertex3d    pos;


  tf->getBareBBox ( box );

  if ( box.isEmpty() || vport.isEmpty() )
  {
    return;
  }

  pos[0] = mouse.x - mouse.dx;
  pos[1] = mouse.y - mouse.dy;
  pos[2] = 0.0;

  fromWindow ( pos, vport );

  origin_[0] = pos[0];
  origin_[1] = pos[1];

  objTransf_   = tf->getTransform ();
  worldTransf_ = getTransformOf ( obj.getParent() );

  objTransf_.transform ( worldTransf_ );

  objCenter_ = matmul  ( objTransf_, box.center() );

  objTransf_.translate ( -objCenter_[0],
                         -objCenter_[1],
                         -objCenter_[2] );

  pos        = project ( cam.getProjection(), objCenter_ );

  objPos_[0] = pos[0];
  objPos_[1] = pos[1];

  if ( ! invert( worldTransf_ ) )
  {
    throw ArithmeticException (
      JEM_FUNC,
      "singular transformation matrix"
    );
  }

  if ( ! orb_ )
  {
    orb_ = newInstance<Orb> ( 2 );
  }

  mode_ = mode;
}


//-----------------------------------------------------------------------
//   rotate_
//-----------------------------------------------------------------------


void Transformer::rotate_

  ( GfxNode&         obj,
    Camera&          cam,
    const Vertex3d&  mpos )

{
  Transformable*  tf = dynamic_cast<Transformable*> ( &obj );

  if ( ! tf )
  {
    return;
  }

  Transform  t = objTransf_;

  if      ( mode_ & ROT_Z )
  {
    zRotate_  ( t, mpos );
  }
  else if ( mode_ & ROT_XY )
  {
    xyRotate_ ( t, mpos );
  }

  t.translate ( objCenter_[0], objCenter_[1], objCenter_[2] );
  t.transform ( worldTransf_ );

  tf->setTransform ( t );

  if ( mode_ & ROT_XY )
  {
    cam.updateZRange ();
  }
}


//-----------------------------------------------------------------------
//   zRotate_
//-----------------------------------------------------------------------


void Transformer::zRotate_

  ( Transform&       t,
    const Vertex3d&  mpos )

{
  Vertex2d  e1, e2;
  GLdouble  a, step;


  e1[0] = origin_[0] - objPos_[0];
  e1[1] = origin_[1] - objPos_[1];

  e2[0] = mpos[0]    - objPos_[0];
  e2[1] = mpos[1]    - objPos_[1];

  a     = getAngle ( e1, e2 );

  if ( mode_ & SNAPPY )
  {
    step = 5.0;
  }
  else
  {
    step = 1.0;
  }

  round    ( a, step );
  t.rotate ( a );

  zeta_  = 0;
  theta_ = (int) a;
}


//-----------------------------------------------------------------------
//   xyRotate_
//-----------------------------------------------------------------------


void Transformer::xyRotate_

  ( Transform&       t,
    const Vertex3d&  mpos )

{
  GLdouble  a, dx, dy, u, v;
  GLdouble  step;


  if ( mode_ & SNAPPY )
  {
    step = 5.0;
  }
  else
  {
    step = 1.0;
  }

  dx = mpos[0] - origin_[0];
  dy = mpos[1] - origin_[1];
  a  = std::sqrt ( dx * dx + dy * dy );

  if ( a > 0.0 )
  {
    theta_ = (int) (std::asin( dy / a ) * DEGREES_PER_RAD);
  }
  else
  {
    theta_ = 0;
  }

  if      ( dx < 0.0 )
  {
    theta_ = 180 - theta_;
  }
  else if ( dy < 0.0 )
  {
    theta_ = 360 + theta_;
  }

  if ( mode_ & SNAPPY )
  {
    theta_ = (int) rounded ( (double) theta_, 45.0 );
  }

  a = theta_ * RADS_PER_DEGREE;
  u = std::cos ( a );
  v = std::sin ( a );
  a = -90.0 * (u * dx + v * dy);

  round    ( a, step );
  t.rotate ( a, v, -u );

  zeta_ = (int) a;
}


//-----------------------------------------------------------------------
//   initTrans_
//-----------------------------------------------------------------------


void Transformer::initTrans_

  ( int               mode,
    const GfxNode&    obj,
    const Camera&     cam,
    const MouseDrag&  mouse )

{
  const Transformable*  tf =

    dynamic_cast<const Transformable*> ( &obj );

  if ( ! tf )
  {
    return;
  }

  const Rect  vport = cam.getViewport ();

  Box         box;
  Vertex3d    pos;


  obj.getBBox ( box );

  if ( box.isEmpty() || vport.isEmpty() )
  {
    return;
  }

  pos[0] = mouse.x - mouse.dx;
  pos[1] = mouse.y - mouse.dy;
  pos[2] = 0.0;

  fromWindow ( pos, vport );

  origin_[0] = pos[0];
  origin_[1] = pos[1];

  objTransf_   = tf->getTransform ();
  worldTransf_ = getTransformOf ( obj.getParent() );
  objCenter_   = matmul ( worldTransf_, box.center() );

  if ( project( pos, cam.getProjection(), objCenter_ ) )
  {
    objPos_[0] = pos[0];
    objPos_[1] = pos[1];
  }

  mode_ = mode;
}


//-----------------------------------------------------------------------
//   translate_
//-----------------------------------------------------------------------


void Transformer::translate_

  ( GfxNode&         obj,
    Camera&          cam,
    const Vertex3d&  mpos )

{
  Transformable*  tf = dynamic_cast<Transformable*> ( &obj );

  if ( ! tf )
  {
    return;
  }

  GLdouble  dx, dy, dz;


  dx = 0.5 * (mpos[0] - origin_[0]);
  dy = 0.5 * (mpos[1] - origin_[1]);
  dz = 0.0;

  if      ( mode_ & TRANS_Z )
  {
    dz = dy;
    dx = 0.0;
    dy = 0.0;
  }
  else if ( mode_ & TRANS_XY )
  {
    dz = 0.0F;

    if ( mode_ & SNAPPY )
    {
      xyTranslate_ ( dx, dy, mpos );
    }
  }
  else
  {
    return;
  }

  tf->setTransform ( objTransf_ );
  cam.translateObj ( obj, dx, dy, dz );

  if ( mode_ & TRANS_Z )
  {
    cam.updateZRange ();
  }
}


//-----------------------------------------------------------------------
//   xyTranslate_
//-----------------------------------------------------------------------


void Transformer::xyTranslate_

  ( GLdouble&        dx,
    GLdouble&        dy,
    const Vertex3d&  mpos )

{
  GLdouble  a, u, v;


  u = mpos[0] - origin_[0];
  v = mpos[1] - origin_[1];
  a = std::sqrt ( u * u + v * v );

  if ( a > 0.0 )
  {
    theta_ = (int) (std::asin( v / a ) * DEGREES_PER_RAD);
  }
  else
  {
    theta_ = 0;
  }

  if      ( u < 0.0 )
  {
    theta_ = 180 - theta_;
  }
  else if ( v < 0.0 )
  {
    theta_ = 360 + theta_;
  }

  if ( mode_ & SNAPPY )
  {
    theta_ = (int) rounded ( (double) theta_, 45.0 );
  }

  a  = theta_ * RADS_PER_DEGREE;
  u  = std::cos ( a );
  v  = std::sin ( a );
  a  = u * dx + v * dy;

  dx = a * u;
  dy = a * v;
}


JEM_END_PACKAGE( gl )
