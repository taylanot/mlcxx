
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
#include <cstring>
#include <jem/base/limits.h>
#include <jem/base/ArithmeticException.h>
#include <jem/io/TextOutput.h>
#include <jem/io/NumberFormatter.h>
#include <jem/util/Properties.h>
#include <jem/gl/Rect.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/Transform.h>


JEM_BEGIN_PACKAGE( gl )


extern const GLdouble  RADS_PER_DEGREE;
extern const GLdouble  DEGREES_PER_RAD;


//=======================================================================
//   class Transform
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


Transform::Transform () noexcept
{
#ifndef NDEBUG

  std::memset ( data_, 0xff, 16 * sizeof(GLdouble) );

#endif
}


Transform::Transform ( const Transform &rhs ) noexcept
{
  std::memcpy ( data_, rhs.data_, 16 * sizeof(GLdouble) );
}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


Transform& Transform::operator = ( const Transform &rhs ) noexcept
{
  if ( this != &rhs )
  {
    std::memcpy ( data_, rhs.data_, 16 * sizeof(GLdouble) );
  }

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void Transform::swap ( Transform& rhs ) noexcept
{
  GLdouble*  myData = data_;
  GLdouble*  rhData = rhs.data_;

  GLdouble   tmp;

  for ( int i = 0; i < 16; i++ )
  {
    tmp       = myData[i];
    myData[i] = rhData[i];
    rhData[i] = tmp;
  }
}


//-----------------------------------------------------------------------
//   setIdentity
//-----------------------------------------------------------------------


void Transform::setIdentity () noexcept
{
  GLdouble* JEM_RESTRICT  tr = data_;

  tr[ 0] = 1.0;
  tr[ 1] = 0.0;
  tr[ 2] = 0.0;
  tr[ 3] = 0.0;

  tr[ 4] = 0.0;
  tr[ 5] = 1.0;
  tr[ 6] = 0.0;
  tr[ 7] = 0.0;

  tr[ 8] = 0.0;
  tr[ 9] = 0.0;
  tr[10] = 1.0;
  tr[11] = 0.0;

  tr[12] = 0.0;
  tr[13] = 0.0;
  tr[14] = 0.0;
  tr[15] = 1.0;
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void Transform::scale ( GLdouble sx, GLdouble sy, GLdouble sz )
{
  GLdouble* JEM_RESTRICT  tr = data_;

  tr[ 0] *= sx;
  tr[ 1] *= sy;
  tr[ 2] *= sz;

  tr[ 4] *= sx;
  tr[ 5] *= sy;
  tr[ 6] *= sz;

  tr[ 8] *= sx;
  tr[ 9] *= sy;
  tr[10] *= sz;

  tr[12] *= sx;
  tr[13] *= sy;
  tr[14] *= sz;
}


//-----------------------------------------------------------------------
//   setScale
//-----------------------------------------------------------------------


void Transform::setScale ( GLdouble sx, GLdouble sy, GLdouble sz )
{
  GLdouble* JEM_RESTRICT  tr = data_;

  GLdouble  s[3] = { sx, sy, sz };
  GLdouble  c;

  for ( int i = 0; i < 3; i++ )
  {
    c = ( tr[i + 0] * tr[i + 0] +
          tr[i + 4] * tr[i + 4] +
          tr[i + 8] * tr[i + 8] );

    c = (GLdouble) std::sqrt ( (double) c );

    if ( c > 0.0 )
    {
      c = s[i] / c;

      tr[i +  0] *= c;
      tr[i +  4] *= c;
      tr[i +  8] *= c;
      tr[i + 12] *= c;
    }
  }
}


//-----------------------------------------------------------------------
//   getScale
//-----------------------------------------------------------------------


GLdouble Transform::getScale () const noexcept
{
  const GLdouble* JEM_RESTRICT  tr = data_;

  GLdouble  s2 = ( tr[0] * tr[0] +
                   tr[4] * tr[4] +
                   tr[8] * tr[8] );

  return (GLdouble) std::sqrt ( (double) s2 );
}


//-----------------------------------------------------------------------
//   rotate
//-----------------------------------------------------------------------


void Transform::rotate ( GLdouble  a ) noexcept
{
  Transform  rot  = rotation ( a );
  Transform  self = *this;

  matmul ( *this, rot, self );
}


void Transform::rotate

  ( GLdouble  a,
    GLdouble  x,
    GLdouble  y )

{
  Transform  rot  = rotation ( a, x, y );
  Transform  self = *this;

  matmul ( *this, rot, self );
}


void Transform::rotate

  ( GLdouble  a,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  Transform  rot  = rotation ( a, x, y, z );
  Transform  self = *this;

  matmul ( *this, rot, self );
}


//-----------------------------------------------------------------------
//   setRotation
//-----------------------------------------------------------------------


void Transform::setRotation ( GLdouble a ) noexcept
{
  GLdouble*  JEM_RESTRICT  tr = data_;

  GLdouble   alpha, beta;


  alpha  = (GLdouble) std::cos ( (double) (a * RADS_PER_DEGREE) );
  beta   = (GLdouble) std::sin ( (double) (a * RADS_PER_DEGREE) );

  tr[ 0] = alpha;
  tr[ 1] =  beta;
  tr[ 2] =   0.0;

  tr[ 4] = -beta;
  tr[ 5] = alpha;
  tr[ 6] =   0.0;

  tr[ 8] =   0.0;
  tr[ 9] =   0.0;
  tr[10] =   1.0;
}


void Transform::setRotation

  ( GLdouble  a,
    GLdouble  x,
    GLdouble  y )

{
  GLdouble*  JEM_RESTRICT  tr = data_;

  GLdouble   alpha, beta, gamma;


  alpha = (GLdouble) std::sqrt ( (double) (x * x + y * y) );

  if ( isTiny( alpha ) )
  {
    throw ArithmeticException (
      JEM_FUNC,
      "rotation axis has zero length"
    );
  }

  alpha  = 1.0 / alpha;

  x     *= alpha;
  y     *= alpha;

  alpha  = (GLdouble) std::cos ( (double) (a * RADS_PER_DEGREE) );
  beta   = (GLdouble) std::sin ( (double) (a * RADS_PER_DEGREE) );
  gamma  = (GLdouble) (1.0 - alpha);

  tr[ 0] = x * x * gamma + alpha;
  tr[ 1] = x * y * gamma;
  tr[ 2] =    -y * beta;

  tr[ 4] = x * y * gamma;
  tr[ 5] = y * y * gamma + alpha;
  tr[ 6] =     x * beta;

  tr[ 8] =     y * beta;
  tr[ 9] =    -x * beta;
  tr[10] =         alpha;
}


void Transform::setRotation

  ( GLdouble  a,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  GLdouble*  JEM_RESTRICT  tr = data_;

  GLdouble   alpha, beta, gamma;


  alpha = (GLdouble) std::sqrt ( (double) (x * x + y * y + z * z) );

  if ( isTiny( alpha ) )
  {
    throw ArithmeticException (
      JEM_FUNC,
      "rotation axis has zero length"
    );
  }

  alpha  = 1.0 / alpha;

  x     *= alpha;
  y     *= alpha;
  z     *= alpha;

  alpha  = (GLdouble) std::cos ( (double) (a * RADS_PER_DEGREE) );
  beta   = (GLdouble) std::sin ( (double) (a * RADS_PER_DEGREE) );
  gamma  = (GLdouble) (1.0 - alpha);

  tr[ 0] = x * x * gamma +     alpha;
  tr[ 1] = x * y * gamma + z * beta;
  tr[ 2] = x * z * gamma - y * beta;

  tr[ 4] = x * y * gamma - z * beta;
  tr[ 5] = y * y * gamma +     alpha;
  tr[ 6] = y * z * gamma + x * beta;

  tr[ 8] = x * z * gamma + y * beta;
  tr[ 9] = y * z * gamma - x * beta;
  tr[10] = z * z * gamma +     alpha;
}


//-----------------------------------------------------------------------
//   transform
//-----------------------------------------------------------------------


void Transform::transform ( const Transform& tr ) noexcept
{
  Transform  self = *this;

  matmul ( *this, tr, self );
}


//-----------------------------------------------------------------------
//   transpose
//-----------------------------------------------------------------------


Transform Transform::transpose () const noexcept
{
  Transform  tr;

  const GLdouble*  JEM_RESTRICT  src = data_;
  GLdouble*        JEM_RESTRICT  dst = tr.data_;

  dst[ 0] = src[ 0];
  dst[ 4] = src[ 1];
  dst[ 8] = src[ 2];
  dst[12] = src[ 3];

  dst[ 1] = src[ 4];
  dst[ 5] = src[ 5];
  dst[ 9] = src[ 6];
  dst[13] = src[ 7];

  dst[ 2] = src[ 8];
  dst[ 6] = src[ 9];
  dst[10] = src[10];
  dst[14] = src[11];

  dst[ 3] = src[12];
  dst[ 7] = src[13];
  dst[11] = src[14];
  dst[15] = src[15];

  return tr;
}


//-----------------------------------------------------------------------
//   identity
//-----------------------------------------------------------------------


Transform Transform::identity () noexcept
{
  Transform  id;

  id.setIdentity ();

  return id;
}


//-----------------------------------------------------------------------
//   scaling
//-----------------------------------------------------------------------


Transform Transform::scaling ( GLdouble s )
{
  Transform  sc;

  sc.setIdentity ();
  sc.scale ( s, s, s );

  return sc;
}


Transform Transform::scaling

  ( GLdouble  sx,
    GLdouble  sy,
    GLdouble  sz )

{
  Transform  sc;

  sc.setIdentity ();
  sc.scale ( sx, sy, sz );

  return sc;
}


//-----------------------------------------------------------------------
//   translation
//-----------------------------------------------------------------------


Transform Transform::translation

  ( GLdouble  dx,
    GLdouble  dy,
    GLdouble  dz ) noexcept

{
  Transform  tr;

  tr.setIdentity ();
  tr.translate   ( dx, dy, dz );

  return tr;
}


//-----------------------------------------------------------------------
//   rotation
//-----------------------------------------------------------------------


Transform Transform::rotation ( GLdouble a ) noexcept
{
  Transform  rot;

  rot.setRotation ( a );

  rot(3,0) = rot(3,1) = rot(3,2) = 0.0;
  rot(0,3) = rot(1,3) = rot(2,3) = 0.0;
  rot(3,3) = 1.0;

  return rot;
}


Transform Transform::rotation

  ( GLdouble  a,
    GLdouble  x,
    GLdouble  y )

{
  Transform  rot;

  rot.setRotation ( a, x, y );

  rot(3,0) = rot(3,1) = rot(3,2) = 0.0;
  rot(0,3) = rot(1,3) = rot(2,3) = 0.0;
  rot(3,3) = 1.0;

  return rot;
}


Transform Transform::rotation

  ( GLdouble  a,
    GLdouble  x,
    GLdouble  y,
    GLdouble  z )

{
  Transform  rot;

  rot.setRotation ( a, x, y, z );

  rot(3,0) = rot(3,1) = rot(3,2) = 0.0;
  rot(0,3) = rot(1,3) = rot(2,3) = 0.0;
  rot(3,3) = 1.0;

  return rot;
}


//-----------------------------------------------------------------------
//   configure2D
//-----------------------------------------------------------------------


void Transform::configure2D ( const Properties&  props )
{
  Vertex2d  t;
  double    a, s;

  // Preserve the current translation

  t[0] = data_[12];
  t[1] = data_[13];

  if ( props.find( s, PropNames::SCALE, minOf( s ), maxOf( s ) ) )
  {
    setScale ( (GLdouble) s );
  }

  if ( props.find( a, PropNames::ROTATION ) )
  {
    // Preserve the current scale

    s = (double) getScale ();

    setRotation ( (GLdouble) a, 0.0, 0.0, 1.0 );
    setScale    ( (GLdouble) s );
  }

  Vertex::configure ( t, PropNames::TRANSLATION, props );
  setTranslation    ( t[0], t[1] );
}


//-----------------------------------------------------------------------
//   getConfig2D
//-----------------------------------------------------------------------


void Transform::getConfig2D ( const Properties& props ) const
{
  Vertex2d  t;
  double    a, s;


  s = (double) getScale ();

  if ( s > 0.0 )
  {
    a = ::asin ( (double) data_[1] / s ) * DEGREES_PER_RAD;
  }
  else
  {
    a = 0.0;
  }

  t[0] = data_[12];
  t[1] = data_[13];

  props.set ( PropNames::SCALE,    s );
  props.set ( PropNames::ROTATION, a );

  Vertex::getConfig ( PropNames::TRANSLATION, props, t );
}


//-----------------------------------------------------------------------
//   listProps2D
//-----------------------------------------------------------------------


void Transform::listProps2D ( const Properties& props )
{
  props.set ( PropNames::SCALE,
              "The scale factor" );
  props.set ( PropNames::ROTATION,
              "The rotation angle (in degrees) around the z-axis" );
  props.set ( PropNames::TRANSLATION,
              "The translation vector" );
}


//-----------------------------------------------------------------------
//   configure3D
//-----------------------------------------------------------------------


void Transform::configure3D ( const Properties& props )
{
  Vertex3d    t, v;
  double      a, s;
  int         i, j;


  // Preserve the current translation

  t[0] = data_[12];
  t[1] = data_[13];
  t[2] = data_[14];

  if ( props.find( s, PropNames::SCALE, minOf( s ), maxOf( s ) ) )
  {
    setScale ( (GLdouble) s );
  }

  if ( props.contains( PropNames::ROTATION ) )
  {
    Properties  rotProps = props.getProps ( PropNames::ROTATION );

    rotProps.get ( a, PropNames::ANGLE );

    v = 0.0;

    if ( ! Vertex::configure( v, PropNames::AXIS, rotProps ) )
    {
      // Force an exception to be thrown.

      rotProps.get ( PropNames::AXIS );
    }

    // Preserve the current scale

    s = (double) getScale ();

    try
    {
      setRotation ( (GLdouble) a, v[0], v[1], v[2] );
    }
    catch ( const ArithmeticException& ex )
    {
      props.propertyError ( PropNames::ROTATION, ex.what() );
    }

    setScale ( (GLdouble) s );
  }
  else if ( props.contains( PropNames::ROTMAT ) )
  {
    Array<double>  r ( 9 );

    props.get ( r, PropNames::ROTMAT );

    if ( r.size() != 9 )
    {
      props.propertyError (
        PropNames::ROTMAT,
        String::format (
          "array has length %d but should have length 9",
          r.size ()
        )
      );
    }

    s = (double) getScale ();

    for ( i = j = 0; i < 9; i += 3, j += 4 )
    {
      a = std::sqrt ( r[i + 0] * r[i + 0] +
                      r[i + 1] * r[i + 1] +
                      r[i + 2] * r[i + 2] );

      if ( isTiny( a ) )
      {
        props.propertyError ( PropNames::ROTMAT,
                              "singular rotation matrix" );
      }

      a = s / a;

      data_[j + 0] = a * r[i + 0];
      data_[j + 1] = a * r[i + 1];
      data_[j + 2] = a * r[i + 2];
    }
  }

  Vertex::configure ( t, PropNames::TRANSLATION, props );
  setTranslation    ( t[0], t[1], t[2] );
}


//-----------------------------------------------------------------------
//   getConfig3D
//-----------------------------------------------------------------------


void Transform::getConfig3D ( const Properties&  props ) const
{
  Array<double>  r ( 9 );

  Vertex3d       t;
  double         a, s;
  int            i, j;


  s = (double) getScale ();

  if ( ! isTiny( s ) )
  {
    a = 1.0 / s;
  }
  else
  {
    a = 1.0;
  }

  for ( i = j = 0; i < 9; i += 3, j += 4 )
  {
    r[i + 0] = a * data_[j + 0];
    r[i + 1] = a * data_[j + 1];
    r[i + 2] = a * data_[j + 2];
  }

  t[0] = data_[12];
  t[1] = data_[13];
  t[2] = data_[14];

  props.set ( PropNames::SCALE,  s );
  props.set ( PropNames::ROTMAT, r );

  Vertex::getConfig ( PropNames::TRANSLATION, props, t );
}


//-----------------------------------------------------------------------
//   listProps3D
//-----------------------------------------------------------------------


void Transform::listProps3D ( const Properties& props )
{
  props.set ( PropNames::SCALE,
              "The scale factor" );
  props.set ( PropNames::ROTMAT,
              "The rotation matrix" );
  props.set ( PropNames::TRANSLATION,
              "The translation vector" );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


void                print

  ( TextOutput&       out,
    const Transform&  tr )

{
  using jem::io::NumberFormatter;

  NumberFormatter  nf;

  nf.nformat.setScientific     ();
  nf.nformat.setFractionDigits (  4 );
  nf.nformat.setFloatWidth     ( 11 );

  for ( int i = 0; i < 4; i++ )
  {
    print ( out, "[ ", nf( tr(i,0) ) );

    for ( int j = 1; j < 4; j++ )
    {
      print ( out, ", ", nf( tr(i,j) ) );
    }

    print ( out, " ]" );

    if ( i < 3 )
    {
      print ( out, '\n' );
    }
  }
}


//-----------------------------------------------------------------------
//   matmul (Transform * Vertex2d)
//-----------------------------------------------------------------------


void                matmul

  ( Vertex2d&         r,
    const Transform&  a,
    const Vertex2d&   b ) noexcept

{
  const GLdouble* JEM_RESTRICT  t = a.addr ();

  r[0] = t[0] * b[0] + t[4] * b[1] + t[12];
  r[1] = t[1] * b[0] + t[5] * b[1] + t[13];
}


//-----------------------------------------------------------------------
//   matmul (Transform * Vertex3d)
//-----------------------------------------------------------------------


void                matmul

  ( Vertex3d&         r,
    const Transform&  a,
    const Vertex3d&   b ) noexcept

{
  const GLdouble* JEM_RESTRICT  t = a.addr ();

  r[0] = t[0] * b[0] + t[4] * b[1] + t[ 8] * b[2] + t[12];
  r[1] = t[1] * b[0] + t[5] * b[1] + t[ 9] * b[2] + t[13];
  r[2] = t[2] * b[0] + t[6] * b[1] + t[10] * b[2] + t[14];
}


//-----------------------------------------------------------------------
//   matmul (Transform * Vertex4d)
//-----------------------------------------------------------------------


void                matmul

  ( Vertex4d&         r,
    const Transform&  a,
    const Vertex4d&   b ) noexcept

{
  const GLdouble* JEM_RESTRICT  t = a.addr ();

  r[0] = t[0] * b[0] + t[4] * b[1] + t[ 8] * b[2] + t[12] * b[3];
  r[1] = t[1] * b[0] + t[5] * b[1] + t[ 9] * b[2] + t[13] * b[3];
  r[2] = t[2] * b[0] + t[6] * b[1] + t[10] * b[2] + t[14] * b[3];
  r[3] = t[3] * b[0] + t[7] * b[1] + t[11] * b[2] + t[15] * b[3];
}


//-----------------------------------------------------------------------
//   matmul (Transorm * Transform)
//-----------------------------------------------------------------------


void                matmul

  ( Transform&        rt,
    const Transform&  at,
    const Transform&  bt ) noexcept

{
        GLdouble* JEM_RESTRICT  r = rt.addr ();
  const GLdouble* JEM_RESTRICT  a = at.addr ();
  const GLdouble* JEM_RESTRICT  b = bt.addr ();

  for ( int j = 0; j < 4; j++ )
  {
    r[0] = (a[ 0] * b[0] + a[ 4] * b[1] +
            a[ 8] * b[2] + a[12] * b[3]);

    r[1] = (a[ 1] * b[0] + a[ 5] * b[1] +
            a[ 9] * b[2] + a[13] * b[3]);

    r[2] = (a[ 2] * b[0] + a[ 6] * b[1] +
            a[10] * b[2] + a[14] * b[3]);

    r[3] = (a[ 3] * b[0] + a[ 7] * b[1] +
            a[11] * b[2] + a[15] * b[3]);

    r   += 4;
    b   += 4;
  }
}


//-----------------------------------------------------------------------
//   project
//-----------------------------------------------------------------------


Vertex3d            project

  ( const Transform&  p,
    const Vertex3d&   v )
{
  Vertex3d  d;

  if ( ! project( d, p, v ) )
  {
    throw ArithmeticException (
      JEM_FUNC,
      "projection failed (zero w-coordinate)"
    );
  }

  return d;
}


bool                project

  ( Vertex3d&         d,
    const Transform&  p,
    const Vertex3d&   v )

{
  const GLdouble* JEM_RESTRICT  t = p.addr ();

  GLdouble  w;

  d[0] = t[0] * v[0] + t[4] * v[1] + t[ 8] * v[2] + t[12];
  d[1] = t[1] * v[0] + t[5] * v[1] + t[ 9] * v[2] + t[13];
  d[2] = t[2] * v[0] + t[6] * v[1] + t[10] * v[2] + t[14];
  w    = t[3] * v[0] + t[7] * v[1] + t[11] * v[2] + t[15];

  if ( w > 0.0 )
  {
    w = 1.0 / w;

    d[0] *= w;
    d[1] *= w;
    d[2] *= w;

    return true;
  }
  else
  {
    return false;
  }
}


bool                project

  ( Vertex2d&         d,
    const Transform&  p,
    const Vertex3d&   v )

{
  const GLdouble* JEM_RESTRICT  t = p.addr ();

  GLdouble  w;

  d[0] = t[0] * v[0] + t[4] * v[1] + t[ 8] * v[2] + t[12];
  d[1] = t[1] * v[0] + t[5] * v[1] + t[ 9] * v[2] + t[13];
  w    = t[3] * v[0] + t[7] * v[1] + t[11] * v[2] + t[15];

  if ( w > 0.0 )
  {
    w = 1.0 / w;

    d[0] *= w;
    d[1] *= w;

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   unproject
//-----------------------------------------------------------------------


bool                unproject

  ( Vertex3d&         v,
    const Transform&  p,
    const Vertex3d&   d )

{
  Transform  pi = p;

  if ( ! invert( pi ) )
  {
    return false;
  }

  return project ( v, pi, d );
}


//-----------------------------------------------------------------------
//   toWindow
//-----------------------------------------------------------------------


void                toWindow

  ( Vertex3d&         v,
    const Rect&       vport )

{
  v[0] = (GLdouble) vport.x + 0.5 * (v[0] + 1.0) * vport.width;
  v[1] = (GLdouble) vport.y + 0.5 * (v[1] + 1.0) * vport.height;
  v[2] =                      0.5 * (v[2] + 1.0);
}


//-----------------------------------------------------------------------
//   fromWindow
//-----------------------------------------------------------------------


void                fromWindow

  ( Vertex2d&         v,
    const Rect&       vport )

{
  v[0] = -1.0 + 2.0 * (v[0] - vport.x) / (GLdouble) vport.width;
  v[1] = -1.0 + 2.0 * (v[1] - vport.y) / (GLdouble) vport.height;
}


void                fromWindow

  ( Vertex3d&         v,
    const Rect&       vport )

{
  v[0] = -1.0 + 2.0 * (v[0] - vport.x) / (GLdouble) vport.width;
  v[1] = -1.0 + 2.0 * (v[1] - vport.y) / (GLdouble) vport.height;
  v[2] = -1.0 + 2.0 *  v[2];
}


//-----------------------------------------------------------------------
//   inverse
//-----------------------------------------------------------------------


Transform  inverse ( const Transform& t )
{
  Transform  r = t;

  if ( ! invert( r ) )
  {
    throw ArithmeticException (
      JEM_FUNC,
      "singular transformation matrix"
    );
  }

  return r;
}


//-----------------------------------------------------------------------
//   invert
//-----------------------------------------------------------------------


bool invert ( Transform& t )
{
  Transform  r;

  GLdouble   a, s;
  int        i, j, k, p;


  r.setIdentity ();

  // Scale all rows so that the largest absolute value on each
  // row equals 1.0.

  for ( i = 0; i < 4; i++ )
  {
    s = (GLdouble) std::fabs ( t(i,0) );

    for ( j = 1; j < 4; j++ )
    {
      a = (GLdouble) std::fabs ( t(i,j) );

      if ( a > s )
      {
        s = a;
      }
    }

    if ( isTiny( s ) )
    {
      return false;
    }

    s = 1.0 / s;

    for ( j = 0; j < 4; j++ )
    {
      t(i,j) *= s;
    }

    r(i,i) *= s;
  }

  // Simple Gauss elimination with row pivoting.

  for ( k = 0; k < 4; k++ )
  {
    p = k;
    s = (GLdouble) std::fabs ( t(k,k) );

    for ( i = k + 1; i < 4; i++ )
    {
      a = (GLdouble) std::fabs ( t(i,k) );

      if ( a > s )
      {
        p = i;
        s = a;
      }
    }

    if ( s < Limits<GLdouble>::EPSILON )
    {
      return false;
    }

    if ( p != k )
    {
      for ( j = k; j < 4; j++ )
      {
        a      = t(k,j);
        t(k,j) = t(p,j);
        t(p,j) = a;
      }

      for ( j = 0; j < 4; j++ )
      {
        a      = r(k,j);
        r(k,j) = r(p,j);
        r(p,j) = a;
      }
    }

    s = 1.0 / t(k,k);

    for ( j = k + 1; j < 4; j++ )
    {
      t(k,j) *= s;
    }

    for ( j = 0; j < 4; j++ )
    {
      r(k,j) *= s;
    }

    for ( i = k + 1; i < 4; i++ )
    {
      a = t(i,k);

      for ( j = k + 1; j < 4; j++ )
      {
        t(i,j) -= a * t(k,j);
      }

      for ( j = 0; j < 4; j++ )
      {
        r(i,j) -= a * r(k,j);
      }
    }
  }

  // Backward substitution

  for ( k = 3; k > 0; k-- )
  {
    for ( i = k - 1; i >= 0; i-- )
    {
      a = t(i,k);

      for ( j = 0; j < 4; j++ )
      {
        r(i,j) -= a * r(k,j);
      }
    }
  }

  t = r;

  return true;
}


//-----------------------------------------------------------------------
//   reldiff
//-----------------------------------------------------------------------


GLdouble            reldiff

  ( const Transform&  a,
    const Transform&  b ) noexcept

{
  GLdouble  rdiff = 0.0;
  GLdouble  scale;
  GLdouble  adiff;


  for ( int i = 0; i < 3; i++ )
  {
    scale = (GLdouble) (std::fabs( a(i,0) ) +
                        std::fabs( a(i,1) ) +
                        std::fabs( a(i,2) ) +
                        std::fabs( b(i,0) ) +
                        std::fabs( b(i,1) ) +
                        std::fabs( b(i,2) )) / 6.0;

    if ( scale > 0.0 )
    {
      scale = 1.0 / scale;

      for ( int j = 0; j < 3; j++ )
      {
        adiff = (GLdouble) ::fabs ( a(i,j) - b(i,j) );
        rdiff = max ( rdiff, scale * adiff );
      }
    }
  }

  scale = (GLdouble) (std::fabs( a(0,3) ) +
                      std::fabs( a(1,3) ) +
                      std::fabs( a(2,3) ) +
                      std::fabs( b(0,3) ) +
                      std::fabs( b(1,3) ) +
                      std::fabs( b(2,3) )) / 6.0;

  if ( scale > 0.0 )
  {
    scale = 1.0 / scale;

    for ( int i = 0; i < 3; i++ )
    {
      adiff = (GLdouble) std::fabs ( a(i,3) - b(i,3) );
      rdiff = max ( rdiff, scale * adiff );
    }
  }

  scale = (GLdouble) (std::fabs( a(3,0) ) +
                      std::fabs( a(3,1) ) +
                      std::fabs( a(3,2) ) +
                      std::fabs( a(3,3) ) +
                      std::fabs( b(3,0) ) +
                      std::fabs( b(3,1) ) +
                      std::fabs( b(3,2) ) +
                      std::fabs( b(3,3) )) / 8.0;

  if ( scale > 0.0 )
  {
    scale = 1.0 / scale;

    for ( int j = 0; j < 4; j++ )
    {
      adiff = (GLdouble) std::fabs ( a(3,j) - b(3,j) );
      rdiff = max ( rdiff, scale * adiff );
    }
  }

  return rdiff;
}


JEM_END_PACKAGE( gl )
