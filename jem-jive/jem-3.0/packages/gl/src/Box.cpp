
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
#include <jem/base/Error.h>
#include <jem/base/tuple/operators.h>
#include <jem/gl/Box.h>
#include <jem/gl/Rect.h>
#include <jem/gl/Color.h>
#include <jem/gl/Transform.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Box
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Box::Box ( const Box& rhs )
{
  rank_       = rhs.rank_;
  corners_[0] = rhs.corners_[0];
  corners_[1] = rhs.corners_[1];

  if ( rhs.transfm_ )
  {
    transfm_ = new Transform ( *rhs.transfm_ );
  }
  else
  {
    transfm_ = 0;
  }
}


Box::Box ( Box&& rhs ) noexcept
{
  rank_        = rhs.rank_;
  corners_[0]  = rhs.corners_[0];
  corners_[1]  = rhs.corners_[1];
  transfm_     = rhs.transfm_;
  rhs.transfm_ = nullptr;
}


Box::~Box ()
{
  delTransfm_ ();
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


Box& Box::operator = ( const Box& rhs )
{
  if ( this != & rhs )
  {
    rank_       = rhs.rank_;
    corners_[0] = rhs.corners_[0];
    corners_[1] = rhs.corners_[1];

    if ( rhs.transfm_ )
    {
      if ( transfm_ )
      {
        *transfm_ = *rhs.transfm_;
      }
      else
      {
        transfm_  = new Transform ( *rhs.transfm_ );
      }
    }
    else
    {
      delTransfm_ ();
    }
  }

  return *this;
}


Box& Box::operator = ( Box&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void Box::swap ( Box& rhs ) noexcept
{
  jem::swap ( rank_,       rhs.rank_       );
  jem::swap ( corners_[0], rhs.corners_[0] );
  jem::swap ( corners_[1], rhs.corners_[1] );
  jem::swap ( transfm_,    rhs.transfm_    );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


double Box::size () const noexcept
{
  double  s;

  if ( rank_ < 1 )
  {
    s = 0.0;
  }
  else
  {
    const GLdouble*  a = corners_[0].addr ();
    const GLdouble*  b = corners_[1].addr ();

    double  dx = (double) (b[0] - a[0]);
    double  dy = (double) (b[1] - a[1]);
    double  dz = (double) (b[2] - a[2]);

    s = std::sqrt ( dx * dx + dy * dy + dz * dz );

    if ( transfm_ )
    {
      Vertex3d  c, v;

      c     = (GLdouble) 0.5 * (corners_[0] + corners_[1]);
      v     = c;
      v[0] += 0.5 * s;

      c     = matmul ( *transfm_, c );
      v     = matmul ( *transfm_, v );

      dx    = (double) (v[0] - c[0]);
      dy    = (double) (v[1] - c[1]);
      dz    = (double) (v[2] - c[2]);

      s     = 2.0 * std::sqrt ( dx * dx + dy * dy + dz * dz );
    }
  }

  return s;
}


//-----------------------------------------------------------------------
//   center
//-----------------------------------------------------------------------


Vertex3d Box::center () const noexcept
{
  Vertex3d  c = (GLdouble) 0.5 * (corners_[0] + corners_[1]);

  if ( transfm_ )
  {
    c = matmul ( *transfm_, c );
  }

  return c;
}


//-----------------------------------------------------------------------
//   align
//-----------------------------------------------------------------------


void Box::align ()
{
  if ( transfm_ )
  {
    const int  rank = rank_;

    Vertex3d   a, b;

    getCorners ( a, b );
    setCorners ( a, b );

    rank_ = rank;
  }
}


//-----------------------------------------------------------------------
//   makeEmpty
//-----------------------------------------------------------------------


void Box::makeEmpty ()
{
  rank_       = 0;
  corners_[0] = 0.0;
  corners_[1] = 0.0;

  delTransfm_ ();
}


//-----------------------------------------------------------------------
//   getCorners
//-----------------------------------------------------------------------


void Box::getCorners

  ( Vertex3d&  a,
    Vertex3d&  b ) const

{
  if ( transfm_ && rank_ > 0 )
  {
    if      ( rank_ == 2 )
    {
      getCorners2D_ ( a, b );
    }
    else if ( rank_ == 3 )
    {
      getCorners3D_ ( a, b );
    }
    else
    {
      throw Error ( JEM_FUNC, "invalid rank: " + String( rank_ ) );
    }
  }
  else
  {
    a = corners_[0];
    b = corners_[1];
  }
}


//-----------------------------------------------------------------------
//   setCorners1D
//-----------------------------------------------------------------------


void Box::setCorners1D

  ( GLdouble  x,
    GLdouble  y )

{
  if ( y < x )
  {
    makeEmpty ();
  }
  else
  {
    GLdouble*  a = corners_[0].addr ();
    GLdouble*  b = corners_[1].addr ();

    rank_ = 1;

    a[0]  = x;
    a[1]  = 0.0;
    a[2]  = 0.0;

    b[0]  = y;
    b[1]  = 0.0;
    b[2]  = 0.0;

    delTransfm_ ();
  }
}


//-----------------------------------------------------------------------
//   setCorners2D
//-----------------------------------------------------------------------


void Box::setCorners2D

  ( const GLfloat*  x,
    const GLfloat*  y )

{
  if ( y[0] < x[0] || y[1] < x[1] )
  {
    makeEmpty ();
  }
  else
  {
    GLdouble*  a = corners_[0].addr ();
    GLdouble*  b = corners_[1].addr ();

    rank_ = 2;

    a[0]  = x[0];
    a[1]  = x[1];
    a[2]  = 0.0;

    b[0]  = y[0];
    b[1]  = y[1];
    b[2]  = 0.0;

    delTransfm_ ();
  }
}


//-----------------------------------------------------------------------
//   setCorners2D
//-----------------------------------------------------------------------


void Box::setCorners2D

  ( const GLdouble*  x,
    const GLdouble*  y )

{
  if ( y[0] < x[0] || y[1] < x[1] )
  {
    makeEmpty ();
  }
  else
  {
    GLdouble*  a = corners_[0].addr ();
    GLdouble*  b = corners_[1].addr ();

    rank_ = 2;

    a[0]  = x[0];
    a[1]  = x[1];
    a[2]  = 0.0;

    b[0]  = y[0];
    b[1]  = y[1];
    b[2]  = 0.0;

    delTransfm_ ();
  }
}


//-----------------------------------------------------------------------
//   setCorners3D
//-----------------------------------------------------------------------


void Box::setCorners3D

  ( const GLfloat*  x,
    const GLfloat*  y )

{
  if ( y[0] < x[0] || y[1] < x[1] || y[2] < x[2] )
  {
    makeEmpty ();
  }
  else
  {
    GLdouble*  a = corners_[0].addr ();
    GLdouble*  b = corners_[1].addr ();

    rank_ = 3;

    a[0]  = x[0];
    a[1]  = x[1];
    a[2]  = x[2];

    b[0]  = y[0];
    b[1]  = y[1];
    b[2]  = y[2];

    delTransfm_ ();
  }
}


//-----------------------------------------------------------------------
//   setCorners3D
//-----------------------------------------------------------------------


void Box::setCorners3D

  ( const GLdouble*  x,
    const GLdouble*  y )

{
  if ( y[0] < x[0] || y[1] < x[1] || y[2] < x[2] )
  {
    makeEmpty ();
  }
  else
  {
    GLdouble*  a = corners_[0].addr ();
    GLdouble*  b = corners_[1].addr ();

    rank_ = 3;

    a[0]  = x[0];
    a[1]  = x[1];
    a[2]  = x[2];

    b[0]  = y[0];
    b[1]  = y[1];
    b[2]  = y[2];

    delTransfm_ ();
  }
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void Box::scale ( GLdouble s )
{
  GLdouble*  a = corners_[0].addr ();
  GLdouble*  b = corners_[1].addr ();

  GLdouble   mid;


  for ( int i = 0; i < rank_; i++ )
  {
    mid  =     0.5 * (a[i] + b[i]);
    a[i] = mid + s * (a[i] - mid);
    b[i] = mid + s * (b[i] - mid);
  }
}


//-----------------------------------------------------------------------
//   enlargeWith
//-----------------------------------------------------------------------


void Box::enlargeWith ( const Box& rhs )
{
  if ( rhs.rank_ == 0 )
  {
    return;
  }

  if ( rank_ == 0 )
  {
    *this = rhs;
  }
  else
  {
    const int  rank = jem::max ( rank_, rhs.rank_ );

    Vertex3d   a, b;
    Vertex3d   x, y;


    this->getCorners ( a, b );
    rhs  .getCorners ( x, y );

    for ( int i = 0; i < 3; i++ )
    {
      if ( x[i] < a[i] )
      {
        a[i] = x[i];
      }
    }

    for ( int i = 0; i < 3; i++ )
    {
      if ( y[i] > b[i] )
      {
        b[i] = y[i];
      }
    }

    setCorners ( a, b );

    rank_ = rank;
  }
}


//-----------------------------------------------------------------------
//   transform
//-----------------------------------------------------------------------


void Box::transform ( const Transform& tr )
{
  if ( rank_ > 0 )
  {
    if ( transfm_ )
    {
      transfm_->transform ( tr );
    }
    else
    {
      transfm_ = new Transform ( tr );
    }

    rank_ = 3;
  }
}


//-----------------------------------------------------------------------
//   transform2D
//-----------------------------------------------------------------------


void Box::transform2D ( const Transform& tr )
{
  if ( rank_ > 0 )
  {
    if ( transfm_ )
    {
      transfm_->transform ( tr );
    }
    else
    {
      transfm_ = new Transform ( tr );
    }

    if ( rank_ == 1 )
    {
      rank_ = 2;
    }
  }
}


//-----------------------------------------------------------------------
//   project
//-----------------------------------------------------------------------


void Box::project ( const Transform& p )
{
  if ( rank_ > 0 )
  {
    rank_ = 3;

    if ( transfm_ )
    {
      Transform  t;

      matmul      ( t, p, *transfm_ );
      project_    ( t );
      delTransfm_ ();
    }
    else
    {
      project_    ( p );
    }
  }
}


void Box::project

  ( const Transform&  p,
    const Rect&       vport )

{
  JEM_PRECHECK ( vport.isValid() );

  project ( p );

  if ( rank_ > 0 )
  {
    toWindow ( corners_[0], vport );
    toWindow ( corners_[1], vport );
  }
}


//-----------------------------------------------------------------------
//   getDrawData
//-----------------------------------------------------------------------


const Transform* Box::getDrawData

  ( Vertex3f&  a,
    Vertex3f&  b ) const

{
  const GLdouble*  c0 = corners_[0].addr ();
  const GLdouble*  c1 = corners_[1].addr ();
  GLfloat*         ap = a          .addr ();
  GLfloat*         bp = b          .addr ();

  ap[0] = (GLfloat) c0[0];
  ap[1] = (GLfloat) c0[1];
  ap[2] = (GLfloat) c0[2];
  bp[0] = (GLfloat) c1[0];
  bp[1] = (GLfloat) c1[1];
  bp[2] = (GLfloat) c1[2];

  return transfm_;
}


//-----------------------------------------------------------------------
//   delTransfm_
//-----------------------------------------------------------------------


inline void Box::delTransfm_ ()
{
  if ( transfm_ )
  {
    delete transfm_;
    transfm_ = 0;
  }
}


//-----------------------------------------------------------------------
//   getCorners2D_
//-----------------------------------------------------------------------


void Box::getCorners2D_

  ( Vertex3d&  x,
    Vertex3d&  y ) const

{
  const Transform&  tr = *transfm_;

  const GLdouble*   a  = corners_[0].addr ();
  const GLdouble*   b  = corners_[1].addr ();

  Vertex3d   v[4];
  Vertex3d   u;
  GLdouble   t;
  int        j = 0;


  u[0] = a[0];
  u[1] = a[1];

  matmul ( v[j], tr, u ); j++;

  u[0] = b[0];

  matmul ( v[j], tr, u ); j++;

  u[1] = b[1];

  matmul ( v[j], tr, u ); j++;

  u[0] = a[0];

  matmul ( v[j], tr, u ); j++;

  for ( int i = 0; i < 2; i++ )
  {
    x[i] = y[i] = v[0][i];
  }

  x[2] = y[2] = 0.0;

  for ( j = 1; j < 4; j++ )
  {
    for ( int i = 0; i < 2; i++ )
    {
      t = v[j][i];

      if      ( t < x[i] )
      {
        x[i] = t;
      }
      else if ( t > y[i] )
      {
        y[i] = t;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getCorners3D_
//-----------------------------------------------------------------------


void Box::getCorners3D_

  ( Vertex3d&  x,
    Vertex3d&  y ) const

{
  const Transform&  tr = *transfm_;

  const GLdouble*   a  = corners_[0].addr ();
  const GLdouble*   b  = corners_[1].addr ();

  Vertex3d   v[8];
  Vertex3d   u;
  GLdouble   t;
  int        j = 0;


  j = 0;
  u = corners_[0];

  for ( int i = 0; i < 2; i++ )
  {
    matmul ( v[j], tr, u ); j++;

    u[0] = b[0];

    matmul ( v[j], tr, u ); j++;

    u[1] = b[1];

    matmul ( v[j], tr, u ); j++;

    u[0] = a[0];

    matmul ( v[j], tr, u ); j++;

    u[1] = a[1];
    u[2] = b[2];
  }

  x = y = v[0];

  for ( j = 1; j < 8; j++ )
  {
    for ( int i = 0; i < 3; i++ )
    {
      t = v[j][i];

      if      ( t < x[i] )
      {
        x[i] = t;
      }
      else if ( t > y[i] )
      {
        y[i] = t;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   project_
//-----------------------------------------------------------------------


void Box::project_ ( const Transform& p )
{
  GLdouble*  a = corners_[0].addr ();
  GLdouble*  b = corners_[1].addr ();

  Vertex3d   v[8];
  Vertex3d   u;
  GLdouble   t;
  int        j, k;


  j = 0;
  u = corners_[0];

  for ( int i = 0; i < 2; i++ )
  {
    if ( jem::gl::project( v[j], p, u ) )
    {
      j++;
    }

    u[0] = b[0];

    if ( jem::gl::project( v[j], p, u ) )
    {
      j++;
    }

    u[1] = b[1];

    if ( jem::gl::project( v[j], p, u ) )
    {
      j++;
    }

    u[0] = a[0];

    if ( jem::gl::project( v[j], p, u ) )
    {
      j++;
    }

    u[1] = a[1];
    u[2] = b[2];
  }

  k = j;

  if ( k < 2 )
  {
    makeEmpty ();
    return;
  }

  for ( int i = 0; i < 3; i++ )
  {
    a[i] = b[i] = v[0][i];
  }

  for ( j = 1; j < k; j++ )
  {
    for ( int i = 0; i < 3; i++ )
    {
      t = v[j][i];

      if      ( t < a[i] )
      {
        a[i] = t;
      }
      else if ( t > b[i] )
      {
        b[i] = t;
      }
    }
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   isVisible
//-----------------------------------------------------------------------


bool                isVisible

  ( const Box&        box,
    const Transform&  proj )

{
  if ( box.isEmpty() )
  {
    return false;
  }

  Box  pbox ( box );

  pbox.project ( proj );

  if ( pbox.isEmpty() )
  {
    return false;
  }
  else
  {
    Vertex3d  a, b;

    pbox.getCorners ( a, b );

    if ( a[0] > 1.0 || b[0] < -1.0 ||
         a[1] > 1.0 || b[1] < -1.0 ||
         a[2] > 1.0 || b[2] < -1.0 )
    {
      return false;
    }
    else
    {
      return true;
    }
  }
}


//-----------------------------------------------------------------------
//   distance
//-----------------------------------------------------------------------


double              distance

  ( const Box&        first,
    const Box&        second )

{
  Vertex3d  a, b, c, d;
  GLdouble  u, v;


  first .getCorners ( a, b );
  second.getCorners ( c, d );

  u = 0.0;
  v = 0.0;

  for ( int i = 0; i < 3; i++ )
  {
    GLdouble  du = a[i] - c[i];
    GLdouble  dv = b[i] - d[i];

    u += du * du;
    v += dv * dv;
  }

  return std::sqrt ( max( u, v ) );
}


//-----------------------------------------------------------------------
//   overlap
//-----------------------------------------------------------------------


bool                overlap

  ( const Box&        first,
    const Box&        second )

{
  Vertex3d  a, b, c, d;

  first .getCorners ( a, b );
  second.getCorners ( c, d );

  for ( int i = 0; i < 3; i++ )
  {
    if ( (d[i] - a[i]) * (b[i] - c[i]) < 0.0 )
    {
      return false;
    }
  }

  return true;
}


JEM_END_PACKAGE( gl )
