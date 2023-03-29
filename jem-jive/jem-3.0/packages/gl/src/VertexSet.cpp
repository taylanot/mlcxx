
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/gl/Box.h>
#include <jem/gl/VertexSet.h>


JEM_DEFINE_CLASS ( jem::gl::VertexSet );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class VertexSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


VertexSet::VertexSet ()
{
  validBBox_ = false;
}


VertexSet::VertexSet ( const GLfloatMatrix& coords ) :

  coords_ ( makeContiguous( coords ) )

{
  validBBox_ = false;
}


VertexSet::~VertexSet ()
{}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> VertexSet::clone () const
{
  return newInstance<VertexSet> ( coords_.clone() );
}


//-----------------------------------------------------------------------
//   reshape
//-----------------------------------------------------------------------


void VertexSet::reshape ( idx_t m, idx_t n )
{
  if ( m != coords_.size(0) || n != coords_.size(1) )
  {
    coords_.reshape ( m, n );

    newShapeEvent.emit ( coords_.shape(), *this );
  }
}


//-----------------------------------------------------------------------
//   setCoords
//-----------------------------------------------------------------------


void VertexSet::setCoords ( const GLfloatMatrix& coords )
{
  bool  newShape = ! equal ( coords.shape(), coords_.shape() );

  coords_.ref ( makeContiguous( coords ) );

  validBBox_ = false;

  if ( newShape )
  {
    newShapeEvent.emit ( coords.shape(), *this );
  }
  else
  {
    newGeomEvent .emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   newCoords
//-----------------------------------------------------------------------


void VertexSet::newCoords ()
{
  validBBox_ = false;

  newGeomEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getBBox
//-----------------------------------------------------------------------


void VertexSet::getBBox ( Box& box ) const
{
  const idx_t  r = rank ();


  if ( r * coords_.size(1) == 0 )
  {
    box.makeEmpty ();
    return;
  }

  if ( ! validBBox_ )
  {
    const_cast<Self*>(this)->updateBBox_ ();
  }

  if      ( r == 1 )
  {
    box.setCorners1D ( bbox_[0], bbox_[3] );
  }
  else if ( r == 2 )
  {
    box.setCorners2D ( bbox_, bbox_ + 3 );
  }
  else
  {
    box.setCorners3D ( bbox_, bbox_ + 3 );
  }
}


//-----------------------------------------------------------------------
//   findVertex
//-----------------------------------------------------------------------


void VertexSet::findVertex

  ( idx_t&    index,
    GLfloat&  dist,
    GLfloat   x,
    GLfloat   y,
    GLfloat   z ) const

{
  const idx_t  r = rank ();

  index = -1;

  if ( r * coords_.size(1) == 0 )
  {
    return;
  }

  switch ( r )
  {
  case 0:

    break;

  case 1:

    findVertex_ ( index, dist, x );
    break;

  case 2:

    findVertex_ ( index, dist, x, y );
    break;

  case 3:

    findVertex_ ( index, dist, x, y, z );
    break;

  default:

    {
      GLfloatVector  pos ( r );

      pos    = (GLfloat) 0.0;
      pos[0] = x;
      pos[1] = y;
      pos[2] = z;

      findVertex_ ( index, dist, pos.addr() );
    }
  }
}


void VertexSet::findVertex

  ( idx_t&          index,
    GLfloat&        dist,
    const GLfloat*  pos ) const

{
  const idx_t  r = rank ();

  index = -1;

  if ( r * coords_.size(1) == 0 )
  {
    return;
  }

  switch ( r )
  {
  case 0:

    break;

  case 1:

    findVertex_ ( index, dist, pos[0] );
    break;

  case 2:

    findVertex_ ( index, dist, pos[0], pos[1] );
    break;

  case 3:

    findVertex_ ( index, dist, pos[0], pos[1], pos[2] );
    break;

  default:

    findVertex_ ( index, dist, pos );
  }
}


//-----------------------------------------------------------------------
//   updateBBox_
//-----------------------------------------------------------------------


void VertexSet::updateBBox_ ()
{
  const GLfloat*  v = coords_.addr ();
  const idx_t     r = coords_.size (0);
  const idx_t     n = coords_.size (1);

  GLfloat         a[3];
  GLfloat         b[3];


  for ( int i = 0; i < 3; i++ )
  {
    a[i] = b[i] = 0.0;
  }

  if ( n > 0_idx )
  {
    idx_t  j = jem::min ( r, 3_idx );

    for ( idx_t i = 0; i < j; i++ )
    {
      a[i] = b[i] = v[i];
    }

    v += r;
  }

  switch ( r )
  {
  case 0:

    break;

  case 1:

    for ( idx_t j = 1; j < n; j++ )
    {
      a[0] = jem::min ( a[0], v[0] );
      b[0] = jem::max ( b[0], v[0] );

      v++;
    }

    break;

  case 2:

    for ( idx_t j = 1; j < n; j++ )
    {
      a[0] = jem::min ( a[0], v[0] );
      a[1] = jem::min ( a[1], v[1] );
      b[0] = jem::max ( b[0], v[0] );
      b[1] = jem::max ( b[1], v[1] );

      v   += 2;
    }

    break;

  default:

    for ( idx_t j = 1; j < n; j++ )
    {
      a[0] = jem::min ( a[0], v[0] );
      a[1] = jem::min ( a[1], v[1] );
      a[2] = jem::min ( a[2], v[2] );
      b[0] = jem::max ( b[0], v[0] );
      b[1] = jem::max ( b[1], v[1] );
      b[2] = jem::max ( b[2], v[2] );

      v   += r;
    }
  }

  bbox_[0]   = a[0];
  bbox_[1]   = a[1];
  bbox_[2]   = a[2];
  bbox_[3]   = b[0];
  bbox_[4]   = b[1];
  bbox_[5]   = b[2];

  validBBox_ = true;
}


//-----------------------------------------------------------------------
//   findVertex_ (1-D case)
//-----------------------------------------------------------------------


void VertexSet::findVertex_

  ( idx_t&     index,
    GLfloat&  dist,
    GLfloat   xpos ) const

{
  const GLfloat*  v = coords_.addr ();
  const idx_t     n = coords_.size (1);

  GLfloat  minDist;
  GLfloat  delta;


  minDist = (GLfloat) std::fabs ( xpos - v[0] );
  index   = 0;

  for ( idx_t i = 1; i < n; i++ )
  {
    delta = (GLfloat) std::fabs ( xpos - v[i] );

    if ( delta < minDist )
    {
      minDist = delta;
      index   = i;
    }
  }

  dist = minDist;
}


//-----------------------------------------------------------------------
//   findVertex_ (2-D case)
//-----------------------------------------------------------------------


void VertexSet::findVertex_

  ( idx_t&    index,
    GLfloat&  dist,
    GLfloat   xpos,
    GLfloat   ypos ) const

{
  const GLfloat*  v = coords_.addr ();
  const idx_t     n = coords_.size (1);

  double  minDist2;
  double  dist2;
  double  dx;
  double  dy;


  dx       = xpos - v[0];
  dy       = ypos - v[1];
  minDist2 = dx * dx + dy * dy;
  index    = 0;
  v       += 2;

  for ( idx_t i = 1; i < n; i++, v += 2 )
  {
    dx    = xpos - v[0];
    dy    = ypos - v[1];
    dist2 = dx * dx + dy * dy;

    if ( dist2 < minDist2 )
    {
      minDist2 = dist2;
      index    = i;
    }
  }

  dist = (GLfloat) std::sqrt ( minDist2 );
}


//-----------------------------------------------------------------------
//   findVertex_ (3-D case)
//-----------------------------------------------------------------------


void VertexSet::findVertex_

  ( idx_t&    index,
    GLfloat&  dist,
    GLfloat   xpos,
    GLfloat   ypos,
    GLfloat   zpos ) const

{
  const GLfloat*  v = coords_.addr ();
  const idx_t     n = coords_.size (1);

  double  minDist2;
  double  dist2;
  double  dx;
  double  dy;
  double  dz;


  dx       = xpos - v[0];
  dy       = ypos - v[1];
  dz       = zpos - v[2];
  minDist2 = dx * dx + dy * dy + dz * dz;
  index    = 0;
  v       += 3;

  for ( idx_t i = 1; i < n; i++, v += 3 )
  {
    dx    = xpos - v[0];
    dy    = ypos - v[1];
    dz    = zpos - v[2];
    dist2 = dx * dx + dy * dy + dz * dz;

    if ( dist2 < minDist2 )
    {
      minDist2 = dist2;
      index    = i;
    }
  }

  dist = (GLfloat) std::sqrt ( minDist2 );
}


//-----------------------------------------------------------------------
//   findVertex_ (general case)
//-----------------------------------------------------------------------


void VertexSet::findVertex_

  ( idx_t&          index,
    GLfloat&        dist,
    const GLfloat*  pos ) const

{
  const GLfloat*  v = coords_.addr ();
  const idx_t     r = coords_.size (0);
  const idx_t     n = coords_.size (1);

  double  minDist2;
  double  dist2;
  double  delta;


  dist2 = 0.0;

  for ( idx_t j = 0; j < r; j++ )
  {
    delta  = pos[j] - v[j];
    dist2 += delta * delta;
  }

  minDist2 = dist2;
  index    = 0;
  v       += r;

  for ( idx_t i = 1; i < n; i++, v += r )
  {
    dist2 = 0.0;

    for ( idx_t j = 0; j < r; j++ )
    {
      delta  = pos[j] - v[j];
      dist2 += delta * delta;
    }

    if ( dist2 < minDist2 )
    {
      minDist2 = dist2;
      index    = i;
    }
  }

  dist = (GLfloat) std::sqrt ( minDist2 );
}


JEM_END_PACKAGE( gl )
