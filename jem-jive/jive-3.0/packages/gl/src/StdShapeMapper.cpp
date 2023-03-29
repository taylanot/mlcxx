
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/ClassTemplate.h>
#include <jive/geom/Shape.h>
#include <jive/geom/ShapeTable.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/BoundaryQuad.h>
#include <jive/geom/BoundaryTriangle.h>
#include <jive/gl/GfxShape.h>
#include <jive/gl/StdShapeMapper.h>


JEM_DEFINE_CLASS( jive::gl::StdShapeMapper );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jive::geom::Shape;


//=======================================================================
//   class StdShapeMapper
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdShapeMapper::StdShapeMapper ()
{}


StdShapeMapper::~StdShapeMapper ()
{}


//-----------------------------------------------------------------------
//   toGfxShape
//-----------------------------------------------------------------------


Ref<GfxShape>  StdShapeMapper::toGfxShape

  ( const Shape&  shape ) const

{
  const idx_t    rank = shape.localRank ();

  Ref<GfxShape>  xshape;

  if      ( rank == 0_idx )
  {
    xshape = newInstance<GfxPoint> ();
  }
  else if ( rank == 1_idx )
  {
    xshape = newLine_ ( shape );
  }
  else if ( rank == 2_idx )
  {
    xshape = newFace_ ( shape );
  }
  else
  {
    xshape = newInstance<GfxShape> ();
  }

  return xshape;
}


//-----------------------------------------------------------------------
//   newLine_
//-----------------------------------------------------------------------


Ref<GfxShape>  StdShapeMapper::newLine_ ( const Shape& shape )
{
  const idx_t   nnd  = shape.nodeCount ();

  Ref<GfxLine>  line = newInstance<GfxLine> ();


  line->inodes.resize ( nnd );

  for ( idx_t i = 0; i < nnd; i++ )
  {
    line->inodes[i] = i;
  }

  return line;
}


//-----------------------------------------------------------------------
//   newFace_
//-----------------------------------------------------------------------


Ref<GfxShape> StdShapeMapper::newFace_ ( const Shape&  shape )
{
  using jive::geom::Geometries;

  const idx_t    typeFlags = ShapeTable::getTypeFlags ( shape );

  Ref<GfxShape>  face;


  if ( typeFlags & (ShapeTable::INTERNAL_TYPE |
                    ShapeTable::BOUNDARY_TYPE) )
  {
    const idx_t  nodeCount = shape.nodeCount   ();
    String       geom      = shape.getGeometry ();

    if      ( nodeCount == 3_idx )
    {
      if ( geom == Geometries::TRIANGLE )
      {
        face = newTriang3_ ();
      }
    }
    else if ( nodeCount == 4_idx )
    {
      if ( geom == Geometries::SQUARE )
      {
        face = newQuad4_ ();
      }
    }
    else if ( nodeCount == 6_idx )
    {
      if      ( geom == Geometries::TRIANGLE )
      {
        face = newTriang6_ ();
      }
      else if ( geom == Geometries::SQUARE )
      {
        face = newQuad6_   ();
      }
    }
    else if ( nodeCount == 8_idx )
    {
      if ( geom == Geometries::SQUARE )
      {
        face = newQuad8_ ();
      }
    }
    else if ( nodeCount == 9_idx )
    {
      if ( geom == Geometries::SQUARE )
      {
        face = newQuad9_ ();
      }
    }
    else if ( nodeCount == 12_idx )
    {
      if ( geom == Geometries::SQUARE )
      {
        face = newQuad12_ ();
      }
    }
  }

  if ( ! face )
  {
    face = newInstance<GfxShape> ();
  }

  return face;
}


//-----------------------------------------------------------------------
//   newQuad4_
//-----------------------------------------------------------------------


Ref<GfxFace>  StdShapeMapper::newQuad4_ ()
{
  using jive::geom::BoundaryQuad4;

  Ref<GfxFace>  face =

    newInstance<GfxFace> ( BoundaryQuad4::getShape() );

  // Initialize the edges.

  face->xedges.resize ( 4 + 1 );
  face->enodes.resize ( 4 * 2 );

  int  i = 0;

  face->xedges[i++] = 0;
  face->xedges[i++] = 2;
  face->xedges[i++] = 4;
  face->xedges[i++] = 6;
  face->xedges[i++] = 8;

  i = 0;

  face->enodes[i++] = 0;
  face->enodes[i++] = 1;
  face->enodes[i++] = 1;
  face->enodes[i++] = 2;
  face->enodes[i++] = 2;
  face->enodes[i++] = 3;
  face->enodes[i++] = 3;
  face->enodes[i++] = 0;

  // Add the extra point.

  face->setXPoint ( 0.0, 0.0 );

  // Initialize the triangles.

  face->triangles.resize ( 3, 4 );

  i = 0;

  face->triangles(0,i) = 0;
  face->triangles(1,i) = 1;
  face->triangles(2,i) = 4;

  i++;

  face->triangles(0,i) = 1;
  face->triangles(1,i) = 2;
  face->triangles(2,i) = 4;

  i++;

  face->triangles(0,i) = 2;
  face->triangles(1,i) = 3;
  face->triangles(2,i) = 4;

  i++;

  face->triangles(0,i) = 3;
  face->triangles(1,i) = 0;
  face->triangles(2,i) = 4;

  return face;
}


//-----------------------------------------------------------------------
//   newQuad6_
//-----------------------------------------------------------------------


Ref<GfxFace>  StdShapeMapper::newQuad6_ ()
{
  using jive::geom::BoundaryQuad6;

  Ref<GfxFace>  face =

    newInstance<GfxFace> ( BoundaryQuad6::getShape() );

  // Initialize the edges.

  face->xedges.resize ( 4 + 1 );
  face->enodes.resize ( 2 * 3 + 2 * 2 );

  int  i = 0;

  face->xedges[i++] =  0;
  face->xedges[i++] =  3;
  face->xedges[i++] =  5;
  face->xedges[i++] =  8;
  face->xedges[i++] = 10;

  i = 0;

  face->enodes[i++] =  0;
  face->enodes[i++] =  1;
  face->enodes[i++] =  2;
  face->enodes[i++] =  2;
  face->enodes[i++] =  3;
  face->enodes[i++] =  3;
  face->enodes[i++] =  4;
  face->enodes[i++] =  5;
  face->enodes[i++] =  5;
  face->enodes[i++] =  0;

  // Add the extra point.

  face->setXPoint ( 0.0, 0.0 );

  // Initialize the triangles.

  face->triangles.resize ( 3, 6 );

  i = 0;

  face->triangles(0,i) = 0;
  face->triangles(1,i) = 1;
  face->triangles(2,i) = 6;

  i++;

  face->triangles(0,i) = 1;
  face->triangles(1,i) = 2;
  face->triangles(2,i) = 6;

  i++;

  face->triangles(0,i) = 2;
  face->triangles(1,i) = 3;
  face->triangles(2,i) = 6;

  i++;

  face->triangles(0,i) = 3;
  face->triangles(1,i) = 4;
  face->triangles(2,i) = 6;

  i++;

  face->triangles(0,i) = 4;
  face->triangles(1,i) = 5;
  face->triangles(2,i) = 6;

  i++;

  face->triangles(0,i) = 5;
  face->triangles(1,i) = 0;
  face->triangles(2,i) = 6;

  return face;
}


//-----------------------------------------------------------------------
//   newQuad8_
//-----------------------------------------------------------------------


Ref<GfxFace>  StdShapeMapper::newQuad8_ ()
{
  using jive::geom::BoundaryQuad8;

  Ref<GfxFace>  face = newQuad9_ ();

  face->shape = BoundaryQuad8::getShape ();

  // Add the extra point.

  face->setXPoint ( 0.0, 0.0 );

  return face;
}


//-----------------------------------------------------------------------
//   newQuad9_
//-----------------------------------------------------------------------


Ref<GfxFace>  StdShapeMapper::newQuad9_ ()
{
  using jive::geom::BoundaryQuad9;

  Ref<GfxFace>  face =

    newInstance<GfxFace> ( BoundaryQuad9::getShape() );

  // Initialize the edges.

  face->xedges.resize ( 4 + 1 );
  face->enodes.resize ( 4 * 3 );

  int  i = 0;

  face->xedges[i++] =  0;
  face->xedges[i++] =  3;
  face->xedges[i++] =  6;
  face->xedges[i++] =  9;
  face->xedges[i++] = 12;

  i = 0;

  face->enodes[i++] =  0;
  face->enodes[i++] =  1;
  face->enodes[i++] =  2;
  face->enodes[i++] =  2;
  face->enodes[i++] =  3;
  face->enodes[i++] =  4;
  face->enodes[i++] =  4;
  face->enodes[i++] =  5;
  face->enodes[i++] =  6;
  face->enodes[i++] =  6;
  face->enodes[i++] =  7;
  face->enodes[i++] =  0;

  // Initialize the triangles.

  face->triangles.resize ( 3, 8 );

  i = 0;

  face->triangles(0,i) = 0;
  face->triangles(1,i) = 1;
  face->triangles(2,i) = 7;

  i++;

  face->triangles(0,i) = 1;
  face->triangles(1,i) = 2;
  face->triangles(2,i) = 3;

  i++;

  face->triangles(0,i) = 3;
  face->triangles(1,i) = 4;
  face->triangles(2,i) = 5;

  i++;

  face->triangles(0,i) = 5;
  face->triangles(1,i) = 6;
  face->triangles(2,i) = 7;

  i++;

  face->triangles(0,i) = 7;
  face->triangles(1,i) = 1;
  face->triangles(2,i) = 8;

  i++;

  face->triangles(0,i) = 1;
  face->triangles(1,i) = 3;
  face->triangles(2,i) = 8;

  i++;

  face->triangles(0,i) = 3;
  face->triangles(1,i) = 5;
  face->triangles(2,i) = 8;

  i++;

  face->triangles(0,i) = 5;
  face->triangles(1,i) = 7;
  face->triangles(2,i) = 8;

  return face;
}


//-----------------------------------------------------------------------
//   newQuad12_
//-----------------------------------------------------------------------


Ref<GfxFace>  StdShapeMapper::newQuad12_ ()
{
  using jive::geom::BoundaryQuad12;

  const double  a    = 1.0 / 3.0;

  Ref<GfxFace>  face =

    newInstance<GfxFace> ( BoundaryQuad12::getShape() );

  Matrix        xp   ( 2, 4 );

  // Initialize the edges.

  face->xedges.resize ( 4 + 1 );
  face->enodes.resize ( 4 * 4 );

  int  i = 0;

  face->xedges[i++] =  0;
  face->xedges[i++] =  4;
  face->xedges[i++] =  8;
  face->xedges[i++] = 12;
  face->xedges[i++] = 16;

  i = 0;

  face->enodes[i++] =  0;
  face->enodes[i++] =  1;
  face->enodes[i++] =  2;
  face->enodes[i++] =  3;
  face->enodes[i++] =  3;
  face->enodes[i++] =  4;
  face->enodes[i++] =  5;
  face->enodes[i++] =  6;
  face->enodes[i++] =  6;
  face->enodes[i++] =  7;
  face->enodes[i++] =  8;
  face->enodes[i++] =  9;
  face->enodes[i++] =  9;
  face->enodes[i++] = 10;
  face->enodes[i++] = 11;
  face->enodes[i++] =  0;

  // Add the extra points.

  xp(0,0) = -a;
  xp(1,0) = -a;

  xp(0,1) =  a;
  xp(1,1) = -a;

  xp(0,2) =  a;
  xp(1,2) =  a;

  xp(0,3) = -a;
  xp(1,3) =  a;

  face->setXPoints ( xp );

  // Initialize the triangles.

  face->triangles.resize ( 3, 18 );

  i = 0;

  face->triangles(0,i) =  0;
  face->triangles(1,i) = 12;
  face->triangles(2,i) = 11;

  i++;

  face->triangles(0,i) =  0;
  face->triangles(1,i) =  1;
  face->triangles(2,i) = 12;

  i++;

  face->triangles(0,i) =  1;
  face->triangles(1,i) = 13;
  face->triangles(2,i) = 12;

  i++;

  face->triangles(0,i) =  1;
  face->triangles(1,i) =  2;
  face->triangles(2,i) = 13;

  i++;

  face->triangles(0,i) =  2;
  face->triangles(1,i) =  4;
  face->triangles(2,i) = 13;

  i++;

  face->triangles(0,i) =  2;
  face->triangles(1,i) =  3;
  face->triangles(2,i) =  4;

  i++;

  face->triangles(0,i) =  4;
  face->triangles(1,i) =  5;
  face->triangles(2,i) = 13;

  i++;

  face->triangles(0,i) = 13;
  face->triangles(1,i) =  5;
  face->triangles(2,i) = 14;

  i++;

  face->triangles(0,i) = 13;
  face->triangles(1,i) = 14;
  face->triangles(2,i) = 12;

  i++;

  face->triangles(0,i) = 12;
  face->triangles(1,i) = 14;
  face->triangles(2,i) = 15;

  i++;

  face->triangles(0,i) = 12;
  face->triangles(1,i) = 15;
  face->triangles(2,i) = 11;

  i++;

  face->triangles(0,i) = 15;
  face->triangles(1,i) = 10;
  face->triangles(2,i) = 11;

  i++;

  face->triangles(0,i) = 10;
  face->triangles(1,i) =  8;
  face->triangles(2,i) =  9;

  i++;

  face->triangles(0,i) = 10;
  face->triangles(1,i) = 15;
  face->triangles(2,i) =  8;

  i++;

  face->triangles(0,i) = 15;
  face->triangles(1,i) =  7;
  face->triangles(2,i) =  8;

  i++;

  face->triangles(0,i) = 15;
  face->triangles(1,i) = 14;
  face->triangles(2,i) =  7;

  i++;

  face->triangles(0,i) = 14;
  face->triangles(1,i) =  6;
  face->triangles(2,i) =  7;

  i++;

  face->triangles(0,i) = 14;
  face->triangles(1,i) =  5;
  face->triangles(2,i) =  6;

  return face;
}


//-----------------------------------------------------------------------
//   newTriang3_
//-----------------------------------------------------------------------


Ref<GfxFace>  StdShapeMapper::newTriang3_ ()
{
  using jive::geom::BoundaryTriangle3;

  Ref<GfxFace>  face =

    newInstance<GfxFace> ( BoundaryTriangle3::getShape() );

  // Initialize the edges.

  face->xedges.resize ( 3 + 1 );
  face->enodes.resize ( 3 * 2 );

  int  i = 0;

  face->xedges[i++] = 0;
  face->xedges[i++] = 2;
  face->xedges[i++] = 4;
  face->xedges[i++] = 6;

  i = 0;

  face->enodes[i++] = 0;
  face->enodes[i++] = 1;
  face->enodes[i++] = 1;
  face->enodes[i++] = 2;
  face->enodes[i++] = 2;
  face->enodes[i++] = 0;

  // Initialize the triangles.

  face->triangles.resize ( 3, 1 );

  face->triangles(0,0) = 0;
  face->triangles(1,0) = 1;
  face->triangles(2,0) = 2;

  return face;
}


//-----------------------------------------------------------------------
//   newTriang6
//-----------------------------------------------------------------------


Ref<GfxFace>  StdShapeMapper::newTriang6_ ()
{
  using jive::geom::BoundaryTriangle6;

  Ref<GfxFace>  face =

    newInstance<GfxFace> ( BoundaryTriangle6::getShape() );

  // Initialize the edges.

  face->xedges.resize ( 3 + 1 );
  face->enodes.resize ( 3 * 3 );

  int  i = 0;

  face->xedges[i++] = 0;
  face->xedges[i++] = 3;
  face->xedges[i++] = 6;
  face->xedges[i++] = 9;

  i = 0;

  face->enodes[i++] = 0;
  face->enodes[i++] = 1;
  face->enodes[i++] = 2;
  face->enodes[i++] = 2;
  face->enodes[i++] = 3;
  face->enodes[i++] = 4;
  face->enodes[i++] = 4;
  face->enodes[i++] = 5;
  face->enodes[i++] = 0;

  // Initialize the triangles.

  face->triangles.resize ( 3, 4 );

  i = 0;

  face->triangles(0,i) = 0;
  face->triangles(1,i) = 1;
  face->triangles(2,i) = 5;

  i++;

  face->triangles(0,i) = 1;
  face->triangles(1,i) = 2;
  face->triangles(2,i) = 3;

  i++;

  face->triangles(0,i) = 3;
  face->triangles(1,i) = 4;
  face->triangles(2,i) = 5;

  i++;

  face->triangles(0,i) = 1;
  face->triangles(1,i) = 3;
  face->triangles(2,i) = 5;

  return face;
}


JIVE_END_PACKAGE( gl )
