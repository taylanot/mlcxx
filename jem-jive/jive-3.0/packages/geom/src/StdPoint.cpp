
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
#include <jive/geom/Geometries.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdPoint.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdPoint );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class StdPoint
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdPoint::GEOMETRY = Geometries::POINT;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdPoint::StdPoint ()
{}


StdPoint::~StdPoint ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdPoint::readFrom ( ObjectInput& )
{}


void StdPoint::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdPoint::rank () const
{
  return (idx_t) RANK;
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdPoint::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdPoint::getGeometry() const
{
  return GEOMETRY;
}


//-----------------------------------------------------------------------
//   isLinear
//-----------------------------------------------------------------------


bool StdPoint::isLinear () const
{
  return true;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdPoint::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdPoint::vertexCoords ()
{
  return Matrix ( RANK, VERTEX_COUNT );
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdPoint::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoint
//-----------------------------------------------------------------------


bool StdPoint::containsPoint ( const Vector& u ) const
{
  JEM_PRECHECK2 ( u.size() == 0,
                  "coordinate array has wrong size" );

  return true;
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdPoint::clipPoint ( const Vector& u ) const
{
  JEM_PRECHECK2 ( u.size() == 0,
                  "coordinate array has wrong size" );

  return false;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdPoint::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  f[0] = 1.0;
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdPoint::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdPoint::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdPoint::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdPoint::declare ()
{
  StdShapeFactory::declare ( "Point", GEOMETRY, & makeNew );
}


JIVE_END_PACKAGE( geom )
