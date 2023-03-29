
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
#include <jive/geom/StdLine.h>
#include <jive/geom/StdTriangle.h>
#include <jive/geom/StdPrism.h>
#include <jive/geom/shfuncs/WedgeFuncs.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/IntegrationSchemes.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdWedge.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdWedge6  );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdWedge18 );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class StdWedge
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdWedge::GEOMETRY = Geometries::WEDGE;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


StdWedge::~StdWedge ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdWedge::rank () const
{
  return (idx_t) RANK;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdWedge::getGeometry () const
{
  return GEOMETRY;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdWedge::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdWedge::vertexCoords ()
{
  return StdPrism::getCoords (
    StdLinearTriangle::vertexCoords (),
    StdLinearLine    ::vertexCoords ()
  );
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdWedge::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoints
//-----------------------------------------------------------------------


bool StdWedge::containsPoint ( const Vector& u ) const
{
  return ( u[0] >=  0.0 && u[1] >= 0.0 && u[0] + u[1] <= 1.0 &&
           u[2] >= -1.0 && u[2] <= 1.0 );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdWedge::clipPoint ( const Vector& u ) const
{
  bool  result = false;

  if ( u[0] < 0.0 )
  {
    u[0] = 0.0; result = true;
  }
  if ( u[1] < 0.0 )
  {
    u[1] = 0.0; result = true;
  }

  double a = u[0] + u[1];

  if ( a > 1.0 )
  {
    a      = 1.0 / a;
    u[0]  *= a;
    u[1]  *= a;
    result = true;
  }

  if ( u[2] < -1.0 )
  {
    u[2] = -1.0; result = true;
  }
  else if ( u[2] > 1.0 )
  {
    u[2] =  1.0; result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   getGaussScheme
//-----------------------------------------------------------------------


Matrix StdWedge::getGaussScheme ( idx_t n )
{
  idx_t  p[2];

  getIpCount_ ( p, n );

  return getGaussScheme ( p[0], p[1] );
}


Matrix StdWedge::getGaussScheme ( idx_t m, idx_t n )
{
  return StdPrism::getIntegrationScheme (
    StdTriangle::getGaussScheme ( m ),
    StdLine    ::getGaussScheme ( n )
  );
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix StdWedge::getIntegrationScheme ( String& s )
{
  String  s0, s1;

  ISchemes::split ( s0, s1, s );

  Matrix  ischeme =

    StdPrism::getIntegrationScheme (
      StdTriangle::getIntegrationScheme ( s0 ),
      StdLine    ::getIntegrationScheme ( s1 )
    );

  s = ISchemes::join ( s0, s1 );

  return ischeme;
}


//-----------------------------------------------------------------------
//   getIpCount_
//-----------------------------------------------------------------------


void StdWedge::getIpCount_ ( idx_t p[2], idx_t n )
{
  if      ( n <= 1 )
  {
    p[0] = 1;
    p[1] = 1;
  }
  else if ( n <= 6 )
  {
    p[0] = 3;
    p[1] = 2;
  }
  else if ( n <= 8 )
  {
    p[0] = 4;
    p[1] = 2;
  }
  else if ( n <= 21 )
  {
    p[0] = 7;
    p[1] = 3;
  }
  else if ( n <= 52 )
  {
    p[0] = 13;
    p[1] =  4;
  }
  else
  {
    p[0] = 19;
    p[1] =  5;
  }
}


//=======================================================================
//   class StdWedge6
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdWedge6::StdWedge6 ()
{}


StdWedge6::~StdWedge6 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdWedge6::readFrom ( ObjectInput& in )
{}


void StdWedge6::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdWedge6::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdWedge6::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_WEDGE6_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdWedge6::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_WEDGE6_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdWedge6::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_WEDGE6_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdWedge6::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdWedge6::declare ()
{
  StdShapeFactory::declare ( "Linear", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdWedge18
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdWedge18::StdWedge18 ()
{}


StdWedge18::~StdWedge18 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdWedge18::readFrom ( ObjectInput& in )
{}


void StdWedge18::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdWedge18::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdWedge18::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdWedge18::vertexCoords ()
{
  return StdPrism::getCoords (
    StdQuadraticTriangle::vertexCoords (),
    StdQuadraticLine    ::vertexCoords ()
  );
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdWedge18::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdWedge18::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_WEDGE18_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdWedge18::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_WEDGE18_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdWedge18::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_WEDGE18_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdWedge18::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdWedge18::declare ()
{
  StdShapeFactory::declare ( "BiQuadratic", GEOMETRY, & makeNew );
}


JIVE_END_PACKAGE( geom )
