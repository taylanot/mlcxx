
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
#include <jive/geom/StdSquare.h>
#include <jive/geom/StdPrism.h>
#include <jive/geom/shfuncs/CubeFuncs.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/IntegrationSchemes.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdCube.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdCube8      );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdCube16     );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdCube20     );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdCube27B    );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdCube3Bx2x2 );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class StdCube
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdCube::GEOMETRY = Geometries::CUBE;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


StdCube::~StdCube ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdCube::rank () const
{
  return (idx_t) RANK;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdCube::getGeometry () const
{
  return GEOMETRY;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdCube::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdCube::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  for ( int j = 0; j < VERTEX_COUNT; j++ )
  {
    for ( int i = 0; i < RANK; i++ )
    {
      u(i,j) = (double) Cube8Funcs::VERTEX_COORDS[i][j];
    }
  }

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdCube::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoints
//-----------------------------------------------------------------------


bool StdCube::containsPoint ( const Vector& u ) const
{
  return ( u[0] >= -1.0 && u[0] <= 1.0 &&
           u[1] >= -1.0 && u[1] <= 1.0 &&
           u[2] >= -1.0 && u[2] <= 1.0 );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdCube::clipPoint ( const Vector& u ) const
{
  bool  result = false;

  for ( int i = 0; i < RANK; i++ )
  {
    if ( u[i] < -1.0 )
    {
      u[i] = -1.0; result = true;
    }
    else if ( u[i] > 1.0 )
    {
      u[i] =  1.0; result = true;
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   getGaussScheme
//-----------------------------------------------------------------------


Matrix StdCube::getGaussScheme ( idx_t n )
{
  idx_t  k = cubicRoot_ ( n );

  return getGaussScheme ( k, k, k );
}


Matrix StdCube::getGaussScheme ( idx_t m, idx_t n, idx_t p )
{
  return StdPrism::getIntegrationScheme (
    StdPrism::getIntegrationScheme (
      StdLine::getGaussScheme ( m ),
      StdLine::getGaussScheme ( n )
    ),
    StdLine::getGaussScheme ( p )
  );
}


//-----------------------------------------------------------------------
//   getNewtonCotesScheme
//-----------------------------------------------------------------------


Matrix StdCube::getNewtonCotesScheme ( idx_t n )
{
  idx_t  k = cubicRoot_ ( n );

  return getNewtonCotesScheme ( k, k, k );
}


Matrix StdCube::getNewtonCotesScheme ( idx_t m, idx_t n, idx_t p )
{
  return StdPrism::getIntegrationScheme (
    StdPrism::getIntegrationScheme (
      StdLine::getNewtonCotesScheme ( m ),
      StdLine::getNewtonCotesScheme ( n )
    ),
    StdLine::getNewtonCotesScheme ( p )
  );
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix StdCube::getIntegrationScheme ( String& s )
{
  String  s0, s1, s2;

  ISchemes::split ( s0, s1, s2, s );

  Matrix  ischeme =

    StdPrism::getIntegrationScheme (
      StdPrism::getIntegrationScheme (
        StdLine::getIntegrationScheme ( s0 ),
        StdLine::getIntegrationScheme ( s1 )
      ),
      StdLine::getIntegrationScheme ( s2 )
    );

  s = ISchemes::join ( s0, s1, s2 );

  return ischeme;
}


//-----------------------------------------------------------------------
//   cubicRoot_
//-----------------------------------------------------------------------


idx_t StdCube::cubicRoot_ ( idx_t n )
{
  idx_t  k;

  for ( k = 1; k < 100; k++ )
  {
    if ( k * k * k >= n )
    {
      break;
    }
  }

  return k;
}


//=======================================================================
//   class StdCube8
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdCube8::StdCube8 ()
{}


StdCube8::~StdCube8 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdCube8::readFrom ( ObjectInput& in )
{}


void StdCube8::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdCube8::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdCube8::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE8_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdCube8::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE8_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdCube8::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE8_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<SShape> StdCube8::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdCube8::declare ()
{
  StdShapeFactory::declare ( "Linear", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdCube16
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdCube16::StdCube16 ()
{}


StdCube16::~StdCube16 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdCube16::readFrom ( ObjectInput& in )
{}


void StdCube16::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdCube16::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdCube16::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdCube16::vertexCoords ()
{
  return StdPrism::getCoords ( StdSquare8::vertexCoords (),
                               StdLine2  ::vertexCoords () );
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdCube16::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdCube16::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE16_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdCube16::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE16_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdCube16::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE16_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<SShape> StdCube16::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdCube16::declare ()
{
  StdShapeFactory::declare ( "Quadratic*Linear",
                             GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdCube20
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdCube20::StdCube20 ()
{}


StdCube20::~StdCube20 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdCube20::readFrom ( ObjectInput& in )
{}


void StdCube20::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdCube20::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdCube20::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdCube20::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  for ( int j = 0; j < VERTEX_COUNT; j++ )
  {
    for ( int i = 0; i < RANK; i++ )
    {
      u(i,j) = (double) Cube20Funcs::VERTEX_COORDS[i][j];
    }
  }

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdCube20::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdCube20::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE20_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdCube20::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE20_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdCube20::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE20_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<SShape> StdCube20::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdCube20::declare ()
{
  StdShapeFactory::declare ( "Quadratic", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdCube27B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdCube27B::StdCube27B ()
{}


StdCube27B::~StdCube27B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdCube27B::readFrom ( ObjectInput& in )
{}


void StdCube27B::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdCube27B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdCube27B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE27B_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdCube27B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE27B_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdCube27B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE27B_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<SShape> StdCube27B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdCube27B::declare ()
{
  StdShapeFactory::declare ( "Bernstein27", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdCube3Bx2x2
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdCube3Bx2x2::StdCube3Bx2x2 ()
{}


StdCube3Bx2x2::~StdCube3Bx2x2 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdCube3Bx2x2::readFrom ( ObjectInput& in )
{}


void StdCube3Bx2x2::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdCube3Bx2x2::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdCube3Bx2x2::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE3BX2X2_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdCube3Bx2x2::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE3BX2X2_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdCube3Bx2x2::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_CUBE3BX2X2_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<SShape> StdCube3Bx2x2::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdCube3Bx2x2::declare ()
{
  StdShapeFactory::declare ( "B3x2x2", GEOMETRY, & makeNew );
}


JIVE_END_PACKAGE( geom )
