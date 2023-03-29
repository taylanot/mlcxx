
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
#include <jive/geom/StdPrism.h>
#include <jive/geom/shfuncs/SquareFuncs.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/IntegrationSchemes.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdSquare.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare4 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare6 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare8 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare9 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare12 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare9B );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare16B );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare25B );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare3Bx2 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare4Bx2 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare4Bx3B );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare5Bx2 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare5Bx3B );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdSquare6Bx3B );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class StdSquare
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdSquare::GEOMETRY = Geometries::SQUARE;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


StdSquare::~StdSquare ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdSquare::rank () const
{
  return (idx_t) RANK;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdSquare::getGeometry () const
{
  return GEOMETRY;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdSquare::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  for ( int j = 0; j < VERTEX_COUNT; j++ )
  {
    for ( int i = 0; i < RANK; i++ )
    {
      u(i,j) = (double) Square4Funcs::VERTEX_COORDS[i][j];
    }
  }

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoints
//-----------------------------------------------------------------------


bool StdSquare::containsPoint ( const Vector& u ) const
{
  return ( u[0] >= -1.0 && u[0] <= 1.0 &&
           u[1] >= -1.0 && u[1] <= 1.0 );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdSquare::clipPoint ( const Vector& u ) const
{
  bool  result = false;

  for ( idx_t i = 0; i < 2; i++ )
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


Matrix StdSquare::getGaussScheme ( idx_t n )
{
  const idx_t  k = squareRoot_ ( n );

  return StdPrism::getIntegrationScheme (
    StdLine::getGaussScheme ( k ),
    StdLine::getGaussScheme ( k )
  );
}


Matrix StdSquare::getGaussScheme ( idx_t m, idx_t n )
{
  return StdPrism::getIntegrationScheme (
    StdLine::getGaussScheme ( m ),
    StdLine::getGaussScheme ( n )
  );
}


//-----------------------------------------------------------------------
//   getNewtonCotesScheme
//-----------------------------------------------------------------------


Matrix StdSquare::getNewtonCotesScheme ( idx_t n )
{
  const idx_t  k = squareRoot_ ( n );

  return StdPrism::getIntegrationScheme (
    StdLine::getNewtonCotesScheme ( k ),
    StdLine::getNewtonCotesScheme ( k )
  );
}


Matrix StdSquare::getNewtonCotesScheme ( idx_t m, idx_t n )
{
  return StdPrism::getIntegrationScheme (
    StdLine::getNewtonCotesScheme ( m ),
    StdLine::getNewtonCotesScheme ( n )
  );
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix StdSquare::getIntegrationScheme ( String& s )
{
  String  s0, s1;

  ISchemes::split ( s0, s1, s );

  Matrix  ischeme =

    StdPrism::getIntegrationScheme (
      StdLine::getIntegrationScheme ( s0 ),
      StdLine::getIntegrationScheme ( s1 )
    );

  s = ISchemes::join ( s0, s1 );

  return ischeme;
}


//-----------------------------------------------------------------------
//   squareRoot_
//-----------------------------------------------------------------------


idx_t StdSquare::squareRoot_ ( idx_t n )
{
  idx_t  k;

  for ( k = 1; k < 100; k++ )
  {
    if ( k * k >= n )
    {
      break;
    }
  }

  return k;
}


//=======================================================================
//   class StdSquare4
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare4::StdSquare4 ()
{}


StdSquare4::~StdSquare4 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare4::readFrom ( ObjectInput& in )
{}


void StdSquare4::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare4::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare4::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare4::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare4::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare4::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare4::declare ()
{
  StdShapeFactory::declare ( "Linear", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare6
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare6::StdSquare6 ()
{}


StdSquare6::~StdSquare6 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare6::readFrom ( ObjectInput& in )
{}


void StdSquare6::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdSquare6::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare6::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare6::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = -1.0; u(1,0) = -1.0;
  u(0,1) =  0.0; u(1,1) = -1.0;
  u(0,2) =  1.0; u(1,2) = -1.0;
  u(0,3) =  1.0; u(1,3) =  1.0;
  u(0,4) =  0.0; u(1,4) =  1.0;
  u(0,5) = -1.0; u(1,5) =  1.0;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare6::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare6::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE6_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare6::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE6_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare6::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE6_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare6::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare6::declare ()
{
  StdShapeFactory::declare ( "Quadratic*Linear",
                             GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare8
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare8::StdSquare8 ()
{}


StdSquare8::~StdSquare8 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare8::readFrom ( ObjectInput& in )
{}


void StdSquare8::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare8::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdSquare8::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare8::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  for ( int j = 0; j < VERTEX_COUNT; j++ )
  {
    for ( int i = 0; i < RANK; i++ )
    {
      u(i,j) = (double) Square8Funcs::VERTEX_COORDS[i][j];
    }
  }

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare8::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare8::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE8_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare8::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE8_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare8::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE8_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare8::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare8::declare ()
{
  StdShapeFactory::declare ( "Quadratic", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare9
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare9::StdSquare9 ()
{}


StdSquare9::~StdSquare9 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare9::readFrom ( ObjectInput& in )
{}


void StdSquare9::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare9::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdSquare9::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare9::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = -1.0; u(1,0) = -1.0;
  u(0,1) =  0.0; u(1,1) = -1.0;
  u(0,2) =  1.0; u(1,2) = -1.0;
  u(0,3) =  1.0; u(1,3) =  0.0;
  u(0,4) =  1.0; u(1,4) =  1.0;
  u(0,5) =  0.0; u(1,5) =  1.0;
  u(0,6) = -1.0; u(1,6) =  1.0;
  u(0,7) = -1.0; u(1,7) =  0.0;
  u(0,8) =  0.0; u(1,8) =  0.0;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare9::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare9::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE9_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare9::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE9_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare9::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE9_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare9::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare9::declare ()
{
  StdShapeFactory::declare ( "BiQuadratic", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare12
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare12::StdSquare12 ()
{}


StdSquare12::~StdSquare12 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare12::readFrom ( ObjectInput& in )
{}


void StdSquare12::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare12::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdSquare12::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare12::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  for ( int j = 0; j < VERTEX_COUNT; j++ )
  {
    for ( int i = 0; i < RANK; i++ )
    {
      u(i,j) = (double) Square12Funcs::VERTEX_COORDS[i][j];
    }
  }

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdSquare12::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare12::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE12_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare12::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE12_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare12::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE12_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare12::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare12::declare ()
{
  StdShapeFactory::declare ( "Cubic", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare9B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare9B::StdSquare9B ()
{}


StdSquare9B::~StdSquare9B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare9B::readFrom ( ObjectInput& in )
{}


void StdSquare9B::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare9B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare9B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE9B_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare9B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE9B_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare9B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE9B_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare9B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare9B::declare ()
{
  StdShapeFactory::declare ( "Bernstein9", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare16B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare16B::StdSquare16B ()
{}


StdSquare16B::~StdSquare16B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare16B::readFrom ( ObjectInput& in )
{}


void StdSquare16B::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare16B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare16B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE16B_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare16B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE16B_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare16B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE16B_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare16B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare16B::declare ()
{
  StdShapeFactory::declare ( "Bernstein16", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare25B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare25B::StdSquare25B ()
{}


StdSquare25B::~StdSquare25B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare25B::readFrom ( ObjectInput& in )
{}


void StdSquare25B::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare25B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare25B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE25B_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare25B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE25B_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare25B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE25B_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare25B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare25B::declare ()
{
  StdShapeFactory::declare ( "Bernstein25", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare3Bx2
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare3Bx2::StdSquare3Bx2 ()
{}


StdSquare3Bx2::~StdSquare3Bx2 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare3Bx2::readFrom ( ObjectInput& in )
{}


void StdSquare3Bx2::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare3Bx2::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare3Bx2::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE3BX2_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare3Bx2::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE3BX2_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare3Bx2::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE3BX2_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare3Bx2::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare3Bx2::declare ()
{
  StdShapeFactory::declare ( "B3x2", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare4Bx2
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare4Bx2::StdSquare4Bx2 ()
{}


StdSquare4Bx2::~StdSquare4Bx2 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare4Bx2::readFrom ( ObjectInput& in )
{}


void StdSquare4Bx2::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare4Bx2::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare4Bx2::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4BX2_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare4Bx2::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4BX2_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare4Bx2::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4BX2_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare4Bx2::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare4Bx2::declare ()
{
  StdShapeFactory::declare ( "B4x2", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare4Bx3B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare4Bx3B::StdSquare4Bx3B ()
{}


StdSquare4Bx3B::~StdSquare4Bx3B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare4Bx3B::readFrom ( ObjectInput& in )
{}


void StdSquare4Bx3B::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare4Bx3B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare4Bx3B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4BX3B_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare4Bx3B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4BX3B_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare4Bx3B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE4BX3B_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare4Bx3B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare4Bx3B::declare ()
{
  StdShapeFactory::declare ( "B4xB3", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare5Bx2
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare5Bx2::StdSquare5Bx2 ()
{}


StdSquare5Bx2::~StdSquare5Bx2 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare5Bx2::readFrom ( ObjectInput& in )
{}


void StdSquare5Bx2::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare5Bx2::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare5Bx2::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE5BX2_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare5Bx2::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE5BX2_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare5Bx2::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE5BX2_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare5Bx2::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare5Bx2::declare ()
{
  StdShapeFactory::declare ( "B5x2", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare5Bx3B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare5Bx3B::StdSquare5Bx3B ()
{}


StdSquare5Bx3B::~StdSquare5Bx3B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare5Bx3B::readFrom ( ObjectInput& in )
{}


void StdSquare5Bx3B::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare5Bx3B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare5Bx3B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE5BX3B_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare5Bx3B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE5BX3B_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare5Bx3B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE5BX3B_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare5Bx3B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare5Bx3B::declare ()
{
  StdShapeFactory::declare ( "B5xB3", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdSquare6Bx3B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdSquare6Bx3B::StdSquare6Bx3B ()
{}


StdSquare6Bx3B::~StdSquare6Bx3B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdSquare6Bx3B::readFrom ( ObjectInput& in )
{}


void StdSquare6Bx3B::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdSquare6Bx3B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdSquare6Bx3B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE6BX3B_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdSquare6Bx3B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE6BX3B_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdSquare6Bx3B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_SQUARE6BX3B_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdSquare6Bx3B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdSquare6Bx3B::declare ()
{
  StdShapeFactory::declare ( "B6xB3", GEOMETRY, & makeNew );
}


JIVE_END_PACKAGE( geom )
