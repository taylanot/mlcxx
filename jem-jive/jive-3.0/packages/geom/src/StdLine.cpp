
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


#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/numeric/Quadrature.h>
#include <jive/geom/shfuncs/LineFuncs.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/IntegrationSchemes.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdLine.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdLine2   );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdLine2C1 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdLine3   );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdLine4   );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdLine3B  );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdLine4B  );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdLine5B  );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdLine6B  );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jem::Error;


//=======================================================================
//   class StdLine
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdLine::GEOMETRY = Geometries::LINE;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


StdLine::~StdLine ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdLine::rank () const
{
  return (idx_t) RANK;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdLine::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdLine::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = -1.0;
  u(0,1) =  1.0;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdLine::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdLine::getGeometry () const
{
  return GEOMETRY;
}


//-----------------------------------------------------------------------
//   containsPoint
//-----------------------------------------------------------------------


bool StdLine::containsPoint ( const Vector& u ) const
{
  return ( u[0] >= -1.0 && u[0] <= 1.0 );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdLine::clipPoint ( const Vector& u ) const
{
  if ( u[0] < -1.0 )
  {
    u[0] = -1.0; return true;
  }
  else if ( u[0] > 1.0 )
  {
    u[0] =  1.0; return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getGaussScheme
//-----------------------------------------------------------------------


Matrix StdLine::getGaussScheme ( idx_t count )
{
  using jem::numeric::Quadrature;

  const idx_t  n = getGaussPointCount ( count );

  Matrix       s ( RANK + 1, n );

  Vector       w ( s(0,ALL) );
  Vector       u ( s(1,ALL) );


  Quadrature::gaussLegendre ( u, w );

  return s;
}


//-----------------------------------------------------------------------
//   getUniformScheme
//-----------------------------------------------------------------------


Matrix StdLine::getUniformScheme ( idx_t count )
{
  const idx_t   n = getUniformPointCount ( count );

  const double  h = 2.0 / (double) n;

  Matrix        s ( RANK + 1, n );

  Vector        w ( s(0,ALL) );
  Vector        u ( s(1,ALL) );


  for ( idx_t i = 0; i < n; i++ )
  {
    u[i] = (0.5 + (double) i) * h - 1.0;
    w[i] = h;
  }

  return s;
}


//-----------------------------------------------------------------------
//   getTrapezoidalScheme
//-----------------------------------------------------------------------


Matrix StdLine::getTrapezoidalScheme ( idx_t count )
{
  const idx_t   n = getTrapezoidalPointCount ( count );

  const idx_t   k = n - 1;
  const double  h = 2.0 / (double) k;

  Matrix        s ( RANK + 1, n );

  Vector        w ( s(0,ALL) );
  Vector        u ( s(1,ALL) );


  w[0] =  0.5 * h;
  u[0] = -1.0;

  for ( idx_t i = 1; i < k; i++ )
  {
    w[i] = h;
    u[i] = (double) i * h - 1.0;
  }

  w[k] = 0.5 * h;
  u[k] = 1.0;

  return s;
}


//-----------------------------------------------------------------------
//   getNewtonCotesScheme
//-----------------------------------------------------------------------


Matrix StdLine::getNewtonCotesScheme ( idx_t count )
{
  const idx_t   n = getNewtonCotesPointCount ( count );

  const double  h = 2.0 / (double) (n - 1);

  Matrix        s ( RANK + 1, n );

  Vector        w ( s(0,ALL) );
  Vector        u ( s(1,ALL) );


  if      ( n == 2_idx )
  {
    w[0] = w[1] = 1.0;
  }
  else if ( n == 3_idx )
  {
    w[0] = 1.0 / 3.0;
    w[1] = 4.0 / 3.0;
    w[2] = w[0];
  }
  else if ( n == 4_idx )
  {
    w[0] = 0.25;
    w[1] = 0.75;
    w[2] = w[1];
    w[3] = w[0];
  }
  else if ( n == 5_idx )
  {
    w[0] =  7.0 / 45.0;
    w[1] = 32.0 / 45.0;
    w[2] = 12.0 / 45.0;
    w[3] = w[1];
    w[4] = w[0];
  }
  else if ( n == 6_idx )
  {
    w[0] = 19.0 / 144.0;
    w[1] = 75.0 / 144.0;
    w[2] = 50.0 / 144.0;
    w[3] = w[2];
    w[4] = w[1];
    w[5] = w[0];
  }
  else
  {
    throw Error (
      JEM_FUNC,
      "unexpected number of Newton-Cotes points: " + String( n )
    );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    u[i] = (double) i * h - 1.0;
  }


  return s;
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix StdLine::getIntegrationScheme ( String& s )
{
  Matrix  (*func) ( idx_t );
  Matrix    ischeme;

  String    name;

  idx_t     n;


  ISchemes::parse ( name, n, s );

  func = 0;

  if      ( name == ""   ||
            name == "G"  ||
            name == ISchemes::GAUSS )
  {
    func = getGaussScheme;
  }
  else if ( name == "U"  ||
            name == ISchemes::UNIFORM )
  {
    func = getUniformScheme;
  }
  else if ( name == "T"  ||
            name == ISchemes::TRAPEZOIDAL )
  {
    func = getTrapezoidalScheme;
  }
  else if ( name == "NC" ||
            name == ISchemes::NEWTON_COTES )
  {
    func = getNewtonCotesScheme;
  }
  else
  {
    throw jem::IllegalInputException (
      JEM_FUNC,
      String::format (

        "invalid line integration scheme: `%s\'; "
        "must be `#\', `G#\', `%s#\', `U#\', `%s#\', "
        "`T#\', `%s#\', `NC#\' or `%s#\', "
        "where # = number of integration points",

        s,
        ISchemes::GAUSS,
        ISchemes::UNIFORM,
        ISchemes::TRAPEZOIDAL,
        ISchemes::NEWTON_COTES
      )
    );
  }

  ischeme.ref ( func( n ) );

  s = name + String ( ischeme.size(1) );

  return ischeme;
}


//-----------------------------------------------------------------------
//   getGaussPointCount
//-----------------------------------------------------------------------


idx_t StdLine::getGaussPointCount ( idx_t n )
{
  if      ( n < 1 )
  {
    return 1;
  }
  else if ( n > 100000_idx )
  {
    return 100000_idx;
  }
  else
  {
    return n;
  }
}


//-----------------------------------------------------------------------
//   getUniformPointCount
//-----------------------------------------------------------------------


idx_t StdLine::getUniformPointCount ( idx_t n )
{
  if ( n < 1 )
  {
    return 1;
  }
  else
  {
    return n;
  }
}


//-----------------------------------------------------------------------
//   getTrapezoidalPointCount
//-----------------------------------------------------------------------


idx_t StdLine::getTrapezoidalPointCount ( idx_t n )
{
  if ( n < 2 )
  {
    return 2;
  }
  else
  {
    return n;
  }
}


//-----------------------------------------------------------------------
//   getNewtonCotesPointCount
//-----------------------------------------------------------------------


idx_t StdLine::getNewtonCotesPointCount ( idx_t n )
{
  if      ( n < 2 )
  {
    return 2;
  }
  else if ( n > 6 )
  {
    return 6;
  }
  else
  {
    return n;
  }
}


//=======================================================================
//   class StdLine2
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdLine2::StdLine2 ()
{}


StdLine2::~StdLine2 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdLine2::readFrom ( ObjectInput& )
{}


void StdLine2::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdLine2::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   isLinear
//-----------------------------------------------------------------------


bool StdLine2::isLinear () const
{
  return true;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdLine2::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE2_FUNCS( f, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdLine2::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE2_GRADS( f, g, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdLine2::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE2_GRADS2( f, g, h, x );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdLine2::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdLine2::declare ()
{
  StdShapeFactory::declare ( "Linear", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdLine2C1
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdLine2C1::StdLine2C1 ()
{}


StdLine2C1::~StdLine2C1 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdLine2C1::readFrom ( ObjectInput& )
{}


void StdLine2C1::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdLine2C1::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdLine2C1::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE2C1_FUNCS( f, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdLine2C1::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE2C1_GRADS( f, g, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdLine2C1::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE2C1_GRADS2( f, g, h, x );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdLine2C1::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdLine2C1::declare ()
{
  StdShapeFactory::declare ( "HermiteC1", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdLine3
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdLine3::StdLine3 ()
{}


StdLine3::~StdLine3 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdLine3::readFrom ( ObjectInput& )
{}


void StdLine3::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdLine3::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdLine3::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdLine3::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = -1.0;
  u(0,1) =  0.0;
  u(0,2) =  1.0;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdLine3::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdLine3::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE3_FUNCS( f, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdLine3::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE3_GRADS( f, g, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdLine3::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE3_GRADS2( f, g, h, x );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdLine3::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdLine3::declare ()
{
  StdShapeFactory::declare ( "Quadratic", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdLine4
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdLine4::StdLine4 ()
{}


StdLine4::~StdLine4 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdLine4::readFrom ( ObjectInput& )
{}


void StdLine4::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdLine4::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdLine4::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdLine4::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = -1.0;
  u(0,1) = -1.0 / 3.0;
  u(0,2) =  1.0 / 3.0;
  u(0,3) =  1.0;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdLine4::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdLine4::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE4_FUNCS( f, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdLine4::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE4_GRADS( f, g, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdLine4::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE4_GRADS2( f, g, h, x );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdLine4::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdLine4::declare ()
{
  StdShapeFactory::declare ( "Cubic", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdLine3B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdLine3B::StdLine3B ()
{}


StdLine3B::~StdLine3B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdLine3B::readFrom ( ObjectInput& )
{}


void StdLine3B::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdLine3B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdLine3B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE3B_FUNCS( f, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdLine3B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE3B_GRADS( f, g, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdLine3B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE3B_GRADS2( f, g, h, x );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdLine3B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdLine3B::declare ()
{
  StdShapeFactory::declare ( "Bernstein3", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdLine4B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdLine4B::StdLine4B ()
{}


StdLine4B::~StdLine4B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdLine4B::readFrom ( ObjectInput& )
{}


void StdLine4B::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdLine4B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdLine4B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE4B_FUNCS( f, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdLine4B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE4B_GRADS( f, g, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdLine4B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE4B_GRADS2( f, g, h, x );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdLine4B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdLine4B::declare ()
{
  StdShapeFactory::declare ( "Bernstein4", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdLine5B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdLine5B::StdLine5B ()
{}


StdLine5B::~StdLine5B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdLine5B::readFrom ( ObjectInput& )
{}


void StdLine5B::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdLine5B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdLine5B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE5B_FUNCS( f, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdLine5B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE5B_GRADS( f, g, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdLine5B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE5B_GRADS2( f, g, h, x );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdLine5B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdLine5B::declare ()
{
  StdShapeFactory::declare ( "Bernstein5", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdLine6B
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdLine6B::StdLine6B ()
{}


StdLine6B::~StdLine6B ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdLine6B::readFrom ( ObjectInput& )
{}


void StdLine6B::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdLine6B::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdLine6B::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE6B_FUNCS( f, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdLine6B::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE6B_GRADS( f, g, x );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdLine6B::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const double  x = u[0];

  JIVE_EVAL_LINE6B_GRADS2( f, g, h, x );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdLine6B::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdLine6B::declare ()
{
  StdShapeFactory::declare ( "Bernstein6", GEOMETRY, & makeNew );
}


JIVE_END_PACKAGE( geom )
