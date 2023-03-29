
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
#include <jive/geom/shfuncs/TetFuncs.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/IntegrationSchemes.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdTetrahedron.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdTetrahedron4  );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdTetrahedron10 );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class StdTetrahedron
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdTetrahedron::GEOMETRY = Geometries::TETRAHEDRON;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


StdTetrahedron::~StdTetrahedron ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdTetrahedron::rank () const
{
  return (idx_t) RANK;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdTetrahedron::getGeometry () const
{
  return GEOMETRY;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdTetrahedron::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdTetrahedron::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = 0.0; u(1,0) = 0.0; u(2,0) = 0.0;
  u(0,1) = 1.0; u(1,1) = 0.0; u(2,1) = 0.0;
  u(0,2) = 0.0; u(1,2) = 1.0; u(2,2) = 0.0;
  u(0,3) = 0.0; u(1,3) = 0.0; u(2,3) = 1.0;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdTetrahedron::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoints
//-----------------------------------------------------------------------


bool StdTetrahedron::containsPoint ( const Vector& u ) const
{
  return ( u[0] >= 0.0 &&
           u[1] >= 0.0 &&
           u[2] >= 0.0 &&
           u[0] + u[1] + u[2] <= 1.0 );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdTetrahedron::clipPoint ( const Vector& u ) const
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
  if ( u[2] < 0.0 )
  {
    u[2] = 0.0; result = true;
  }

  double a = u[0] + u[1] + u[2];

  if ( a > 1.0 )
  {
    a      = 1.0 / a;
    u[0]  *= a;
    u[1]  *= a;
    u[2]  *= a;
    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   getGaussScheme
//-----------------------------------------------------------------------


Matrix StdTetrahedron::getGaussScheme ( idx_t count )
{
  const idx_t  n = getGaussPointCount ( count );

  Matrix       s ( RANK + 1, n );

  Vector       w ( s(0,ALL) );
  Matrix       u ( s(slice(1,END),ALL) );

  double       a, b, c, d, e, f;


  if      ( n == 1_idx )
  {
    w[0]   = 1.0 / 6.0;

    u(0,0) = u(1,0) = u(2,0) = 0.25;
  }
  else if ( n == 4_idx )
  {
    w[0]   = w[1] = w[2] = w[3] = 0.25 / 6.0;

    a      = 0.5854101966249685;
    b      = 0.1381966011250105;

    u(0,0) = b;
    u(1,0) = b;
    u(2,0) = b;

    u(0,1) = a;
    u(1,1) = b;
    u(2,1) = b;

    u(0,2) = b;
    u(1,2) = a;
    u(2,2) = b;

    u(0,3) = b;
    u(1,3) = b;
    u(2,3) = a;
  }
  else if ( n == 5_idx )
  {
    w[0]   = -4.0 / 30.0;
    w[1]   = w[2] = w[3]
           = w[4] = 9.0 / 120.0;

    a      = 0.25;
    b      = 0.50;
    c      = 0.1666666666666667;

    u(0,0) = a;
    u(1,0) = a;
    u(2,0) = a;

    u(0,1) = b;
    u(1,1) = c;
    u(2,1) = c;

    u(0,2) = c;
    u(1,2) = b;
    u(2,2) = c;

    u(0,3) = c;
    u(1,3) = c;
    u(2,3) = b;

    u(0,4) = c;
    u(1,4) = c;
    u(2,4) = c;
  }
  else if ( n == 11_idx )
  {
    w[0]    = -74.0 / 5625.0;
    w[1]    = w[2] = w[3]  = w[4] = 343.0 / 45000.0;
    w[5]    = w[6] = w[7]  = w[8] =
              w[9] = w[10] = 56.0 / 2250.0;

    a       = 0.25;
    b       = 11.0 / 14.0;
    c       =  1.0 / 14.0;
    d       = 0.3994035761667992;
    e       = 0.1005964238332008;

    u(0,0)  = a;
    u(1,0)  = a;
    u(2,0)  = a;

    u(0,1)  = b;
    u(1,1)  = c;
    u(2,1)  = c;

    u(0,2)  = c;
    u(1,2)  = b;
    u(2,2)  = c;

    u(0,3)  = c;
    u(1,3)  = c;
    u(2,3)  = b;

    u(0,4)  = c;
    u(1,4)  = c;
    u(2,4)  = c;

    u(0,5)  = d;
    u(1,5)  = d;
    u(2,5)  = e;

    u(0,6)  = d;
    u(1,6)  = e;
    u(2,6)  = d;

    u(0,7)  = d;
    u(1,7)  = e;
    u(2,7)  = e;

    u(0,8)  = e;
    u(1,8)  = d;
    u(2,8)  = d;

    u(0,9)  = e;
    u(1,9)  = d;
    u(2,9)  = e;

    u(0,10) = e;
    u(1,10) = e;
    u(2,10) = d;
  }
  else if ( n == 14_idx )
  {
    w[0]    = w[1]  = w[2]  = w[3]  = 0.1126879257180162 / 6.0;
    w[4]    = w[5]  = w[6]  = w[7]  = 0.0734930431163619 / 6.0;
    w[8]    = w[9]  = w[10] =
              w[11] = w[12] = w[13] = 0.0425460207770812 / 6.0;

    a       = 0.0673422422100983;
    b       = 0.3108859192633005;
    c       = 0.7217942490673264;
    d       = 0.0927352503108912;
    e       = 0.4544962958743506;
    f       = 0.0455037041256494;

    u(0,0)  = a;
    u(1,0)  = b;
    u(2,0)  = b;

    u(0,1)  = b;
    u(1,1)  = a;
    u(2,1)  = b;

    u(0,2)  = b;
    u(1,2)  = b;
    u(2,2)  = a;

    u(0,3)  = b;
    u(1,3)  = b;
    u(2,3)  = b;

    u(0,4)  = c;
    u(1,4)  = d;
    u(2,4)  = d;

    u(0,5)  = d;
    u(1,5)  = c;
    u(2,5)  = d;

    u(0,6)  = d;
    u(1,6)  = d;
    u(2,6)  = c;

    u(0,7)  = d;
    u(1,7)  = d;
    u(2,7)  = d;

    u(0,8)  = e;
    u(1,8)  = e;
    u(2,8)  = f;

    u(0,9)  = e;
    u(1,9)  = f;
    u(2,9)  = e;

    u(0,10) = e;
    u(1,10) = f;
    u(2,10) = f;

    u(0,11) = f;
    u(1,11) = e;
    u(2,11) = e;

    u(0,12) = f;
    u(1,12) = e;
    u(2,12) = f;

    u(0,13) = f;
    u(1,13) = f;
    u(2,13) = e;
  }
  else
  {
    throw jem::Error (
      JEM_FUNC,
      "unexpected number of Gauss points: " + String( n )
    );
  }

  return s;
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix StdTetrahedron::getIntegrationScheme ( String& s )
{
  Matrix  (*func) ( idx_t );

  Matrix    ischeme;
  String    name;
  idx_t     n;


  ISchemes::parse ( name, n, s );

  func = 0;

  if ( name == ""  ||
       name == "G" ||
       name == ISchemes::GAUSS )
  {
    func = getGaussScheme;
  }
  else
  {
    throw jem::IllegalInputException (
      JEM_FUNC,
      String::format (

        "invalid tetrahedron integration scheme: `%s\'; "
        "must be `#\', `G#\' or `%s#\', "
        "where # = number of integration points",

        s, ISchemes::GAUSS
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


idx_t StdTetrahedron::getGaussPointCount ( idx_t n )
{
  if      ( n <= 1 )
  {
    return 1;
  }
  else if ( n <= 4 )
  {
    return 4;
  }
  else if ( n == 5 )
  {
    return 5;
  }
  else if ( n <= 11 )
  {
    return 11;
  }
  else
  {
    return 14;
  }
}


//=======================================================================
//   class StdTetrahedron4
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdTetrahedron4::StdTetrahedron4 ()
{}


StdTetrahedron4::~StdTetrahedron4 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdTetrahedron4::readFrom ( ObjectInput& in )
{}


void StdTetrahedron4::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdTetrahedron4::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   isLinear
//-----------------------------------------------------------------------


bool StdTetrahedron4::isLinear () const
{
  return true;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdTetrahedron4::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_TET4_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdTetrahedron4::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_TET4_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdTetrahedron4::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_TET4_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdTetrahedron4::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdTetrahedron4::declare ()
{
  StdShapeFactory::declare ( "Linear", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdTetrahedron10
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdTetrahedron10::StdTetrahedron10 ()
{}


StdTetrahedron10::~StdTetrahedron10 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdTetrahedron10::readFrom ( ObjectInput& in )
{}


void StdTetrahedron10::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdTetrahedron10::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdTetrahedron10::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdTetrahedron10::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = 0.0; u(1,0) = 0.0; u(2,0) = 0.0;
  u(0,1) = 0.5; u(1,1) = 0.0; u(2,1) = 0.0;
  u(0,2) = 1.0; u(1,2) = 0.0; u(2,2) = 0.0;
  u(0,3) = 0.5; u(1,3) = 0.5; u(2,3) = 0.0;
  u(0,4) = 0.0; u(1,4) = 1.0; u(2,4) = 0.0;
  u(0,5) = 0.0; u(1,5) = 0.5; u(2,5) = 0.0;
  u(0,6) = 0.0; u(1,6) = 0.0; u(2,6) = 0.5;
  u(0,7) = 0.5; u(1,7) = 0.0; u(2,7) = 0.5;
  u(0,8) = 0.0; u(1,8) = 0.5; u(2,8) = 0.5;
  u(0,9) = 0.0; u(1,9) = 0.0; u(2,9) = 1.0;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdTetrahedron10::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdTetrahedron10::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_TET10_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdTetrahedron10::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_TET10_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdTetrahedron10::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_TET10_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdTetrahedron10::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdTetrahedron10::declare ()
{
  StdShapeFactory::declare ( "Quadratic", GEOMETRY, & makeNew );
}


JIVE_END_PACKAGE( geom )
