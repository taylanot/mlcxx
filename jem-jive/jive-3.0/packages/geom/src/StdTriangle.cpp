
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
#include <jem/base/array/operators.h>
#include <jive/geom/shfuncs/TriangFuncs.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/IntegrationSchemes.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdTriangle.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdTriangle3 );
JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdTriangle6 );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class StdTriangle
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdTriangle::GEOMETRY = Geometries::TRIANGLE;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


StdTriangle::~StdTriangle ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdTriangle::rank () const
{
  return (idx_t) RANK;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdTriangle::getGeometry () const
{
  return GEOMETRY;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdTriangle::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdTriangle::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = 0.0; u(1,0) = 0.0;
  u(0,1) = 1.0; u(1,1) = 0.0;
  u(0,2) = 0.0; u(1,2) = 1.0;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdTriangle::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoints
//-----------------------------------------------------------------------


bool StdTriangle::containsPoint ( const Vector& u ) const
{
  return ( u[0] >= 0.0 && u[1] >= 0.0 && u[0] + u[1] <= 1.0 );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdTriangle::clipPoint ( const Vector& u ) const
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

  return result;
}


//-----------------------------------------------------------------------
//   getGaussScheme
//-----------------------------------------------------------------------


Matrix StdTriangle::getGaussScheme ( idx_t count )
{
  if ( count > 19 )
  {
    idx_t  n;

    for ( n = 1; n <= 1000; n++ )
    {
      if ( n * n >= count )
      {
        break;
      }
    }

    Matrix  s = StdLine::getGaussScheme ( n );

    return getIntegrationScheme ( s, s );
  }
  else
  {
    const idx_t   n = getGaussPointCount ( count );

    const double  q = 0.666666666667;
    const double  r = 0.166666666667;

    Matrix        s ( RANK + 1, n );

    Vector        w ( s(0,ALL) );
    Matrix        u ( s(slice(1,END),ALL) );

    if      ( n == 1_idx )
    {
      w[0]   = 0.5;

      u(0,0) = u(1,0) = 0.333333333333333;
    }
    else if ( n == 3_idx )
    {
      w[0]   = w[1] = w[2] = 0.166666666666667;

      u(0,0) = r;
      u(1,0) = r;

      u(0,1) = q;
      u(1,1) = r;

      u(0,2) = r;
      u(1,2) = q;
    }
    else if ( n == 4_idx )
    {
      w[0]   =               -27.0 / 96.0;
      w[1]   = w[2] = w[3] =  25.0 / 96.0;

      u(0,0) = u(1,0) = 0.333333333333333;

      u(0,1) = 0.2;
      u(1,1) = 0.2;

      u(0,2) = 0.6;
      u(1,2) = 0.2;

      u(0,3) = 0.2;
      u(1,3) = 0.6;
    }
    else if ( n == 7_idx )
    {
      w[0]   = w[1] = w[2] = 0.0629695902724;
      w[3]   = w[4] = w[5] = 0.06619707639425;
      w[6]   =               0.1125;

      u(0,0) = 0.1012865073235;
      u(1,0) = u(0,0);

      u(0,1) = 0.7974269853531;
      u(1,1) = u(0,0);

      u(0,2) = u(0,0);
      u(1,2) = u(0,1);

      u(0,3) = 0.4701420641051;
      u(1,3) = 0.0597158717898;

      u(0,4) = u(0,3);
      u(1,4) = u(0,3);

      u(0,5) = u(1,3);
      u(1,5) = u(0,3);

      u(0,6) = 0.3333333333333;
      u(1,6) = u(0,6);
    }
    else if ( n == 13_idx )
    {
      w[0]    = w[1]  = w[2]  =         0.0266736178044;
      w[3]    = w[4]  = w[5]  =
                w[6]  = w[7]  = w[8]  = 0.0385568804452;
      w[9]    = w[10] = w[11] =         0.0878076287166;
      w[12]   =                        -0.0747850222339;

      u(0,0)  = 0.0651301029022;
      u(1,0)  = u(0,0);

      u(0,1)  = 0.8697397941956;
      u(1,1)  = u(0,0);

      u(0,2)  = u(0,0);
      u(1,2)  = u(0,1);

      u(0,3)  = 0.3128654960049;
      u(1,3)  = 0.0486903154253;

      u(0,4)  = 0.6384441885698;
      u(1,4)  = u(0,3);

      u(0,5)  = u(1,3);
      u(1,5)  = u(0,4);

      u(0,6)  = u(0,4);
      u(1,6)  = u(0,5);

      u(0,7)  = u(0,3);
      u(1,7)  = u(0,4);

      u(0,8)  = u(0,5);
      u(1,8)  = u(0,3);

      u(0,9)  = 0.2603459660790;
      u(1,9)  = u(0,9);

      u(0,10) = 0.4793080678419;
      u(1,10) = u(0,9);

      u(0,11) = u(0,9);
      u(1,11) = u(0,10);

      u(0,12) = 0.3333333333333;
      u(1,12) = u(0,12);
    }
    else if ( n == 19_idx )
    {
      w[0]    =                         0.097135796282796102;
      w[1]    = w[2]  = w[3]  =         0.031334700227139835;
      w[4]    = w[5]  = w[6]  =         0.07782754100477543;
      w[7]    = w[8]  = w[9]  =         0.079647738927209097;
      w[10]   = w[11] = w[12] =         0.025577675658698101;
      w[13]   = w[14] = w[15] =
                w[16] = w[17] = w[18] = 0.043283539377289404;

      w      *= 0.5;

      u(0,0)  = u(1,0) = 0.33333333333333331;

      u(0,1)  = 0.020634961602525929;
      u(1,1)  = 0.48968251919873701;

      u(0,2)  = u(1,2) = u(1,1);

      u(0,3)  = u(1,1);
      u(1,3)  = u(0,1);

      u(0,4)  = 0.125820817014129;
      u(1,4)  = 0.43708959149293553;

      u(0,5)  = u(1,5) = u(1,4);

      u(0,6)  = u(1,4);
      u(1,6)  = u(0,4);

      u(0,7)  = 0.62359292876193562;
      u(1,7)  = 0.18820353561903219;

      u(0,8)  = u(1,8) = u(1,7);

      u(0,9)  = u(1,7);
      u(1,9)  = u(0,7);

      u(0,10) = 0.91054097321109406;
      u(1,10) = 0.044729513394452969;

      u(0,11) = u(1,11) = u(1,10);

      u(0,12) = u(1,10);
      u(1,12) = u(0,10);

      u(0,13) = 0.036838412054736258;
      u(1,13) = 0.74119859878449801;

      u(0,14) = u(0,13);
      u(1,14) = 0.22196298916076571;

      u(0,15) = u(1,13);
      u(1,15) = u(1,14);

      u(0,16) = u(1,13);
      u(1,16) = u(0,13);

      u(0,17) = u(1,14);
      u(1,17) = u(0,13);

      u(0,18) = u(1,14);
      u(1,18) = u(1,13);
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
}


//-----------------------------------------------------------------------
//   getUniformScheme
//-----------------------------------------------------------------------


Matrix StdTriangle::getUniformScheme ( idx_t count )
{
  const idx_t  n = getUniformPointCount ( count );

  Matrix       s ( RANK + 1, n );
  Matrix       t ( RANK, 3 * n );
  Matrix       v ( RANK, 3 * n );

  double       a;

  idx_t        i, j, k;


  t(0,0) = 0.0;
  t(1,0) = 0.0;
  t(0,1) = 1.0;
  t(1,1) = 0.0;
  t(0,2) = 0.0;
  t(1,2) = 1.0;

  k = 3;

  while ( k < 3 * n )
  {
    for ( i = j = 0; i < k; i += 3, j += 6 )
    {
      for ( idx_t r = 0; r < 2; r++ )
      {
        a = 0.5 * (t(r,i + 1) + t(r,i + 2));

        v(r,j + 0) = a;
        v(r,j + 1) = t(r,i + 0);
        v(r,j + 2) = t(r,i + 1);
        v(r,j + 3) = a;
        v(r,j + 4) = t(r,i + 2);
        v(r,j + 5) = t(r,i + 0);
      }
    }

    k = j;

    t.swap ( v );
  }

  a = 1.0 / (2.0 * (double) n);

  for ( i = j = 0; i < k; i += 3, j++ )
  {
    s(0,j) = a;
    s(1,j) = (t(0,i + 0) + t(0,i + 1) + t(0,i + 2)) / 3.0;
    s(2,j) = (t(1,i + 0) + t(1,i + 1) + t(1,i + 2)) / 3.0;
  }

  return s;
}


//-----------------------------------------------------------------------
//   getIntegrationScheme (given a string)
//-----------------------------------------------------------------------


Matrix StdTriangle::getIntegrationScheme ( String& s )
{
  Matrix  ischeme;

  if ( s.find( '*' ) >= 0 )
  {
    String  s0, s1;

    ISchemes::split ( s0, s1, s );

    ischeme.ref (
      getIntegrationScheme (
        StdLine::getIntegrationScheme ( s0 ),
        StdLine::getIntegrationScheme ( s1 )
      )
    );

    s = ISchemes::join ( s0, s1 );
  }
  else
  {
    Matrix  (*func) ( idx_t );

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
    else
    {
      throw jem::IllegalInputException (
        JEM_FUNC,
        String::format (

          "invalid triangle integration scheme: `%s\'; "
          "must be `#\', `G#\', `%s#\', `U#\' or `%s#\', "
          "where # = number of integration points",

          s,
          ISchemes::GAUSS,
          ISchemes::UNIFORM
        )
      );
    }

    ischeme.ref ( func( n ) );

    s = name + String ( ischeme.size(1) );
  }

  return ischeme;
}


//-----------------------------------------------------------------------
//   getIntegrationScheme (given two 1-D integration schemes)
//-----------------------------------------------------------------------


Matrix StdTriangle::getIntegrationScheme

  ( const Matrix&  s0,
    const Matrix&  s1 )

{
  JEM_PRECHECK2 ( s0.size(0) == 2 && s1.size(0) == 2,
                  "Array shape mismatch" );

  const idx_t  m  = s0.size (1);
  const idx_t  n  = s1.size (1);

  Matrix       s  ( RANK + 1, m * n );

  Vector       w  ( s(0,ALL) );
  Matrix       u  ( s(slice(1,END),ALL) );

  double       x, y;
  idx_t        i, j, k;


  for ( j = k = 0; j < n; j++ )
  {
    y = s1(1,j);

    for ( i = 0; i < m; i++, k++ )
    {
      x      = s0(1,i);
      w[k]   = (1.0 - x) * s0(0,i) * s1(0,j) / 8.0;
      u(0,k) = (1.0 + x) / 2.0;
      u(1,k) = (1.0 - x) * (1.0 + y) / 4.0;
    }
  }

  return s;
}


//-----------------------------------------------------------------------
//   getGaussPointCount
//-----------------------------------------------------------------------


idx_t StdTriangle::getGaussPointCount ( idx_t n )
{
  if      ( n <= 1 )
  {
    return 1;
  }
  else if ( n <= 3 )
  {
    return 3;
  }
  else if ( n <= 4 )
  {
    return 4;
  }
  else if ( n <= 7 )
  {
    return 7;
  }
  else if ( n <= 13 )
  {
    return 13;
  }
  else if ( n <= 19 )
  {
    return 19;
  }
  else
  {
    idx_t  k;

    for ( k = 1; k <= 1000; k++ )
    {
      if ( k * k >= n )
      {
        break;
      }
    }

    return k * k;
  }
}


//-----------------------------------------------------------------------
//   getUniformPointCount
//-----------------------------------------------------------------------


idx_t StdTriangle::getUniformPointCount ( idx_t n )
{
  idx_t  k = 1;

  for ( idx_t i = 0; i < 20; i++ )
  {
    if ( k >= n )
    {
      break;
    }

    k *= 2;
  }

  return k;
}


//=======================================================================
//   class StdTriangle3
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdTriangle3::StdTriangle3 ()
{}


StdTriangle3::~StdTriangle3 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdTriangle3::readFrom ( ObjectInput& )
{}


void StdTriangle3::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdTriangle3::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   isLinear
//-----------------------------------------------------------------------


bool StdTriangle3::isLinear () const
{
  return true;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdTriangle3::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_TRIANG3_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdTriangle3::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_TRIANG3_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdTriangle3::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_TRIANG3_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdTriangle3::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdTriangle3::declare ()
{
  StdShapeFactory::declare ( "Linear", GEOMETRY, & makeNew );
}


//=======================================================================
//   class StdTriangle6
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdTriangle6::StdTriangle6 ()
{}


StdTriangle6::~StdTriangle6 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdTriangle6::readFrom ( ObjectInput& )
{}


void StdTriangle6::writeTo  ( ObjectOutput& ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdTriangle6::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdTriangle6::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdTriangle6::vertexCoords ()
{
  Matrix  u ( RANK, VERTEX_COUNT );

  u(0,0) = 0.0; u(1,0) = 0.0;
  u(0,1) = 0.5; u(1,1) = 0.0;
  u(0,2) = 1.0; u(1,2) = 0.0;
  u(0,3) = 0.5; u(1,3) = 0.5;
  u(0,4) = 0.0; u(1,4) = 1.0;
  u(0,5) = 0.0; u(1,5) = 0.5;

  return u;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdTriangle6::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdTriangle6::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_TRIANG6_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdTriangle6::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_TRIANG6_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdTriangle6::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_TRIANG6_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdTriangle6::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdTriangle6::declare ()
{
  StdShapeFactory::declare ( "Quadratic", GEOMETRY, & makeNew );
}


JIVE_END_PACKAGE( geom )
