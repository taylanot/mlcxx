
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
#include <jive/geom/StdCube.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdPrism.h>
#include <jive/geom/shfuncs/HyperCubeFuncs.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/IntegrationSchemes.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdHyperCube.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdHyperCube16  );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class StdHyperCube
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdHyperCube::GEOMETRY = Geometries::HYPER_CUBE;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


StdHyperCube::~StdHyperCube ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdHyperCube::rank () const
{
  return (idx_t) RANK;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdHyperCube::getGeometry () const
{
  return GEOMETRY;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdHyperCube::vertexCount () const
{
  return (idx_t) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdHyperCube::vertexCoords ()
{
  return StdPrism::getCoords ( StdCube8::vertexCoords (),
                               StdLine2::vertexCoords () );
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdHyperCube::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoints
//-----------------------------------------------------------------------


bool StdHyperCube::containsPoint ( const Vector& u ) const
{
  return ( u[0] >= -1.0 && u[0] <= 1.0 &&
           u[1] >= -1.0 && u[1] <= 1.0 &&
           u[2] >= -1.0 && u[2] <= 1.0 &&
           u[3] >= -1.0 && u[3] <= 1.0 );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdHyperCube::clipPoint ( const Vector& u ) const
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


Matrix StdHyperCube::getGaussScheme ( idx_t n )
{
  const idx_t  k = quarticRoot_ ( n );

  return getGaussScheme ( k, k, k, k );
}


Matrix StdHyperCube::getGaussScheme

  ( idx_t  m,
    idx_t  n,
    idx_t  p,
    idx_t  q )

{
  return StdPrism::getIntegrationScheme (
    StdCube::getGaussScheme ( m, n, p ),
    StdLine::getGaussScheme ( q )
  );
}


//-----------------------------------------------------------------------
//   getNewtonCotesScheme
//-----------------------------------------------------------------------


Matrix StdHyperCube::getNewtonCotesScheme ( idx_t n )
{
  const idx_t  k = quarticRoot_ ( n );

  return getNewtonCotesScheme ( k, k, k, k );
}


Matrix StdHyperCube::getNewtonCotesScheme

  ( idx_t  m,
    idx_t  n,
    idx_t  p,
    idx_t  q )

{
  return StdPrism::getIntegrationScheme (
    StdCube::getNewtonCotesScheme ( m, n, p ),
    StdLine::getNewtonCotesScheme ( q )
  );
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix StdHyperCube::getIntegrationScheme ( String& s )
{
  String  s0, s1;

  ISchemes::split ( s0, s1, s );

  Matrix  ischeme =

    StdPrism::getIntegrationScheme (
      StdCube::getIntegrationScheme ( s0 ),
      StdLine::getIntegrationScheme ( s1 )
    );

  s = ISchemes::join ( s0, s1 );

  return ischeme;
}


//-----------------------------------------------------------------------
//   quarticRoot_
//-----------------------------------------------------------------------


idx_t StdHyperCube::quarticRoot_ ( idx_t n )
{
  idx_t  k;

  for ( k = 1; k < 50; k++ )
  {
    if ( (k * k * k * k) >= n )
    {
      break;
    }
  }

  return k;
}


//=======================================================================
//   class StdHyperCube16
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdHyperCube16::StdHyperCube16 ()
{}


StdHyperCube16::~StdHyperCube16 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdHyperCube16::readFrom ( ObjectInput& in )
{}


void StdHyperCube16::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdHyperCube16::shapeFuncCount () const
{
  return (idx_t) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdHyperCube16::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  JIVE_EVAL_HYPERCUBE16_FUNCS( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdHyperCube16::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  JIVE_EVAL_HYPERCUBE16_GRADS( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdHyperCube16::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  JIVE_EVAL_HYPERCUBE16_GRADS2( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<SShape> StdHyperCube16::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdHyperCube16::declare ()
{
  StdShapeFactory::declare ( "Linear", GEOMETRY, & makeNew );
}


JIVE_END_PACKAGE( geom )
