
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/geom/error.h>
#include <jive/geom/utilities.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/ParametricEdge.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::ParametricEdge );


JIVE_BEGIN_PACKAGE( geom )


using jem::numeric::dotProduct;


//=======================================================================
//   private (non-member) functions
//=======================================================================

//-----------------------------------------------------------------------
//   getNormal
//-----------------------------------------------------------------------


static inline double      getNormal

  ( Vec2&                   t,
    const ParametricEdge&   edge )

{
  double  a, b, c;

  a = std::sqrt ( dotProduct( t, t ) );

  if ( jem::isTiny( a ) )
  {
    zeroVectorError ( edge.getContext(), "normal" );
  }

  b    =  1.0 / a;
  c    =  t[0];
  t[0] =  b * t[1];
  t[1] = -b * c;

  return a;
}


//=======================================================================
//   class ParametricEdge
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ParametricEdge::ParametricEdge ()
{}


ParametricEdge::ParametricEdge

  ( const String&         name,
    const Matrix&         ischeme,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, xshape, sshape )

{
  JEM_PRECHECK2 ( xshape->rank() == 1,
                  "invalid geometric shape functions" );
}


ParametricEdge::~ParametricEdge ()
{}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void ParametricEdge::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c ) const

{
  const idx_t  ipCount = ipCount_;

  Vec2         t;


  updateFuncs_ ( ipoints_ );

  for ( idx_t ip = 0; ip < ipCount; ip++ )
  {
    geom::getGlobalPoint ( t, ipoints_.xfuncs, ip, c );

    x(0,ip) = t[0];
    x(1,ip) = t[1];
  }
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void ParametricEdge::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  const Cubix&  xgrads    = ipoints_.xgrads;

  const idx_t   nodeCount = nodeCount_;
  const idx_t   ipCount   = ipCount_;

  Vec2          t;
  double        xg;


  updateGrads_ ( ipoints_ );

  for ( idx_t ip = 0; ip < ipCount; ip++ )
  {
    t = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      xg    = xgrads(0,i,ip);

      t[0] += c(0,i) * xg;
      t[1] += c(1,i) * xg;
    }

    w[ip] = iweights_[ip] * std::sqrt ( dotProduct( t, t ) );
  }
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void ParametricEdge::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Vector&  xsf = scratch_.xfuncs;

  Vec2           t;


  xshape_->evalShapeFunctions ( xsf, u );
  geom::getGlobalPoint     ( t, xsf, c );

  x[0] = t[0];
  x[1] = t[1];
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool ParametricEdge::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  const StdShape&  xsh       = * xshape_;
  const double     eps2      = eps * eps;
  const idx_t      nodeCount = nodeCount_;

  const Vector&    xsf       = scratch_.xfuncs;
  const Matrix&    xgrads    = scratch_.xgrads;

  Vec2             gx;
  double           px;
  Vec2             dx;
  double           du;
  double           dv;

  double           ddx, ddx0;
  double           delta;
  double           xg;
  int              iiter;


  // Execute a Newton-Raphson procedure to find the local
  // point. Start in the geometric center of the shape.

  iiter = 0;
  delta = jem::maxOf ( delta );
  ddx0  = 0.0;
  u     = center_;

  while ( true )
  {
    xsh.evalShapeGradients ( xsf, xgrads, u );
    geom::getGlobalPoint   ( dx,  xsf,    c );

    dx[0] = x[0] - dx[0];
    dx[1] = x[1] - dx[1];

    ddx   = dotProduct ( dx, dx );

    if ( iiter == 0 )
    {
      ddx0 = ddx;
    }

    if ( delta < NR_DELTA_ * eps2 ||
         ddx   < NR_ALPHA_ * eps2 ||
         ddx   > NR_BETA_  * ddx0 ||
         iiter > MAX_NR_ITER_ )
    {
      break;
    }

    gx = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      xg     = xgrads(0,i);

      gx[0] += c(0,i) * xg;
      gx[1] += c(1,i) * xg;
    }

    px = dotProduct ( gx, gx );
    dv = dotProduct ( gx, dx );

    if ( jem::isTiny( px ) )
    {
      break;
    }

    du    = dv / px;
    u[0] += du;
    delta = px * du * du;

    iiter++;
  }

  if ( xsh.clipPoint( u ) )
  {
    xsh.evalShapeFunctions (     xsf, u );
    geom::getGlobalPoint   ( dx, xsf, c );

    dx[0] = x[0] - dx[0];
    dx[1] = x[1] - dx[1];

    ddx   = dotProduct ( dx, dx );
  }

  return (ddx < eps2);
}


//-----------------------------------------------------------------------
//   evalNormal
//-----------------------------------------------------------------------


void ParametricEdge::evalNormal

  ( const Vector&  q,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Matrix&  xgrads    = scratch_.xgrads;
  const idx_t    nodeCount = nodeCount_;

  Vec2           t;
  double         xg;


  xshape_->evalShapeGradients ( scratch_.xfuncs, xgrads, u );

  t = 0.0;

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    xg    = xgrads(0,i);

    t[0] += c(0,i) * xg;
    t[1] += c(1,i) * xg;
  }

  getNormal ( t, *this );

  q[0] = t[0];
  q[1] = t[1];
}


//-----------------------------------------------------------------------
//   calcNormals
//-----------------------------------------------------------------------


void ParametricEdge::calcNormals

  ( const Matrix&    q,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&  xgrads    = points.xgrads;

  const idx_t   nodeCount = nodeCount_;
  const idx_t   pcount    = q.size(1);

  Vec2          t;
  double        xg;
  double        a;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    t = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      xg    = xgrads(0,i,ip);

      t[0] += c(0,i) * xg;
      t[1] += c(1,i) * xg;
    }

    a = getNormal ( t, *this );

    q(0,ip) = t[0];
    q(1,ip) = t[1];

    if ( w )
    {
      (*w)[ip] = a;
    }
  }
}


JIVE_END_PACKAGE( geom )
