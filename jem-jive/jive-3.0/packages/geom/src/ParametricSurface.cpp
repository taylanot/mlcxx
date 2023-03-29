
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
#include <jem/base/tuple/operators.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/geom/error.h>
#include <jive/geom/utilities.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/ParametricSurface.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::ParametricSurface );


JIVE_BEGIN_PACKAGE( geom )


using jem::numeric::dotProduct;
using jem::numeric::crossProduct;


//=======================================================================
//   private (non-member) functions
//=======================================================================

//-----------------------------------------------------------------------
//   getNormal
//-----------------------------------------------------------------------


static inline double        getNormal

  ( Vec3&                     n,
    const Vec3&               s,
    const Vec3&               t,
    const ParametricSurface&  surf )

{
  double  a;

  n = crossProduct ( s, t );
  a = std::sqrt ( dotProduct( n, n ) );

  if ( jem::isTiny( a ) )
  {
    zeroVectorError ( surf.getContext(), "normal" );
  }

  n = n * (1.0 / a);

  return a;
}


//=======================================================================
//   class ParametricSurface
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ParametricSurface::ParametricSurface ()
{}


ParametricSurface::ParametricSurface

  ( const String&         name,
    const Matrix&         ischeme,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, xshape, sshape )

{
  JEM_PRECHECK2 ( xshape->rank() == 2,
                  "invalid geometric shape functions" );
}


ParametricSurface::~ParametricSurface ()
{}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void ParametricSurface::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c ) const

{
  const idx_t  ipCount = ipCount_;

  Vec3         t;


  updateFuncs_ ( ipoints_ );

  for ( idx_t ip = 0; ip < ipCount; ip++ )
  {
    geom::getGlobalPoint ( t, ipoints_.xfuncs, ip, c );

    x(0,ip) = t[0];
    x(1,ip) = t[1];
    x(2,ip) = t[2];
  }
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void ParametricSurface::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  const Cubix&  xgrads    = ipoints_.xgrads;

  const idx_t   nodeCount = nodeCount_;
  const idx_t   ipCount   = ipCount_;

  Vec3          s, t, n;
  double        xg0, xg1;


  updateGrads_ ( ipoints_ );

  for ( idx_t ip = 0; ip < ipCount; ip++ )
  {
    s = 0.0;
    t = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      xg0   = xgrads(0,i,ip);
      xg1   = xgrads(1,i,ip);

      s[0] += c(0,i) * xg0;
      s[1] += c(1,i) * xg0;
      s[2] += c(2,i) * xg0;

      t[0] += c(0,i) * xg1;
      t[1] += c(1,i) * xg1;
      t[2] += c(2,i) * xg1;
    }

    n = crossProduct ( s, t );

    w[ip] = iweights_[ip] * std::sqrt( dotProduct( n, n ) );
  }
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void ParametricSurface::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Vector&  xsf = scratch_.xfuncs;

  Vec3           t;

  xshape_->evalShapeFunctions ( xsf, u );
  geom::getGlobalPoint     ( t, xsf, c );

  x[0] = t[0];
  x[1] = t[1];
  x[2] = t[2];
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool ParametricSurface::getLocalPoint

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

  Mat2x2           px;
  Vec3             s, t;
  Vec3             dx;
  Vec2             du;
  Vec2             dv;

  double           ddx, ddx0;
  double           delta;
  double           det;
  double           xg[2];
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
    dx[2] = x[2] - dx[2];

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

    s = 0.0;
    t = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      xg[0] = xgrads(0,i);
      xg[1] = xgrads(1,i);

      s[0] += c(0,i) * xg[0];
      s[1] += c(1,i) * xg[0];
      s[2] += c(2,i) * xg[0];

      t[0] += c(0,i) * xg[1];
      t[1] += c(1,i) * xg[1];
      t[2] += c(2,i) * xg[1];
    }

    px(0,0) = dotProduct ( s, s );
    px(0,1) = dotProduct ( s, t );
    px(1,0) = px(0,1);
    px(1,1) = dotProduct ( t, t );

    dv[0]   = dotProduct ( s, dx );
    dv[1]   = dotProduct ( t, dx );

    det     = px(0,0) * px(1,1) - px(0,1) * px(1,0);

    if ( jem::isTiny( det ) )
    {
      break;
    }

    det   = 1.0 / det;

    du[0] = det * (px(1,1) * dv[0] - px(0,1) * dv[1]);
    du[1] = det * (px(0,0) * dv[1] - px(1,0) * dv[0]);

    u[0] += du[0];
    u[1] += du[1];

    delta = px(0,0) * du[0] * du[0] + px(1,1) * du[1] * du[1];

    iiter++;
  }

  if ( xsh.clipPoint( u ) )
  {
    xsh.evalShapeFunctions (     xsf, u );
    geom::getGlobalPoint   ( dx, xsf, c );

    dx[0] = x[0] - dx[0];
    dx[1] = x[1] - dx[1];
    dx[2] = x[2] - dx[2];

    ddx   = dotProduct ( dx, dx );
  }

  return (ddx < eps2);
}


//-----------------------------------------------------------------------
//   evalNormal
//-----------------------------------------------------------------------


void ParametricSurface::evalNormal

  ( const Vector&  q,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Matrix&  xgrads    = scratch_.xgrads;
  const idx_t    nodeCount = nodeCount_;

  Vec3           s, t, n;
  double         xg0, xg1;


  xshape_->evalShapeGradients ( scratch_.xfuncs, xgrads, u );

  s = 0.0;
  t = 0.0;

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    xg0   = xgrads(0,i);
    xg1   = xgrads(1,i);

    s[0] += c(0,i) * xg0;
    s[1] += c(1,i) * xg0;
    s[2] += c(2,i) * xg0;

    t[0] += c(0,i) * xg1;
    t[1] += c(1,i) * xg1;
    t[2] += c(2,i) * xg1;
  }

  getNormal ( n, s, t, *this );

  q[0] = n[0];
  q[1] = n[1];
  q[2] = n[2];
}


//-----------------------------------------------------------------------
//   calcNormals
//-----------------------------------------------------------------------


void ParametricSurface::calcNormals

  ( const Matrix&    q,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&  xgrads    = points.xgrads;

  const idx_t   nodeCount = nodeCount_;
  const idx_t   pcount    = q.size(1);

  Vec3          s, t, n;
  double        xg0, xg1;
  double        a;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    s = 0.0;
    t = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      xg0   = xgrads(0,i,ip);
      xg1   = xgrads(1,i,ip);

      s[0] += c(0,i) * xg0;
      s[1] += c(1,i) * xg0;
      s[2] += c(2,i) * xg0;

      t[0] += c(0,i) * xg1;
      t[1] += c(1,i) * xg1;
      t[2] += c(2,i) * xg1;
    }

    a = getNormal ( n, s, t, *this );

    q(0,ip) = n[0];
    q(1,ip) = n[1];
    q(2,ip) = n[2];

    if ( w )
    {
      (*w)[ip] = a;
    }
  }
}


JIVE_END_PACKAGE( geom )
