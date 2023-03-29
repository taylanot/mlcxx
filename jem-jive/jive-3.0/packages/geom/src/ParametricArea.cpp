
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
#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jive/util/error.h>
#include <jive/geom/error.h>
#include <jive/geom/utilities.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/ParametricArea.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::ParametricArea );


JIVE_BEGIN_PACKAGE( geom )


using jem::numeric::invert;
using jem::numeric::dotProduct;


//=======================================================================
//   class ParametricArea
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ParametricArea::ParametricArea ()
{}


ParametricArea::ParametricArea

  ( const String&         name,
    const Matrix&         ischeme,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, xshape, sshape )

{
  JEM_PRECHECK2 ( xshape->rank() == RANK,
                  "invalid geometric shape functions" );
}


ParametricArea::ParametricArea

  ( const String&         name,
    const Matrix&         ischeme,
    const Boundary&       boundary,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, boundary, xshape, sshape )

{
  JEM_PRECHECK2 ( xshape->rank() == RANK,
                  "invalid geometric shape functions" );
}


ParametricArea::~ParametricArea ()
{}


//-----------------------------------------------------------------------
//   getBoundaryCoords
//-----------------------------------------------------------------------


void ParametricArea::getBoundaryCoords

  ( const Matrix&  x,
    idx_t          ibnd,
    const Matrix&  c ) const

{
  using jive::util::indexError;

  const Topology&  topo = boundary_.topology;

  JEM_ASSERT2 ( topo.isContiguous(),
                "non-contiguous boundary Topology" );

  if ( ibnd < 0 || ibnd >= topo.size(0) )
  {
    indexError ( getContext(), "boundary", ibnd, topo.size(0) );
  }

  const idx_t* JEM_RESTRICT  offsets = topo.getOffsetPtr ();
  const idx_t* JEM_RESTRICT  indices = topo.getIndexPtr  ();

  const idx_t  first = offsets[ibnd];
  const idx_t  last  = offsets[ibnd + 1];
  const idx_t  count = last - first;

JEM_IVDEP

  for ( idx_t i = 0; i < count; i++ )
  {
    idx_t  j = indices[i + first];

    x(0,i) = c(0,j);
    x(1,i) = c(1,j);
  }
}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void ParametricArea::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c )  const

{
  Vec2  t;

  updateFuncs_ ( ipoints_ );

  for ( idx_t ip = 0; ip < ipCount_; ip++ )
  {
    geom::getGlobalPoint ( t, ipoints_.xfuncs, ip, c );

    x(0,ip) = t[0];
    x(1,ip) = t[1];
  }
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void ParametricArea::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  using jem::numeric::det;

  Mat2x2  ja;

  updateGrads_ ( ipoints_ );

  for ( idx_t ip = 0; ip < ipCount_; ip++ )
  {
    getJacobi ( ja, ipoints_.xgrads, ip, c );

    w[ip] = iweights_[ip] * std::fabs ( det( ja ) );
  }
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void ParametricArea::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Vector&  xsf = scratch_.xfuncs;

  Vec2  t;


  xshape_->evalShapeFunctions ( xsf, u );
  geom::getGlobalPoint     ( t, xsf, c );

  x[0] = t[0];
  x[1] = t[1];
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool ParametricArea::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  const StdShape&  xsh       = * xshape_;
  const double     eps2      = eps * eps;

  const Vector&    xsf       = scratch_.xfuncs;
  const Matrix&    xgrads    = scratch_.xgrads;

  Mat2x2           gx;
  Vec2             du;
  Vec2             dx;

  double           ddx, ddx0;
  double           jdet;
  int              iiter;


  // Execute a Newton-Raphson procedure to find the local
  // point. Start in the geometric center of the shape.

  iiter = 0;
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

    if ( ddx   < NR_ALPHA_ * eps2 ||
         ddx   > NR_BETA_  * ddx0 ||
         iiter > MAX_NR_ITER_ )
    {
      break;
    }

    getJacobi ( gx, c, xgrads );

    jdet = gx(0,0) * gx(1,1) - gx(1,0) * gx(0,1);

    if ( jem::isTiny( jdet ) )
    {
      break;
    }

    jdet  = 1.0 / jdet;

    du[0] = jdet * (gx(1,1) * dx[0] - gx(0,1) * dx[1]);
    du[1] = jdet * (gx(0,0) * dx[1] - gx(1,0) * dx[0]);

    u[0] += du[0];
    u[1] += du[1];

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
//   evalShapeGradients
//-----------------------------------------------------------------------


void ParametricArea::evalShapeGradients

  ( const Matrix&  g,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Matrix&  xgrads = scratch_.xgrads;
  const Matrix&  sgrads = scratch_.sgrads;

  Mat2x2         ja;


  xshape_->evalShapeGradients ( scratch_.xfuncs, xgrads, u );

  if ( xshape_ != sshape_ )
  {
    sshape_->evalShapeGradients ( scratch_.sfuncs, sgrads, u );
  }

  getJacobi ( ja, xgrads, c );

  if ( ! invert( ja ) )
  {
    singularMatrixError ( getContext(), "Jacobi" );
  }

  applyJacobi ( g, ja, sgrads );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void ParametricArea::evalShapeGradGrads

  ( const Matrix&  g2,
    const Matrix&  g,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Matrix&  xgrads  = scratch_.xgrads;
  const Matrix&  xgrads2 = scratch_.xgrads2;
  const Matrix&  sgrads  = scratch_.sgrads;
  const Matrix&  sgrads2 = scratch_.sgrads2;

  Mat2x2  ja;
  Mat3x2  ja2;
  Mat3x3  qu;


  xshape_->evalShapeGradGrads ( scratch_.xfuncs,
                                xgrads, xgrads2, u );

  if ( xshape_ != sshape_ )
  {
    sshape_->evalShapeGradGrads ( scratch_.sfuncs,
                                  sgrads, sgrads2, u );
  }

  getJacobi ( ja,  xgrads,  c );
  getJacobi ( ja2, xgrads2, c );
  getQujobi ( qu,  ja );

  if ( ! invert( ja ) )
  {
    singularMatrixError ( getContext(), "Jacobi" );
  }

  if ( ! invert( qu ) )
  {
    singularMatrixError ( getContext(), "Qujobi" );
  }

  applyJacobi ( g,  ja,  sgrads );
  applyJacobi ( g2, ja2, g      );

  sgrads2 -= g2;

  applyJacobi ( g2, qu, sgrads2 );
}


//-----------------------------------------------------------------------
//   calcGradients
//-----------------------------------------------------------------------


void ParametricArea::calcGradients

  ( const Cubix&     g,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&  xgrads    = points.xgrads;
  const Cubix&  sgrads    = points.sgrads;

  const idx_t   pcount    = xgrads.size(2);

  Mat2x2        ja;
  double        jdet;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    getJacobi ( ja, xgrads, ip, c );

    if ( ! invert( ja, jdet ) )
    {
      singularMatrixError ( getContext(), "Jacobi" );
    }

    applyJacobi ( g, ja, sgrads, ip );

    if ( w )
    {
      (*w)[ip] = jdet;
    }
  }
}


//-----------------------------------------------------------------------
//   calcGradGrads
//-----------------------------------------------------------------------


void ParametricArea::calcGradGrads

  ( const Cubix&     g2,
    const Cubix&     g,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&   xgrads  = points.xgrads;
  const Cubix&   xgrads2 = points.xgrads2;
  const Cubix&   sgrads  = points.sgrads;
  const Cubix&   sgrads2 = points.sgrads2;
  const Matrix&  tgrads2 = scratch_.sgrads2;

  const idx_t    pcount  = xgrads.size(2);

  Mat2x2         ja;
  Mat3x2         ja2;
  Mat3x3         qu;
  double         jdet;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    getJacobi ( ja,  xgrads,  ip, c );
    getJacobi ( ja2, xgrads2, ip, c );
    getQujobi ( qu,  ja );

    if ( ! invert( ja, jdet ) )
    {
      singularMatrixError ( getContext(), "Jacobi" );
    }

    if ( ! invert( qu ) )
    {
      singularMatrixError ( getContext(), "Qujobi" );
    }

    applyJacobi ( g,       ja,  sgrads, ip );
    applyJacobi ( tgrads2, ja2, g[ip] );

    tgrads2 = sgrads2[ip] - tgrads2;

    applyJacobi ( g2[ip], qu, tgrads2 );

    if ( w )
    {
      (*w)[ip] = jdet;
    }
  }
}


JIVE_END_PACKAGE( geom )
