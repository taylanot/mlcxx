
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
#include <jive/util/error.h>
#include <jive/geom/error.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/ParametricLine.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::ParametricLine );


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class ParametricLine
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ParametricLine::ParametricLine ()
{}


ParametricLine::ParametricLine

  ( const String&         name,
    const Matrix&         ischeme,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, xshape, sshape )

{
  JEM_PRECHECK2 ( xshape->rank() == RANK,
                  "invalid geometric shape functions" );
}


ParametricLine::ParametricLine

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


ParametricLine::~ParametricLine ()
{}


//-----------------------------------------------------------------------
//   getBoundaryCoords
//-----------------------------------------------------------------------


void ParametricLine::getBoundaryCoords

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

JEM_IVDEP

  for ( idx_t j = first; j < last; j++ )
  {
    x(0,j - first) = c(0,indices[j]);
  }
}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void ParametricLine::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c )  const

{
  const Matrix&  xsf       = ipoints_.xfuncs;

  const idx_t    ipCount   = ipCount_;
  const idx_t    nodeCount = nodeCount_;

  double         u;


  updateFuncs_ ( ipoints_ );

  if      ( nodeCount == 1_idx )
  {
JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      x(0,ip) = c(0,0) * xsf(0,ip);
    }
  }
  else if ( nodeCount == 2_idx )
  {
JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      x(0,ip) = c(0,0) * xsf(0,ip) +
                c(0,1) * xsf(1,ip);
    }
  }
  else if ( nodeCount == 3_idx )
  {
JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      x(0,ip) = c(0,0) * xsf(0,ip) +
                c(0,1) * xsf(1,ip) +
                c(0,2) * xsf(2,ip);
    }
  }
  else if ( nodeCount >= 4_idx )
  {
    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      u = 0.0;

      for ( idx_t i = 0; i < nodeCount; i++ )
      {
        u += c(0,i) * xsf(i,ip);
      }

      x(0,ip) = u;
    }
  }
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void ParametricLine::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  const Cubix&  xgrads    = ipoints_.xgrads;

  const idx_t   ipCount   = ipCount_;
  const idx_t   nodeCount = nodeCount_;

  double        ja;


  updateGrads_ ( ipoints_ );

  if      ( nodeCount == 1_idx )
  {
JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      ja    = xgrads(0,0,ip) * c(0,0);
      w[ip] = iweights_[ip]  * std::fabs( ja );
    }
  }
  else if ( nodeCount == 2_idx )
  {
JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      ja    = xgrads(0,0,ip) * c(0,0) +
              xgrads(0,1,ip) * c(0,1);
      w[ip] = iweights_[ip]  * std::fabs( ja );
    }
  }
  else if ( nodeCount == 3_idx )
  {
JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      ja    = xgrads(0,0,ip) * c(0,0) +
              xgrads(0,1,ip) * c(0,1) +
              xgrads(0,2,ip) * c(0,2);
      w[ip] = iweights_[ip]  * std::fabs( ja );
    }
  }
  else if ( nodeCount >= 4_idx )
  {
    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      ja = 0.0;

      for ( idx_t i = 0; i < nodeCount; i++ )
      {
        ja += xgrads(0,i,ip) * c(0,i);
      }

      w[ip] = iweights_[ip] * std::fabs( ja );
    }
  }
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void ParametricLine::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Vector&  xsf       = scratch_.xfuncs;
  const idx_t    nodeCount = nodeCount_;

  double         a;


  xshape_->evalShapeFunctions ( xsf, u );

  a = 0.0;

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    a += c(0,i) * xsf[i];
  }

  x[0] = a;
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool ParametricLine::getLocalPoint

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

  double           gx;
  double           dx;
  double           ddx, ddx0;
  int              iiter;


  // Execute a Newton-Raphson procedure to find the local
  // point. Start in the geometric center of the shape.

  iiter = 0;
  ddx0  = 0.0;
  u     = center_;

  while ( true )
  {
    xsh.evalShapeGradients ( xsf, xgrads, u );

    dx = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      dx += c(0,i) * xsf[i];
    }

    dx  = x[0] - dx;
    ddx = dx * dx;

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

    gx = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      gx += c(0,i) * xgrads(0,i);
    }

    if ( jem::isTiny( gx ) )
    {
      break;
    }

    u[0] += (dx / gx);

    iiter++;
  }

  if ( xsh.clipPoint( u ) )
  {
    xsh.evalShapeFunctions ( xsf, u );

    dx = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      dx += c(0,i) * xsf[i];
    }

    dx  = x[0] - dx;
    ddx = dx * dx;
  }

  return (ddx < eps2);
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void ParametricLine::evalShapeGradients

  ( const Matrix&  g,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Matrix&  xgrads    = scratch_.xgrads;
  const Matrix&  sgrads    = scratch_.sgrads;

  const idx_t    nodeCount = nodeCount_;
  const idx_t    funcCount = funcCount_;

  double         ja;
  double         invJa;


  xshape_->evalShapeGradients ( scratch_.xfuncs, xgrads, u );

  if ( xshape_ != sshape_ )
  {
    sshape_->evalShapeGradients ( scratch_.sfuncs, sgrads, u );
  }

  ja = 0.0;

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    ja += xgrads(0,i) * c(0,i);
  }

  if ( jem::isTiny( ja ) )
  {
    singularMatrixError ( getContext(), "Jacobi" );
  }

  invJa = 1.0 / ja;

JEM_IVDEP

  for ( idx_t i = 0; i < funcCount; i++ )
  {
    g(0,i) = invJa * sgrads(0,i);
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void ParametricLine::evalShapeGradGrads

  ( const Matrix&  g2,
    const Matrix&  g,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Matrix&  xgrads    = scratch_.xgrads;
  const Matrix&  xgrads2   = scratch_.xgrads2;
  const Matrix&  sgrads    = scratch_.sgrads;
  const Matrix&  sgrads2   = scratch_.sgrads2;

  const idx_t    nodeCount = nodeCount_;
  const idx_t    funcCount = funcCount_;

  double         ja, ja2;
  double         invJa;
  double         invQu;


  xshape_->evalShapeGradGrads ( scratch_.xfuncs,
                                xgrads, xgrads2, u );

  if ( xshape_ != sshape_ )
  {
    sshape_->evalShapeGradGrads ( scratch_.sfuncs,
                                  sgrads, sgrads2, u );
  }

  ja = ja2 = 0.0;

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    ja  += xgrads (0,i) * c(0,i);
    ja2 += xgrads2(0,i) * c(0,i);
  }

  if ( jem::isTiny( ja ) )
  {
    singularMatrixError ( getContext(), "Jacobi" );
  }

  invJa = 1.0 / ja;
  invQu = invJa * invJa;

JEM_IVDEP

  for ( idx_t i = 0; i < funcCount; i++ )
  {
    g (0,i) = invJa * sgrads(0,i);
    g2(0,i) = invQu * (sgrads2(0,i) - ja2 * g(0,i));
  }
}


//-----------------------------------------------------------------------
//   calcGradients
//-----------------------------------------------------------------------


void ParametricLine::calcGradients

  ( const Cubix&     g,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&  xgrads    = points.xgrads;
  const Cubix&  sgrads    = points.sgrads;

  const idx_t   pcount    = xgrads.size(2);
  const idx_t   nodeCount = nodeCount_;
  const idx_t   funcCount = funcCount_;

  double        ja;
  double        invJa;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    ja = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      ja += xgrads(0,i,ip) * c(0,i);
    }

    if ( jem::isTiny( ja ) )
    {
      singularMatrixError ( getContext(), "Jacobi" );
    }

    invJa = 1.0 / ja;

JEM_IVDEP

    for ( idx_t i = 0; i < funcCount; i++ )
    {
      g(0,i,ip) = invJa * sgrads(0,i,ip);
    }

    if ( w )
    {
      (*w)[ip] = ja;
    }
  }
}


//-----------------------------------------------------------------------
//   calcGradGrads
//-----------------------------------------------------------------------


void ParametricLine::calcGradGrads

  ( const Cubix&     g2,
    const Cubix&     g,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&  xgrads    = points.xgrads;
  const Cubix&  xgrads2   = points.xgrads2;
  const Cubix&  sgrads    = points.sgrads;
  const Cubix&  sgrads2   = points.sgrads2;

  const idx_t   pcount    = xgrads.size(2);
  const idx_t   nodeCount = nodeCount_;
  const idx_t   funcCount = funcCount_;

  double        ja, ja2;
  double        invJa;
  double        invQu;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    ja = ja2 = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      ja  += xgrads (0,i,ip) * c(0,i);
      ja2 += xgrads2(0,i,ip) * c(0,i);
    }

    if ( jem::isTiny( ja ) )
    {
      singularMatrixError ( getContext(), "Jacobi" );
    }

    invJa = 1.0 / ja;
    invQu = invJa * invJa;

JEM_IVDEP

    for ( idx_t i = 0; i < funcCount; i++ )
    {
      g (0,i,ip) = invJa * sgrads(0,i,ip);
      g2(0,i,ip) = invQu * (sgrads2(0,i,ip) - ja2 * g(0,i,ip));
    }

    if ( w )
    {
      (*w)[ip] = ja;
    }
  }
}


JIVE_END_PACKAGE( geom )
