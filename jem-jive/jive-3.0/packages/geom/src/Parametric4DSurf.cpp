
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
#include <jive/geom/error.h>
#include <jive/geom/Tuple.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/Parametric4DSurf.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::Parametric4DSurf );


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   getNormal
//-----------------------------------------------------------------------


static inline double    getNormal

  ( Vec4&                 n,
    const Vec4&           s,
    const Vec4&           t,
    const Vec4&           v )

{
  double  a[3][4];

  a[0][1] = t[0] * v[1] - t[1] * v[0];
  a[0][2] = t[0] * v[2] - t[2] * v[0];
  a[0][3] = t[0] * v[3] - t[3] * v[0];
  a[1][2] = t[1] * v[2] - t[2] * v[1];
  a[1][3] = t[1] * v[3] - t[3] * v[1];
  a[2][3] = t[2] * v[3] - t[3] * v[2];

  n[0] = -s[1] * a[2][3] + s[2] * a[1][3] - s[3] * a[1][2];
  n[1] =  s[0] * a[2][3] - s[2] * a[0][3] + s[3] * a[0][2];
  n[2] = -s[0] * a[1][3] + s[1] * a[0][3] - s[3] * a[0][1];
  n[3] =  s[0] * a[1][2] - s[1] * a[0][2] + s[2] * a[0][1];

  return std::sqrt ( n[0] * n[0] + n[1] * n[1] +
                     n[2] * n[2] + n[3] * n[3] );
}


//=======================================================================
//   class Parametric4DSurf
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Parametric4DSurf::Parametric4DSurf ()
{}


Parametric4DSurf::Parametric4DSurf

  ( const String&         name,
    const Matrix&         ischeme,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, xshape, sshape )

{
  JEM_PRECHECK2 ( xshape->rank() == 3,
                  "invalid geometric shape functions" );
}


Parametric4DSurf::~Parametric4DSurf ()
{}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void Parametric4DSurf::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  const Cubix&  xgrads    = ipoints_.xgrads;

  const idx_t   nodeCount = nodeCount_;
  const idx_t   ipCount   = ipCount_;

  Vec4          s, t, v, n;
  double        xg0, xg1, xg2;
  double        a;


  updateGrads_ ( ipoints_ );

  for ( idx_t ip = 0; ip < ipCount; ip++ )
  {
    s = 0.0;
    t = 0.0;
    v = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      xg0   = xgrads(0,i,ip);
      xg1   = xgrads(1,i,ip);
      xg2   = xgrads(2,i,ip);

      s[0] += c(0,i) * xg0;
      s[1] += c(1,i) * xg0;
      s[2] += c(2,i) * xg0;
      s[3] += c(3,i) * xg0;

      t[0] += c(0,i) * xg1;
      t[1] += c(1,i) * xg1;
      t[2] += c(2,i) * xg1;
      t[3] += c(3,i) * xg1;

      v[0] += c(0,i) * xg2;
      v[1] += c(1,i) * xg2;
      v[2] += c(2,i) * xg2;
      v[3] += c(3,i) * xg2;
    }

    a     = getNormal ( n, s, t, v );
    w[ip] = a * iweights_[ip];
  }
}


//-----------------------------------------------------------------------
//   evalNormal
//-----------------------------------------------------------------------


void Parametric4DSurf::evalNormal

  ( const Vector&  q,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Matrix&  xgrads    = scratch_.xgrads;

  const idx_t    nodeCount = nodeCount_;

  Vec4           s, t, v, n;
  double         xg0, xg1, xg2;
  double         a;


  xshape_->evalShapeGradients ( scratch_.xfuncs, xgrads, u );

  s = 0.0;
  t = 0.0;
  v = 0.0;

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    xg0   = xgrads(0,i);
    xg1   = xgrads(1,i);
    xg2   = xgrads(2,i);

    s[0] += c(0,i) * xg0;
    s[1] += c(1,i) * xg0;
    s[2] += c(2,i) * xg0;
    s[3] += c(3,i) * xg0;

    t[0] += c(0,i) * xg1;
    t[1] += c(1,i) * xg1;
    t[2] += c(2,i) * xg1;
    t[3] += c(3,i) * xg1;

    v[0] += c(0,i) * xg2;
    v[1] += c(1,i) * xg2;
    v[2] += c(2,i) * xg2;
    v[3] += c(3,i) * xg2;
  }

  a = getNormal ( n, s, t, v );

  if ( jem::isTiny( a ) )
  {
    zeroVectorError ( getContext(), "normal" );
  }

  a    = 1.0 / a;

  q[0] = a * n[0];
  q[1] = a * n[1];
  q[2] = a * n[2];
  q[3] = a * n[3];
}


//-----------------------------------------------------------------------
//   calcNormals
//-----------------------------------------------------------------------


void Parametric4DSurf::calcNormals

  ( const Matrix&    q,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&   xgrads    = points.xgrads;

  const idx_t    nodeCount = nodeCount_;
  const idx_t    pcount    = q.size(1);

  Vec4           s, t, v, n;
  double         xg0, xg1, xg2;
  double         a, b;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    s = 0.0;
    t = 0.0;
    v = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      xg0   = xgrads(0,i,ip);
      xg1   = xgrads(1,i,ip);
      xg2   = xgrads(2,i,ip);

      s[0] += c(0,i) * xg0;
      s[1] += c(1,i) * xg0;
      s[2] += c(2,i) * xg0;
      s[3] += c(3,i) * xg0;

      t[0] += c(0,i) * xg1;
      t[1] += c(1,i) * xg1;
      t[2] += c(2,i) * xg1;
      t[3] += c(3,i) * xg1;

      v[0] += c(0,i) * xg2;
      v[1] += c(1,i) * xg2;
      v[2] += c(2,i) * xg2;
      v[3] += c(3,i) * xg2;
    }

    a = getNormal ( n, s, t, v );

    if ( jem::isTiny( a ) )
    {
      zeroVectorError ( getContext(), "normal" );
    }

    b       = 1.0 / a;

    q(0,ip) = b * n[0];
    q(1,ip) = b * n[1];
    q(2,ip) = b * n[2];
    q(3,ip) = b * n[3];

    if ( w )
    {
      (*w)[ip] = a;
    }
  }
}


JIVE_END_PACKAGE( geom )
