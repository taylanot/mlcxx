
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


#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdEdge.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdEdge );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class StdEdge
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdEdge::StdEdge ()
{}


StdEdge::StdEdge

  ( const Matrix&         xnodes,
    const Ref<StdShape>&  sshape ) :

    Super ( xnodes, newInstance<StdLine2>(), sshape )

{}


StdEdge::StdEdge

  ( const Ref<Super>&  parent,
    const Matrix&      xnodes ) :

    Super ( parent, xnodes )

{}


StdEdge::~StdEdge ()
{}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdEdge::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const Matrix&  xnodes = xnodeCoords_;

  const double   a = 1.0 - u[0];
  const double   b = 1.0 + u[0];


  point_[0] = 0.5 * (a * xnodes(0,0) + b * xnodes(0,1));
  point_[1] = 0.5 * (a * xnodes(1,0) + b * xnodes(1,1));

  sshape_->evalShapeFunctions ( f, point_ );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdEdge::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const Matrix&  xnodes    = xnodeCoords_;
  const Matrix&  sgrads    = sshapeGrads_;

  const idx_t    funcCount = funcCount_;

  const double   a = 1.0 - u[0];
  const double   b = 1.0 + u[0];

  double         xu, yu;


  point_[0] = 0.5 * (a * xnodes(0,0) + b * xnodes(0,1));
  point_[1] = 0.5 * (a * xnodes(1,0) + b * xnodes(1,1));

  xu        = 0.5 * (xnodes(0,1) - xnodes(0,0));
  yu        = 0.5 * (xnodes(1,1) - xnodes(1,0));

  sshape_->evalShapeGradients ( f, sgrads, point_ );

JEM_IVDEP

  for ( idx_t i = 0; i < funcCount; i++ )
  {
    g(0,i) = xu * sgrads(0,i) + yu * sgrads(1,i);
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdEdge::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const Matrix&  xnodes    = xnodeCoords_;
  const Matrix&  sgrads    = sshapeGrads_;
  const Matrix&  sgrads2   = sshapeGrads2_;

  const idx_t    funcCount = funcCount_;

  const double   a = 1.0 - u[0];
  const double   b = 1.0 + u[0];

  double         xu, yu;


  point_[0] = 0.5 * (a * xnodes(0,0) + b * xnodes(0,1));
  point_[1] = 0.5 * (a * xnodes(1,0) + b * xnodes(1,1));

  xu        = 0.5 * (xnodes(0,1) - xnodes(0,0));
  yu        = 0.5 * (xnodes(1,1) - xnodes(1,0));

  sshape_->evalShapeGradGrads ( f, sgrads, sgrads2, point_ );

JEM_IVDEP

  for ( idx_t i = 0; i < funcCount; i++ )
  {
    g(0,i) = xu * sgrads(0,i) + yu * sgrads(1,i);

    h(0,i) =

      xu * xu * sgrads2(0,i) +
      yu * yu * sgrads2(1,i) +
      xu * yu * sgrads2(2,i) * 2.0;
  }
}


//-----------------------------------------------------------------------
//   mapBoundaryPoint
//-----------------------------------------------------------------------


void StdEdge::mapBoundaryPoint

  ( const Vector&  v,
    const Vector&  u ) const

{
  const Matrix&  xnodes = xnodeCoords_;

  const double   a      = 1.0 - u[0];
  const double   b      = 1.0 + u[0];

  v[0] = 0.5 * (a * xnodes(0,0) + b * xnodes(0,1));
  v[1] = 0.5 * (a * xnodes(1,0) + b * xnodes(1,1));
}


JIVE_END_PACKAGE( geom )
