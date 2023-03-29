
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
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/geom/utilities.h>
#include <jive/geom/StdBoundary.h>


JEM_DEFINE_SERIABLE_CLASS ( jive::geom::StdBoundary );


JIVE_BEGIN_PACKAGE( geom )


using jem::numeric::matmul;


//=======================================================================
//   class StdBoundary
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdBoundary::StdBoundary () :

  rank_      ( -1 ),
  nodeCount_ ( -1 ),
  funcCount_ ( -1 )

{}


StdBoundary::StdBoundary

  ( const Matrix&         xnodes,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    xshape_      ( xshape ),
    sshape_      ( sshape ),
    xnodeCoords_ ( xnodes )

{
  JEM_PRECHECK2 ( xshape && sshape, "NULL shape" );

  JEM_PRECHECK2 ( xnodes.size(0) == sshape->rank()      &&
                  xnodes.size(0) == xshape->rank() + 1  &&
                  xnodes.size(1) == xshape->shapeFuncCount(),
                  "Array shape mismatch" );

  init_ ();
}


StdBoundary::StdBoundary

  ( const Ref<Self>&  parent,
    const Matrix&     xnodes ) :

    parent_      ( parent ),
    xnodeCoords_ ( xnodes )

{
  JEM_PRECHECK2 ( parent, "NULL parent shape" );

  xshape_ = parent->xshape_;
  sshape_ = parent->sshape_;

  if ( xnodes.size(0) != xshape_->rank() + 1 ||
       xnodes.size(1) != xshape_->shapeFuncCount() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "coordinate array has wrong shape: (%d,%d); "
        "should be (%d,%d)",
        xnodes.size(0),
        xnodes.size(1),
        xshape_->rank() + 1,
        xshape_->shapeFuncCount ()
      )
    );
  }

  init_ ();
}


StdBoundary::~StdBoundary ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdBoundary::readFrom ( ObjectInput& in )
{
  decode ( in, parent_, xshape_, sshape_, xnodeCoords_ );
  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdBoundary::writeTo ( ObjectOutput& out ) const
{
  encode ( out, parent_, xshape_, sshape_, xnodeCoords_ );
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdBoundary::rank () const
{
  return rank_;
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdBoundary::shapeFuncCount () const
{
  return funcCount_;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdBoundary::getGeometry () const
{
  return xshape_->getGeometry ();
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdBoundary::vertexCount () const
{
  return xshape_->vertexCount ();
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdBoundary::getVertexCoords () const
{
  return xshape_->getVertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoint
//-----------------------------------------------------------------------


bool StdBoundary::containsPoint ( const Vector& u ) const
{
  return xshape_->containsPoint ( u );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdBoundary::clipPoint ( const Vector& u ) const
{
  return xshape_->clipPoint ( u );
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdBoundary::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const Matrix&  xnodes    = xnodeCoords_;
  const Vector&  xfuncs    = xshapeFuncs_;
  const Vector&  point     = point_;

  const idx_t    rank      = rank_;
  const idx_t    nodeCount = nodeCount_;

  double         x, y, z;
  double         xf;


  if      ( rank == 0_idx )
  {
    point[0] = xnodes(0,0);

    sshape_->evalShapeFunctions ( f, point );
  }
  else if ( rank == 1_idx )
  {
    xshape_->evalShapeFunctions ( xfuncs, u );

    x = y = 0.0;

    for ( idx_t j = 0; j < nodeCount; j++ )
    {
      xf = xfuncs[j];

      x += xf * xnodes(0,j);
      y += xf * xnodes(1,j);
    }

    point[0] = x;
    point[1] = y;

    sshape_->evalShapeFunctions ( f, point );
  }
  else if ( rank == 2_idx )
  {
    xshape_->evalShapeFunctions ( xfuncs, u );

    x = y = z = 0.0;

    for ( idx_t j = 0; j < nodeCount; j++ )
    {
      xf = xfuncs[j];

      x += xf * xnodes(0,j);
      y += xf * xnodes(1,j);
      z += xf * xnodes(2,j);
    }

    point[0] = x;
    point[1] = y;
    point[2] = z;

    sshape_->evalShapeFunctions ( f, point );
  }
  else
  {
    xshape_->evalShapeFunctions ( xfuncs, u );

    matmul ( point, xnodes, xfuncs );

    sshape_->evalShapeFunctions ( f, point );
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdBoundary::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const Matrix&  xnodes    = xnodeCoords_;
  const Vector&  xfuncs    = xshapeFuncs_;
  const Matrix&  xgrads    = xshapeGrads_;
  const Matrix&  sgrads    = sshapeGrads_;
  const Vector&  point     = point_;
  const Matrix&  ja        = jacobi_;

  const idx_t    rank      = rank_;
  const idx_t    nodeCount = nodeCount_;
  const idx_t    funcCount = funcCount_;

  double         x, y, z;
  double         xu, xv;
  double         yu, yv;
  double         zu, zv;
  double         xf;
  idx_t          i, j, k;


  if      ( rank == 0_idx )
  {
    point[0] = xnodes(0,0);

    sshape_->evalShapeFunctions ( f, point );
  }
  else if ( rank == 1_idx )
  {
    xshape_->evalShapeGradients ( xfuncs, xgrads, u );

    x  =  y = 0.0;
    xu = yu = 0.0;

    for ( j = 0; j < nodeCount; j++ )
    {
      xf  = xfuncs[j];

      x  += xf          * xnodes(0,j);
      xu += xgrads(0,j) * xnodes(0,j);

      y  += xf          * xnodes(1,j);
      yu += xgrads(0,j) * xnodes(1,j);
    }

    point[0] = x;
    point[1] = y;

    sshape_->evalShapeGradients ( f, sgrads, point );

JEM_IVDEP

    for ( j = 0; j < funcCount; j++ )
    {
      g(0,j) = xu * sgrads(0,j) + yu * sgrads(1,j);
    }
  }
  else if ( rank == 2_idx )
  {
    xshape_->evalShapeGradients ( xfuncs, xgrads, u );

    x  =  y =  z = 0.0;
    xu = xv = yu = yv = zu = zv = 0.0;

    for ( j = 0; j < nodeCount; j++ )
    {
      xf  = xfuncs[j];

      x  += xf          * xnodes(0,j);
      xu += xgrads(0,j) * xnodes(0,j);
      xv += xgrads(1,j) * xnodes(0,j);

      y  += xf          * xnodes(1,j);
      yu += xgrads(0,j) * xnodes(1,j);
      yv += xgrads(1,j) * xnodes(1,j);

      z  += xf          * xnodes(2,j);
      zu += xgrads(0,j) * xnodes(2,j);
      zv += xgrads(1,j) * xnodes(2,j);
    }

    point[0] = x;
    point[1] = y;
    point[2] = z;

    sshape_->evalShapeGradients ( f, sgrads, point );

JEM_IVDEP

    for ( j = 0; j < funcCount; j++ )
    {
      g(0,j) = xu * sgrads(0,j) + yu * sgrads(1,j) +
               zu * sgrads(2,j);

      g(1,j) = xv * sgrads(0,j) + yv * sgrads(1,j) +
               zv * sgrads(2,j);
    }
  }
  else
  {
    xshape_->evalShapeGradients ( xfuncs, xgrads, u );

    for ( k = 0; k <= rank; k++ )
    {
      point[k] = 0.0;

      for ( i = 0; i < rank; i++ )
      {
        ja(i,k) = 0.0;
      }
    }

    for ( j = 0; j < nodeCount; j++ )
    {
      xf = xfuncs[j];

      for ( k = 0; k <= rank; k++ )
      {
        x         = xnodes(k,j);
        point[k] += xf * x;

        for ( i = 0; i < rank; i++ )
        {
          ja(i,k) += xgrads(i,j) * x;
        }
      }
    }

    sshape_->evalShapeGradients ( f, sgrads, point );

    for ( j = 0; j < funcCount; j++ )
    {
      for ( i = 0; i < rank; i++ )
      {
        x = 0.0;

        for ( k = 0; k <= rank; k++ )
        {
          x += ja(i,k) * sgrads(k,j);
        }

        g(i,j) = x;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdBoundary::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const Matrix&  xnodes     = xnodeCoords_;
  const Vector&  xfuncs     = xshapeFuncs_;
  const Matrix&  xgrads     = xshapeGrads_;
  const Matrix&  xgrads2    = xshapeGrads2_;
  const Matrix&  sgrads     = sshapeGrads_;
  const Matrix&  sgrads2    = sshapeGrads2_;
  const Vector&  point      = point_;

  const idx_t    rank       = rank_;
  const idx_t    nodeCount  = nodeCount_;
  const idx_t    funcCount  = funcCount_;


  if ( rank == 0_idx )
  {
    point[0] = xnodes(0,0);

    sshape_->evalShapeFunctions ( f, point );

    return;
  }

  xshape_->evalShapeGradGrads ( xfuncs, xgrads, xgrads2, u );

  if ( rank == 1_idx )
  {
    double  x,  y;
    double  xu, xu2;
    double  yu, yu2;
    double  xf;
    idx_t   j;

    x   = y   = 0.0;
    xu  = yu  = 0.0;
    xu2 = yu2 = 0.0;

    for ( j = 0; j < nodeCount; j++ )
    {
      xf  = xfuncs[j];

      x   += xf           * xnodes(0,j);
      xu  += xgrads (0,j) * xnodes(0,j);
      xu2 += xgrads2(0,j) * xnodes(0,j);

      y   += xf           * xnodes(1,j);
      yu  += xgrads (0,j) * xnodes(1,j);
      yu2 += xgrads2(0,j) * xnodes(1,j);
    }

    point[0] = x;
    point[1] = y;

    sshape_->evalShapeGradGrads ( f, sgrads, sgrads2, point );

JEM_IVDEP

    for ( j = 0; j < funcCount; j++ )
    {
      g(0,j) = xu * sgrads(0,j) + yu * sgrads(1,j);

      h(0,j) =

        xu * xu * sgrads2(0,j)       +
        yu * yu * sgrads2(1,j)       +
        xu * yu * sgrads2(2,j) * 2.0 +
        xu2     * sgrads(0,j)        +
        yu2     * sgrads(1,j);
    }
  }
  else
  {
    const Matrix&  ja     = jacobi_;
    const Matrix&  qu     = qujobi_;
    const Matrix&  ja2    = jacobi2_;

    const idx_t    xrank2 = (rank * rank + rank) / 2;
    const idx_t    srank2 = xrank2 + rank + 1;

    double         x, xf;
    idx_t          i, j, k;


    for ( k = 0; k <= rank; k++ )
    {
      point[k] = 0.0;

      for ( i = 0; i < rank; i++ )
      {
        ja(i,k) = 0.0;
      }

      for ( i = 0; i < xrank2; i++ )
      {
        ja2(i,k) = 0.0;
      }
    }

    for ( j = 0; j < nodeCount; j++ )
    {
      xf = xfuncs[j];

      for ( k = 0; k <= rank; k++ )
      {
        x         = xnodes(k,j);
        point[k] += xf * x;

        for ( i = 0; i < rank; i++ )
        {
          ja(i,k) += xgrads(i,j) * x;
        }

        for ( i = 0; i < xrank2; i++ )
        {
          ja2(i,k) += xgrads2(i,j) * x;
        }
      }
    }

    getQujobi ( qu, ja );

    sshape_->evalShapeGradGrads ( f, sgrads, sgrads2, point );

    for ( j = 0; j < funcCount; j++ )
    {
      for ( i = 0; i < rank; i++ )
      {
        x = 0.0;

        for ( k = 0; k <= rank; k++ )
        {
          x += ja(i,k) * sgrads(k,j);
        }

        g(i,j) = x;
      }

      for ( i = 0; i < xrank2; i++ )
      {
        x = 0.0;

        for ( k = 0; k < srank2; k++ )
        {
          x += qu(i,k) * sgrads2(k,j);
        }

        for ( k = 0; k <= rank; k++ )
        {
          x += ja2(i,k) * sgrads(k,j);
        }

        h(i,j) = x;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* StdBoundary::getExtByID ( ExtensionID id ) const
{
  return sshape_->getExtByID ( id );
}


//-----------------------------------------------------------------------
//   mapBoundaryPoint
//-----------------------------------------------------------------------


void StdBoundary::mapBoundaryPoint

  ( const Vector&  v,
    const Vector&  u ) const

{
  JEM_PRECHECK2 ( v.size() == u.size() + 1, "Array size mismatch" );

  const Matrix&  xnodes    = xnodeCoords_;
  const Vector&  xfuncs    = xshapeFuncs_;

  const idx_t    rank      = rank_;
  const idx_t    nodeCount = nodeCount_;

  double         x, y, z;
  double         xf;


  if      ( rank == 0_idx )
  {
    v[0] = xnodes(0,0);
  }
  else if ( rank == 1_idx )
  {
    xshape_->evalShapeFunctions ( xfuncs, u );

    x = y = 0.0;

    for ( idx_t j = 0; j < nodeCount; j++ )
    {
      xf = xfuncs[j];

      x += xf * xnodes(0,j);
      y += xf * xnodes(1,j);
    }

    v[0] = x;
    v[1] = y;
  }
  else if ( rank == 2_idx )
  {
    xshape_->evalShapeFunctions ( xfuncs, u );

    x = y = z = 0.0;

    for ( idx_t j = 0; j < nodeCount; j++ )
    {
      xf = xfuncs[j];

      x += xf * xnodes(0,j);
      y += xf * xnodes(1,j);
      z += xf * xnodes(2,j);
    }

    v[0] = x;
    v[1] = y;
    v[2] = z;
  }
  else
  {
    xshape_->evalShapeFunctions ( xfuncs, u );

    matmul ( v, xnodes, xfuncs );
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void StdBoundary::init_ ()
{
  rank_      = xshape_->rank           ();
  nodeCount_ = xshape_->shapeFuncCount ();
  funcCount_ = sshape_->shapeFuncCount ();

  if ( parent_ )
  {
    xshapeFuncs_ .ref ( parent_->xshapeFuncs_  );
    xshapeGrads_ .ref ( parent_->xshapeGrads_  );
    xshapeGrads2_.ref ( parent_->xshapeGrads2_ );

    sshapeGrads_ .ref ( parent_->sshapeGrads_  );
    sshapeGrads2_.ref ( parent_->sshapeGrads2_ );

    jacobi_      .ref ( parent_->jacobi_       );
    qujobi_      .ref ( parent_->qujobi_       );
    jacobi2_     .ref ( parent_->jacobi2_      );

    point_       .ref ( parent_->point_        );
  }
  else
  {
    const idx_t  xrank2 = (rank_ * rank_ + rank_) / 2;
    const idx_t  srank2 = xrank2 + rank_ + 1;


    xshapeFuncs_ .resize (            nodeCount_ );
    xshapeGrads_ .resize ( rank_,     nodeCount_ );
    xshapeGrads2_.resize ( xrank2,    nodeCount_ );

    sshapeGrads_ .resize ( rank_ + 1, funcCount_ );
    sshapeGrads2_.resize ( srank2,    funcCount_ );

    jacobi_ .resize      ( rank_,     rank_ + 1  );
    qujobi_ .resize      ( xrank2,    srank2     );
    jacobi2_.resize      ( xrank2,    rank_ + 1  );

    point_  .resize      ( rank_ + 1 );

    if ( rank_ == 0 )
    {
      point_[0] = xnodeCoords_(0,0);
    }
  }

  connect ( sshape_->changeEvent, this, &Self::shapeChanged_ );
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void StdBoundary::shapeChanged_ ()
{
  changeEvent.emit ( *this );
}


JIVE_END_PACKAGE( geom )
