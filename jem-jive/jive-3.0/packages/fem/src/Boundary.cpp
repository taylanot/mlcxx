
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


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jive/fem/Boundary.h>


JEM_DEFINE_CLASS( jive::fem::Boundary );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class Boundary
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Boundary::NEW_SHAPE_  = 1 << 0;
const int  Boundary::NEW_POINTS_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Boundary::Boundary

  ( const BoundarySet&    b,
    const Ref<IElement>&  e ) :

    bounds ( b ),
    elem_  ( e )

{
  if ( ! elem_ )
  {
    elem_ = newInstance<IElement> ( b.getElements() );
  }

  pntType_ = IPOINTS;

  Self::setShape ( elem_->shape.get(), -1 );
}


Boundary::~Boundary ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void Boundary::reset ()
{
  elem_->reset ();

  setShape ( elem_->shape.get(), -1 );
}


//-----------------------------------------------------------------------
//   setShape
//-----------------------------------------------------------------------


void Boundary::setShape ( IShape* s, idx_t iloc )
{
  IElement&  e = * elem_;


  if ( s != e.shape )
  {
    e.setShape ( s );
  }

  ibound    = -1_idx;
  ilocal    = iloc;
  newShape  = true;
  newShape_ = true;

  if ( ilocal >= 0 )
  {
    shape     = s->getBoundaryShape ( ilocal );
    localRank = e.rank - 1;
    nodeCount = shape->nodeCount      ();
    funcCount = shape->shapeFuncCount ();

    inodes.ref ( e.shape->getBoundaryNodes( ilocal ) );
  }
  else
  {
    shape     = nullptr;
    localRank = 0;
    nodeCount = 0;
    funcCount = 0;

    inodes.resize ( 0 );
  }

  xnodes.resize ( e.rank, nodeCount );
  point_.resize ( e.rank );

  setupChanged_ ( NEW_SHAPE_ );
}


//-----------------------------------------------------------------------
//   loadData
//-----------------------------------------------------------------------


void Boundary::loadData ( idx_t ibound, idx_t ielem )
{
  JEM_PRECHECK ( this->ilocal >= 0 );

  loadDataFast ( ibound, ielem );
}


//-----------------------------------------------------------------------
//   setLocalPoints
//-----------------------------------------------------------------------


void Boundary::setLocalPoints ( const Matrix& coords )
{
  JEM_PRECHECK ( coords.size(0) == localRank );

  if ( pntType_ != CUSTOM )
  {
    pntType_ = CUSTOM;

    setupChanged_ ( NEW_POINTS_ );
  }

  pntCount = coords.size (1);

  upoints.ref ( coords );
}


//-----------------------------------------------------------------------
//   setGlobalPoints
//-----------------------------------------------------------------------


void Boundary::setGlobalPoints

  ( const Matrix&  coords,
    double         eps )

{
  JEM_PRECHECK ( this->ibound >= 0 );

  if ( pntType_ != CUSTOM )
  {
    pntType_ = CUSTOM;

    setupChanged_ ( NEW_POINTS_ );
  }

  pntCount = coords.size (1);

  if ( upoints.size(1) != pntCount )
  {
    upoints.resize ( localRank, pntCount );
  }

  xpoints.ref ( coords );

  for ( idx_t i = 0; i < pntCount; i++ )
  {
    bool  found = shape->getLocalPoint ( upoints[i],
                                         xpoints[i],
                                         eps,
                                         xnodes );

    if ( ! found )
    {
      idx_t  bndID = bounds.getBoundaryID ( ibound );

      throw jem::IllegalInputException (
        JEM_FUNC,
        String::format (
          "unable to find global points on boundary %d",
          bndID
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   getGlobalPoints
//-----------------------------------------------------------------------


const Matrix&  Boundary::getGlobalPoints ()
{
  JEM_PRECHECK ( this->ibound >= 0 );

  if ( pntType_ == IPOINTS )
  {
    shape->getGlobalIntegrationPoints ( xpoints, xnodes );
  }
  else
  {
    if ( xpoints.size(1) != pntCount )
    {
      xpoints.resize ( elem_->rank, pntCount );
    }

    for ( idx_t i = 0; i < pntCount; i++ )
    {
      shape->getGlobalPoint ( xpoints[i], upoints[i], xnodes );
    }
  }

  return xpoints;
}


//-----------------------------------------------------------------------
//   getPointWeights
//-----------------------------------------------------------------------


const Vector&  Boundary::getPointWeights ()
{
  JEM_PRECHECK ( this->ibound >= 0 );

  if ( pntType_ == IPOINTS )
  {
    shape->getIntegrationWeights ( pntWeights, xnodes );
  }
  else
  {
    if ( pntWeights.size() != pntCount )
    {
      pntWeights.resize ( pntCount );
    }

    pntWeights = 0.0;
  }

  return pntWeights;
}


//-----------------------------------------------------------------------
//   getNormals
//-----------------------------------------------------------------------


const Matrix&  Boundary::getNormals ()
{
  JEM_PRECHECK ( this->ibound >= 0 );

  if      ( pntType_ == IPOINTS )
  {
    shape->getNormals       ( normals, pntWeights, xnodes );
  }
  else if ( pntType_ == VERTICES )
  {
    shape->getVertexNormals ( normals, xnodes );
  }
  else
  {
    if ( normals.size(1) != pntCount )
    {
      normals.resize ( elem_->rank, pntCount );
    }

    for ( idx_t i = 0; i < pntCount; i++ )
    {
      shape->evalNormal ( normals[i], upoints[i], xnodes );
    }
  }

  return normals;
}


//-----------------------------------------------------------------------
//   getShapeFuncs
//-----------------------------------------------------------------------


const Matrix&  Boundary::getShapeFuncs ()
{
  JEM_PRECHECK ( this->ibound >= 0 );

  if      ( pntType_ == IPOINTS )
  {
    shapeFuncs = shape->getShapeFunctions ();
  }
  else if ( pntType_ == VERTICES )
  {
    shapeFuncs = shape->getVertexFunctions ();
  }
  else if ( pntType_ == CUSTOM )
  {
    if ( shapeFuncs.size(1) != pntCount )
    {
      shapeFuncs.resize ( funcCount, pntCount );
    }

    for ( idx_t i = 0; i < pntCount; i++ )
    {
      shape->evalShapeFunctions ( shapeFuncs[i], upoints[i] );
    }
  }

  return shapeFuncs;
}


//-----------------------------------------------------------------------
//   getShapeGrads
//-----------------------------------------------------------------------


const Cubix&  Boundary::getShapeGrads ()
{
  JEM_PRECHECK ( this->ibound >= 0 );

  IElement&  e = * elem_;

  if      ( pntType_ == IPOINTS )
  {
    e.shape->getBoundaryGradients ( shapeGrads,
                                    ilocal,
                                    e.xnodes );
  }
  else
  {
    if ( shapeGrads.size(2) != pntCount )
    {
      shapeGrads.resize ( elem_->rank, funcCount, pntCount );
    }

    for ( idx_t i = 0; i < pntCount; i++ )
    {
      e.shape->mapBoundaryPoint   ( point_, ilocal, upoints[i] );
      e.shape->evalShapeGradients ( shapeGrads[i], point_, e.xnodes );
    }
  }

  return shapeGrads;
}


//-----------------------------------------------------------------------
//   setupChanged_
//-----------------------------------------------------------------------


void Boundary::setupChanged_ ( int flags )
{
  IElement&  e = * elem_;

  if      ( pntType_ == CUSTOM || ilocal < 0 )
  {
    pntCount = 0;

    // This is necessary to make sure that these arrays do not share
    // data with other arrays.

    upoints.resize ( localRank, pntCount );
  }
  else if ( pntType_ == IPOINTS )
  {
    upoints.ref ( shape->getIntegrationScheme() (slice(1,END),ALL) );

    pntCount = upoints.size (1);
  }
  else if ( pntType_ == VERTICES )
  {
    upoints.ref ( shape->getVertexCoords() );

    pntCount = upoints.size (1);
  }

  xpoints   .resize ( e.rank,            pntCount );
  shapeFuncs.resize (         funcCount, pntCount );
  pntWeights.resize (                    pntCount );
  normals   .resize ( e.rank,            pntCount );
  shapeGrads.resize ( e.rank, funcCount, pntCount );
}


JIVE_END_PACKAGE( fem )
