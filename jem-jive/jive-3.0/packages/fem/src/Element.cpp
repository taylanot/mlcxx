
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
#include <jem/base/RuntimeException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Event.h>
#include <jive/fem/Element.h>


JEM_DEFINE_CLASS( jive::fem::Element );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class Element
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Element::NEW_SHAPE_  = 1 << 0;
const int  Element::NEW_POINTS_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Element::Element

  ( const ElementSet&  e,
    const Ref<Shape>&  s ) :

    elems ( e ),
    nodes ( e.getNodes() )

{
  using jem::util::connect;

  rank     = nodes.rank ();
  pntType_ = IPOINTS;

  Self::setShape ( s.get() );

  connect ( nodes.newRankEvent(), this, & Self::rankChanged_ );
}


Element::~Element ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void Element::reset ()
{
  setShape ( 0 );
}


//-----------------------------------------------------------------------
//   setShape
//-----------------------------------------------------------------------


void Element::setShape ( Shape* s )
{
  ielem     = -1_idx;
  shape     = s;

  newShape  = true;
  newShape_ = true;

  if ( s )
  {
    if ( s->globalRank() != rank && nodes.size() )
    {
      shapeRankError_ ();
    }

    localRank = s->localRank      ();
    nodeCount = s->nodeCount      ();
    funcCount = s->shapeFuncCount ();
  }
  else
  {
    localRank = 0;
    nodeCount = 0;
    funcCount = 0;
  }

  inodes.resize (       nodeCount );
  xnodes.resize ( rank, nodeCount );

  setupChanged_ ( NEW_SHAPE_ );
}


//-----------------------------------------------------------------------
//   loadData
//-----------------------------------------------------------------------


void Element::loadData ( idx_t ielem )
{
  JEM_PRECHECK ( this->shape );

  loadDataFast ( ielem );
}


//-----------------------------------------------------------------------
//   setLocalPoints
//-----------------------------------------------------------------------


void Element::setLocalPoints ( const Matrix& coords )
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


void Element::setGlobalPoints

  ( const Matrix&  coords,
    double         eps )

{
  JEM_PRECHECK ( this->ielem >= 0 && coords.size(0) == rank );

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
      idx_t  elemID = elems.getElemID ( ielem );

      throw jem::IllegalInputException (
        JEM_FUNC,
        String::format (
          "unable to find global points within element %d",
          elemID
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   getGlobalPoints
//-----------------------------------------------------------------------


const Matrix& Element::getGlobalPoints ()
{
  JEM_PRECHECK ( this->ielem >= 0 );

  if ( pntType_ == IPOINTS )
  {
    shape->getGlobalIntegrationPoints ( xpoints, xnodes );
  }
  else
  {
    if ( xpoints.size(1) != pntCount )
    {
      xpoints.resize ( rank, pntCount );
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


const Vector& Element::getPointWeights ()
{
  JEM_PRECHECK ( this->ielem >= 0 );

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
//   getShapeFuncs
//-----------------------------------------------------------------------


const Matrix& Element::getShapeFuncs ()
{
  JEM_PRECHECK ( this->ielem >= 0 );

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
//   setupChanged_
//-----------------------------------------------------------------------


void Element::setupChanged_ ( int flags )
{
  if      ( pntType_ == CUSTOM || shape == nullptr )
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

  xpoints   .resize ( rank,      pntCount );
  shapeFuncs.resize ( funcCount, pntCount );
  pntWeights.resize ( pntCount );
}


//-----------------------------------------------------------------------
//   shapeRankError_
//-----------------------------------------------------------------------


void Element::shapeRankError_ () const
{
  String  name = shape->getName ();

  throw jem::RuntimeException (
    CLASS_NAME,
    String::format (
      "rank mismatch: shape `%s\' has rank %d, but mesh has rank %d",
      name,
      shape->globalRank (),
      nodes .rank       ()
    )
  );
}


//-----------------------------------------------------------------------
//   rankChanged_
//-----------------------------------------------------------------------


void Element::rankChanged_ ( idx_t newRank )
{
  if ( shape                               &&
       newRank      != shape->globalRank() &&
       nodes.size() >  0 )
  {
    shapeRankError_ ();
  }

  rank = newRank;

  nodes.resetEvents ();
}


JIVE_END_PACKAGE( fem )
