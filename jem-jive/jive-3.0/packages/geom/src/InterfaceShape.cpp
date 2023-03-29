
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
#include <jem/base/array/operators.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/geom/InterfaceShape.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::InterfaceShape );


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class InterfaceShape
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


InterfaceShape::InterfaceShape ()
{}


InterfaceShape::InterfaceShape

  ( const Ref<BShape>&  bshape ) :

    bshape_ ( bshape )

{
  JEM_PRECHECK2 ( bshape, "NULL boundary shape object" );

  myName_ = bshape->getName ();

  init_ ();
}


InterfaceShape::InterfaceShape

  ( const String&       name,
    const Ref<BShape>&  bshape ) :

    Super   ( name   ),
    bshape_ ( bshape )

{
  JEM_PRECHECK2 ( bshape, "NULL boundary shape object" );

  init_ ();
}


InterfaceShape::~InterfaceShape ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void InterfaceShape::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, bshape_ );
  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void InterfaceShape::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, bshape_ );
}


//-----------------------------------------------------------------------
//   globalRank
//-----------------------------------------------------------------------


idx_t InterfaceShape::globalRank () const
{
  return midCoords_.size (0);
}


//-----------------------------------------------------------------------
//   localRank
//-----------------------------------------------------------------------


idx_t InterfaceShape::localRank () const
{
  return midCoords_.size(0) - 1;
}

//-----------------------------------------------------------------------
//   nodeCount
//-----------------------------------------------------------------------


idx_t InterfaceShape::nodeCount () const
{
  return 2 * midCoords_.size (1);
}


//-----------------------------------------------------------------------
//   paramCount
//-----------------------------------------------------------------------


idx_t InterfaceShape::vertexCount () const
{
  return bshape_->vertexCount ();
}


//-----------------------------------------------------------------------
//   ipointCount
//-----------------------------------------------------------------------


idx_t InterfaceShape::ipointCount () const
{
  return bshape_->ipointCount ();
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t InterfaceShape::shapeFuncCount () const
{
  return bshape_->shapeFuncCount ();
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String InterfaceShape::getGeometry () const
{
  return bshape_->getGeometry ();
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix InterfaceShape::getVertexCoords () const
{
  return bshape_->getVertexCoords ();
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix InterfaceShape::getIntegrationScheme () const
{
  return bshape_->getIntegrationScheme ();
}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void InterfaceShape::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c ) const

{
  bshape_->getGlobalIntegrationPoints ( x, getMidCoords( c ) );
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void InterfaceShape::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  bshape_->getIntegrationWeights ( w, getMidCoords( c ) );
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void InterfaceShape::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  bshape_->getGlobalPoint ( x, u, getMidCoords( c ) );
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool InterfaceShape::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  return bshape_->getLocalPoint ( u, x, eps, getMidCoords( c ) );
}


//-----------------------------------------------------------------------
//   containsLocalPoint
//-----------------------------------------------------------------------


bool InterfaceShape::containsLocalPoint ( const Vector& u ) const
{
  return bshape_->containsLocalPoint ( u );
}


//-----------------------------------------------------------------------
//   getShapeFunctions
//-----------------------------------------------------------------------


Matrix InterfaceShape::getShapeFunctions () const
{
  return bshape_->getShapeFunctions ();
}


//-----------------------------------------------------------------------
//   getVertexFunctions
//-----------------------------------------------------------------------


Matrix InterfaceShape::getVertexFunctions () const
{
  return bshape_->getVertexFunctions ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void InterfaceShape::evalShapeFunctions

  ( const Vector&  h,
    const Vector&  u ) const

{
  bshape_->evalShapeFunctions ( h, u );
}


//-----------------------------------------------------------------------
//   evalNormal
//-----------------------------------------------------------------------


void InterfaceShape::evalNormal

  ( const Vector&  q,
    const Vector&  u,
    const Matrix&  c ) const

{
  bshape_->evalNormal ( q, u, getMidCoords( c ) );
}


//-----------------------------------------------------------------------
//   getNormals
//-----------------------------------------------------------------------


void InterfaceShape::getNormals

  ( const Matrix&  q,
    const Vector&  w,
    const Matrix&  c ) const

{
  bshape_->getNormals ( q, w, getMidCoords( c ) );
}


//-----------------------------------------------------------------------
//   getVertexNormals
//-----------------------------------------------------------------------


void InterfaceShape::getVertexNormals

  ( const Matrix&  q,
    const Matrix&  c ) const

{
  bshape_->getVertexNormals ( q, getMidCoords( c ) );
}


//-----------------------------------------------------------------------
//   getMidCoords
//-----------------------------------------------------------------------


void InterfaceShape::getMidCoords

  ( const Matrix&  x,
    const Matrix&  c ) const

{
  JEM_ASSERT2 ( c.size(0) == x.size(0) &&
                c.size(1) == x.size(1) * 2,
                "Array shape mismatch" );

  if ( x.isContiguous() && c.isContiguous() )
  {
    const idx_t     n = x.size(0) * x.size(1);

    const double* JEM_RESTRICT  cp0 = c.addr ();
    const double* JEM_RESTRICT  cp1 = cp0 + n;
    double*       JEM_RESTRICT  xp  = x.addr ();

    idx_t         i;

    for ( i = 0; i < n; i++ )
    {
      xp[i] = 0.5 * (cp0[i] + cp1[i]);
    }
  }
  else
  {
    const idx_t  n = x.size (1);

    x = 0.5 * ( c[slice(BEGIN,n)] + c[slice(n,END)] );
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* InterfaceShape::getExtByID ( ExtensionID extID ) const
{
  return bshape_->getExtByID ( extID );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void InterfaceShape::init_ ()
{
  using jem::util::connect;

  midCoords_.resize ( bshape_->globalRank(),
                      bshape_->nodeCount () );

  connect ( bshape_->changeEvent, this, & Self::shapeChanged_ );
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void InterfaceShape::shapeChanged_ ()
{
  changeEvent.emit ( *this );
}


JIVE_END_PACKAGE( geom )
