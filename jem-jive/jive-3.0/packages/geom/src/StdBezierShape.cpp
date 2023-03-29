
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
#include <jem/base/Array.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/geom/utilities.h>
#include <jive/geom/Names.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdBezierShape.h>


JEM_DEFINE_SERIABLE_CLASS ( jive::geom::StdBezierShape );


JIVE_BEGIN_PACKAGE( geom )


using jem::ArithmeticException;
using jem::numeric::matmul;


//=======================================================================
//   class StdBezierShape
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdBezierShape::StdBezierShape ()
{
  isRational_ = false;
}


StdBezierShape::StdBezierShape ( const Ref<StdShape>& basis )
{
  JEM_PRECHECK2 ( basis, "NULL basis functions" );

  const idx_t  n = basis->shapeFuncCount ();

  isRational_      = false;
  extractor_.shape = n;

  init_ ( basis );
}


StdBezierShape::StdBezierShape

  ( const Ref<StdShape>&  basis,
    idx_t                 sfCount )

{
  JEM_PRECHECK2 ( basis,
                  "NULL basis functions" );
  JEM_PRECHECK2 ( sfCount >= 0,
                  "invalid number of Bezier shape functions" );

  isRational_      = false;
  extractor_.shape = jem::shape ( sfCount,
                                  basis->shapeFuncCount() );

  init_ ( basis );
}

StdBezierShape::StdBezierShape

  ( const Ref<StdShape>&    basis,
    const BezierExtractor&  bez )

{
  JEM_PRECHECK2 ( basis,
                  "NULL basis functions" );
  JEM_PRECHECK2 ( bez.shape[1] == basis->shapeFuncCount(),
                  "Bezier extraction operator shape mismatch" );

  extractor_  = bez;
  isRational_ = false;

  init_ ( basis );
}


StdBezierShape::StdBezierShape

  ( const Ref<StdShape>&    basis,
    const BezierExtractor&  bez,
    const Vector&           wgt )

{
  JEM_PRECHECK2 ( basis,
                  "NULL basis functions" );
  JEM_PRECHECK2 ( bez.shape[0] == wgt.size() &&
                  bez.shape[1] == basis->shapeFuncCount(),
                  "Bezier extraction operator shape mismatch" );

  weights_.ref ( wgt );

  extractor_  = bez;
  isRational_ = true;

  init_ ( basis );
}


StdBezierShape::~StdBezierShape ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdBezierShape::readFrom ( ObjectInput& in )
{
  Ref<StdShape>  basis;

  decode ( in, basis, weights_, extractor_, isRational_ );
  init_  ( basis );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdBezierShape::writeTo ( ObjectOutput& out ) const
{
  encode ( out, basis_, weights_, extractor_, isRational_ );
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdBezierShape::rank () const
{
  return basis_->rank ();
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdBezierShape::shapeFuncCount () const
{
  return extractor_.shape[0];
}


//-----------------------------------------------------------------------
//   basisFuncCount
//-----------------------------------------------------------------------


idx_t StdBezierShape::basisFuncCount () const
{
  return extractor_.shape[1];
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdBezierShape::getGeometry () const
{
  return basis_->getGeometry ();
}


//-----------------------------------------------------------------------
//   isLinear
//-----------------------------------------------------------------------


bool StdBezierShape::isLinear () const
{
  if ( isRational_ )
  {
    return false;
  }
  else
  {
    return basis_->isLinear ();
  }
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdBezierShape::vertexCount () const
{
  return basis_->vertexCount ();
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdBezierShape::getVertexCoords () const
{
  return basis_->getVertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoint
//-----------------------------------------------------------------------


bool StdBezierShape::containsPoint ( const Vector& u ) const
{
  return basis_->containsPoint ( u );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdBezierShape::clipPoint ( const Vector& u ) const
{
  return basis_->clipPoint ( u );
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdBezierShape::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  basis_->evalShapeFunctions ( shapeFuncs_, u );

  matmul ( f, extractor_, shapeFuncs_ );

  if ( isRational_ )
  {
    applyWeights_ ( f );
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdBezierShape::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  basis_->evalShapeGradients ( shapeFuncs_, shapeGrads_, u );

  matmul ( f, extractor_, shapeFuncs_ );
  matmul ( g, extractor_, shapeGrads_ );

  if ( isRational_ )
  {
    applyWeights_ ( f, g );
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdBezierShape::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  basis_->evalShapeGradGrads ( shapeFuncs_,
                               shapeGrads_, shapeGrads2_, u );

  matmul ( f, extractor_, shapeFuncs_ );
  matmul ( g, extractor_, shapeGrads_ );
  matmul ( h, extractor_, shapeGrads2_ );

  if ( isRational_ )
  {
    applyWeights_ ( f, g, h );
  }
}


//-----------------------------------------------------------------------
//   setBezierExtractor
//-----------------------------------------------------------------------


void StdBezierShape::setBezierExtractor

  ( const BezierExtractor&  bez )

{
  if ( bez.shape[0] != extractor_.shape[0] ||
       bez.shape[1] != extractor_.shape[1] )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "Bezier extraction operator has wrong shape: (%d,%d); "
        "should be (%d,%d)",
        bez.shape[0],
        bez.shape[1],
        extractor_.shape[0],
        extractor_.shape[1]
      )
    );
  }

  extractor_ = bez;

  if ( isRational_ )
  {
    weights_.ref ( Vector() );

    isRational_ = false;
  }

  changeEvent.emit ( *this );
}


void StdBezierShape::setBezierExtractor

  ( const BezierExtractor&  bez,
    const Vector&           wgt )

{
  if ( bez.shape[0] != extractor_.shape[0] ||
       bez.shape[1] != extractor_.shape[1] )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "Bezier extraction operator has wrong shape: (%d,%d); "
        "should be (%d,%d)",
        bez.shape[0],
        bez.shape[1],
        extractor_.shape[0],
        extractor_.shape[1]
      )
    );
  }

  if ( wgt.size() != bez.shape[0] )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "Bezier weight vector has wrong length: %d; "
        "should be %d",
        wgt.size (),
        bez.shape[0]
      )
    );
  }

  extractor_  = bez;

  if ( ! isRational_ )
  {
    gradWeights_ .resize ( shapeGrads_ .size(0) );
    grad2Weights_.resize ( shapeGrads2_.size(0) );

    isRational_ = true;
  }

  weights_.ref ( wgt );

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* StdBezierShape::getExtByID ( ExtensionID id ) const
{
  if ( id == BezierExtension::ID )
  {
    BezierExtension*  ext = const_cast<Self*> ( this );

    return ext;
  }
  else
  {
    return basis_->getExtByID ( id );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdBezierShape::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  Ref<StdShape>    basis = StdShapeFactory::newInstance (
    geom,
    conf.makeProps ( PropNames::BASIS ),
    props.getProps ( PropNames::BASIS )
  );

  BezierExtractor  bez;
  Vector           wgt;

  if ( props.contains( PropNames::EXTRACTOR ) )
  {
    bez.configure ( props.getProps( PropNames::EXTRACTOR ) );
  }
  else
  {
    bez.setIdentity ( basis->shapeFuncCount() );
  }

  bez.getConfig ( conf.makeProps( PropNames::EXTRACTOR ) );

  if ( bez.shape[1] != basis->shapeFuncCount() )
  {
    props.propertyError (
      PropNames::EXTRACTOR,
      String::format (
        "Bezier extraction operator has wrong shape: (%d,%d); "
        "should be (N,%d)",
        bez.shape[0],
        bez.shape[1],
        basis->shapeFuncCount()
      )
    );
  }

  if ( props.find( wgt, PropNames::WEIGHTS ) )
  {
    if ( wgt.size() != bez.shape[0] )
    {
      props.propertyError (
        PropNames::WEIGHTS,
        String::format (
          "non-rational weight array has wrong length: %d; "
          "should have length %d",
          wgt.size (),
          bez.shape[0]
        )
      );
    }

    conf.set ( PropNames::WEIGHTS, wgt );

    return jem::newInstance<Self> ( basis, bez, wgt );
  }
  else
  {
    return jem::newInstance<Self> ( basis, bez );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdBezierShape::declare ()
{
  StdShapeFactory::declare ( "Bezier", "*", & makeNew );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void StdBezierShape::init_ ( const Ref<StdShape>& basis )
{
  using jem::util::connect;

  const idx_t  rank      = basis->rank ();
  const idx_t  rank2     = (rank * rank + rank) / 2;
  const idx_t  funcCount = basis->shapeFuncCount ();

  basis_ = basis;

  shapeFuncs_ .resize ( funcCount );
  shapeGrads_ .resize ( rank,  funcCount );
  shapeGrads2_.resize ( rank2, funcCount );

  if ( isRational_ )
  {
    gradWeights_ .resize ( rank );
    grad2Weights_.resize ( rank2 );
  }

  connect ( basis->changeEvent, this, &Self::shapeChanged_ );
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void StdBezierShape::shapeChanged_ ()
{
  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   applyWeights_
//-----------------------------------------------------------------------


double StdBezierShape::applyWeights_

  ( const Vector&  f ) const

{
  double  w = dot ( f, weights_ );

  if ( w <= 0.0 )
  {
    throw ArithmeticException (
      JEM_FUNC,
      "non-positive weight in the evaluation of rational "
      "Bezier functions"
    );
  }

  w  = 1.0 / w;
  f *= weights_ * w;

  return w;
}


void StdBezierShape::applyWeights_

  ( const Vector&  f,
    const Matrix&  g ) const

{
  const idx_t  rank      = g.size (0);
  const idx_t  funcCount = g.size (1);

  double       gw0, gw1, gw2;
  double       w;


  w = applyWeights_ ( f );

  matmul ( gradWeights_, g, weights_ );

  if     ( rank == 1_idx )
  {
    gw0 = w * gradWeights_[0];

    for ( idx_t k = 0; k < funcCount; k++ )
    {
      g(0,k) = w * weights_[k] * g(0,k) - f[k] * gw0;
    }
  }
  else if ( rank == 2_idx )
  {
    gw0 = w * gradWeights_[0];
    gw1 = w * gradWeights_[1];

    for ( idx_t k = 0; k < funcCount; k++ )
    {
      double  wk = w * weights_[k];
      double  fk = f[k];

      g(0,k) = wk * g(0,k) - fk * gw0;
      g(1,k) = wk * g(1,k) - fk * gw1;
    }
  }
  else if ( rank == 3_idx )
  {
    gw0 = w * gradWeights_[0];
    gw1 = w * gradWeights_[1];
    gw2 = w * gradWeights_[2];

    for ( idx_t k = 0; k < funcCount; k++ )
    {
      double  wk = w * weights_[k];
      double  fk = f[k];

      g(0,k) = wk * g(0,k) - fk * gw0;
      g(1,k) = wk * g(1,k) - fk * gw1;
      g(2,k) = wk * g(2,k) - fk * gw2;
    }
  }
  else if ( rank >= 4_idx )
  {
    gradWeights_ *= w;

    for ( idx_t k = 0; k < funcCount; k++ )
    {
      double  wk = w * weights_[k];
      double  fk = f[k];

      for ( idx_t i = 0; i < rank; i++ )
      {
        g(i,k) = wk * g(i,k) - fk * gradWeights_[i];
      }
    }
  }
}


void StdBezierShape::applyWeights_

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h ) const

{
  const idx_t  rank      = g.size (0);
  const idx_t  funcCount = g.size (1);

  double       hw0, hw1, hw2, hw3, hw4, hw5;
  double       gw0, gw1, gw2;
  double       w;


  w = applyWeights_ ( f );

  matmul ( gradWeights_,  g, weights_ );
  matmul ( grad2Weights_, h, weights_ );

  if      ( rank == 1_idx )
  {
    gw0 = w * gradWeights_ [0];
    hw0 = w * grad2Weights_[0];

    for ( idx_t k = 0; k < funcCount; k++ )
    {
      double  wk  = w * weights_[k];
      double  fk  = f[k];
      double  gk0 = g(0,k);

      h(0,k) = wk * (h(0,k) - 2.0 * gk0 * gw0) -
               fk * (hw0    - 2.0 * gw0 * gw0);

      g(0,k) = wk * gk0 - fk * gw0;
    }
  }
  else if ( rank == 2_idx )
  {
    gw0 = w * gradWeights_ [0];
    gw1 = w * gradWeights_ [1];
    hw0 = w * grad2Weights_[0];
    hw1 = w * grad2Weights_[1];
    hw2 = w * grad2Weights_[2];

    for ( idx_t k = 0; k < funcCount; k++ )
    {
      double  wk  = w * weights_[k];
      double  fk  = f[k];
      double  gk0 = g(0,k);
      double  gk1 = g(1,k);

      h(0,k) = wk * (h(0,k) - 2.0 * gk0 * gw0) -
               fk * (hw0    - 2.0 * gw0 * gw0);
      h(1,k) = wk * (h(1,k) - 2.0 * gk1 * gw1) -
               fk * (hw1    - 2.0 * gw1 * gw1);
      h(2,k) = wk * (h(2,k) -       gk0 * gw1 - gk1 * gw0) -
               fk * (hw2    - 2.0 * gw0 * gw1);

      g(0,k) = wk * gk0 - fk * gw0;
      g(1,k) = wk * gk1 - fk * gw1;
    }
  }
  else if ( rank == 3_idx )
  {
    gw0 = w * gradWeights_ [0];
    gw1 = w * gradWeights_ [1];
    gw2 = w * gradWeights_ [2];
    hw0 = w * grad2Weights_[0];
    hw1 = w * grad2Weights_[1];
    hw2 = w * grad2Weights_[2];
    hw3 = w * grad2Weights_[3];
    hw4 = w * grad2Weights_[4];
    hw5 = w * grad2Weights_[5];

    for ( idx_t k = 0; k < funcCount; k++ )
    {
      double  wk  = w * weights_[k];
      double  fk  = f[k];
      double  gk0 = g(0,k);
      double  gk1 = g(1,k);
      double  gk2 = g(2,k);

      h(0,k) = wk * (h(0,k) - 2.0 * gk0 * gw0) -
               fk * (hw0    - 2.0 * gw0 * gw0);
      h(1,k) = wk * (h(1,k) - 2.0 * gk1 * gw1) -
               fk * (hw1    - 2.0 * gw1 * gw1);
      h(2,k) = wk * (h(2,k) - 2.0 * gk2 * gw2) -
               fk * (hw2    - 2.0 * gw2 * gw2);
      h(3,k) = wk * (h(3,k) -       gk0 * gw1 - gk1 * gw0) -
               fk * (hw3    - 2.0 * gw0 * gw1);
      h(4,k) = wk * (h(4,k) -       gk0 * gw2 - gk2 * gw0) -
               fk * (hw4    - 2.0 * gw0 * gw2);
      h(5,k) = wk * (h(5,k) -       gk1 * gw2 - gk2 * gw1) -
               fk * (hw5    - 2.0 * gw1 * gw2);

      g(0,k) = wk * gk0 - fk * gw0;
      g(1,k) = wk * gk1 - fk * gw1;
      g(2,k) = wk * gk2 - fk * gw2;
    }
  }
  else if ( rank >= 4_idx )
  {
    gradWeights_  *= w;
    grad2Weights_ *= w;

    for ( idx_t k = 0; k < funcCount; k++ )
    {
      double  wk = w * weights_[k];
      double  fk = f[k];
      idx_t   p;

      for ( idx_t i = 0; i < rank; i++ )
      {
        double  gwi = gradWeights_[i];

        h(i,k) = wk * (h(i,k) - 2.0 * g(i,k) * gwi) -
                 fk * (grad2Weights_[i] - 2.0 * gwi * gwi);
      }

      p = rank;

      for ( idx_t i = 0; i < rank; i++ )
      {
        for ( idx_t j = i + 1; j < rank; j++, p++ )
        {
          double  gwi = gradWeights_[i];
          double  gwj = gradWeights_[j];

          h(p,k) = wk * (h(p,k) - g(i,k) * gwj - g(j,k) * gwi) -
                   fk * (grad2Weights_[p] - 2.0 * gwi * gwj);
        }
      }

      for ( idx_t i = 0; i < rank; i++ )
      {
        g(i,k) = wk * g(i,k) - fk * gradWeights_[i];
      }
    }
  }
}


JIVE_END_PACKAGE( geom )
