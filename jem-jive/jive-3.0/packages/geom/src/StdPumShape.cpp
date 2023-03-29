
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
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/geom/utilities.h>
#include <jive/geom/Names.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/StdPumShape.h>


JEM_DEFINE_SERIABLE_CLASS ( jive::geom::StdPumShape );


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jem::IllegalArgumentException;


//=======================================================================
//   class StdPumShape::Basis_
//=======================================================================


class StdPumShape::Basis_ : public Collectable
{
 public:

  explicit                Basis_

    ( const Ref<StdShape>&  sh );

  inline void             evalFunctions

    ( const Vector&         u )      const;

  inline void             evalGradients

    ( const Vector&         u )      const;

  inline void             evalGradGrads

    ( const Vector&         u )      const;


 public:

  Ref<StdShape>           shape;

  Vector                  shapeFuncs;
  Matrix                  shapeGrads;
  Matrix                  shapeGrads2;

  idx_t                   order;
  idx_t                   offset;
  idx_t                   funcCount;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


StdPumShape::Basis_::Basis_ ( const Ref<StdShape>& sh ) :

  shape ( sh )

{
  const idx_t  rank  = shape->rank ();
  const idx_t  rank2 = (rank * rank + rank) / 2;

  order     = 1;
  offset    = 0;
  funcCount = shape->shapeFuncCount ();

  shapeFuncs .resize (        funcCount );
  shapeGrads .resize ( rank,  funcCount );
  shapeGrads2.resize ( rank2, funcCount );
}


//-----------------------------------------------------------------------
//   evalFunctions
//-----------------------------------------------------------------------


inline void StdPumShape::Basis_::evalFunctions

  ( const Vector&  u ) const

{
  shape->evalShapeFunctions ( shapeFuncs, u );
}


//-----------------------------------------------------------------------
//   evalGradients
//-----------------------------------------------------------------------


inline void StdPumShape::Basis_::evalGradients

  ( const Vector&  u ) const

{
  shape->evalShapeGradients ( shapeFuncs, shapeGrads, u );
}


//-----------------------------------------------------------------------
//   evalGradGrads
//-----------------------------------------------------------------------


inline void StdPumShape::Basis_::evalGradGrads

  ( const Vector&  u ) const

{
  shape->evalShapeGradGrads ( shapeFuncs,
                              shapeGrads, shapeGrads2, u );
}


//=======================================================================
//   class StdPumShape
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdPumShape::StdPumShape () :

  rank_      ( -1 ),
  order_     ( -1 ),
  funcCount_ ( -1 )

{}


StdPumShape::StdPumShape ( const SShapeVector& shapes )
{
  JEM_PRECHECK2 ( shapes.size() > 0,
                  "no shape functions specified" );

  checkShapes_ ( CLASS_NAME, shapes );
  init_        ( shapes );
}


StdPumShape::~StdPumShape ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdPumShape::readFrom ( ObjectInput& in )
{
  SShapeVector  shapes;

  decode ( in, shapes );
  init_  (     shapes );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdPumShape::writeTo ( ObjectOutput& out ) const
{
  SShapeVector  shapes ( bases_.size() );

  for ( idx_t i = 0; i < bases_.size(); i++ )
  {
    shapes[i] = bases_[i]->shape;
  }

  encode ( out, shapes );
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdPumShape::rank () const
{
  return rank_;
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdPumShape::shapeFuncCount () const
{
  return funcCount_;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdPumShape::getGeometry () const
{
  return bases_[0]->shape->getGeometry ();
}


//-----------------------------------------------------------------------
//   isLinear
//-----------------------------------------------------------------------


bool StdPumShape::isLinear () const
{
  return false;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdPumShape::vertexCount () const
{
  return bases_[0]->shape->vertexCount ();
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdPumShape::getVertexCoords () const
{
  return bases_[0]->shape->getVertexCoords ();
}


//-----------------------------------------------------------------------
//   containsPoint
//-----------------------------------------------------------------------


bool StdPumShape::containsPoint ( const Vector& u ) const
{
  return bases_[0]->shape->containsPoint ( u );
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdPumShape::clipPoint ( const Vector& u ) const
{
  return bases_[0]->shape->clipPoint ( u );
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdPumShape::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const Basis_&  basis0     = * bases_[0];

  const idx_t    order      = order_;
  const idx_t    basisCount = bases_.size ();

  double         f0;
  idx_t          ifunc, jfunc;


  ifunc = 0;

  basis0.evalFunctions ( u );

  for ( idx_t i = 0; i < basis0.funcCount; i++, ifunc += order )
  {
    f[ifunc] = basis0.shapeFuncs[i];
  }

  for ( idx_t ib = 1; ib < basisCount; ib++ )
  {
    const Basis_&  basis = * bases_[ib];

    basis.evalFunctions ( u );

    jfunc = 0;

    for ( idx_t i = 0; i < basis0.funcCount; i++ )
    {
      f0    = basis0.shapeFuncs[i];
      ifunc = basis.offset + i * order;

      for ( idx_t j = 0; j < basis.order; j++ )
      {
        f[ifunc] = f0 * basis.shapeFuncs[jfunc];

        ifunc++;
        jfunc++;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdPumShape::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const Basis_&  basis0     = * bases_[0];

  const idx_t    rank       = rank_;
  const idx_t    order      = order_;
  const idx_t    basisCount = bases_.size ();

  double         f0, ff;
  idx_t          ifunc, jfunc;


  ifunc = 0;

  basis0.evalGradients ( u );

  for ( idx_t i = 0; i < basis0.funcCount; i++, ifunc += order )
  {
    f[ifunc] = basis0.shapeFuncs[i];

    for ( idx_t k = 0; k < rank; k++ )
    {
      g(k,ifunc) = basis0.shapeGrads(k,i);
    }
  }

  for ( idx_t ib = 1; ib < basisCount; ib++ )
  {
    const Basis_&  basis = * bases_[ib];

    basis.evalGradients ( u );

    jfunc = 0;

    for ( idx_t i = 0; i < basis0.funcCount; i++ )
    {
      f0    = basis0.shapeFuncs[i];
      ifunc = basis.offset + i * order;

      for ( idx_t j = 0; j < basis.order; j++ )
      {
        ff       = basis.shapeFuncs[jfunc];
        f[ifunc] = f0 * ff;

        for ( idx_t k = 0; k < rank; k++ )
        {
          g(k,ifunc) =

            f0 * basis .shapeGrads(k,jfunc) +
            ff * basis0.shapeGrads(k,i);
        }

        ifunc++;
        jfunc++;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdPumShape::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const Basis_&  basis0     = * bases_[0];

  const idx_t    rank       =  rank_;
  const idx_t    rank2      = (rank * rank + rank) / 2;
  const idx_t    order      = order_;
  const idx_t    basisCount = bases_.size ();

  idx_t          ifunc, jfunc;


  ifunc = 0;

  basis0.evalGradGrads ( u );

  ifunc = 0;

  for ( idx_t i = 0; i < basis0.funcCount; i++, ifunc += order )
  {
    f[ifunc] = basis0.shapeFuncs[i];

    for ( idx_t k = 0; k < rank; k++ )
    {
      g(k,ifunc) = basis0.shapeGrads(k,i);
    }

    for ( idx_t k = 0; k < rank2; k++ )
    {
      h(k,ifunc) = basis0.shapeGrads2(k,i);
    }
  }

  for ( idx_t ib = 1; ib < basisCount; ib++ )
  {
    const Basis_&  basis = * bases_[ib];

    basis.evalGradGrads ( u );

    jfunc = 0;

    for ( idx_t i = 0; i < basis0.funcCount; i++ )
    {
      double  f0 = basis0.shapeFuncs[i];
      ifunc      = basis.offset + i * order;

      for ( idx_t j = 0; j < basis.order; j++ )
      {
        double  ff = basis.shapeFuncs[jfunc];
        f[ifunc]   = f0 * ff;

        for ( idx_t k = 0; k < rank; k++ )
        {
          g(k,ifunc) =

            f0 * basis .shapeGrads(k,jfunc) +
            ff * basis0.shapeGrads(k,i);
        }

        for ( idx_t k = 0; k < rank2; k++ )
        {
          idx_t  ii = grad2Map_(k,0);
          idx_t  jj = grad2Map_(k,1);

          h(k,ifunc) =

            f0                      * basis.shapeGrads2(k,jfunc) +
            basis0.shapeGrads(ii,i) * basis.shapeGrads(jj,jfunc) +
            basis0.shapeGrads(jj,i) * basis.shapeGrads(ii,jfunc) +
            basis0.shapeGrads2(k,i) * ff;
        }

        ifunc++;
        jfunc++;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* StdPumShape::getExtByID ( ExtensionID id ) const
{
  const idx_t  basisCount = bases_.size ();

  for ( idx_t ib = 0; ib < basisCount; ib++ )
  {
    void*  ext = bases_[ib]->shape->getExtByID ( id );

    if ( ext )
    {
      return ext;
    }
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdPumShape::makeNew

  ( const String&      geom,
    const Properties&  conf,
    const Properties&  props )

{
  const idx_t    MAX_BASES = 100;

  SShapeVector   basisShapes ( MAX_BASES );
  Properties     basisProps;
  String         basisName;
  idx_t          funcCount;
  idx_t          rank;
  idx_t          ib;


  basisName = PropNames::BASIS + String ( 0 );

  props.get ( basisProps, basisName );

  basisShapes[0] = StdShapeFactory::newInstance (
    geom,
    conf.makeProps( basisName ),
    basisProps
  );

  rank      = basisShapes[0]->rank           ();
  funcCount = basisShapes[0]->shapeFuncCount ();

  if ( funcCount == 0 )
  {
    props.propertyError (
      basisName,
      "basis shape 0 has no shape functions"
    );
  }

  for ( ib = 1; ib <= MAX_BASES; ib++ )
  {
    basisName = PropNames::BASIS + String ( ib );

    if ( ! props.find( basisProps, basisName ) )
    {
      break;
    }

    if ( ib >= MAX_BASES )
    {
      props.propertyError (
        basisName,
        "too many basis functions specified"
      );
    }

    basisShapes[ib] = StdShapeFactory::newInstance (
      geom,
      conf.makeProps( basisName ),
      basisProps
    );

    if ( basisShapes[ib]->rank() != rank )
    {
      props.propertyError (
        basisName,
        String::format (
          "basis shape %d has rank %d but should have rank %d",
          ib,
          basisShapes[ib]->rank (),
          rank
        )
      );
    }

    if ( basisShapes[ib]->shapeFuncCount() % funcCount > 0 )
    {
      props.propertyError (
        basisName,
        String::format (
          "number of shape functions (%d) is not divisible by %d",
          basisShapes[ib]->shapeFuncCount (),
          funcCount
        )
      );
    }
  }

  basisShapes.reshape ( ib );

  return newInstance<Self> ( basisShapes );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdPumShape::declare ()
{
  StdShapeFactory::declare ( "Pum", "*", & makeNew );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void StdPumShape::init_ ( const SShapeVector& shapes )
{
  using jem::util::connect;

  const idx_t  basisCount = shapes.size ();

  Ref<Basis_>  basis0;
  Ref<Basis_>  basis;
  idx_t        rank2;


  bases_.resize ( basisCount );

  basis0     = newInstance<Basis_> ( shapes[0] );
  bases_[0]  = basis0;
  rank_      = shapes[0]->rank ();
  order_     = basis0->order;
  funcCount_ = basis0->funcCount;

  connect ( shapes[0]->changeEvent, this, &Self::shapeChanged_ );

  for ( idx_t ib = 1; ib < basisCount; ib++ )
  {
    basis         = newInstance<Basis_> ( shapes[ib] );
    basis->order  = basis->funcCount / basis0->funcCount;
    basis->offset = order_;

    bases_[ib]    = basis;
    order_       += basis->order;
    funcCount_   += basis->funcCount;

    connect ( shapes[ib]->changeEvent, this, &Self::shapeChanged_ );
  }

  rank2 = (rank_ * rank_ + rank_) / 2;

  grad2Map_.resize ( rank2, 2 );

  getGrad2Map ( grad2Map_[0], grad2Map_[1], rank_ );
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void StdPumShape::shapeChanged_ ()
{
  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   checkShapes_
//-----------------------------------------------------------------------


void StdPumShape::checkShapes_

  ( const String&        context,
    const SShapeVector&  shapes )

{
  const idx_t  n = shapes.size ();

  idx_t  funcCount;
  idx_t  rank;


  if ( n == 0 )
  {
    throw IllegalArgumentException (
      context,
      "zero basis shapes specified (at least one required)"
    );
  }

  for ( idx_t ib = 0; ib < n; ib++ )
  {
    if ( ! shapes[ib] )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "no basis shape %d specified (null pointer)", ib
        )
      );
    }
  }

  funcCount = shapes[0]->shapeFuncCount ();
  rank      = shapes[0]->rank           ();

  if ( funcCount == 0 )
  {
    throw IllegalArgumentException (
      context,
      "basis shape 0 has no shape functions"
    );
  }

  for ( idx_t ib = 1; ib < n; ib++ )
  {
    const StdShape& sh = * shapes[ib];

    if ( sh.rank() != rank )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "basis shape %d has rank %d but should have rank %d",
          ib,
          sh.rank (),
          rank
        )
      );
    }

    if ( sh.shapeFuncCount() % funcCount > 0 )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "basis shape %d has an invalid number (%d) "
          "of shape functions",
          ib,
          sh.shapeFuncCount ()
        )
      );
    }
  }
}


JIVE_END_PACKAGE( geom )
