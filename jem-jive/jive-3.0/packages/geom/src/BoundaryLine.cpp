
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
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/operators.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Properties.h>
#include <jive/geom/error.h>
#include <jive/geom/Names.h>
#include <jive/geom/shfuncs/LineFuncs.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/ParametricEdge.h>
#include <jive/geom/BShapeFactory.h>
#include <jive/geom/BoundaryLine.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jem::numeric::dotProduct;


//=======================================================================
//   class BLine2Shape (declaration)
//=======================================================================


class BLine2Shape : public ParametricEdge
{
 public:

  JEM_DECLARE_CLASS       ( BLine2Shape, ParametricEdge );

  static const int          NODE_COUNT = 2;


  explicit inline           BLine2Shape

    ( const String&           name = "" );

  inline                    BLine2Shape

    ( const String&           name,
      const Matrix&           ischeme,
      const Ref<StdShape>&    sshape );

  virtual void              getGlobalIntegrationPoints

    ( const Matrix&           x,
      const Matrix&           c )          const override;

  virtual void              getIntegrationWeights

    ( const Vector&           w,
      const Matrix&           c )          const override;

  virtual void              getGlobalPoint

    ( const Vector&           x,
      const Vector&           u,
      const Matrix&           c )          const override;

  virtual bool              getLocalPoint

    ( const Vector&           u,
      const Vector&           x,
      double                  eps,
      const Matrix&           c )          const override;

  virtual void              evalNormal

    ( const Vector&           q,
      const Vector&           u,
      const Matrix&           c )          const override;

  virtual void              getNormals

    ( const Matrix&           q,
      const Vector&           w,
      const Matrix&           c )          const override;

  virtual void              getVertexNormals

    ( const Matrix&           q,
      const Matrix&           c )          const override;


 protected:

  virtual                  ~BLine2Shape ();


 private:

  inline double             getNormal_

    ( Vec2&                   q,
      const Matrix&           c )          const;

};


//=======================================================================
//   class BLine2Shape (implementation)
//=======================================================================


JIVE_END_PACKAGE( geom )

JEM_DEFINE_SERIABLE_CLASS( jive::geom::BLine2Shape );

JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline BLine2Shape::BLine2Shape

  ( const String&  name ) :

    Super ( name,
            StdLine::getGaussScheme ( 1 ),
            newInstance<StdLine2>   (), nullptr )

{}


inline BLine2Shape::BLine2Shape

  ( const String&         name,
    const Matrix&         ischeme,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, newInstance<StdLine2>(), sshape )

{}


BLine2Shape::~BLine2Shape ()
{}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void BLine2Shape::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c ) const

{
  const idx_t  ipCount = ipCount_;

  double       a, b;


  if ( ipCount == 1_idx )
  {
    a = 0.5 * (1.0 - ipoints_.coords(0,0));
    b = 1.0 - a;

    x(0,0) = a * c(0,0) + b * c(0,1);
    x(1,0) = a * c(1,0) + b * c(1,1);
  }
  else
  {
    const double  c00 = c(0,0);
    const double  c01 = c(0,1);
    const double  c10 = c(1,0);
    const double  c11 = c(1,1);

JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      a = 0.5 * (1.0 - ipoints_.coords(0,ip));
      b = 1.0 - a;

      x(0,ip) = a * c00 + b * c01;
      x(1,ip) = a * c10 + b * c11;
    }
  }
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void BLine2Shape::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  const idx_t  ipCount = ipCount_;

  double       t0, t1;
  double       a;


  t0 = c(0,1) - c(0,0);
  t1 = c(1,1) - c(1,0);

  a = 0.5 * std::sqrt ( t0 * t0 + t1 * t1 );

  if      ( ipCount == 1_idx )
  {
    w[0] = a * iweights_[0];
  }
  else if ( ipCount == 2_idx )
  {
    w[0] = a * iweights_[0];
    w[1] = a * iweights_[1];
  }
  else if ( ipCount == 3_idx )
  {
    w[0] = a * iweights_[0];
    w[1] = a * iweights_[1];
    w[2] = a * iweights_[2];
  }
  else if ( ipCount >= 4_idx )
  {
JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      w[ip] = a * iweights_[ip];
    }
  }
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void BLine2Shape::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  Vec2  xsf;

  JIVE_EVAL_LINE2_FUNCS( xsf, u[0] );

  for ( int i = 0; i < 2; i++ )
  {
    x[i] = c(i,0) * xsf[0] + c(i,1) * xsf[1];
  }
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool BLine2Shape::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  Vec2    t;
  Vec2    dx;
  double  px;
  double  du;


  t[0]  = c(0,1) - c(0,0);
  t[1]  = c(1,1) - c(1,0);

  dx[0] = x[0] - c(0,0);
  dx[1] = x[1] - c(1,0);

  px    = dotProduct ( t, t );

  if ( jem::isTiny( px ) )
  {
    return false;
  }

  du = dotProduct( t, dx ) / px;

  if      ( du < 0.0 )
  {
    du = 0.0;
  }
  else if ( du > 1.0 )
  {
    du = 1.0;
  }

  dx[0] = dx[0] - du * t[0];
  dx[1] = dx[1] - du * t[1];

  if ( dotProduct( dx, dx ) < eps * eps )
  {
    u[0] = 2.0 * du - 1.0;

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   evalNormal
//-----------------------------------------------------------------------


void BLine2Shape::evalNormal

  ( const Vector&  q,
    const Vector&  u,
    const Matrix&  c ) const

{
  Vec2  n;

  getNormal_ ( n, c );

  q[0] = n[0];
  q[1] = n[1];
}


//-----------------------------------------------------------------------
//   getNormals
//-----------------------------------------------------------------------


void BLine2Shape::getNormals

  ( const Matrix&  q,
    const Vector&  w,
    const Matrix&  c ) const

{
  const idx_t  ipCount = ipCount_;

  Vec2         n;
  double       a;


  a = getNormal_ ( n, c );

  if      ( ipCount == 1_idx )
  {
    w[0]   = a * iweights_[0];

    q(0,0) = n[0];
    q(1,0) = n[1];
  }
  else if ( ipCount == 2_idx )
  {
    w[0]   = a * iweights_[0];
    w[1]   = a * iweights_[1];

    q(0,0) = n[0];
    q(1,0) = n[1];

    q(0,1) = n[0];
    q(1,1) = n[1];
  }
  else if ( ipCount == 3_idx )
  {
    w[0]   = a * iweights_[0];
    w[1]   = a * iweights_[1];
    w[2]   = a * iweights_[2];

    q(0,0) = n[0];
    q(1,0) = n[1];

    q(0,1) = n[0];
    q(1,1) = n[1];

    q(0,2) = n[0];
    q(1,2) = n[1];
  }
  else
  {
JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      w[ip]   = a * iweights_[ip];

      q(0,ip) = n[0];
      q(1,ip) = n[1];
    }
  }
}


//-----------------------------------------------------------------------
//   getVertexNormals
//-----------------------------------------------------------------------


void BLine2Shape::getVertexNormals

  ( const Matrix&  q,
    const Matrix&  c ) const

{
  Vec2  n;

  getNormal_ ( n, c );

  q(0,0) = q(0,1) = n[0];
  q(1,0) = q(1,1) = n[1];
}


//-----------------------------------------------------------------------
//   getNormal_
//-----------------------------------------------------------------------


inline double BLine2Shape::getNormal_

  ( Vec2&          q,
    const Matrix&  c ) const

{
  Vec2    t;
  double  a, b;

  t[0] = c(0,1) - c(0,0);
  t[1] = c(1,1) - c(1,0);

  a = std::sqrt ( t[0] * t[0] + t[1] * t[1] );

  if ( jem::isTiny( a ) )
  {
    zeroVectorError ( getContext(), "normal" );
  }

  b    = 1.0 / a;

  q[0] =  b * t[1];
  q[1] = -b * t[0];

  return (0.5 * a);
}


//=======================================================================
//   class BoundaryLine
//=======================================================================


class BoundaryLine
{
 public:

  typedef Ref<BShape>     (*BShapeCtor)

    ( const String&           name,
      const Matrix&           ischeme,
      const Ref<SShape>&      sfuncs );


  static Ref<BShape>        getShape

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      BShapeCtor              ctor,
      String&                 ischeme,
      const String&           sfType );

};


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryLine::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    BShapeCtor         ctor,
    String&            ischeme,
    const String&      sfType )

{
  Ref<SShape>  sfuncs;
  Ref<BShape>  shape;


  sfuncs = StdShapeFactory::findShapeFuncs (
    StdLine::GEOMETRY,
    conf, props, sfType,
    StdLine::RANK
  );

  props.find ( ischeme, PropNames::ISCHEME );

  try
  {
    shape = ctor ( name,
                   StdLine::getIntegrationScheme ( ischeme ),
                   sfuncs );
  }
  catch ( jem::IllegalInputException& ex )
  {
    ex.setContext ( props.getContext( PropNames::ISCHEME ) );
    throw;
  }

  conf.set ( PropNames::ISCHEME, ischeme );

  return shape;
}


//=======================================================================
//   class BoundaryLine2
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryLine2::TYPE_NAME = "BLine2";
const char*  BoundaryLine2::ISCHEME   = "Gauss2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryLine2::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryLine2::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<BLine2Shape> (
    name,
    StdLine::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryLine2::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<BLine2Shape> ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryLine2::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryLine::getShape ( name,
                                  conf .makeProps ( name ),
                                  props.findProps ( name ),
                                  & getShape,
                                  ischeme,
                                  "Linear" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryLine2::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                   & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryLine2", & getShape );
}


//=======================================================================
//   class BoundaryLine3
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryLine3::TYPE_NAME = "BLine3";
const char*  BoundaryLine3::ISCHEME   = "Gauss3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryLine3::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryLine3::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricEdge> (
    name,
    StdLine::getIntegrationScheme ( ischeme ),
    newInstance<StdLine3> (),
    sfuncs
  );
}


Ref<BShape> BoundaryLine3::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricEdge> (
    name,
    ischeme,
    newInstance<StdLine3> (),
    sfuncs
  );
}


Ref<BShape> BoundaryLine3::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryLine::getShape ( name,
                                  conf .makeProps ( name ),
                                  props.findProps ( name ),
                                  & getShape,
                                  ischeme,
                                  "Quadratic" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryLine3::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                   & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryLine3", & getShape );
}


//=======================================================================
//   class BoundaryLine4
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryLine4::TYPE_NAME = "BLine4";
const char*  BoundaryLine4::ISCHEME   = "Gauss4";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryLine4::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryLine4::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricEdge> (
    name,
    StdLine::getIntegrationScheme ( ischeme ),
    newInstance<StdLine4> (),
    sfuncs
  );
}


Ref<BShape> BoundaryLine4::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricEdge> (
    name,
    ischeme,
    newInstance<StdLine4> (),
    sfuncs
  );
}


Ref<BShape> BoundaryLine4::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryLine::getShape ( name,
                                  conf .makeProps ( name ),
                                  props.findProps ( name ),
                                  & getShape,
                                  ischeme,
                                  "Cubic" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryLine4::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                   & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryLine4", & getShape );
}


JIVE_END_PACKAGE( geom )
