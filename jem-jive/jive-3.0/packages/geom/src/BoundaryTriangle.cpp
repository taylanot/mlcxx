
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
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Properties.h>
#include <jive/geom/error.h>
#include <jive/geom/Names.h>
#include <jive/geom/shfuncs/TriangFuncs.h>
#include <jive/geom/StdTriangle.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/ParametricSurface.h>
#include <jive/geom/BShapeFactory.h>
#include <jive/geom/BoundaryTriangle.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jem::numeric::dotProduct;
using jem::numeric::crossProduct;


//=======================================================================
//   class BTriangle3Shape (declaration)
//=======================================================================


class BTriangle3Shape : public ParametricSurface
{
 public:

  JEM_DECLARE_CLASS     ( BTriangle3Shape, ParametricSurface );

  static const int        NODE_COUNT = 3;


  explicit inline         BTriangle3Shape

    ( const String&         name = "" );

  inline                  BTriangle3Shape

    ( const String&         name,
      const Matrix&         ischeme,
      const Ref<SShape>&    sshape );

  virtual void            getGlobalIntegrationPoints

    ( const Matrix&         x,
      const Matrix&         c )              const override;

  virtual void            getIntegrationWeights

    ( const Vector&         w,
      const Matrix&         c )              const override;

  virtual void            getGlobalPoint

    ( const Vector&         x,
      const Vector&         u,
      const Matrix&         c )              const override;

  virtual bool            getLocalPoint

    ( const Vector&         u,
      const Vector&         x,
      double                eps,
      const Matrix&         c )              const override;

  virtual void            evalNormal

    ( const Vector&         q,
      const Vector&         u,
      const Matrix&         c )              const override;

  virtual void            getNormals

    ( const Matrix&         q,
      const Vector&         w,
      const Matrix&         c )              const override;

  virtual void            getVertexNormals

    ( const Matrix&         q,
      const Matrix&         c )              const override;


 protected:

  virtual                ~BTriangle3Shape ();


 private:

  inline double           getNormal_

    ( Vec3&                 n,
      const Matrix&         c )              const;

};


//=======================================================================
//   class BTriangle3Shape (implementation)
//=======================================================================


JIVE_END_PACKAGE( geom )

JEM_DEFINE_SERIABLE_CLASS( jive::geom::BTriangle3Shape );

JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline BTriangle3Shape::BTriangle3Shape

  ( const String& name ) :

  Super ( name,
          StdTriangle::getGaussScheme ( 1 ),
          newInstance<StdTriangle3>   (), nullptr )

{}


inline BTriangle3Shape::BTriangle3Shape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sshape ) :

    Super ( name, ischeme, newInstance<StdTriangle3>(), sshape )

{}


BTriangle3Shape::~BTriangle3Shape ()
{}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void BTriangle3Shape::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c ) const

{
  const Matrix&  xsf     = ipoints_.xfuncs;
  const idx_t    ipCount = ipCount_;


  updateFuncs_ ( ipoints_ );

  if ( ipCount == 1_idx )
  {
    x(0,0) = xsf(0,0) * c(0,0) +
             xsf(1,0) * c(0,1) +
             xsf(2,0) * c(0,2);

    x(1,0) = xsf(0,0) * c(1,0) +
             xsf(1,0) * c(1,1) +
             xsf(2,0) * c(1,2);

    x(2,0) = xsf(0,0) * c(2,0) +
             xsf(1,0) * c(2,1) +
             xsf(2,0) * c(2,2);
  }
  else
  {
    const double  c00 = c(0,0);
    const double  c01 = c(0,1);
    const double  c02 = c(0,2);
    const double  c10 = c(1,0);
    const double  c11 = c(1,1);
    const double  c12 = c(1,2);
    const double  c20 = c(2,0);
    const double  c21 = c(2,1);
    const double  c22 = c(2,2);

JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      x(0,ip) = xsf(0,ip) * c00 +
                xsf(1,ip) * c01 +
                xsf(2,ip) * c02;

      x(1,ip) = xsf(0,ip) * c10 +
                xsf(1,ip) * c11 +
                xsf(2,ip) * c12;

      x(2,ip) = xsf(0,ip) * c20 +
                xsf(1,ip) * c21 +
                xsf(2,ip) * c22;
    }
  }
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void BTriangle3Shape::getIntegrationWeights

  ( const Vector& w,
    const Matrix& c ) const

{
  const idx_t  ipCount = ipCount_;

  Vec3         n, s, t;
  double       a;


  for ( int i = 0; i < 3; i++ )
  {
    s[i] = c(i,1) - c(i,0);
    t[i] = c(i,2) - c(i,0);
  }

  n = crossProduct ( s, t );
  a = std::sqrt ( dotProduct( n, n ) );

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


void BTriangle3Shape::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  Vec3  xsf;

  JIVE_EVAL_TRIANG3_FUNCS( xsf, u );

  for ( int i = 0; i < 3; i++ )
  {
    x[i] = c(i,0) * xsf[0] + c(i,1) * xsf[1] + c(i,2) * xsf[2];
  }
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool BTriangle3Shape::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  Mat2x2  px;

  Vec3    s, t;
  Vec3    dx;
  Vec2    du;
  Vec2    dv;

  double  tmp;
  double  det;


  for ( int i = 0; i < 3; i++ )
  {
    tmp   = c(i,0);
    s[i]  = c(i,1) - tmp;
    t[i]  = c(i,2) - tmp;
    dx[i] = x[i]   - tmp;
  }

  px(0,0) = dotProduct ( s, s );
  px(0,1) = dotProduct ( s, t );
  px(1,0) = px(0,1);
  px(1,1) = dotProduct ( t, t );

  dv[0]   = dotProduct ( s, dx );
  dv[1]   = dotProduct ( t, dx );

  det     = px(0,0) * px(1,1) - px(0,1) * px(1,0);

  if ( jem::isTiny( det ) )
  {
    return false;
  }

  tmp   = 1.0 / det;

  du[0] = tmp * (px(1,1) * dv[0] - px(0,1) * dv[1]);
  du[1] = tmp * (px(0,0) * dv[1] - px(1,0) * dv[0]);

  if ( du[0] < 0.0 )
  {
    du[0] = 0.0;
  }
  if ( du[1] < 0.0 )
  {
    du[1] = 0.0;
  }

  tmp = du[0] + du[1];

  if ( tmp > 1.0 )
  {
    tmp    = 1.0 / tmp;
    du[0] *= tmp;
    du[1] *= tmp;
  }

  for ( int i = 0; i < 3; i++ )
  {
    dx[i] = dx[i] - du[0] * s[i] - du[1] * t[i];
  }

  if ( dotProduct( dx, dx ) < eps * eps )
  {
    u[0] = du[0];
    u[1] = du[1];

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


void BTriangle3Shape::evalNormal

  ( const Vector&  q,
    const Vector&  u,
    const Matrix&  c ) const

{
  Vec3  n;

  getNormal_ ( n, c );

  q[0] = n[0];
  q[1] = n[1];
  q[2] = n[2];
}


//-----------------------------------------------------------------------
//   getNormals
//-----------------------------------------------------------------------


void BTriangle3Shape::getNormals

  ( const Matrix&  q,
    const Vector&  w,
    const Matrix&  c ) const

{
  const idx_t  ipCount = ipCount_;

  Vec3         n;
  double       a;


  a = getNormal_ ( n, c );

  if ( ipCount == 1_idx )
  {
    w[0]   = a * iweights_[0];
    q(0,0) = n[0];
    q(1,0) = n[1];
    q(2,0) = n[2];
  }
  else
  {

JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      w[ip]   = a * iweights_[ip];
      q(0,ip) = n[0];
      q(1,ip) = n[1];
      q(2,ip) = n[2];
    }
  }
}


//-----------------------------------------------------------------------
//   getVertexNormals
//-----------------------------------------------------------------------


void BTriangle3Shape::getVertexNormals

  ( const Matrix&  q,
    const Matrix&  c ) const

{
  Vec3  n;


  getNormal_ ( n, c );

  q(0,0) = q(0,1) = q(0,2) = n[0];
  q(1,0) = q(1,1) = q(1,2) = n[1];
  q(2,0) = q(2,1) = q(2,2) = n[2];
}


//-----------------------------------------------------------------------
//   getNormal_
//-----------------------------------------------------------------------


inline double BTriangle3Shape::getNormal_

  ( Vec3&          q,
    const Matrix&  c ) const

{
  Vec3    n, s, t;
  double  a, b;

  for ( int i = 0; i < 3; i++ )
  {
    s[i] = c(i,1) - c(i,0);
    t[i] = c(i,2) - c(i,0);
  }

  n = crossProduct ( s, t );
  a = std::sqrt ( dotProduct( n, n ) );

  if ( jem::isTiny( a ) )
  {
    zeroVectorError ( getContext(), "normal" );
  }

  b = 1.0 / a;

  for ( int i = 0; i < 3; i++ )
  {
    q[i] = b * n[i];
  }

  return a;
}


//=======================================================================
//   class BoundaryTriangle
//=======================================================================


class BoundaryTriangle
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


Ref<BShape> BoundaryTriangle::getShape

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
    StdTriangle::GEOMETRY,
    conf, props, sfType,
    StdTriangle::RANK
  );

  props.find ( ischeme, PropNames::ISCHEME );

  try
  {
    shape = ctor (
      name,
      StdTriangle::getIntegrationScheme ( ischeme ),
      sfuncs
    );
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
//   class BoundaryTriangle3
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryTriangle3::TYPE_NAME = "BTriangle3";
const char*  BoundaryTriangle3::ISCHEME   = "Gauss3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryTriangle3::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryTriangle3::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<BTriangle3Shape> (
    name,
    StdTriangle::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryTriangle3::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<BTriangle3Shape> ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryTriangle3::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryTriangle::getShape ( name,
                                      conf .makeProps ( name ),
                                      props.findProps ( name ),
                                      & getShape,
                                      ischeme,
                                      "Linear" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryTriangle3::declare ()
{
  BShapeFactory::declare ( TYPE_NAME, & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryTriangle3",
                           & getShape );
}


//=======================================================================
//   class BoundaryTriangle6
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryTriangle6::TYPE_NAME = "BTriangle6";
const char*  BoundaryTriangle6::ISCHEME   = "Gauss7";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryTriangle6::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryTriangle6::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return getShape (
    name,
    StdTriangle::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryTriangle6::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricSurface> (
    name,
    ischeme,
    newInstance<StdTriangle6> (),
    sfuncs
  );
}


Ref<BShape> BoundaryTriangle6::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryTriangle::getShape ( name,
                                      conf .makeProps ( name ),
                                      props.findProps ( name ),
                                      & getShape,
                                      ischeme,
                                      "Quadratic" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryTriangle6::declare ()
{
  BShapeFactory::declare ( TYPE_NAME, & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryTriangle6",
                           & getShape );
}


JIVE_END_PACKAGE( geom )
