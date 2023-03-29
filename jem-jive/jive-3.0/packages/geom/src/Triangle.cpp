
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
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/array/select.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/geom/Names.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/shfuncs/TriangFuncs.h>
#include <jive/geom/StdEdge.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdTriangle.h>
#include <jive/geom/ParametricArea.h>
#include <jive/geom/ParametricEdge.h>
#include <jive/geom/BoundaryLine.h>
#include <jive/geom/IShapeFactory.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/Triangle.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jem::numeric::dotProduct;
using jive::util::joinNames;
using jive::util::Topology;


//=======================================================================
//   class Triangle3Shape (declaration)
//=======================================================================


class Triangle3Shape : public ParametricArea
{
 public:

  JEM_DECLARE_CLASS       ( Triangle3Shape, ParametricArea );

  static const int          RANK = 2;


  explicit inline           Triangle3Shape

    ( const String&           name = "" );

  inline                    Triangle3Shape

    ( const String&           name,
      const Matrix&           ischeme,
      const Boundary&         boundary,
      const Ref<StdShape>&    xshape,
      const Ref<StdShape>&    sshape = nullptr );

  virtual void              getGlobalIntegrationPoints

    ( const Matrix&           x,
      const Matrix&           c )             const override;

  virtual void              getIntegrationWeights

    ( const Vector&           w,
      const Matrix&           c )             const override;

  virtual void              getGlobalPoint

    ( const Vector&           x,
      const Vector&           u,
      const Matrix&           c )             const override;

  virtual bool              getLocalPoint

    ( const Vector&           u,
      const Vector&           x,
      double                  eps,
      const Matrix&           c )             const override;


 protected:

  virtual                  ~Triangle3Shape ();

  static inline double      getArea_

    ( const Matrix&           c );

};


//=======================================================================
//   class Triangle3Shape (implementation)
//=======================================================================


JIVE_END_PACKAGE( geom )

JEM_DEFINE_SERIABLE_CLASS( jive::geom::Triangle3Shape );

JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline Triangle3Shape::Triangle3Shape ( const String& name ) :

  Super ( name,
          StdTriangle::getGaussScheme ( 1 ),
          newInstance<StdTriangle3>   () )

{}


inline Triangle3Shape::Triangle3Shape

  ( const String&         name,
    const Matrix&         ischeme,
    const Boundary&       boundary,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, boundary, xshape, sshape )

{
  JEM_PRECHECK2 ( xshape && xshape->shapeFuncCount() == 3,
                  "invalid geometric shape functions" );
}


Triangle3Shape::~Triangle3Shape ()
{}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void Triangle3Shape::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c )  const

{
  const Matrix&  xsf     = ipoints_.xfuncs;
  const idx_t    ipCount = ipCount_;


  updateFuncs_ ( ipoints_ );

  if ( ipCount == 1 )
  {
    x(0,0) = xsf(0,0) * c(0,0) +
             xsf(1,0) * c(0,1) +
             xsf(2,0) * c(0,2);
    x(1,0) = xsf(0,0) * c(1,0) +
             xsf(1,0) * c(1,1) +
             xsf(2,0) * c(1,2);
  }
  else
  {
    const double  c00 = c(0,0);
    const double  c01 = c(0,1);
    const double  c02 = c(0,2);
    const double  c10 = c(1,0);
    const double  c11 = c(1,1);
    const double  c12 = c(1,2);

JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      x(0,ip) = xsf(0,ip) * c00 +
                xsf(1,ip) * c01 +
                xsf(2,ip) * c02;
      x(1,ip) = xsf(0,ip) * c10 +
                xsf(1,ip) * c11 +
                xsf(2,ip) * c12;
    }
  }
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void Triangle3Shape::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  const idx_t   ipCount = ipCount_;
  const double  a       = getArea_ ( c );


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


void Triangle3Shape::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  Tuple<double,3>  xsf;

  JIVE_EVAL_TRIANG3_FUNCS( xsf, u );

  for ( idx_t i = 0; i < RANK; i++ )
  {
    x[i] = c(i,0) * xsf[0] + c(i,1) * xsf[1] + c(i,2) * xsf[2];
  }
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool Triangle3Shape::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  Mat2x2  gx;
  Vec2    dx;
  Vec2    du;

  double  tmp;
  double  det;


  for ( int i = 0; i < RANK; i++ )
  {
    tmp     = c(i,0);
    gx(i,0) = c(i,1) - tmp;
    gx(i,1) = c(i,2) - tmp;
    dx[i]   = x[i]   - tmp;
  }

  det = gx(0,0) * gx(1,1) - gx(0,1) * gx(1,0);

  if ( jem::isTiny( det ) )
  {
    return false;
  }

  tmp   = 1.0 / det;

  du[0] = tmp * (gx(1,1) * dx[0] - gx(0,1) * dx[1]);
  du[1] = tmp * (gx(0,0) * dx[1] - gx(1,0) * dx[0]);

  if ( du[0] >= 0.0 && du[1] >= 0.0 && du[0] + du[1] <= 1.0 )
  {
    u[0] = du[0];
    u[1] = du[1];

    return true;
  }

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

  for ( int i = 0; i < RANK; i++ )
  {
    dx[i] = dx[i] - du[0] * gx(i,0) - du[1] * gx(i,1);
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
//   getArea_
//-----------------------------------------------------------------------


inline double Triangle3Shape::getArea_ ( const Matrix& c )
{
  double a;

  a  = c(0,1) * c(1,2) - c(1,1) * c(0,2);
  a += c(1,0) * c(0,2) - c(0,0) * c(1,2);
  a += c(0,0) * c(1,1) - c(1,0) * c(0,1);

  return std::fabs ( a );
}


//=======================================================================
//   class Triangle::Utils_
//=======================================================================


class Triangle::Utils_
{
 public:

  typedef Ref<BShape>     (*BShapeCtor)

    ( const String&           name,
      const Matrix&           ischeme,
      const Ref<SShape>&      sfuncs );

  typedef Ref<IShape>     (*IShapeCtor)

    ( const String&           name,
      const MatrixVector&     ischeme,
      const Ref<SShape>&      sfuncs );


  static ShapeBoundary      getBoundary

    ( const String&           name,
      const MatrixVector&     ischeme,
      const Ref<SShape>&      xfuncs,
      const Ref<SShape>&      sfuncs );

  static ShapeBoundary      getBoundary

    ( const String&           name,
      BShapeCtor              ctor,
      const MatrixVector&     ischeme,
      const Topology&         topo,
      const Ref<SShape>&      sfuncs );

  static Ref<IShape>        getShape

    ( const String&           name,
      IShapeCtor              ctor,
      String&                 ischeme,
      String&                 bscheme,
      const Ref<SShape>&      sfuncs );

  static Ref<IShape>        getShape

    ( const String&           name,
      IShapeCtor              ctor,
      const Properties&       conf,
      const Properties&       props,
      const StringVector&     ischeme,
      const String&           sfType );

};


//-----------------------------------------------------------------------
//   getBoundary
//-----------------------------------------------------------------------


ShapeBoundary Triangle::Utils_::getBoundary

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   xfuncs,
    const Ref<SShape>&   sfuncs )

{
  const idx_t    nodeCount = xfuncs->shapeFuncCount ();

  BShapeVector   bounds    ( BOUNDARY_COUNT );
  SBShapeVector  bxfuncs   ( BOUNDARY_COUNT );
  SBShapeVector  bsfuncs   ( BOUNDARY_COUNT );

  IdxVector      offsets   ( BOUNDARY_COUNT + 1 );
  IdxVector      inodes    =

    Triangle3::getBoundaryTopology().getColumnIndices ();

  Matrix         u         = StdTriangle3::vertexCoords ();
  Matrix         v         = Matrix ( u[inodes] );

  String         baseName  = joinNames ( name, "boundary" );

  idx_t          ipos;


  bxfuncs[0] = newInstance<StdEdge> (
    v[slice(0,2)],
    xfuncs
  );

  for ( int i = 1; i < BOUNDARY_COUNT; i++ )
  {
    int  j = 2 * i;

    bxfuncs[i] = newInstance<StdEdge> ( bxfuncs[0],
                                        v[slice(j,j + 2)] );
  }

  if ( sfuncs && sfuncs != xfuncs )
  {
    bsfuncs[0] = newInstance<StdEdge> (
      v[slice(0,2)],
      sfuncs
    );

    for ( int i = 1; i < BOUNDARY_COUNT; i++ )
    {
      int  j = 2 * i;

      bsfuncs[i] = newInstance<StdEdge> ( bsfuncs[0],
                                          v[slice(j,j + 2)] );
    }
  }

  for ( int i = 0; i < BOUNDARY_COUNT; i++ )
  {
    bounds[i] = newInstance<ParametricEdge> (
      baseName + String( i ),
      ischeme[i],
      bxfuncs[i],
      bsfuncs[i]
    );
  }

  inodes.resize ( BOUNDARY_COUNT * nodeCount );

  ipos = 0;

  for ( int i = 0; i < BOUNDARY_COUNT; i++ )
  {
    offsets[i] = ipos;

    for ( idx_t j = 0; j < nodeCount; j++ )
    {
      inodes[ipos++] = j;
    }
  }

  offsets[BOUNDARY_COUNT] = ipos;

  return ShapeBoundary (
    bounds,
    bxfuncs,
    Topology (
      jem::shape ( BOUNDARY_COUNT, nodeCount ),
      offsets,
      inodes
    )
  );
}

//-----------------------------------------------------------------------
//   getBoundary
//-----------------------------------------------------------------------


ShapeBoundary Triangle::Utils_::getBoundary

  ( const String&        name,
    BShapeCtor           ctor,
    const MatrixVector&  ischeme,
    const Topology&      topo,
    const Ref<SShape>&   sfuncs )

{
  BShapeVector   bounds   ( BOUNDARY_COUNT );
  SBShapeVector  bfuncs   ( BOUNDARY_COUNT );

  String         baseName = joinNames ( name, "boundary" );

  Matrix         u        ( 2, 3 + 1 );


  u[slice(0,3)] = StdTriangle3::vertexCoords ();
  u[3]          = u[0];

  bfuncs[0] = newInstance<StdEdge> ( u[slice(0,2)],   sfuncs );
  bfuncs[1] = newInstance<StdEdge> ( bfuncs[0], u[slice(1,3)] );
  bfuncs[2] = newInstance<StdEdge> ( bfuncs[0], u[slice(2,4)] );

  for ( int i = 0; i < BOUNDARY_COUNT; i++ )
  {
    bounds[i] = ctor ( baseName + String ( i ),
                       ischeme[i], bfuncs[i] );
  }

  return ShapeBoundary ( bounds, bfuncs, topo );
}


//-----------------------------------------------------------------------
//   getShape (given integration schemes)
//-----------------------------------------------------------------------


Ref<IShape> Triangle::Utils_::getShape

  ( const String&        name,
    IShapeCtor           ctor,
    String&              ischeme,
    String&              bscheme,
    const Ref<SShape>&   sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );

  s[0].ref ( StdTriangle::getIntegrationScheme( ischeme ) );
  s[1].ref ( StdLine    ::getIntegrationScheme( bscheme ) );
  s[2].ref ( s[1] );
  s[3].ref ( s[1] );

  return ctor ( name, s, sfuncs );
}


//-----------------------------------------------------------------------
//   getShape (given properties)
//-----------------------------------------------------------------------


Ref<IShape> Triangle::Utils_::getShape

  ( const String&        name,
    IShapeCtor           ctor,
    const Properties&    conf,
    const Properties&    props,
    const StringVector&  ischeme,
    const String&        sfType )

{
  JEM_ASSERT2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                "invalid integration scheme" );

  MatrixVector  s ( BOUNDARY_COUNT + 1 );
  StringVector  t ( ischeme );

  Ref<SShape>   sfuncs;
  Ref<IShape>   shape;


  sfuncs = StdShapeFactory::findShapeFuncs (
    GEOMETRY,
    conf, props, sfType,
    StdTriangle::RANK
  );

  if ( props.find( t, PropNames::ISCHEME ) )
  {
    if      ( t.size() == 1 )
    {
      ischeme[0] = t[0];
    }
    else if ( t.size() == 2 )
    {
      ischeme[0]            = t[0];
      ischeme[slice(1,END)] = t[1];
    }
    else if ( t.size() == s.size() )
    {
      ischeme = t;
    }
    else
    {
      props.propertyError (
        PropNames::ISCHEME,
        "array must have length 1, 2 or 4"
      );
    }
  }

  try
  {
    s[0].ref ( StdTriangle::getIntegrationScheme( ischeme[0] ) );

    for ( idx_t i = 1; i < s.size(); i++ )
    {
      s[i].ref ( StdLine::getIntegrationScheme( ischeme[i] ) );
    }
  }
  catch ( jem::IllegalInputException& ex )
  {
    ex.setContext ( props.getContext( PropNames::ISCHEME ) );
    throw;
  }

  conf.set ( PropNames::ISCHEME, ischeme );

  return ctor ( name, s, sfuncs );
}


//=======================================================================
//   class Triangle
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Triangle::GEOMETRY  = Geometries::TRIANGLE;
const char*  Triangle::TYPE_NAME = "Triangle";

#ifndef _MSC_EXTENSIONS
const int    Triangle::BOUNDARY_COUNT;
#endif


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Triangle::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  xfuncs,
    const Ref<SShape>&  sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );

  s[0].ref ( StdTriangle::getIntegrationScheme( ischeme ) );
  s[1].ref ( StdLine    ::getIntegrationScheme( bscheme ) );

  for ( idx_t i = 2; i < s.size(); i++ )
  {
    s[i].ref ( s[1] );
  }

  return getShape ( name, s, xfuncs, sfuncs );
}


Ref<IShape> Triangle::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   xfuncs,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( xfuncs &&
                  xfuncs->getGeometry() == GEOMETRY,
                  "invalid geometric shape functions" );
  JEM_PRECHECK2 ( sfuncs == nullptr  ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return newInstance<ParametricArea> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      ischeme[slice(1,END)],
      xfuncs,
      sfuncs
    ),
    xfuncs,
    sfuncs
  );
}


Ref<IShape> Triangle::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  Properties    myProps = props.getProps  ( name );
  Properties    myConf  = conf .makeProps ( name );

  Ref<SShape>   sfuncs  =

    StdShapeFactory::newInstance (
      GEOMETRY,
      myConf .makeProps ( PropNames::SHAPE_FUNCS ),
      myProps.getProps  ( PropNames::SHAPE_FUNCS )
    );

  MatrixVector  ischeme ( BOUNDARY_COUNT + 1 );
  StringVector  ispecs  ( BOUNDARY_COUNT + 1 );
  StringVector  t;

  myProps.get ( t, PropNames::ISCHEME );

  if      ( t.size() == 2 )
  {
    ispecs[0]            = t[0];
    ispecs[slice(1,END)] = t[1];
  }
  else if ( t.size() == ispecs.size() )
  {
    ispecs = t;
  }
  else
  {
    myProps.propertyError (
      PropNames::ISCHEME,
      String::format (
        "array must have length 2 or %d",
        ispecs.size ()
      )
    );
  }

  try
  {
    ischeme[0].ref (
      StdTriangle::getIntegrationScheme ( ispecs[0] )
    );

    for ( idx_t i = 1; i < ischeme.size(); i++ )
    {
      ischeme[i].ref ( StdLine::getIntegrationScheme( ispecs[i] ) );
    }
  }
  catch ( jem::IllegalInputException& ex )
  {
    ex.setContext ( myProps.getContext( PropNames::ISCHEME ) );
    throw;
  }

  myConf.set ( PropNames::ISCHEME, ispecs );

  return getShape ( name, ischeme, sfuncs );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Triangle::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,              & getShape );
  IShapeFactory::declare ( "jive::geom::Triangle", & getShape );
}


//=======================================================================
//   class Triangle3
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Triangle3::TYPE_NAME = "Triangle3";
const char*  Triangle3::ISCHEME   = "Gauss3";
const char*  Triangle3::BSCHEME   = "Gauss2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Triangle3::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  bscheme = BSCHEME;

  return Utils_::getShape ( name,    getShape,
                            ischeme, bscheme, sfuncs );
}


Ref<IShape> Triangle3::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name,    getShape,
                            ischeme, bscheme, sfuncs );
}


Ref<IShape> Triangle3::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );


  Ref<SShape>  xshape = newInstance<StdTriangle3> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<Triangle3Shape> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      & BoundaryLine2::getShape,
      ischeme[slice(1,END)],
      getBoundaryTopology (),
      sshape
    ),
    xshape,
    sshape
  );
}


Ref<IShape> Triangle3::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0] = ISCHEME;
  ischeme[1] = BSCHEME;
  ischeme[2] = BSCHEME;
  ischeme[3] = BSCHEME;

  return Utils_::getShape ( name,
                            getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "Linear" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Triangle3::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( BOUNDARY_COUNT * NODES_PER_BOUNDARY );
  idx_t      i;


  offsets[0] = 0;

  for ( i = 0; i < BOUNDARY_COUNT; i++ )
  {
    offsets[i + 1] = offsets[i] + NODES_PER_BOUNDARY;
  }

  i = 0;

  indices[i++] = 0;
  indices[i++] = 1;

  indices[i++] = 1;
  indices[i++] = 2;

  indices[i++] = 2;
  indices[i++] = 0;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Triangle3::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,               & getShape );
  IShapeFactory::declare ( "jive::geom::Triangle3", & getShape );
}


//=======================================================================
//   class Triangle6
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Triangle6::TYPE_NAME = "Triangle6";
const char*  Triangle6::ISCHEME   = "Gauss7";
const char*  Triangle6::BSCHEME   = "Gauss3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Triangle6::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  bscheme = BSCHEME;

  return Utils_::getShape ( name,    getShape,
                            ischeme, bscheme, sfuncs );
}


Ref<IShape> Triangle6::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name,    getShape,
                            ischeme, bscheme, sfuncs );
}


Ref<IShape> Triangle6::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid shape functions" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );


  Ref<SShape>  xshape = newInstance<StdTriangle6> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricArea> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      & BoundaryLine3::getShape,
      ischeme[slice(1,END)],
      getBoundaryTopology (),
      sshape
    ),
    xshape,
    sshape
  );
}


Ref<IShape> Triangle6::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0] = ISCHEME;
  ischeme[1] = BSCHEME;
  ischeme[2] = BSCHEME;
  ischeme[3] = BSCHEME;

  return Utils_::getShape ( name,
                            getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "Quadratic" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Triangle6::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( BOUNDARY_COUNT * NODES_PER_BOUNDARY );
  idx_t      i;


  offsets[0] = 0;

  for ( i = 0; i < BOUNDARY_COUNT; i++ )
  {
    offsets[i + 1] = offsets[i] + NODES_PER_BOUNDARY;
  }

  i = 0;

  indices[i++] = 0;
  indices[i++] = 1;
  indices[i++] = 2;

  indices[i++] = 2;
  indices[i++] = 3;
  indices[i++] = 4;

  indices[i++] = 4;
  indices[i++] = 5;
  indices[i++] = 0;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Triangle6::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,               & getShape );
  IShapeFactory::declare ( "jive::geom::Triangle6", & getShape );
}


JIVE_END_PACKAGE( geom )
