
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
#include <jem/io/Serializable.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/geom/Names.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/shfuncs/LineFuncs.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdPoint.h>
#include <jive/geom/StdBoundary.h>
#include <jive/geom/ParametricLine.h>
#include <jive/geom/BoundaryPoint.h>
#include <jive/geom/IShapeFactory.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/Line.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::Topology;


//=======================================================================
//   class Line2Shape (declaration)
//=======================================================================


class Line2Shape : public ParametricLine
{
 public:

  JEM_DECLARE_CLASS       ( Line2Shape, ParametricLine );

  explicit inline           Line2Shape

    ( const String&           name = "" );

  inline                    Line2Shape

    ( const String&           name,
      const Matrix&           ischeme,
      const Boundary&         boundary,
      const Ref<StdShape>&    xshape,
      const Ref<StdShape>&    sshape = nullptr );

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


 protected:

  virtual                  ~Line2Shape  ();

};


//=======================================================================
//   class Line2Shape (implementation)
//=======================================================================


JIVE_END_PACKAGE( geom )

JEM_DEFINE_SERIABLE_CLASS( jive::geom::Line2Shape );

JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline Line2Shape::Line2Shape ( const String& name ) :

  Super ( name,
          StdLine::getGaussScheme ( 1 ),
          newInstance<StdLine2>   (),  nullptr )

{}


inline Line2Shape::Line2Shape

  ( const String&         name,
    const Matrix&         ischeme,
    const Boundary&       boundary,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super ( name, ischeme, boundary, xshape, sshape )

{
  JEM_PRECHECK2 ( xshape && xshape->shapeFuncCount() == 2,
                  "invalid geometric shape functions" );
}


Line2Shape::~Line2Shape ()
{}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void Line2Shape::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c )  const

{
  const Matrix&  xsf     = ipoints_.xfuncs;
  const idx_t    ipCount = ipCount_;


  updateFuncs_ ( ipoints_ );

  if ( ipCount == 1 )
  {
    x(0,0) = xsf(0,0) * c(0,0) + xsf(1,0) * c(0,1);
  }
  else
  {
    const double  c00 = c(0,0);
    const double  c01 = c(0,1);

JEM_IVDEP

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      x(0,ip) = xsf(0,ip) * c00 + xsf(1,ip) * c01;
    }
  }
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void Line2Shape::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  const idx_t  ipCount = ipCount_;

  double       a;


  a = 0.5 * std::fabs ( c(0,1) - c(0,0) );

  switch ( ipCount )
  {
  case 0:

    break;

  case 1:

    w[0] = a * iweights_[0];

    break;

  case 2:

    w[0] = a * iweights_[0];
    w[1] = a * iweights_[1];

    break;

  case 3:

    w[0] = a * iweights_[0];
    w[1] = a * iweights_[1];
    w[2] = a * iweights_[2];

    break;

  default:

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


void Line2Shape::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  Vec2  xsf;

  JIVE_EVAL_LINE2_FUNCS( xsf, u[0] );

  x[0] = c(0,0) * xsf[0] + c(0,1) * xsf[1];
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool Line2Shape::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  double  gx;
  double  dx;
  double  v;


  gx = c(0,1) - c(0,0);

  if ( jem::isTiny( gx ) )
  {
    return false;
  }

  v = 2.0 * (x[0] - c(0,0)) / gx - 1.0;

  if ( v >= -1.0 && v <= 1.0 )
  {
    u[0] = v;

    return true;
  }
  else
  {
    if ( v < -1.0 )
    {
      v  = -1.0;
      dx = x[0] - c(0,0);
    }
    else
    {
      v  = 1.0;
      dx = x[0] - c(0,1);
    }

    u[0] = v;

    return ( (dx * dx) < (eps * eps) );
  }
}


//=======================================================================
//   class Line::Utils_
//=======================================================================


class Line::Utils_
{
 public:

  typedef Ref<IShape>     (*IShapeCtor)

    ( const String&           name,
      const Matrix&           ischeme,
      const Ref<SShape>&      sfuncs );


  static ShapeBoundary      getBoundary

    ( const String&           name,
      const Ref<SShape>&      xfuncs,
      const Ref<SShape>&      sfuncs );

  static ShapeBoundary      getBoundary

    ( const String&           name,
      const Topology&         topo,
      const Ref<SShape>&      sfuncs );

  static Ref<IShape>        getShape

    ( const String&           name,
      IShapeCtor              ctor,
      const Properties&       conf,
      const Properties&       props,
      String&                 ischeme,
      const String&           sfType );

};


//-----------------------------------------------------------------------
//   getBoundary
//-----------------------------------------------------------------------


ShapeBoundary Line::Utils_::getBoundary

  ( const String&       name,
    const Ref<SShape>&  xfuncs,
    const Ref<SShape>&  sfuncs )

{
  const idx_t    nodeCount = xfuncs->shapeFuncCount ();

  BShapeVector   bounds    ( BOUNDARY_COUNT );
  SBShapeVector  bxfuncs   ( BOUNDARY_COUNT );

  IdxVector      offsets   ( BOUNDARY_COUNT + 1 );
  IdxVector      inodes    ( BOUNDARY_COUNT * nodeCount );

  Matrix         u         = StdLine::vertexCoords ();

  idx_t          ipos;


  bounds[0] = newInstance<BoundaryPoint> (
    joinNames ( name, "point0" ),
    -1,
    -1.0,
    xfuncs,
    sfuncs
  );

  bounds[1] = newInstance<BoundaryPoint> (
    joinNames ( name, "point1" ),
    1,
    1.0,
    xfuncs,
    sfuncs
  );

  bxfuncs[0] = newInstance<StdBoundary> (
    u(ALL,slice(0,1)),
    newInstance<StdPoint> (),
    xfuncs
  );

  bxfuncs[1] = newInstance<StdBoundary> (
    bxfuncs[0],
    u(ALL,slice(1,2))
  );

  ipos = 0;

  for ( idx_t i = 0; i < BOUNDARY_COUNT; i++ )
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


ShapeBoundary Line::Utils_::getBoundary

  ( const String&       name,
    const Topology&     topo,
    const Ref<SShape>&  sfuncs )

{
  BShapeVector   bounds ( BOUNDARY_COUNT );
  SBShapeVector  bfuncs ( BOUNDARY_COUNT );

  Matrix         u      = StdLine::vertexCoords ();


  bounds[0] = newInstance<BoundaryPoint> (
    joinNames ( name, "point0" ),
    -1,
    -1.0,
    nullptr,
    sfuncs
  );

  bounds[1] = newInstance<BoundaryPoint> (
    joinNames ( name, "point1" ),
    1,
    1.0,
    nullptr,
    sfuncs
  );

  bfuncs[0] = newInstance<StdBoundary> (
    u(ALL,slice(0,1)),
    newInstance<StdPoint> (),
    sfuncs
  );

  bfuncs[1] = newInstance<StdBoundary> (
    u(ALL,slice(1,2)),
    newInstance<StdPoint> (),
    sfuncs
  );

  return ShapeBoundary ( bounds, bfuncs, topo );
}


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Line::Utils_::getShape

  ( const String&      name,
    IShapeCtor         ctor,
    const Properties&  conf,
    const Properties&  props,
    String&            ischeme,
    const String&      sfType )

{
  Ref<SShape>  sfuncs;
  Ref<IShape>  shape;


  sfuncs = StdShapeFactory::findShapeFuncs (
    GEOMETRY,
    conf, props, sfType,
    StdLine::RANK
  );

  props.find ( ischeme, PropNames::ISCHEME );

  try
  {
    shape = ctor (
      name,
      StdLine::getIntegrationScheme ( ischeme ),
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
//   class Line
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Line::GEOMETRY  = Geometries::LINE;
const char*  Line::TYPE_NAME = "Line";

#ifndef _MSC_EXTENSIONS
const int    Line::BOUNDARY_COUNT;
#endif


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Line::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  xfuncs,
    const Ref<SShape>&  sfuncs )

{
  return getShape (
    name,
    StdLine::getIntegrationScheme ( ischeme ),
    xfuncs,
    sfuncs
  );
}


Ref<IShape> Line::getShape

  ( const String&        name,
    const Matrix&        ischeme,
    const Ref<SShape>&   xfuncs,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( xfuncs  &&
                  xfuncs->getGeometry() == GEOMETRY,
                  "invalid geometric shape functions" );
  JEM_PRECHECK2 ( sfuncs == nullptr  ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return newInstance<ParametricLine> (
    name,
    ischeme,
    Utils_::getBoundary (
      name,
      xfuncs,
      sfuncs == nullptr ? xfuncs : sfuncs
    ),
    xfuncs,
    sfuncs
  );
}


Ref<IShape> Line::getShape

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

  Matrix        ischeme;
  String        ispecs;

  myProps.get ( ispecs, PropNames::ISCHEME );

  try
  {
    ischeme.ref ( StdLine::getIntegrationScheme( ispecs ) );
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


void Line::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,          & getShape );
  IShapeFactory::declare ( "jive::geom::Line", & getShape );
}


//=======================================================================
//   class Line2
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Line2::TYPE_NAME = "Line2";
const char*  Line2::ISCHEME   = "Gauss2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Line2::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<IShape> Line2::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return getShape (
    name,
    StdLine::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<IShape> Line2::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  Ref<SShape>  xshape = newInstance<StdLine2> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<Line2Shape> (
    name,
    ischeme,
    Utils_::getBoundary ( name, getBoundaryTopology(), sshape ),
    xshape,
    sshape
  );
}


Ref<IShape> Line2::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return Utils_::getShape ( name,
                            & getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "Linear" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Line2::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( BOUNDARY_COUNT * NODES_PER_BOUNDARY );
  idx_t      i;


  offsets[0] = 0;

  for ( i = 0; i < BOUNDARY_COUNT; i++ )
  {
    offsets[i + 1] = offsets[i] + NODES_PER_BOUNDARY;
  }

  indices[0] = 0;
  indices[1] = 1;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Line2::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,           & getShape );
  IShapeFactory::declare ( "jive::geom::Line2", & getShape );
}


//=======================================================================
//   class Line3
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Line3::TYPE_NAME = "Line3";
const char*  Line3::ISCHEME   = "Gauss3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Line3::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<IShape> Line3::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return getShape (
    name,
    StdLine::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<IShape> Line3::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  Ref<SShape>  xshape = newInstance<StdLine3> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricLine> (
    name,
    ischeme,
    Utils_::getBoundary ( name, getBoundaryTopology(), sshape ),
    xshape,
    sshape
  );
}


Ref<IShape> Line3::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return Utils_::getShape ( name,
                            & getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "Quadratic" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Line3::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( BOUNDARY_COUNT * NODES_PER_BOUNDARY );
  idx_t      i;


  offsets[0] = 0;

  for ( i = 0; i < BOUNDARY_COUNT; i++ )
  {
    offsets[i + 1] = offsets[i] + NODES_PER_BOUNDARY;
  }

  indices[0] = 0;
  indices[1] = 2;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Line3::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,           & getShape );
  IShapeFactory::declare ( "jive::geom::Line3", & getShape );
}


//=======================================================================
//   class Line4
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Line4::TYPE_NAME = "Line4";
const char*  Line4::ISCHEME   = "Gauss4";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Line4::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<IShape> Line4::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return getShape (
    name,
    StdLine::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<IShape> Line4::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  Ref<SShape>  xshape = newInstance<StdLine4> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricLine> (
    name,
    ischeme,
    Utils_::getBoundary ( name, getBoundaryTopology(), sshape ),
    xshape,
    sshape
  );
}


Ref<IShape> Line4::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return Utils_::getShape ( name,
                            & getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "Cubic" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Line4::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( BOUNDARY_COUNT * NODES_PER_BOUNDARY );
  idx_t      i;


  offsets[0] = 0;

  for ( i = 0; i < BOUNDARY_COUNT; i++ )
  {
    offsets[i + 1] = offsets[i] + NODES_PER_BOUNDARY;
  }

  indices[0] = 0;
  indices[1] = 3;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Line4::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,           & getShape );
  IShapeFactory::declare ( "jive::geom::Line4", & getShape );
}


JIVE_END_PACKAGE( geom )
