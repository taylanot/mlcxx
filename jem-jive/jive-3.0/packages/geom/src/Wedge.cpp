
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
#include <jem/base/array/select.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/geom/Names.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/StdBoundary.h>
#include <jive/geom/StdTriangle.h>
#include <jive/geom/StdSquare.h>
#include <jive/geom/StdWedge.h>
#include <jive/geom/ParametricVolume.h>
#include <jive/geom/ParametricSurface.h>
#include <jive/geom/BoundaryTriangle.h>
#include <jive/geom/BoundaryQuad.h>
#include <jive/geom/IShapeFactory.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/Wedge.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::Topology;


//=======================================================================
//   class Wedge::Utils_
//=======================================================================


class Wedge::Utils_
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
      BShapeCtor              tctor,
      BShapeCtor              qctor,
      const MatrixVector&     ischeme,
      const Topology&         topo,
      const Ref<SShape>&      sfuncs );

  static Ref<IShape>        getShape

    ( const String&           name,
      IShapeCtor              ctor,
      String&                 ischeme,
      String&                 tscheme,
      String&                 qscheme,
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


ShapeBoundary Wedge::Utils_::getBoundary

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

    Wedge6::getBoundaryTopology().getColumnIndices ();

  Matrix         u         = StdWedge6::vertexCoords ();
  Matrix         v         = Matrix ( u[inodes] );

  String         baseName  = joinNames ( name, "boundary" );

  idx_t          ipos;
  int            j = 0;


  // The first triangular boundary.

  bxfuncs[0] = newInstance<StdBoundary> (
    v[slice(j,j + 3)],
    newInstance<StdTriangle3> (),
    xfuncs
  );

  j += 3;

  // The three quadrilateral boundaries.

  bxfuncs[1] = newInstance<StdBoundary> (
    v[slice(j,j + 4)],
    newInstance<StdSquare4> (),
    xfuncs
  );

  j += 4;

  for ( int i = 2; i < 4; i++, j += 4 )
  {
    bxfuncs[i] = newInstance<StdBoundary> (
      bxfuncs[1],
      v[slice(j,j + 4)]
    );
  }

  // The last triangular boundary.

  bxfuncs[4] = newInstance<StdBoundary> (
    bxfuncs[0],
    v[slice(j,j + 3)]
  );

  if ( sfuncs && sfuncs != xfuncs )
  {
    j = 0;

    bsfuncs[0] = newInstance<StdBoundary> (
      v[slice(j,j + 3)],
      newInstance<StdTriangle3> (),
      sfuncs
    );

    j += 3;

    bsfuncs[1] = newInstance<StdBoundary> (
      v[slice(j,j + 4)],
      newInstance<StdSquare4> (),
      sfuncs
    );

    j += 4;

    for ( int i = 2; i < 4; i++, j += 4 )
    {
      bsfuncs[i] = newInstance<StdBoundary> (
        bsfuncs[1],
        v[slice(j,j + 4)]
      );
    }

    bsfuncs[4] = newInstance<StdBoundary> (
      bsfuncs[0],
      v[slice(j,j + 3)]
    );
  }

  for ( int i = 0; i < BOUNDARY_COUNT; i++ )
  {
    bounds[i] = newInstance<ParametricSurface> (
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


ShapeBoundary Wedge::Utils_::getBoundary

  ( const String&        name,
    BShapeCtor           tctor,
    BShapeCtor           qctor,
    const MatrixVector&  ischeme,
    const Topology&      topo,
    const Ref<SShape>&   sfuncs )

{
  BShapeVector   bounds   ( BOUNDARY_COUNT );
  SBShapeVector  bfuncs   ( BOUNDARY_COUNT );

  String         baseName = joinNames ( name, "boundary" );

  Matrix         u        = StdWedge6::vertexCoords ();
  Matrix         v;

  IdxVector      inodes;
  int            j = 0;


  if ( topo.size(1) == Wedge6::NODE_COUNT )
  {
    inodes.ref ( topo.getColumnIndices() );
  }
  else
  {
    inodes.ref (
      Wedge6::getBoundaryTopology().getColumnIndices ()
    );
  }

  v.ref ( Matrix( u[inodes] ) );

  // The first triangular boundary.

  bfuncs[0] = newInstance<StdBoundary> (
    v[slice(j,j + 3)],
    newInstance<StdTriangle3> (),
    sfuncs
  );

  j += 3;

  bounds[0] = tctor ( baseName  + '0',
                      ischeme[0], bfuncs[0] );

  // The three quadrilateral boundaries.

  bfuncs[1] = newInstance<StdBoundary> (
    v[slice(j,j + 4)],
    newInstance<StdSquare4> (),
    sfuncs
  );

  j += 4;

  for ( int i = 2; i < 4; i++, j += 4 )
  {
    bfuncs[i] = newInstance<StdBoundary> (
      bfuncs[1],
      v[slice(j,j + 4)]
    );
  }

  for ( int i = 1; i < 4; i++ )
  {
    bounds[i] = qctor ( baseName  + String ( i ),
                        ischeme[i], bfuncs[i] );
  }

  // The last triangular boundary.

  bfuncs[4] = newInstance<StdBoundary> (
    bfuncs[0],
    v[slice(j,j + 3)]
  );

  bounds[4] = tctor ( baseName  + '4',
                      ischeme[4], bfuncs[4] );

  return ShapeBoundary ( bounds, bfuncs, topo );
}


//-----------------------------------------------------------------------
//   getShape (given integration schemes)
//-----------------------------------------------------------------------


Ref<IShape> Wedge::Utils_::getShape

  ( const String&        name,
    IShapeCtor           ctor,
    String&              ischeme,
    String&              tscheme,
    String&              qscheme,
    const Ref<SShape>&   sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );

  s[0].ref ( StdWedge   ::getIntegrationScheme( ischeme ) );
  s[1].ref ( StdTriangle::getIntegrationScheme( tscheme ) );
  s[5].ref ( s[1] );
  s[2].ref ( StdSquare  ::getIntegrationScheme( qscheme ) );
  s[3].ref ( s[2] );
  s[4].ref ( s[2] );

  return ctor ( name, s, sfuncs );
}


//-----------------------------------------------------------------------
//   getShape (given properties)
//-----------------------------------------------------------------------


Ref<IShape> Wedge::Utils_::getShape

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
    StdWedge::RANK
  );

  if ( props.find( t, PropNames::ISCHEME ) )
  {
    if      ( t.size() == 1 )
    {
      ischeme[0] = t[0];
    }
    else if ( t.size() == 3 )
    {
      ischeme[0] = t[0];
      ischeme[1] = t[1];
      ischeme[2] = t[2];
      ischeme[3] = t[2];
      ischeme[4] = t[2];
      ischeme[5] = t[1];
    }
    else if ( t.size() == s.size() )
    {
      ischeme = t;
    }
    else
    {
      props.propertyError (
        PropNames::ISCHEME,
        "array must have length 1, 3 or 6"
      );
    }
  }

  try
  {
    s[0].ref ( StdWedge   ::getIntegrationScheme( ischeme[0] ) );
    s[1].ref ( StdTriangle::getIntegrationScheme( ischeme[1] ) );
    s[5].ref ( StdTriangle::getIntegrationScheme( ischeme[5] ) );

    for ( int i = 2; i < 5; i++ )
    {
      s[i].ref ( StdSquare::getIntegrationScheme( ischeme[i] ) );
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
//   class Wedge
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Wedge::GEOMETRY  = Geometries::WEDGE;
const char*  Wedge::TYPE_NAME = "Wedge";

#ifndef _MSC_EXTENSIONS
const int    Wedge::BOUNDARY_COUNT;
#endif


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Wedge::getShape

  ( const String&       name,
    String&             ischeme,
    String&             tscheme,
    String&             qscheme,
    const Ref<SShape>&  xfuncs,
    const Ref<SShape>&  sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );

  s[0].ref ( StdWedge   ::getIntegrationScheme( ischeme ) );
  s[1].ref ( StdTriangle::getIntegrationScheme( tscheme ) );
  s[5].ref ( s[1] );
  s[2].ref ( StdSquare  ::getIntegrationScheme( qscheme ) );
  s[3].ref ( s[2] );
  s[4].ref ( s[2] );

  return getShape ( name, s, xfuncs, sfuncs );
}


Ref<IShape> Wedge::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   xfuncs,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( xfuncs  &&
                  xfuncs->getGeometry() == GEOMETRY,
                  "invalid geometric shape functions" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return newInstance<ParametricVolume> (
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


Ref<IShape> Wedge::getShape

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

  if      ( t.size() == 3 )
  {
    ispecs[0] = t[0];
    ispecs[1] = t[1];
    ispecs[2] = t[2];
    ispecs[3] = t[2];
    ispecs[4] = t[2];
    ispecs[5] = t[1];
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
        "array must have length 3 or %d",
        ispecs.size ()
      )
    );
  }

  try
  {
    ischeme[0].ref ( StdWedge   ::getIntegrationScheme( ispecs[0] ) );
    ischeme[1].ref ( StdTriangle::getIntegrationScheme( ispecs[1] ) );
    ischeme[5].ref ( StdTriangle::getIntegrationScheme( ispecs[5] ) );

    for ( int i = 2; i < 5; i++ )
    {
      ischeme[i].ref ( StdSquare::getIntegrationScheme( ispecs[i] ) );
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


void Wedge::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,           & getShape );
  IShapeFactory::declare ( "jive::geom::Wedge", & getShape );
}


//=======================================================================
//   class Wedge6
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Wedge6::TYPE_NAME = "Wedge6";
const char*  Wedge6::ISCHEME   = "3*2";
const char*  Wedge6::TSCHEME   = "3";
const char*  Wedge6::QSCHEME   = "2*2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Wedge6::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  tscheme = TSCHEME;
  String  qscheme = QSCHEME;

  return Utils_::getShape ( name,    getShape,
                            ischeme, tscheme, qscheme, sfuncs );
}


Ref<IShape> Wedge6::getShape

  ( const String&       name,
    String&             ischeme,
    String&             tscheme,
    String&             qscheme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name,    getShape,
                            ischeme, tscheme, qscheme, sfuncs );
}


Ref<IShape> Wedge6::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );


  Ref<SShape>  xshape = newInstance<StdWedge6> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricVolume> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      & BoundaryTriangle3::getShape,
      & BoundaryQuad4    ::getShape,
      ischeme[slice(1,END)],
      getBoundaryTopology (),
      sshape
    ),
    xshape,
    sshape
  );
}


Ref<IShape> Wedge6::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0] = ISCHEME;
  ischeme[1] = TSCHEME;
  ischeme[2] = QSCHEME;
  ischeme[3] = QSCHEME;
  ischeme[4] = QSCHEME;
  ischeme[5] = TSCHEME;

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


Topology Wedge6::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( 3 * 4 + 2 * 3 );
  idx_t      i;


  i = 0;

  offsets[i++] = 0;
  offsets[i++] = 3;
  offsets[i++] = 7;
  offsets[i++] = 11;
  offsets[i++] = 15;
  offsets[i++] = 18;

  i = 0;

  // First boundary

  indices[i++] = 0;
  indices[i++] = 2;
  indices[i++] = 1;

  // Second boundary

  indices[i++] = 0;
  indices[i++] = 1;
  indices[i++] = 4;
  indices[i++] = 3;

  // Third boundary

  indices[i++] = 1;
  indices[i++] = 2;
  indices[i++] = 5;
  indices[i++] = 4;

  // Fourth boundary

  indices[i++] = 2;
  indices[i++] = 0;
  indices[i++] = 3;
  indices[i++] = 5;

  // Fifth boundary

  indices[i++] = 3;
  indices[i++] = 4;
  indices[i++] = 5;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Wedge6::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,             & getShape );
  IShapeFactory::declare ( "jive::geom::Wedge6",  & getShape );
}


//=======================================================================
//   class Wedge18
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Wedge18::TYPE_NAME = "Wedge18";
const char*  Wedge18::ISCHEME   = "7*3";
const char*  Wedge18::TSCHEME   = "7";
const char*  Wedge18::QSCHEME   = "3*3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Wedge18::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  tscheme = TSCHEME;
  String  qscheme = QSCHEME;

  return Utils_::getShape ( name,    getShape,
                            ischeme, tscheme, qscheme, sfuncs );
}


Ref<IShape> Wedge18::getShape

  ( const String&       name,
    String&             ischeme,
    String&             tscheme,
    String&             qscheme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name,    getShape,
                            ischeme, tscheme, qscheme, sfuncs );
}


Ref<IShape> Wedge18::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );


  Ref<SShape>  xshape = newInstance<StdWedge18> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricVolume> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      & BoundaryTriangle6::getShape,
      & BoundaryQuad9    ::getShape,
      ischeme[slice(1,END)],
      getBoundaryTopology (),
      sshape
    ),
    xshape,
    sshape
  );
}


Ref<IShape> Wedge18::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0] = ISCHEME;
  ischeme[1] = TSCHEME;
  ischeme[2] = QSCHEME;
  ischeme[3] = QSCHEME;
  ischeme[4] = QSCHEME;
  ischeme[5] = TSCHEME;

  return Utils_::getShape ( name,
                            & getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "BiQuadratic" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Wedge18::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( 3 * 9 + 2 * 6 );
  idx_t      i;


  i = 0;

  offsets[i++] = 0;
  offsets[i++] = 6;
  offsets[i++] = 15;
  offsets[i++] = 24;
  offsets[i++] = 33;
  offsets[i++] = 39;

  i = 0;

  // First boundary

  indices[i++] = 0;
  indices[i++] = 5;
  indices[i++] = 4;
  indices[i++] = 3;
  indices[i++] = 2;
  indices[i++] = 1;

  // Second boundary

  indices[i++] = 0;
  indices[i++] = 1;
  indices[i++] = 2;
  indices[i++] = 8;
  indices[i++] = 14;
  indices[i++] = 13;
  indices[i++] = 12;
  indices[i++] = 6;
  indices[i++] = 7;

  // Third boundary

  indices[i++] = 2;
  indices[i++] = 3;
  indices[i++] = 4;
  indices[i++] = 10;
  indices[i++] = 16;
  indices[i++] = 15;
  indices[i++] = 14;
  indices[i++] = 8;
  indices[i++] = 9;

  // Fourth boundary

  indices[i++] = 4;
  indices[i++] = 5;
  indices[i++] = 0;
  indices[i++] = 6;
  indices[i++] = 12;
  indices[i++] = 17;
  indices[i++] = 16;
  indices[i++] = 10;
  indices[i++] = 11;

  // Fifth boundary

  indices[i++] = 12;
  indices[i++] = 13;
  indices[i++] = 14;
  indices[i++] = 15;
  indices[i++] = 16;
  indices[i++] = 17;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Wedge18::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,             & getShape );
  IShapeFactory::declare ( "jive::geom::Wedge18", & getShape );
}


JIVE_END_PACKAGE( geom )
