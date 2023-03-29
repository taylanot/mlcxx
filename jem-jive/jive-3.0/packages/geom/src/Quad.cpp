
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
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/select.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/geom/Names.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/StdEdge.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdSquare.h>
#include <jive/geom/ParametricArea.h>
#include <jive/geom/ParametricEdge.h>
#include <jive/geom/BoundaryLine.h>
#include <jive/geom/IShapeFactory.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/Quad.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::shape;
using jem::newInstance;
using jive::util::joinNames;
using jive::util::Topology;


//=======================================================================
//   class Quad::Utils_
//=======================================================================


class Quad::Utils_
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

  static ShapeBoundary      getBoundary

    ( const String&           name,
      BShapeCtor              ctor1,
      BShapeCtor              ctor2,
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
      String&                 ischeme,
      String&                 bscheme1,
      String&                 bscheme2,
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


ShapeBoundary Quad::Utils_::getBoundary

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

    Quad4::getBoundaryTopology().getColumnIndices ();

  Matrix         u         = StdSquare4::vertexCoords ();
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


inline ShapeBoundary Quad::Utils_::getBoundary

  ( const String&        name,
    BShapeCtor           ctor,
    const MatrixVector&  ischeme,
    const Topology&      topo,
    const Ref<SShape>&   sfuncs )

{
  return getBoundary ( name, ctor, ctor, ischeme, topo, sfuncs );
}


ShapeBoundary Quad::Utils_::getBoundary

  ( const String&        name,
    BShapeCtor           ctor1,
    BShapeCtor           ctor2,
    const MatrixVector&  ischeme,
    const Topology&      topo,
    const Ref<SShape>&   sfuncs )

{
  BShapeVector   bounds   ( BOUNDARY_COUNT );
  SBShapeVector  bfuncs   ( BOUNDARY_COUNT );

  String         baseName = joinNames ( name, "boundary" );

  Matrix         u        ( 2, 4 + 1 );


  u[slice(0,4)] = StdSquare4::vertexCoords ();
  u[4]          = u[0];

  bfuncs[0] = newInstance<StdEdge> ( u[slice(0,2)], sfuncs );

  for ( int i = 1; i < BOUNDARY_COUNT; i++ )
  {
    bfuncs[i] = newInstance<StdEdge> ( bfuncs[0], u[slice(i, i + 2)] );
  }

  bounds[0] = ctor1 ( baseName + '0', ischeme[0], bfuncs[0] );
  bounds[1] = ctor2 ( baseName + '1', ischeme[1], bfuncs[1] );
  bounds[2] = ctor1 ( baseName + '2', ischeme[2], bfuncs[2] );
  bounds[3] = ctor2 ( baseName + '3', ischeme[3], bfuncs[3] );

  return ShapeBoundary ( bounds, bfuncs, topo );
}


//-----------------------------------------------------------------------
//   getShape (given integration schemes)
//-----------------------------------------------------------------------


Ref<IShape> Quad::Utils_::getShape

  ( const String&        name,
    IShapeCtor           ctor,
    String&              ischeme,
    String&              bscheme,
    const Ref<SShape>&   sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );


  s[0].ref ( StdSquare::getIntegrationScheme( ischeme ) );
  s[1].ref ( StdLine  ::getIntegrationScheme( bscheme ) );

  for ( idx_t i = 2; i < s.size(); i++ )
  {
    s[i].ref ( s[1] );
  }

  return ctor ( name, s, sfuncs );
}


Ref<IShape> Quad::Utils_::getShape

  ( const String&       name,
    IShapeCtor          ctor,
    String&             ischeme,
    String&             bscheme1,
    String&             bscheme2,
    const Ref<SShape>&  sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );

  s[0].ref ( StdSquare::getIntegrationScheme( ischeme  ) );
  s[1].ref ( StdLine  ::getIntegrationScheme( bscheme1 ) );
  s[2].ref ( StdLine  ::getIntegrationScheme( bscheme2 ) );
  s[3].ref ( s[1] );
  s[4].ref ( s[2] );

  return ctor ( name, s, sfuncs );
}


//-----------------------------------------------------------------------
//   getShape (given properties)
//-----------------------------------------------------------------------


Ref<IShape> Quad::Utils_::getShape

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
    StdSquare::RANK
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
    else if ( t.size() == 3 )
    {
      ischeme[0] = t[0];
      ischeme[1] = t[1];
      ischeme[2] = t[2];
      ischeme[3] = t[1];
      ischeme[4] = t[2];
    }
    else if ( t.size() == s.size() )
    {
      ischeme = t;
    }
    else
    {
      props.propertyError (
        PropNames::ISCHEME,
        "array must have length 1, 2, 3 or 5"
      );
    }
  }

  try
  {
    s[0].ref ( StdSquare::getIntegrationScheme( ischeme[0] ) );

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
//   class Quad
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Quad::GEOMETRY  = Geometries::SQUARE;
const char*  Quad::TYPE_NAME = "Quad";

#ifndef _MSC_EXTENSIONS
const int    Quad::BOUNDARY_COUNT;
#endif


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Quad::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  xfuncs,
    const Ref<SShape>&  sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );

  s[0].ref ( StdSquare::getIntegrationScheme( ischeme ) );
  s[1].ref ( StdLine  ::getIntegrationScheme( bscheme ) );

  for ( idx_t i = 2; i < s.size(); i++ )
  {
    s[i].ref ( s[1] );
  }

  return getShape ( name, s, xfuncs, sfuncs );
}


Ref<IShape> Quad::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   xfuncs,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == (BOUNDARY_COUNT + 1),
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( xfuncs  &&
                  xfuncs->getGeometry() == GEOMETRY,
                  "invalid geometric shape functions" );
  JEM_PRECHECK2 ( sfuncs == nullptr  ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape function" );

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


Ref<IShape> Quad::getShape

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
    ischeme[0].ref ( StdSquare::getIntegrationScheme( ispecs[0] ) );

    for ( idx_t i = 1; i < ispecs.size(); i++ )
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


void Quad::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,          & getShape );
  IShapeFactory::declare ( "jive::geom::Quad", & getShape );
}


//=======================================================================
//   class Quad4
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Quad4::TYPE_NAME = "Quad4";
const char*  Quad4::ISCHEME   = "2*2";
const char*  Quad4::BSCHEME   = "2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Quad4::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  bscheme = BSCHEME;

  return Utils_::getShape ( name, getShape, ischeme, bscheme, sfuncs );
}


Ref<IShape> Quad4::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name, getShape, ischeme, bscheme, sfuncs );
}


Ref<IShape> Quad4::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  Ref<SShape>  xshape = newInstance<StdSquare4> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricArea> (
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


Ref<IShape> Quad4::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0]            = ISCHEME;
  ischeme[slice(1,END)] = BSCHEME;

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


Topology Quad4::getBoundaryTopology ()
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
  indices[i++] = 3;

  indices[i++] = 3;
  indices[i++] = 0;

  return Topology ( shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Quad4::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,           & getShape );
  IShapeFactory::declare ( "jive::geom::Quad4", & getShape );
}


//=======================================================================
//   class Quad6
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Quad6::TYPE_NAME = "Quad6";
const char*  Quad6::ISCHEME   = "3*2";
const char*  Quad6::BSCHEME1  = "3";
const char*  Quad6::BSCHEME2  = "2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Quad6::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme  = ISCHEME;
  String  bscheme1 = BSCHEME1;
  String  bscheme2 = BSCHEME2;

  return Utils_::getShape ( name, getShape, ischeme,
                            bscheme1, bscheme2, sfuncs );
}


Ref<IShape> Quad6::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme1,
    String&             bscheme2,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name, getShape, ischeme,
                            bscheme1, bscheme2, sfuncs );
}


Ref<IShape> Quad6::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shaope functions" );

  Ref<SShape>  xshape = newInstance<StdSquare6> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricArea> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      & BoundaryLine3::getShape,
      & BoundaryLine2::getShape,
      ischeme[slice(1,END)],
      getBoundaryTopology (),
      sshape
    ),
    xshape,
    sshape
  );
}


Ref<IShape> Quad6::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0] = ISCHEME;
  ischeme[1] = BSCHEME1;
  ischeme[2] = BSCHEME2;
  ischeme[3] = BSCHEME1;
  ischeme[4] = BSCHEME2;

  return Utils_::getShape ( name,
                            getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "Quadratic*Linear" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Quad6::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( 10 );
  idx_t      i;


  offsets[0] =  0;
  offsets[1] =  3;
  offsets[2] =  5;
  offsets[3] =  8;
  offsets[4] = 10;

  i = 0;

  indices[i++] = 0;
  indices[i++] = 1;
  indices[i++] = 2;

  indices[i++] = 2;
  indices[i++] = 3;

  indices[i++] = 3;
  indices[i++] = 4;
  indices[i++] = 5;

  indices[i++] = 5;
  indices[i++] = 0;

  return Topology ( shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Quad6::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,           & getShape );
  IShapeFactory::declare ( "jive::geom::Quad6", & getShape );
}


//=======================================================================
//   class Quad8
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Quad8::TYPE_NAME = "Quad8";
const char*  Quad8::ISCHEME   = "3*3";
const char*  Quad8::BSCHEME   = "3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Quad8::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  bscheme = BSCHEME;

  return Utils_::getShape ( name, getShape, ischeme, bscheme, sfuncs );
}


Ref<IShape> Quad8::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name, getShape, ischeme, bscheme, sfuncs );
}


Ref<IShape> Quad8::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                 "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  Ref<SShape>  xshape = newInstance<StdSquare8> ();
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


Ref<IShape> Quad8::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0]            = ISCHEME;
  ischeme[slice(1,END)] = BSCHEME;

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


Topology Quad8::getBoundaryTopology ()
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
  indices[i++] = 6;

  indices[i++] = 6;
  indices[i++] = 7;
  indices[i++] = 0;

  return Topology ( shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Quad8::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,           & getShape );
  IShapeFactory::declare ( "jive::geom::Quad8", & getShape );
}


//=======================================================================
//   class Quad9
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Quad9::TYPE_NAME = "Quad9";
const char*  Quad9::ISCHEME   = "3*3";
const char*  Quad9::BSCHEME   = "3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Quad9::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  bscheme = BSCHEME;

  return Utils_::getShape ( name, getShape, ischeme, bscheme, sfuncs );
}


Ref<IShape> Quad9::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name, getShape, ischeme, bscheme, sfuncs );
}


Ref<IShape> Quad9::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  Ref<SShape>  xshape = newInstance<StdSquare9> ();
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


Ref<IShape> Quad9::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0]            = ISCHEME;
  ischeme[slice(1,END)] = BSCHEME;

  return Utils_::getShape ( name,
                            getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "BiQuadratic" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Quad9::getBoundaryTopology ()
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
  indices[i++] = 6;

  indices[i++] = 6;
  indices[i++] = 7;
  indices[i++] = 0;

  return Topology ( shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Quad9::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,           & getShape );
  IShapeFactory::declare ( "jive::geom::Quad9", & getShape );
}


//=======================================================================
//   class Quad12
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Quad12::TYPE_NAME = "Quad12";
const char*  Quad12::ISCHEME   = "4*4";
const char*  Quad12::BSCHEME   = "4";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Quad12::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  bscheme = BSCHEME;

  return Utils_::getShape ( name, getShape, ischeme, bscheme, sfuncs );
}


Ref<IShape> Quad12::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return Utils_::getShape ( name, getShape, ischeme, bscheme, sfuncs );
}


Ref<IShape> Quad12::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  Ref<SShape>  xshape = newInstance<StdSquare12> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricArea> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      & BoundaryLine4::getShape,
      ischeme[slice(1,END)],
      getBoundaryTopology (),
      sshape
    ),
    xshape,
    sshape
  );
}


Ref<IShape> Quad12::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0]            = ISCHEME;
  ischeme[slice(1,END)] = BSCHEME;

  return Utils_::getShape ( name,
                            getShape,
                            conf .makeProps ( name ),
                            props.findProps ( name ),
                            ischeme,
                            "Cubic" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Quad12::getBoundaryTopology ()
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

  indices[i++] =  0;
  indices[i++] =  1;
  indices[i++] =  2;
  indices[i++] =  3;

  indices[i++] =  3;
  indices[i++] =  4;
  indices[i++] =  5;
  indices[i++] =  6;

  indices[i++] =  6;
  indices[i++] =  7;
  indices[i++] =  8;
  indices[i++] =  9;

  indices[i++] =  9;
  indices[i++] = 10;
  indices[i++] = 11;
  indices[i++] =  0;

  return Topology ( shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Quad12::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,            & getShape );
  IShapeFactory::declare ( "jive::geom::Quad12", & getShape );
}


JIVE_END_PACKAGE( geom )
