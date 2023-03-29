
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
#include <jive/geom/StdTetrahedron.h>
#include <jive/geom/ParametricVolume.h>
#include <jive/geom/ParametricSurface.h>
#include <jive/geom/BoundaryTriangle.h>
#include <jive/geom/IShapeFactory.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/Tetrahedron.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::Topology;


//=======================================================================
//   class Tetrahedron::Utils_
//=======================================================================


class Tetrahedron::Utils_
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


ShapeBoundary Tetrahedron::Utils_::getBoundary

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

    Tetrahedron4::getBoundaryTopology().getColumnIndices ();

  Matrix         u         = StdTetrahedron4::vertexCoords ();
  Matrix         v         = Matrix ( u[inodes] );

  String         baseName  = joinNames ( name, "boundary" );

  idx_t          ipos;


  bxfuncs[0] = newInstance<StdBoundary> (
    v[slice(0,3)],
    newInstance<StdTriangle3> (),
    xfuncs
  );

  for ( int i = 1; i < BOUNDARY_COUNT; i++ )
  {
    int  j = 3 * i;

    bxfuncs[i] = newInstance<StdBoundary> ( bxfuncs[0],
                                            v[slice(j,j + 3)] );
  }

  if ( sfuncs && sfuncs != xfuncs )
  {
    bsfuncs[0] = newInstance<StdBoundary> (
      v[slice(0,3)],
      newInstance<StdTriangle3> (),
      sfuncs
    );

    for ( int i = 1; i < BOUNDARY_COUNT; i++ )
    {
      int  j = 3 * i;

      bsfuncs[i] = newInstance<StdBoundary> ( bsfuncs[0],
                                              v[slice(j,j + 3)] );
    }
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


ShapeBoundary Tetrahedron::Utils_::getBoundary

  ( const String&        name,
    BShapeCtor           ctor,
    const MatrixVector&  ischeme,
    const Topology&      topo,
    const Ref<SShape>&   sfuncs )

{
  BShapeVector   bounds   ( BOUNDARY_COUNT );
  SBShapeVector  bfuncs   ( BOUNDARY_COUNT );

  String         baseName = joinNames ( name, "boundary" );

  Matrix         u        = StdTetrahedron::vertexCoords ();
  Matrix         v;

  IdxVector      inodes;


  if ( topo.size(1) == Tetrahedron4::NODE_COUNT )
  {
    inodes.ref ( topo.getColumnIndices() );
  }
  else
  {
    inodes.ref (
      Tetrahedron4::getBoundaryTopology().getColumnIndices ()
    );
  }

  v.ref ( Matrix( u[inodes] ) );

  bfuncs[0] = newInstance<StdBoundary> (
    v[slice(0,3)],
    newInstance<StdTriangle3> (),
    sfuncs
  );

  for ( int i = 1; i < BOUNDARY_COUNT; i++ )
  {
    int  j = 3 * i;

    bfuncs[i] = newInstance<StdBoundary> (
      bfuncs[0],
      v[slice(j,j + 3)]
    );
  }

  for ( int i = 0; i < BOUNDARY_COUNT; i++ )
  {
    bounds[i] = ctor ( baseName  + String ( i ),
                       ischeme[i], bfuncs[i] );
  }

  return ShapeBoundary ( bounds, bfuncs, topo );
}


//-----------------------------------------------------------------------
//   getShape (given integration schemes)
//-----------------------------------------------------------------------


Ref<IShape> Tetrahedron::Utils_::getShape

  ( const String&        name,
    IShapeCtor           ctor,
    String&              ischeme,
    String&              bscheme,
    const Ref<SShape>&   sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );


  s[0].ref ( StdTetrahedron::getIntegrationScheme( ischeme ) );
  s[1].ref ( StdTriangle   ::getIntegrationScheme( bscheme ) );

  for ( idx_t i = 2; i < s.size(); i++ )
  {
    s[i].ref ( s[1] );
  }

  return ctor ( name, s, sfuncs );
}


//-----------------------------------------------------------------------
//   getShape (given properties)
//-----------------------------------------------------------------------


Ref<IShape> Tetrahedron::Utils_::getShape

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
    StdTetrahedron::RANK
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
        "array must have length 1, 2 or 5"
      );
    }
  }

  try
  {
    s[0].ref ( StdTetrahedron::getIntegrationScheme( ischeme[0] ) );

    for ( idx_t i = 1; i < s.size(); i++ )
    {
      s[i].ref ( StdTriangle::getIntegrationScheme( ischeme[i] ) );
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
//   class Tetrahedron
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Tetrahedron::GEOMETRY  = Geometries::TETRAHEDRON;
const char*  Tetrahedron::TYPE_NAME = "Tet";

#ifndef _MSC_EXTENSIONS
const int    Tetrahedron::BOUNDARY_COUNT;
#endif


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Tetrahedron::getShape

  ( const String&       name,
    String&             ischeme,
    String&             bscheme,
    const Ref<SShape>&  xfuncs,
    const Ref<SShape>&  sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );

  s[0].ref ( StdTetrahedron::getIntegrationScheme( ischeme ) );
  s[1].ref ( StdTriangle   ::getIntegrationScheme( bscheme ) );

  for ( idx_t i = 2; i < s.size(); i++ )
  {
    s[i].ref ( s[1] );
  }

  return getShape ( name, s, xfuncs, sfuncs );
}


Ref<IShape> Tetrahedron::getShape

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


Ref<IShape> Tetrahedron::getShape

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
      StdTetrahedron::getIntegrationScheme ( ispecs[0] )
    );

    for ( idx_t i = 1; i < ischeme.size(); i++ )
    {
      ischeme[i].ref (
        StdTriangle::getIntegrationScheme ( ispecs[i] )
      );
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


void Tetrahedron::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,                 & getShape );
  IShapeFactory::declare ( "jive::geom::Tetrahedron", & getShape );
}


//=======================================================================
//   class Tetrahedron4
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Tetrahedron4::TYPE_NAME = "Tet4";
const char*  Tetrahedron4::ISCHEME   = "Gauss4";
const char*  Tetrahedron4::BSCHEME   = "Gauss3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Tetrahedron4::getShape

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


Ref<IShape> Tetrahedron4::getShape

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


Ref<IShape> Tetrahedron4::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  Ref<SShape>  xshape = newInstance<StdTetrahedron4> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricVolume> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      & BoundaryTriangle3::getShape,
      ischeme[slice(1,END)],
      getBoundaryTopology (),
      sshape
    ),
    xshape,
    sshape
  );
}


Ref<IShape> Tetrahedron4::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0]            = ISCHEME;
  ischeme[slice(1,END)] = BSCHEME;

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


Topology Tetrahedron4::getBoundaryTopology ()
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

  // First boundary

  indices[i++] = 0;
  indices[i++] = 2;
  indices[i++] = 1;

  // Second boundary

  indices[i++] = 0;
  indices[i++] = 1;
  indices[i++] = 3;

  // Third boundary

  indices[i++] = 1;
  indices[i++] = 2;
  indices[i++] = 3;

  // Fourth boundary

  indices[i++] = 0;
  indices[i++] = 3;
  indices[i++] = 2;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Tetrahedron4::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,                  & getShape );
  IShapeFactory::declare ( "jive::geom::Tetrahedron4", & getShape );
}


//=======================================================================
//   class Tetrahedron10
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Tetrahedron10::TYPE_NAME = "Tet10";
const char*  Tetrahedron10::ISCHEME   = "Gauss11";
const char*  Tetrahedron10::BSCHEME   = "Gauss7";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Tetrahedron10::getShape

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


Ref<IShape> Tetrahedron10::getShape

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


Ref<IShape> Tetrahedron10::getShape

  ( const String&        name,
    const MatrixVector&  ischeme,
    const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );


  Ref<SShape>  xshape = newInstance<StdTetrahedron10> ();
  Ref<SShape>  sshape = sfuncs ? sfuncs : xshape;

  return newInstance<ParametricVolume> (
    name,
    ischeme[0],
    Utils_::getBoundary (
      name,
      & BoundaryTriangle6::getShape,
      ischeme[slice(1,END)],
      getBoundaryTopology (),
      sshape
    ),
    xshape,
    sshape
  );
}


Ref<IShape> Tetrahedron10::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0]            = ISCHEME;
  ischeme[slice(1,END)] = BSCHEME;

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


Topology Tetrahedron10::getBoundaryTopology ()
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
  indices[i++] = 7;
  indices[i++] = 9;
  indices[i++] = 6;

  // Third boundary

  indices[i++] = 2;
  indices[i++] = 3;
  indices[i++] = 4;
  indices[i++] = 8;
  indices[i++] = 9;
  indices[i++] = 7;

  // Fourth boundary

  indices[i++] = 0;
  indices[i++] = 6;
  indices[i++] = 9;
  indices[i++] = 8;
  indices[i++] = 4;
  indices[i++] = 5;

  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Tetrahedron10::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,                   & getShape );
  IShapeFactory::declare ( "jive::geom::Tetrahedron10", & getShape );
}


JIVE_END_PACKAGE( geom )
