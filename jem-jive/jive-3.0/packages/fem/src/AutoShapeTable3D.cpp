
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


#include <jem/base/ClassTemplate.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/GroupSet.h>
#include <jive/geom/Wedge.h>
#include <jive/geom/Names.h>
#include <jive/geom/Hexahedron.h>
#include <jive/geom/Tetrahedron.h>
#include <jive/geom/BoundaryQuad.h>
#include <jive/geom/BoundaryTriangle.h>
#include <jive/geom/ShapeTableFactory.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/AutoShapeTable3D.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::AutoShapeTable3D );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::GroupSet;
using jive::geom::IShape;
using jive::geom::BShape;


//=======================================================================
//   class AutoShapeTable3D
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  AutoShapeTable3D::TYPE_NAME = "Auto3D";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


AutoShapeTable3D::AutoShapeTable3D ()
{}


AutoShapeTable3D::AutoShapeTable3D

  ( const String&        name,
    const ElementSet&    elems,
    idx_t                maxPrec ) :

    Super ( name, elems, maxPrec )

{}


AutoShapeTable3D::AutoShapeTable3D

  ( const String&        name,
    const ElementGroup&  belems,
    const ElementGroup&  felems,
    idx_t                  maxPrec ) :

    Super ( name, belems, felems, maxPrec )

{}


AutoShapeTable3D::~AutoShapeTable3D ()
{}


//-----------------------------------------------------------------------
//   initTetProps
//-----------------------------------------------------------------------


void AutoShapeTable3D::initTetProps

  ( const Properties&  props,
    idx_t              prec,
    idx_t              nodeCount )

{
  using jive::geom::PropNames;

  StringVector  ischeme ( 2 );

  idx_t         k = prec + nodeCount / 4;


  ischeme[0] = String ( k * k * k );
  ischeme[1] = String ( k * k );

  props.set ( PropNames::ISCHEME, ischeme );
}


//-----------------------------------------------------------------------
//   initHexProps
//-----------------------------------------------------------------------


void AutoShapeTable3D::initHexProps

  ( const Properties&  props,
    idx_t              prec,
    idx_t              nodeCount )

{
  using jive::geom::PropNames;

  StringVector  ischeme ( 2 );

  idx_t         k = prec + 2;

  if ( nodeCount > 8 )
  {
    k++;
  }

  ischeme[0] = String::format ( "%d*%d*%d", k, k, k );
  ischeme[1] = String::format ( "%d*%d",    k, k );

  props.set ( PropNames::ISCHEME, ischeme );
}


//-----------------------------------------------------------------------
//   initWedgeProps
//-----------------------------------------------------------------------


void AutoShapeTable3D::initWedgeProps

  ( const Properties&  props,
    idx_t                prec,
    idx_t                nodeCount )

{
  using jive::geom::PropNames;

  StringVector  ischeme ( 3 );

  idx_t         k = prec + 1;
  idx_t         m;

  if ( nodeCount > 6 )
  {
    k++;
  }

  m = k + 1;

  ischeme[0] = String::format ( "%d*%d", k * k, m );
  ischeme[1] = String         ( k * k );
  ischeme[2] = String::format ( "%d*%d", m, m );

  props.set ( PropNames::ISCHEME, ischeme );
}


//-----------------------------------------------------------------------
//   initBTriangProps
//-----------------------------------------------------------------------


void AutoShapeTable3D::initBTriangProps

  ( const Properties&  props,
    idx_t              prec,
    idx_t              nodeCount )

{
  using jive::geom::PropNames;

  idx_t  k = prec + nodeCount / 3;

  props.set ( PropNames::ISCHEME, String( k * k ) );
}


//-----------------------------------------------------------------------
//   initBQuadProps
//-----------------------------------------------------------------------


void AutoShapeTable3D::initBQuadProps

  ( const Properties&  props,
    idx_t              prec,
    idx_t              nodeCount )

{
  using jive::geom::PropNames;

  idx_t  k = prec + 2;


  if ( nodeCount > 4 )
  {
    k++;
  }

  props.set ( PropNames::ISCHEME,
              String::format ( "%d*%d", k, k ) );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void AutoShapeTable3D::declare ()
{
  using jive::geom::ShapeTableFactory;

  ShapeTableFactory::declare ( TYPE_NAME,  & makeNew );
  ShapeTableFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<ShapeTable> AutoShapeTable3D::makeNew

  ( const String&        name,
    const Properties&    conf,
    const Properties&    props,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<GroupSet>  elems = jem::dynamicCast<GroupSet> ( items );

  if ( elems && ElementSet::isValidData( elems ) )
  {
    return newInstance<Self> ( name, ElementSet( elems ) );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   makeIShapes_
//-----------------------------------------------------------------------


bool AutoShapeTable3D::makeIShapes_

  ( PrintWriter&        log,
    const ShapeVector&  shapes,
    idx_t               nodeCount )

{
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::outdent;
  using jive::geom::Wedge6;
  using jive::geom::Wedge18;
  using jive::geom::Hexahedron8;
  using jive::geom::Hexahedron16;
  using jive::geom::Hexahedron20;
  using jive::geom::Tetrahedron4;
  using jive::geom::Tetrahedron10;

  Ref<IShape>          (*ctorFunc)

    ( const String&        name,
      const Properties&    conf,
      const Properties&    props );

  void                 (*initFunc)

    ( const Properties&    props,
      idx_t                prec,
      idx_t                nodeCount );


  const idx_t  n     = shapes.size ();

  Properties   props ( "shapeProps" );
  Properties   conf;

  String       type;
  String       name;


  ctorFunc = 0;
  initFunc = 0;

  switch ( nodeCount )
  {
  case 4:

    type     = "Tet4";
    ctorFunc = & Tetrahedron4::getShape;
    initFunc = & initTetProps;

    break;

  case 6:

    type     = "Wedge6";
    ctorFunc = & Wedge6::getShape;
    initFunc = & initWedgeProps;

    break;

  case 8:

    type     = "Hex8";
    ctorFunc = & Hexahedron8::getShape;
    initFunc = & initHexProps;

    break;

  case 10:

    type     = "Tet10";
    ctorFunc = & Tetrahedron10::getShape;
    initFunc = & initTetProps;

    break;

  case 16:

    type     = "Hex16";
    ctorFunc = & Hexahedron16::getShape;
    initFunc = & initHexProps;

    break;

  case 18:

    type     = "Wedge18";
    ctorFunc = & Wedge18::getShape;
    initFunc = & initWedgeProps;

    break;

  case 20:

    type     = "Hex20";
    ctorFunc = & Hexahedron20::getShape;
    initFunc = & initHexProps;

    break;
  }

  if ( ! ctorFunc )
  {
    return false;
  }

  name = joinNames ( myName_, type.toLower() );

  print ( log, "creating ", n, " shapes of type `" );
  print ( log, type, "\' ...\n", indent );

  for ( idx_t i = 0; i < n; i++ )
  {
    initFunc ( props.makeProps( name ), i, nodeCount );

    shapes[i] = ctorFunc ( name, conf, props );

    print ( log, "\nprecision ", i, ":\n\n" );
    print ( log, indent, conf.findProps( name ), outdent, endl );
  }

  print ( log, outdent, endl );

  return true;
}


//-----------------------------------------------------------------------
//   makeBShapes_
//-----------------------------------------------------------------------


bool AutoShapeTable3D::makeBShapes_

  ( PrintWriter&        log,
    const ShapeVector&  shapes,
    idx_t               nodeCount )

{
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::outdent;
  using jive::geom::BoundaryQuad4;
  using jive::geom::BoundaryQuad8;
  using jive::geom::BoundaryQuad9;
  using jive::geom::BoundaryTriangle3;
  using jive::geom::BoundaryTriangle6;

  Ref<BShape>          (*ctorFunc)

    ( const String&        name,
      const Properties&    conf,
      const Properties&    props );

  void                 (*initFunc)

    ( const Properties&    props,
      idx_t                  prec,
      idx_t                  nodeCount );


  const idx_t  n     = shapes.size ();

  Properties   props ( "shapeProps" );
  Properties   conf;

  String       type;
  String       name;


  ctorFunc = 0;
  initFunc = 0;

  switch ( nodeCount )
  {
  case 3:

    type     = "BTriangle3";
    ctorFunc = & BoundaryTriangle3::getShape;
    initFunc = & initBTriangProps;

    break;

  case 4:

    type     = "BQuad4";
    ctorFunc = & BoundaryQuad4::getShape;
    initFunc = & initBQuadProps;

    break;

  case 6:

    type     = "BTriangle6";
    ctorFunc = & BoundaryTriangle6::getShape;
    initFunc = & initBTriangProps;

    break;

  case 8:

    type     = "BQuad8";
    ctorFunc = & BoundaryQuad8::getShape;
    initFunc = & initBQuadProps;

    break;

  case 9:

    type     = "BQuad9";
    ctorFunc = & BoundaryQuad9::getShape;
    initFunc = & initBQuadProps;

    break;
  }

  if ( ! ctorFunc )
  {
    return false;
  }

  name = joinNames ( myName_, type.toLower() );

  print ( log, "creating ", n, " shapes of type `" );
  print ( log, type, "\' ...\n", indent );

  for ( idx_t i = 0; i < n; i++ )
  {
    initFunc ( props.makeProps( name ), i, nodeCount );

    shapes[i] = ctorFunc ( name, conf, props );

    print ( log, "\nprecision ", i, ":\n\n" );
    print ( log, indent, conf.findProps( name ), outdent, endl );
  }

  print ( log, outdent, endl );

  return true;
}


JIVE_END_PACKAGE( fem )
