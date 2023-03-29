
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
#include <jive/geom/Quad.h>
#include <jive/geom/Names.h>
#include <jive/geom/Triangle.h>
#include <jive/geom/BoundaryLine.h>
#include <jive/geom/ShapeTableFactory.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/AutoShapeTable2D.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::AutoShapeTable2D );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::GroupSet;
using jive::geom::IShape;
using jive::geom::BShape;


//=======================================================================
//   class AutoShapeTable2D
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  AutoShapeTable2D::TYPE_NAME = "Auto2D";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


AutoShapeTable2D::AutoShapeTable2D ()
{}


AutoShapeTable2D::AutoShapeTable2D

  ( const String&        name,
    const ElementSet&    elems,
    idx_t                maxPrec ) :

    Super ( name, elems, maxPrec )

{}


AutoShapeTable2D::AutoShapeTable2D

  ( const String&        name,
    const ElementGroup&  belems,
    const ElementGroup&  felems,
    idx_t                maxPrec ) :

    Super ( name, belems, felems, maxPrec )

{}


AutoShapeTable2D::~AutoShapeTable2D ()
{}


//-----------------------------------------------------------------------
//   initTriangProps
//-----------------------------------------------------------------------


void AutoShapeTable2D::initTriangProps

  ( const Properties&  props,
    idx_t              prec,
    idx_t              nodeCount )

{
  using jive::geom::PropNames;

  StringVector  ischeme ( 2 );

  idx_t         k = prec + nodeCount / 3;


  ischeme[0] = String ( k * k );
  ischeme[1] = String ( k );

  props.set ( PropNames::ISCHEME, ischeme );
}


//-----------------------------------------------------------------------
//   initQuadProps
//-----------------------------------------------------------------------


void AutoShapeTable2D::initQuadProps

  ( const Properties&  props,
    idx_t              prec,
    idx_t              nodeCount )

{
  using jive::geom::PropNames;

  StringVector  ischeme ( 2 );

  idx_t         k = prec + 2;


  if ( nodeCount > 4 )
  {
    k++;
  }

  ischeme[0] = String::format ( "%d*%d", k, k );
  ischeme[1] = String         ( k );

  props.set ( PropNames::ISCHEME, ischeme );
}


//-----------------------------------------------------------------------
//   initBLineProps
//-----------------------------------------------------------------------


void AutoShapeTable2D::initBLineProps

  ( const Properties&  props,
    idx_t              prec,
    idx_t              nodeCount )

{
  using jive::geom::PropNames;

  String  ischeme ( prec + nodeCount - 1 );

  props.set ( PropNames::ISCHEME, ischeme );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void AutoShapeTable2D::declare ()
{
  using jive::geom::ShapeTableFactory;

  ShapeTableFactory::declare ( TYPE_NAME,  & makeNew );
  ShapeTableFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<ShapeTable> AutoShapeTable2D::makeNew

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


bool AutoShapeTable2D::makeIShapes_

  ( PrintWriter&         log,
    const ShapeVector&   shapes,
    idx_t                nodeCount )

{
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::outdent;
  using jive::geom::Quad4;
  using jive::geom::Quad8;
  using jive::geom::Quad9;
  using jive::geom::Quad12;
  using jive::geom::Triangle3;
  using jive::geom::Triangle6;

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
  case 3:

    type     = "Triangle3";
    ctorFunc = & Triangle3::getShape;
    initFunc = & initTriangProps;

    break;

  case 4:

    type     = "Quad4";
    ctorFunc = & Quad4::getShape;
    initFunc = & initQuadProps;

    break;

  case 6:

    type     = "Triangle6";
    ctorFunc = & Triangle6::getShape;
    initFunc = & initTriangProps;

    break;

  case 8:

    type     = "Quad8";
    ctorFunc = & Quad8::getShape;
    initFunc = & initQuadProps;

    break;

  case 9:

    type     = "Quad9";
    ctorFunc = & Quad9::getShape;
    initFunc = & initQuadProps;

    break;

  case 12:

    type     = "Quad12";
    ctorFunc = & Quad12::getShape;
    initFunc = & initQuadProps;

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


bool AutoShapeTable2D::makeBShapes_

  ( PrintWriter&        log,
    const ShapeVector&  shapes,
    idx_t               nodeCount )

{
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::outdent;
  using jive::geom::BoundaryLine2;
  using jive::geom::BoundaryLine3;
  using jive::geom::BoundaryLine4;


  Ref<BShape>          (*ctorFunc)

    ( const String&        name,
      const Properties&    conf,
      const Properties&    props );


  const idx_t  n     = shapes.size ();

  Properties   props ( "shapeProps" );
  Properties   conf;

  String       type;
  String       name;


  ctorFunc = 0;

  switch ( nodeCount )
  {
  case 2:

    type     = "BLine2";
    ctorFunc = & BoundaryLine2::getShape;

    break;

  case 3:

    type     = "BLine3";
    ctorFunc = & BoundaryLine3::getShape;

    break;

  case 4:

    type     = "BLine4";
    ctorFunc = & BoundaryLine4::getShape;

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
    initBLineProps ( props.makeProps( name ), i, nodeCount );

    shapes[i] = ctorFunc ( name, conf, props );

    print ( log, "\nprecision ", i, ":\n\n" );
    print ( log, indent, conf.findProps( name ), outdent, endl );
  }

  print ( log, outdent, endl );

  return true;
}


JIVE_END_PACKAGE( fem )
