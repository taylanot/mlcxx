
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
#include <jive/geom/Line.h>
#include <jive/geom/Names.h>
#include <jive/geom/ShapeTableFactory.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/AutoShapeTable1D.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::AutoShapeTable1D );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::GroupSet;
using jive::geom::IShape;


//=======================================================================
//   class AutoShapeTable1D
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  AutoShapeTable1D::TYPE_NAME = "Auto1D";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


AutoShapeTable1D::AutoShapeTable1D ()
{}


AutoShapeTable1D::AutoShapeTable1D

  ( const String&        name,
    const ElementSet&    elems,
    idx_t                maxPrec ) :

    Super ( name, elems, maxPrec )

{}


AutoShapeTable1D::AutoShapeTable1D

  ( const String&        name,
    const ElementGroup&  belems,
    const ElementGroup&  felems,
    idx_t                maxPrec ) :

    Super ( name, belems, felems, maxPrec )

{}


AutoShapeTable1D::~AutoShapeTable1D ()
{}


//-----------------------------------------------------------------------
//   initLineProps
//-----------------------------------------------------------------------


void AutoShapeTable1D::initLineProps

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


void AutoShapeTable1D::declare ()
{
  using jive::geom::ShapeTableFactory;

  ShapeTableFactory::declare ( TYPE_NAME,  & makeNew );
  ShapeTableFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<ShapeTable> AutoShapeTable1D::makeNew

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


bool AutoShapeTable1D::makeIShapes_

  ( PrintWriter&         log,
    const ShapeVector&   shapes,
    idx_t                nodeCount )

{
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::outdent;
  using jive::geom::Line2;
  using jive::geom::Line3;
  using jive::geom::Line4;

  Ref<IShape>          (*ctorFunc)

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

  idx_t        i;


  ctorFunc = 0;
  initFunc = 0;

  switch ( nodeCount )
  {
  case 2:

    type     = "Line2";
    ctorFunc = & Line2::getShape;
    initFunc = & initLineProps;

    break;

  case 3:

    type     = "Line3";
    ctorFunc = & Line3::getShape;
    initFunc = & initLineProps;

    break;

  case 4:

    type     = "Line4";
    ctorFunc = & Line4::getShape;
    initFunc = & initLineProps;

    break;
  }

  if ( ! ctorFunc )
  {
    return false;
  }

  name = joinNames ( myName_, type.toLower() );

  print ( log, "creating ", n, " shapes of type `" );
  print ( log, type, "\' ...\n", indent );

  for ( i = 0; i < n; i++ )
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


bool AutoShapeTable1D::makeBShapes_

  ( PrintWriter&        log,
    const ShapeVector&  shapes,
    idx_t               nodeCount )

{
  return false;
}


JIVE_END_PACKAGE( fem )
