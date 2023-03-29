
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
#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/ItemSet.h>
#include <jive/util/PointSet.h>
#include <jive/util/GroupSet.h>
#include <jive/util/EmptyItemGroup.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/geom/InternalShape.h>
#include <jive/geom/InterfaceShape.h>
#include <jive/geom/ShapeTableFactory.h>
#include <jive/fem/Names.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/AutoShapeTable.h>


JEM_DEFINE_CLASS( jive::fem::AutoShapeTable );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jem::System;
using jive::util::EmptyItemGroup;
using jive::geom::Shape;
using jive::geom::IShape;
using jive::geom::BShape;
using jive::geom::FShape;


//=======================================================================
//   class AutoShapeTable
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  AutoShapeTable::TYPE_NAME = "Auto";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


AutoShapeTable::AutoShapeTable ()
{}


AutoShapeTable::AutoShapeTable

  ( const String&        name,
    const ElementSet&    elems,
    idx_t                maxPrec ) :

    Super ( name, elems.getData(), maxPrec )

{
  belems_ = newInstance<EmptyItemGroup> ( elems.getData() );
  felems_ = newInstance<EmptyItemGroup> ( elems.getData() );

  connect_ ();
}


AutoShapeTable::AutoShapeTable

  ( const String&        name,
    const ElementGroup&  belems,
    const ElementGroup&  felems,
    idx_t                maxPrec ) :

    Super   ( name, belems.getElements().getData(), maxPrec ),

    belems_ ( belems.getData() ),
    felems_ ( felems.getData() )

{
  bname_ = fname_ = "unknown";

  connect_ ();
}


AutoShapeTable::~AutoShapeTable ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void AutoShapeTable::readFrom ( ObjectInput& in )
{
  idx_t  m;

  decode ( in, items_,  belems_, felems_ );
  decode ( in, myName_, bname_,  fname_, m );

  map_   .resize ( 0 );
  shapes_.resize ( m, 0 );

  Super::connect_ ();
  Self ::connect_ ();

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void AutoShapeTable::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items_,  belems_, felems_ );
  encode ( out, myName_, bname_,  fname_, shapes_.size(0) );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void AutoShapeTable::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties      myProps = props.findProps ( myName_ );

    Ref<ItemGroup>  belems;
    Ref<ItemGroup>  felems;

    String          bname;
    String          fname;

    idx_t           maxPrec;


    maxPrec = maxPrecision ();

    myProps.find ( bname,   PropNames::BOUNDARY_ELEMS );
    myProps.find ( fname,   PropNames::INTERFACE_ELEMS );
    myProps.find ( maxPrec, PropNames::MAX_PRECISION, 0, 100 );

    if ( ! bname.size() )
    {
      belems = newInstance<EmptyItemGroup> ( getItems() );
    }
    else
    {
      belems = ItemGroup::get ( bname,   getItems   (),
                                globdat, getContext () );
    }

    if ( ! fname.size() )
    {
      felems = newInstance<EmptyItemGroup> ( getItems() );
    }
    else
    {
      felems = ItemGroup::get ( fname,   getItems   (),
                                globdat, getContext () );
    }

    if ( maxPrec != maxPrecision() ||
         belems  != belems_        ||
         felems  != felems_ )
    {
      shapes_.resize ( maxPrec + 1, 0 );
      map_   .resize ( 0 );

      belems_.swap ( belems );
      felems_.swap ( felems );
      bname_ .swap ( bname  );
      fname_ .swap ( fname  );

      changeEvent.emit ( *this );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void AutoShapeTable::getConfig

  ( const Properties&  conf ) const

{
  Properties   myConf  = conf.makeProps ( myName_ );

  const idx_t  maxPrec = shapes_.size(0) - 1;

  myConf.set ( PropNames::BOUNDARY_ELEMS,  bname_  );
  myConf.set ( PropNames::INTERFACE_ELEMS, fname_  );
  myConf.set ( PropNames::MAX_PRECISION,   maxPrec );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void AutoShapeTable::declare ()
{
  using jive::geom::ShapeTableFactory;

  ShapeTableFactory::declare ( TYPE_NAME,  & makeNew );
  ShapeTableFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<ShapeTable> AutoShapeTable::makeNew

  ( const String&        name,
    const Properties&    conf,
    const Properties&    props,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  using jem::dynamicCast;
  using jive::util::PointSet;
  using jive::util::GroupSet;
  using jive::geom::ShapeTableFactory;


  GroupSet*  elems = dynamicCast<GroupSet*> ( items );

  if ( elems )
  {
    PointSet*  nodes =

      dynamic_cast<PointSet*> ( elems->getGroupedItems() );

    if ( nodes )
    {
      const idx_t      rank = nodes->rank ();

      Ref<ShapeTable>  st;
      String           type;


      type = String::format ( "Auto%dD", rank );

      st   = ShapeTableFactory::newInstance ( type,  name,
                                              conf,  props,
                                              items, globdat );

      if ( st )
      {
        return st;
      }

      type = String::format ( "%s%dD", CLASS_NAME, rank );

      return ShapeTableFactory::newInstance ( type,  name,
                                              conf,  props,
                                              items, globdat );
    }
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void AutoShapeTable::update_ ()
{
  using jem::Class;

  ElementGroup  belems     = ElementGroup  ( belems_ );
  ElementGroup  felems     = ElementGroup  ( felems_ );
  ElementSet    elems      = belems.getElements     ();

  const idx_t   elemCount  = elems  .size           ();
  const idx_t   belemCount = belems .size           ();
  const idx_t   felemCount = felems .size           ();
  const idx_t   maxNodes   = elems.maxElemNodeCount ();
  const idx_t   firstElem  = map_   .size           ();
  const idx_t   maxPrec    = shapes_.size(0) - 1;

  IdxMatrix     nodeMaps   ( maxNodes + 1, 3 );

  IdxVector     inodeMap   ( nodeMaps[0] );
  IdxVector     bnodeMap   ( nodeMaps[1] );
  IdxVector     fnodeMap   ( nodeMaps[2] );

  Shape*        shape;
  Class*        type;
  Class*        iType;
  Class*        bType;
  Class*        fType;

  idx_t         shapeCount;
  idx_t         nodeCount;

  idx_t         j, n;


  print ( System::debug( myName_ ),
          myName_, " : updating shape table ...\n" );

  shapeCount = shapes_.size (1);
  nodeMaps   = jem::minOf<idx_t> ();

  iType =              IShape::getType ();
  bType = belemCount ? BShape::getType () : nullptr;
  fType = felemCount ? FShape::getType () : nullptr;

  for ( j = 0; j < shapeCount; j++ )
  {
    shape = shapes_(0,j).get ();

    if ( shape )
    {
      nodeCount = shape->nodeCount ();
      type      = shape->getClass  ();

      if      ( iType->isBaseOf( type ) )
      {
        inodeMap[nodeCount] = j;
      }
      else if ( bType && bType->isBaseOf( type ) )
      {
        bnodeMap[nodeCount] = j;
      }
      else if ( fType && fType->isBaseOf( type ) )
      {
        fnodeMap[nodeCount] = j;
      }
    }
  }

  map_.reshape ( elemCount );

  for ( idx_t ielem = firstElem; ielem < elemCount; ielem++ )
  {
    nodeCount = elems.getElemNodeCount ( ielem );

    if      ( belemCount && belems.contains( ielem ) )
    {
      if ( bnodeMap[nodeCount] < 0 )
      {
        bnodeMap[nodeCount] = shapeCount++;
      }

      j = bnodeMap[nodeCount];
    }
    else if ( felemCount && felems.contains( ielem ) )
    {
      if ( fnodeMap[nodeCount] < 0 )
      {
        fnodeMap[nodeCount] = shapeCount++;
      }

      j = fnodeMap[nodeCount];
    }
    else
    {
      if ( inodeMap[nodeCount] < 0 )
      {
        inodeMap[nodeCount] = shapeCount++;
      }

      j = inodeMap[nodeCount];
    }

    map_[ielem] = j;
  }

  n = shapes_.size (1);

  if ( shapeCount > n )
  {
    Ref<PrintWriter>  log =

      newInstance<PrintWriter> ( & System::debug( myName_ ) );

    log->incrIndentLevel ();

    shapes_.reshape ( maxPrec + 1, shapeCount );

    for ( nodeCount = 0; nodeCount <= maxNodes; nodeCount++ )
    {
      j = inodeMap[nodeCount];

      if ( j >= n )
      {
        log->printLine ();

        makeIShapes_ ( *log, shapes_[j], nodeCount );
      }

      j = bnodeMap[nodeCount];

      if ( j >= n )
      {
        log->printLine ();

        makeBShapes_ ( *log, shapes_[j], nodeCount );
      }

      j = fnodeMap[nodeCount];

      if ( j >= n )
      {
        log->printLine ();

        makeFShapes_ ( *log, shapes_[j], nodeCount );
      }
    }

    log->printLine ();
    log->flush     ();
  }

  for ( idx_t ielem = firstElem; ielem < elemCount; ielem++ )
  {
    j = map_[ielem];

    if ( j >= 0 && shapes_(0,j) == nullptr )
    {
      map_[ielem] = -1;
    }
  }

  items_->resetEvents ();
}


//-----------------------------------------------------------------------
//   makeFShapes_
//-----------------------------------------------------------------------


bool AutoShapeTable::makeFShapes_

  ( PrintWriter&        out,
    const ShapeVector&  shapes,
    idx_t               nodeCount )

{
  using jem::staticCast;

  bool  result;


  // Number of nodes must be even:

  if ( nodeCount % 2 )
  {
    return false;
  }

  print ( out, "trying to create ", shapes.size(),
          " interface shapes ...\n\n" );

  out.incrIndentLevel ();

  result = makeBShapes_ ( out, shapes, nodeCount / 2 );

  out.decrIndentLevel ();

  if ( result )
  {
    print ( out, "wrapping boundary shapes in "
            "interface shapes ...\n" );

    for ( idx_t i = 0; i < shapes.size(); i++ )
    {
      shapes[i] =

        newInstance<FShape> ( staticCast<BShape>( shapes[i] ) );
    }
  }
  else
  {
    print ( out, "unknown boundary shape; "
            "no interface shapes created\n" );
  }

  return result;
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void AutoShapeTable::connect_ ()
{
  using jem::util::connect;
  using jive::util::GroupSet;

  GroupSet*  elems = static_cast<GroupSet*> ( Super::getItems() );

  connect ( elems->newTopoEvent, this, & Self::invalidate_ );
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void AutoShapeTable::invalidate_ ()
{
  const idx_t  m = shapes_.size (0);

  map_   .resize ( 0 );
  shapes_.resize ( m, 0 );

  changeEvent.emit ( *this );
}


JIVE_END_PACKAGE( fem )
