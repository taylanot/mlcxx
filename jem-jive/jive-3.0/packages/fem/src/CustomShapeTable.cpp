
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
#include <jem/base/IllegalInputException.h>
#include <jem/util/Event.h>
#include <jive/util/GroupSet.h>
#include <jive/geom/Shape.h>
#include <jive/geom/ShapeTableFactory.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/CustomShapeTable.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::CustomShapeTable );


JIVE_BEGIN_PACKAGE( fem )


using jive::util::GroupSet;


//=======================================================================
//   class CustomShapeTable
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  CustomShapeTable::TYPE_NAME = "Custom";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


CustomShapeTable::CustomShapeTable ()
{}


CustomShapeTable::CustomShapeTable

  ( const String&      name,
    const ElementSet&  elems ) :

    Super ( name, elems.getData() )

{
  connect_ ();
}


CustomShapeTable::~CustomShapeTable ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void CustomShapeTable::readFrom ( ObjectInput& in )
{
  Super::readFrom ( in );

  connect_ ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void CustomShapeTable::declare ()
{
  using jive::geom::ShapeTableFactory;

  ShapeTableFactory::declare ( TYPE_NAME,  & makeNew );
  ShapeTableFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<ShapeTable> CustomShapeTable::makeNew

  ( const String&        name,
    const Properties&    conf,
    const Properties&    props,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<GroupSet>  elems = jem::dynamicCast<GroupSet> ( items );

  if ( elems && ElementSet::isValidData( elems ) )
  {
    return jem::newInstance<Self> ( name, ElementSet( elems ) );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   checkShapesFor_
//-----------------------------------------------------------------------


void CustomShapeTable::checkShapeFor_

  ( const geom::Shape&  shape,
    const ItemGroup&    igroup ) const

{
  const idx_t   nodeCount = shape.nodeCount    ();

  ElementGroup  egroup    ( const_cast<ItemGroup*>( &igroup ) );

  ElementSet    elems     = egroup.getElements ();
  IdxVector     ielems    = egroup.getIndices  ();

  idx_t         i, n;


  if ( elems.getNodes().rank() != shape.globalRank() )
  {
    String  name = shape.getName ();

    throw jem::IllegalInputException (
      getContext (),
      String::format (
        "shape/mesh mismatch: "
        "shape `%s\' has rank %d, but mesh has rank %d",
        name,
        shape.globalRank      (),
        elems.getNodes().rank ()
      )
    );
  }

  for ( i = 0; i < ielems.size(); i++ )
  {
    n = elems.getElemNodeCount ( ielems[i] );

    if ( n != nodeCount )
    {
      String  name   = shape.getName   ();
      idx_t     elemID = elems.getElemID ( ielems[i] );

      throw jem::IllegalInputException (
        getContext (),
        String::format (
          "shape/element mismatch: "
          "shape `%s\' has %d nodes, but element %d has %d nodes",
          name,
          nodeCount,
          elemID,
          n
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void CustomShapeTable::connect_ ()
{
  using jem::util::connect;

  GroupSet*  elems = static_cast<GroupSet*> ( Super::getItems() );

  connect ( elems->newTopoEvent, this, & Self::invalidate_ );
}


JIVE_END_PACKAGE( fem )
