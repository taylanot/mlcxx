
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


#include <jem/base/Class.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/util/AllItemGroup.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/util/EmptyItemGroup.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/BoundaryGroup.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class BoundaryGroup
//=======================================================================

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


BoundaryGroup BoundaryGroup::get

  ( const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat,
    const String&       context )

{
  String  name;

  props.get ( name, "boundaries" );
  conf .set ( "boundaries", name );

  Ref<Data>  data =

    Data::find ( name, Globdat::BOUNDARIES, globdat );

  if ( data == nullptr || ! isValidData( data ) )
  {
    throw jem::IllegalInputException (
      context,
      String::format (
        "undefined boundary group: `%s\'", name
      )
    );
  }

  return BoundaryGroup ( data );
}


//-----------------------------------------------------------------------
//   isValidData
//-----------------------------------------------------------------------


bool BoundaryGroup::isValidData ( const Ref<Data>& data )
{
  using jem::Class;

  typedef NodeSet    ::Data  NodeData;
  typedef ElementSet ::Data  ElemData;
  typedef BoundarySet::Data  BndData;

  if ( ! data )
  {
    return true;
  }

  ItemSet*  items = data ->getItems ();
  Class*    type  = items->getClass ();

  if ( ! BndData::getType()->isBaseOf( type ) )
  {
    return false;
  }

  items = static_cast<BndData*>(items)->getCompoundItems ();
  type  = items->getClass ();

  if ( ! ElemData::getType()->isBaseOf( type ) )
  {
    return false;
  }

  items = static_cast<ElemData*>( items )->getGroupedItems ();
  type  = items->getClass ();

  return NodeData::getType()->isBaseOf ( type );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newBoundaryGroup
//-----------------------------------------------------------------------


BoundaryGroup         newBoundaryGroup

  ( const BoundarySet&  bounds )

{
  using jive::util::EmptyItemGroup;

  JEM_PRECHECK ( bounds );

  return BoundaryGroup
    ( newInstance<EmptyItemGroup>( bounds.getData() ) );
}


BoundaryGroup         newBoundaryGroup

  ( const SliceAll&     all,
    const BoundarySet&  bounds )

{
  using jive::util::AllItemGroup;

  JEM_PRECHECK ( bounds );

  return BoundaryGroup
    ( newInstance<AllItemGroup>( bounds.getData() ) );
}


BoundaryGroup         newBoundaryGroup

  ( const IdxVector&    ibounds,
    const BoundarySet&  bounds )

{
  using jive::util::ArrayItemGroup;

  JEM_PRECHECK ( bounds );

  return BoundaryGroup
    ( newInstance<ArrayItemGroup>( ibounds, bounds.getData() ) );
}


JIVE_END_PACKAGE( fem )
