
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
#include <jem/base/Class.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jive/util/Globdat.h>
#include <jem/util/Properties.h>
#include <jem/util/HashDictionary.h>
#include <jive/fem/error.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/OrderedBndGroup.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jem::util::Dict;
using jem::util::HashDict;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   getOrderedBndGroups
//-----------------------------------------------------------------------


static Ref<Dict>        getOrderedBndGroups

  ( const BoundarySet&    bounds,
    const Properties&     globdat )

{
  Properties  bnddat =

    Globdat::getDataFor ( bounds.getData(), globdat );

  Ref<Dict>   groups;

  if ( ! bnddat.find( groups, Globdat::BOUNDARY_GROUPS ) )
  {
    groups = newInstance<HashDict> ();

    bnddat.set ( Globdat::BOUNDARY_GROUPS, groups );
  }

  return groups;
}


//=======================================================================
//   class OrderedBndGroup
//=======================================================================

//-----------------------------------------------------------------------
//   getColors
//-----------------------------------------------------------------------


IdxVector OrderedBndGroup::getColors

  ( const IdxVector&    ibounds,
    const BoundarySet&  bounds )

{
  JEM_PRECHECK ( bounds );

  ElementSet   elems     = bounds.getElements ();
  const idx_t  ibndCount = ibounds.size       ();

  IdxVector    colors    ( ibndCount );

  idx_t        ilocal;
  idx_t        ibound;
  idx_t        ielem;
  idx_t        i;


  for ( i = 0; i < ibndCount; i++ )
  {
    ibound = ibounds[i];

    bounds.getBoundary ( ielem, ilocal, ibound );

    colors[i] = ilocal;
  }

  return colors;
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void OrderedBndGroup::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  Ref<Dict>  dict = getOrderedBndGroups ( getBoundaries(), globdat );

  data_->store  ( name, globdat );
  dict ->insert ( name, data_   );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


OrderedBndGroup OrderedBndGroup::find

  ( const String&       name,
    const BoundarySet&  bounds,
    const Properties&   globdat )

{
  JEM_PRECHECK ( bounds );

  using jem::staticCast;

  Ref<Dict>   dict = getOrderedBndGroups ( bounds, globdat );
  Ref<Data>   data = staticCast<Data>    ( dict->get( name ) );


  if ( data )
  {
    return Self ( data );
  }

  Ref<ItemGroup>  igroup =

    ItemGroup::find ( name, bounds.getData(), globdat );

  if ( igroup )
  {
    Self  bgroup =

      newOrderedBndGroup ( igroup->getIndices(), bounds );

    dict->insert ( name, bgroup.getData() );

    return bgroup;
  }
  else
  {
    return Self ();
  }
}


OrderedBndGroup OrderedBndGroup::find

  ( const String&      name,
    const String&      bounds,
    const Properties&  globdat )

{
  idx_t  i = name.find ( '/' );

  if ( i >= 0 )
  {
    String  setName = name[slice(BEGIN,i)].stripWhite ();
    String  grpName = name[slice(i+1,END)].stripWhite ();

    BoundarySet  bndSet = BoundarySet::find ( setName, globdat );

    if ( ! bndSet )
    {
      return find ( grpName, bndSet, globdat );
    }
  }
  else
  {
    BoundarySet  bndSet = BoundarySet::find ( bounds, globdat );

    if ( ! bndSet )
    {
      return find ( name, bndSet, globdat );
    }
  }

  return OrderedBndGroup ();
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


OrderedBndGroup OrderedBndGroup::get

  ( const String&       name,
    const BoundarySet&  bounds,
    const Properties&   globdat,
    const String&       context )

{
  OrderedBndGroup  group = find ( name, bounds, globdat );

  if ( ! group )
  {
    throw jem::IllegalInputException (
      context,
      String::format (
        "undefined boundary group: `%s\'", name
      )
    );
  }

  return group;
}


OrderedBndGroup OrderedBndGroup::get

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const String&      context )

{
  String   name;

  props.get ( name, "boundaries" );
  conf .set ( "boundaries", name );

  OrderedBndGroup  group =

    find ( name, Globdat::BOUNDARIES, globdat );

  if ( ! group )
  {
    throw jem::IllegalInputException (
      context,
      String::format (
        "undefined boundary group: `%s\'", name
      )
    );
  }

  return group;
}


//-----------------------------------------------------------------------
//   isValidData
//-----------------------------------------------------------------------


bool OrderedBndGroup::isValidData ( const Ref<Data>& data )
{
  using   jem::Class;

  typedef BoundarySet::Data  BndData;
  typedef ElementSet::Data   ElemData;
  typedef NodeSet::Data      NodeData;

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
  type = items->getClass ();

  return NodeData::getType()->isBaseOf ( type );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newOrderedBndGroup
//-----------------------------------------------------------------------


OrderedBndGroup       newOrderedBndGroup

  ( const BoundarySet&  bounds )

{
  const idx_t   bndCount = bounds.size ();

  IdxVector   ibounds ( bndCount );
  idx_t         ibound;


  for ( ibound = 0; ibound < bndCount; ibound++ )
  {
    ibounds[ibound] = ibound;
  }

  return OrderedBndGroup (
    newInstance<util::ColoredItemGroup> (
      OrderedBndGroup::getColors ( ibounds, bounds ),
      ibounds,
      bounds.getData ()
    )
  );
}


OrderedBndGroup       newOrderedBndGroup

  ( const IdxVector&    ibounds,
    const BoundarySet&  bounds )

{
  return OrderedBndGroup (
    newInstance<util::ColoredItemGroup> (
      OrderedBndGroup::getColors ( ibounds, bounds ),
      ibounds,
      bounds.getData ()
    )
  );
}


JIVE_END_PACKAGE( fem )
