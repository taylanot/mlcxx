
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/HashDictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/util/Properties.h>
#include <jem/xml/print.h>
#include <jem/xml/utilities.h>
#include <jive/util/typedefs.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/Globdat.h>
#include <jive/util/AllItemGroup.h>
#include <jive/util/EmptyItemGroup.h>
#include <jive/util/ItemGroup.h>


JEM_DEFINE_CLASS( jive::util::ItemGroup );


JIVE_BEGIN_PACKAGE( util )


using jem::staticCast;
using jem::dynamicCast;
using jem::newInstance;


//=======================================================================
//   class ItemGroup
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  ItemGroup::GROUP_REORDERED = 1;
const int  ItemGroup::ITEMS_REORDERED = 2;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


ItemGroup::~ItemGroup ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void ItemGroup::resetEvents ()
{}


//-----------------------------------------------------------------------
//   getIDs
//-----------------------------------------------------------------------


IdxVector ItemGroup::getIDs () const
{
  IdxVector  iitems  ( getIndices()  );
  IdxVector  itemIDs ( iitems.size() );

  getItems()->getItemMap()->getItemIDs ( itemIDs, iitems );

  return itemIDs;
}


//-----------------------------------------------------------------------
//   getIndex
//-----------------------------------------------------------------------


idx_t ItemGroup::getIndex ( idx_t i ) const
{
  if ( i < 0 || i >= size() )
  {
    indexError ( JEM_FUNC, "item", i, size() );
  }

  return getIndices()[i];
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool ItemGroup::contains ( idx_t iitem ) const
{
  IdxVector    iitems ( getIndices() );

  const idx_t  n = iitems.size ();

  for ( idx_t  i = 0; i < n; i++ )
  {
    if ( iitems[i] == iitem )
    {
      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   filter
//-----------------------------------------------------------------------


void ItemGroup::filter ( const IdxVector& iitems ) const
{
  const idx_t  n = iitems.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( ! contains( iitems[i] ) )
    {
      iitems[i] = -iitems[i] - 1;
    }
  }
}


//-----------------------------------------------------------------------
//   findMembers
//-----------------------------------------------------------------------


idx_t ItemGroup::findMembers

  ( const IdxVector&  jitems,
    const IdxVector&  iitems ) const

{
  JEM_PRECHECK2 ( jitems.size() >= iitems.size(),
                  "output array is too small" );

  const idx_t  n = iitems.size ();

  if ( n > jitems.size() )
  {
    jitems[slice(BEGIN,n)] = iitems;
    filter ( jitems[slice(BEGIN,n)] );
  }
  else
  {
    jitems = iitems;
    filter ( jitems );
  }

  idx_t  j = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  iitem = jitems[i];

    if ( iitem >= 0 )
    {
      jitems[j++] = iitem;
    }
  }

  return j;
}


//-----------------------------------------------------------------------
//   findNonMembers
//-----------------------------------------------------------------------


idx_t ItemGroup::findNonMembers

  ( const IdxVector&  jitems,
    const IdxVector&  iitems ) const

{
  JEM_PRECHECK2 ( jitems.size() >= iitems.size(),
                  "output array is too small" );

  const idx_t  n = iitems.size ();

  if ( n < jitems.size() )
  {
    jitems[slice(BEGIN,n)] = iitems;
    filter ( jitems[slice(BEGIN,n)] );
  }
  else
  {
    jitems = iitems;
    filter ( jitems );
  }

  idx_t j = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( jitems[i] < 0 )
    {
      jitems[j++] = iitems[i];
    }
  }

  return j;
}


//-----------------------------------------------------------------------
//   getUnion
//-----------------------------------------------------------------------


IdxVector ItemGroup::getUnion

  ( const ItemGroup&  first,
    const ItemGroup&  second )

{
  JEM_PRECHECK2 ( first.getItems() == second.getItems(),
                  "item groups associated with different item sets" );

  const idx_t       firstSize  = first .size ();
  const idx_t       secondSize = second.size ();

  const ItemGroup*  larger;
  const ItemGroup*  smaller;

  IdxVector         iitems ( firstSize + secondSize );

  idx_t             m, n;


  if ( firstSize < secondSize )
  {
    larger  = & second;
    smaller = & first;
    m       = secondSize;
  }
  else
  {
    larger  = & first;
    smaller = & second;
    m       = firstSize;
  }

  iitems[slice(BEGIN,m)] = larger->getIndices ();

  n = larger->findNonMembers (
    iitems[slice(m,END)],
    smaller->getIndices ()
  );

  return iitems[slice(BEGIN,m + n)];
}


//-----------------------------------------------------------------------
//   getIntersection
//-----------------------------------------------------------------------


IdxVector ItemGroup::getIntersection

  ( const ItemGroup&  first,
    const ItemGroup&  second )

{
  JEM_PRECHECK2 ( first.getItems() == second.getItems(),
                  "item groups associated with different item sets" );

  const idx_t       firstSize  = first .size ();
  const idx_t       secondSize = second.size ();

  const ItemGroup*  larger;
  const ItemGroup*  smaller;

  IdxVector         iitems ( jem::min( firstSize, secondSize ) );

  idx_t             m, n;


  if ( firstSize < secondSize )
  {
    larger  = & second;
    smaller = & first;
    m       = firstSize;
  }
  else
  {
    larger  = & first;
    smaller = & second;
    m       = secondSize;
  }

  n = larger->findMembers (
    iitems[slice(BEGIN,m)],
    smaller->getIndices ()
  );

  return iitems[slice(BEGIN,n)];
}


//-----------------------------------------------------------------------
//   printGroup
//-----------------------------------------------------------------------


void ItemGroup::printGroup

  ( PrintWriter&      out,
    const ItemSet&    items,
    const IdxVector&  iitems )

{
  using jem::io::space;

  const idx_t  n = iitems.size ();

  IdxVector    itemIDs ( n );


  items.getItemMap()->getItemIDs ( itemIDs, iitems );

  print ( out, '{' );

  if ( n > 0 )
  {
    print        ( out, space );
    printIntList ( out, itemIDs );
    print        ( out, space );
  }

  print ( out, '}' );
}


//-----------------------------------------------------------------------
//   printSome
//-----------------------------------------------------------------------


void ItemGroup::printSome

  ( PrintWriter&   out,
    const Dict&    groups,
    const String&  tag )

{
  using jem::xml::beginStartTag;
  using jem::xml::endStartTag;
  using jem::xml::endTag;
  using jem::xml::attribute;

  Ref<DictEnum>  e = groups.enumerate ();


  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<Object>  obj   = e->getValue ();
    ItemGroup*   group = dynamicCast<ItemGroup*> ( obj );

    if ( ! group )
    {
      continue;
    }

    if ( dynamicCast<AllItemGroup*>( obj ) )
    {
      continue;
    }

    String  groupName = e->getKey ();
    String  tagName   = tag;

    if ( tagName.size() == 0 )
    {
      tagName = group->getTagName ();
    }

    print ( out, beginStartTag( tagName ),
            attribute( "name", groupName ), endStartTag );

    group->printTo ( out );

    print ( out, endTag( tagName ) );
  }
}


//-----------------------------------------------------------------------
//   printAll
//-----------------------------------------------------------------------


void ItemGroup::printAll

  ( PrintWriter&       out,
    const Properties&  globdat )

{
  Ref<Dict>      list = ItemSet::getAll ( globdat );
  Ref<DictEnum>  e    = list->enumerate ();

  for ( ; ! e->atEnd(); e->toNext() )
  {
    Ref<ItemSet>   items = dynamicCast<ItemSet> ( e->getValue() );

    if ( items )
    {
      printSome ( out, * getFor( items, globdat ) );
    }
  }
}


//-----------------------------------------------------------------------
//   getTagName
//-----------------------------------------------------------------------


String ItemGroup::getTagName () const
{
  return getTagName ( getItems()->getItemName() );
}


String ItemGroup::getTagName ( const String& itemName )
{
  return ( jem::xml::makeTag ( itemName ) + "Group" );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void ItemGroup::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  getFor( getItems(), globdat )->insert (
    name,
    Ref<Object> ( const_cast<Self*>( this ) )
  );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<ItemGroup> ItemGroup::find

  ( const String&      name,
    const String&      items,
    const Properties&  globdat )

{
  idx_t  i = name.find ( '/' );

  if ( i >= 0 )
  {
    String  setName = name[slice(BEGIN,i)].stripWhite ();
    String  grpName = name[slice(i+1,END)].stripWhite ();

    Ref<ItemSet>  set = ItemSet::find ( setName, globdat );

    if ( set )
    {
      return find ( grpName, set, globdat );
    }
  }
  else
  {
    Ref<ItemSet>  set = ItemSet::find ( items, globdat );

    if ( set )
    {
      return find ( name, set, globdat );
    }
  }

  return nullptr;
}


Ref<ItemGroup> ItemGroup::find

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Ref<Dict>  groups = findFor ( items, globdat );

  if ( ! groups )
  {
    if ( name == "all" || name == "none" )
    {
      groups = getFor ( items, globdat );
    }
    else
    {
      return nullptr;
    }
  }

  return dynamicCast<ItemGroup> ( groups->get( name ) );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<ItemGroup> ItemGroup::get

  ( const String&      name,
    const String&      items,
    const Properties&  globdat,
    const String&      context )

{
  idx_t  i = items.find ( '/' );

  if ( i >= 0 )
  {
    String  setName = name[slice(BEGIN,i)].stripWhite ();
    String  grpName = name[slice(i+1,END)].stripWhite ();

    return get ( grpName,
                 ItemSet::get ( setName, globdat, context ),
                 globdat, context );
  }
  else
  {
    return get ( name,
                 ItemSet::get ( items, globdat, context ),
                 globdat, context );
  }
}


Ref<ItemGroup> ItemGroup::get

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const Properties&    globdat,
    const String&        context )

{
  Ref<ItemGroup>  group = find ( name, items, globdat );

  if ( ! group )
  {
    String  itemName;

    if ( items )
    {
      itemName = items->getItemName ();
    }
    else
    {
      itemName = "item";
    }

    throw jem::IllegalInputException (
      context,
      String::format ( "undefined %s group: `%s\'",
                       itemName, name )
    );
  }

  return group;
}


//-----------------------------------------------------------------------
//   findFor
//-----------------------------------------------------------------------


Ref<Dict> ItemGroup::findFor

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Ref<Dict>  groups;

  Globdat::findFor ( groups, Globdat::ITEM_GROUPS, items, globdat );

  return groups;
};


//-----------------------------------------------------------------------
//   getFor
//-----------------------------------------------------------------------


Ref<Dict> ItemGroup::getFor

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  using jem::util::HashDict;

  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Properties  itemdat ( Globdat::getDataFor( items, globdat ) );
  Ref<Dict>   groups;


  if ( ! itemdat.find( groups, Globdat::ITEM_GROUPS ) )
  {
    groups = newInstance<HashDict> ();

    groups->insert ( "all",  newInstance<AllItemGroup>  ( items ) );
    groups->insert ( "none", newInstance<EmptyItemGroup>( items ) );

    itemdat.set ( Globdat::ITEM_GROUPS, groups );
  }

  return groups;
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void ItemGroup::emitEvents_ ( int events )
{}


JIVE_END_PACKAGE( util )
