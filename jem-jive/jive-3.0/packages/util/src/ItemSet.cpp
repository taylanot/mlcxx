
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
#include <jem/base/array/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/Properties.h>
#include <jem/util/HashDictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/xml/utilities.h>
#include <jive/util/typedefs.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>


JEM_DEFINE_CLASS( jive::util::ItemSet );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;


//=======================================================================
//   class ItemSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ItemSet::ItemSet ( const String& name ) :

  Super ( name )

{}


ItemSet::~ItemSet ()
{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String ItemSet::getContext () const
{
  const String  itemName = getItemName ();

  return NamedObject::makeContext ( itemName + " set", myName_ );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t ItemSet::size () const
{
  return getItemMap()->size ();
}


//-----------------------------------------------------------------------
//   getTagName
//-----------------------------------------------------------------------


String ItemSet::getTagName () const
{
  return jem::xml::makeTag ( getItemsName() );
}


//-----------------------------------------------------------------------
//   getItemsName
//-----------------------------------------------------------------------


String ItemSet::getItemsName () const
{
  const String  name = getItemName ();
  const idx_t   n    = name.size   ();

  if ( n == 0 )
  {
    return name;
  }

  if ( name.back() == 'y' )
  {
    return name[slice(BEGIN,n - 1)] + "ies";
  }

  if ( name.back() == 's' )
  {
    return name;
  }

  if ( name.back() == 'x' )
  {
    if ( n > 1 && (name[n - 2] == 'e' || name[n - 2] == 'i') )
    {
      return name[slice(BEGIN,n - 2)] + "ices";
    }
    else
    {
      return name + "es";
    }
  }

  return name + 's';
}


//-----------------------------------------------------------------------
//   getItemPath
//-----------------------------------------------------------------------


String ItemSet::getItemPath ( idx_t id ) const
{
  return String::format ( "%s[%d]", myName_, id );
}


//-----------------------------------------------------------------------
//   findItem
//-----------------------------------------------------------------------


idx_t ItemSet::findItem ( idx_t id ) const
{
  return getItemMap()->findItem ( id );
}


//-----------------------------------------------------------------------
//   getItemID
//-----------------------------------------------------------------------


idx_t ItemSet::getItemID ( idx_t iitem ) const
{
  return getItemMap()->getItemID ( iitem );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void ItemSet::printTo ( PrintWriter& out ) const
{
  IdxVector  iitems ( jem::iarray( this->size() ) );

  this->printSubset ( out, iitems );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void ItemSet::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  Ref<Self>  self ( const_cast<Self*>( this ) );

  getAll ( globdat ) -> insert ( name, self );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<ItemSet> ItemSet::find

  ( const String&      name,
    const Properties&  globdat )

{
  Ref<Dict>  dict;

  if ( globdat.find( dict, Globdat::ITEM_SETS ) )
  {
    return jem::dynamicCast<ItemSet> ( dict->get( name ) );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<ItemSet> ItemSet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  Ref<ItemSet>  iset = find ( name, globdat );

  if ( ! iset )
  {
    throw jem::IllegalInputException (
      context,
      String::format ( "undefined item set: `%s\'", name )
    );
  }

  return iset;
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void ItemSet::erase

  ( const String&      name,
    const Properties&  globdat )

{
  Ref<Dict>  dict;

  if ( globdat.find( dict, Globdat::ITEM_SETS ) )
  {
    dict->erase ( name );
  }
}


//-----------------------------------------------------------------------
//   getAll
//-----------------------------------------------------------------------


Ref<Dict> ItemSet::getAll ( const Properties& globdat )
{
  using jem::util::HashDict;

  Ref<Dict>  dict;

  if ( ! globdat.find( dict, Globdat::ITEM_SETS ) )
  {
    dict = newInstance<HashDict> ();

    globdat.set ( Globdat::ITEM_SETS, dict );
  }

  return dict;
}


//-----------------------------------------------------------------------
//   listAll
//-----------------------------------------------------------------------


StringVector ItemSet::listAll ( const Properties& globdat )
{
  Ref<Dict>     dict;
  StringVector  list;


  if ( globdat.find( dict, Globdat::ITEM_SETS ) )
  {
    Ref<DictEnum>  e = dict->enumerate ();

    list.resize ( dict->size() );

    for ( idx_t i = 0; ! e->atEnd(); e->toNext(), i++ )
    {
      list[i] = e->getKey ();
    }
  }

  return list;
}


JIVE_END_PACKAGE( util )
