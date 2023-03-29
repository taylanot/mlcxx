
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


#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/HashMap.h>
#include <jem/util/FastAllocator.h>
#include <jive/util/error.h>
#include <jive/util/Reordering.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/HashItemMap.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::HashItemMap );


using jem::newInstance;
using jem::util::Flex;
using jem::util::HashMap;
using jem::util::FastAllocator;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class HashItemMap::IndexMap_
//=======================================================================


class HashItemMap::IndexMap_ :

  public HashMap < idx_t, idx_t, FastAllocator > {};


//=======================================================================
//   class HashItemMap
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


HashItemMap::HashItemMap () : nextID_ ( 0 )
{
  indexMap_ = new IndexMap_ ();
}


HashItemMap::HashItemMap ( const Self& rhs ) :

  itemIDs_ ( rhs.itemIDs_ ),
  nextID_  ( rhs.nextID_  )

{
  indexMap_ = new IndexMap_ ( * rhs.indexMap_ );
}


HashItemMap::~HashItemMap ()
{
  delete indexMap_;
  indexMap_ = nullptr;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void HashItemMap::readFrom ( ObjectInput& in )
{
  decode ( in, itemIDs_, nextID_ );

  idx_t  n = itemIDs_.size ();

  indexMap_->clear   ();
  indexMap_->reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    indexMap_->insert ( itemIDs_[i], i );
  }
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void HashItemMap::writeTo ( ObjectOutput& out ) const
{
  encode ( out, itemIDs_, nextID_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> HashItemMap::clone () const
{
  return newInstance<Self> ( * this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t HashItemMap::size () const
{
  return itemIDs_.size ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void HashItemMap::clear ()
{
  indexMap_->clear ();
  itemIDs_  .clear ();

  nextID_ = 0;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void HashItemMap::reserve ( idx_t n )
{
  indexMap_->reserve ( n );
  itemIDs_  .reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void HashItemMap::trimToSize ()
{
  indexMap_->trimToSize ();
  itemIDs_  .trimToSize ();
}


//-----------------------------------------------------------------------
//   addItem
//-----------------------------------------------------------------------


idx_t HashItemMap::addItem ()
{
  using jem::maxOf;

  const idx_t  id    = nextID_;
  const idx_t  iitem = itemIDs_.size ();

  if ( nextID_ == maxOf( nextID_ ) )
  {
    throw ItemIDException (
      JEM_FUNC,
      "failed to find a free item ID",
      maxOf ( nextID_ )
    );
  }

  nextID_++;

  itemIDs_.pushBack ( id );
  indexMap_->insert ( id, iitem );

  return iitem;
}


idx_t HashItemMap::addItem ( idx_t itemID )
{
  const idx_t  iitem = itemIDs_.size ();

  if ( itemID >= nextID_ )
  {
    nextID_ = itemID + 1;
  }
  else if ( indexMap_->contains( itemID ) )
  {
    throw ItemIDException (
      JEM_FUNC,
      String::format ( "item ID %d is already in use", itemID ),
      itemID
    );
  }

  itemIDs_.pushBack ( itemID );
  indexMap_->insert ( itemID, iitem );

  return iitem;
}


//-----------------------------------------------------------------------
//   addItems
//-----------------------------------------------------------------------


idx_t HashItemMap::addItems ( idx_t count )
{
  using jem::maxOf;

  JEM_PRECHECK2 ( count >= 0, "invalid number of items" );

  const idx_t  id    = nextID_;
  const idx_t  iitem = itemIDs_.size ();

  if ( nextID_ > (maxOf( nextID_ ) - count) )
  {
    throw ItemIDException (
      JEM_FUNC,
      "failed to find a free item ID",
      maxOf ( nextID_ )
    );
  }

  nextID_ += count;

  for ( idx_t i = 0; i < count; i++ )
  {
    itemIDs_.pushBack ( id + i );
    indexMap_->insert ( id + i, iitem + i );
  }

  return iitem;
}


//-----------------------------------------------------------------------
//   reorderItems
//-----------------------------------------------------------------------


void HashItemMap::reorderItems ( const Reordering& reord )
{
  JEM_PRECHECK2 ( reord.oldSize() == size(), "invalid Reordering" );

  const idx_t*  JEM_RESTRICT  rmap    = reord.rmapPtr ();
  idx_t*        JEM_RESTRICT  itemIDs = itemIDs_.addr ();

  const idx_t   start    = reord.start   ();
  const idx_t   newSize  = reord.newSize ();

  IndexMap_&    indexMap = * indexMap_;


  indexMap.clear ();

  if ( reord.isSimple() )
  {
    for ( idx_t i = start; i < newSize; i++ )
    {
      itemIDs[i] = itemIDs[rmap[i]];
    }

    itemIDs_.resize ( newSize );
  }
  else
  {
    Flex<idx_t>  tmpIDs ( newSize );

    idx_t* JEM_RESTRICT   newIDs = tmpIDs.addr ();


    if ( start > 0 )
    {
      std::memcpy ( newIDs, itemIDs, (size_t) start * sizeof(idx_t) );
    }

    for ( idx_t i = start; i < newSize; i++ )
    {
      newIDs[i] = itemIDs[rmap[i]];
    }

    itemIDs_.swap ( tmpIDs );
  }

  itemIDs = itemIDs_.addr ();

  idx_t  j  = 0;

  for ( idx_t i = 0; i < newSize; i++ )
  {
    idx_t  itemID = itemIDs[i];

    indexMap[itemID] = i;

    if ( itemID >= j )
    {
      j = itemID + 1;
    }
  }

  nextID_ = j;
}


//-----------------------------------------------------------------------
//   findItem
//-----------------------------------------------------------------------


idx_t HashItemMap::findItem ( idx_t itemID ) const
{
  IndexMap_::Iterator itr = indexMap_->find ( itemID );

  if ( itr == indexMap_->end() )
  {
    return -1_idx;
  }
  else
  {
    return itr->second;
  }
}


//-----------------------------------------------------------------------
//   findItems
//-----------------------------------------------------------------------


idx_t HashItemMap::findItems

  ( const IdxVector&  iitems,
    const IdxVector&  itemIDs ) const

{
  JEM_PRECHECK2 ( iitems.size() >= itemIDs.size(),
                  "Array size mismatch" );

  const idx_t  n = itemIDs.size ();

  IndexMap_::Iterator  last = indexMap_->end ();
  IndexMap_::Iterator  itr;

  idx_t  k = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    itr = indexMap_->find ( itemIDs[i] );

    if ( itr == last )
    {
      iitems[i] = -1_idx;
    }
    else
    {
      iitems[i] = itr->second;
      k++;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getItemID
//-----------------------------------------------------------------------


idx_t HashItemMap::getItemID ( idx_t iitem ) const
{
  if ( iitem < 0 || iitem >= itemIDs_.size() )
  {
    indexError ( JEM_FUNC, "item", iitem, itemIDs_.size() );
  }

  return itemIDs_[iitem];
}


//-----------------------------------------------------------------------
//   getItemIDs
//-----------------------------------------------------------------------


void HashItemMap::getItemIDs

  ( const IdxVector&  itemIDs,
    const IdxVector&  iitems ) const

{
  JEM_PRECHECK2 ( itemIDs.size() >= iitems.size(),
                  "item ID array is too small" );

  const idx_t  n         = iitems  .size ();
  const idx_t  itemCount = itemIDs_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  iitem = iitems[i];

    if ( iitem < 0 || iitem >= itemCount )
    {
      indexError ( JEM_FUNC, "item", iitem, itemCount );
    }

    itemIDs[i] = itemIDs_[iitem];
  }
}


JIVE_END_PACKAGE ( util )
