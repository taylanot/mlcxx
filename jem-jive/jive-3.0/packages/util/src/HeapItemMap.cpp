
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/error.h>
#include <jive/util/Reordering.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/HeapItemMap.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::HeapItemMap );


JIVE_BEGIN_PACKAGE( util )


using jem::iarray;
using jem::newInstance;


//=======================================================================
//   private utility functions
//=======================================================================

//-----------------------------------------------------------------------
//   binarySearch
//-----------------------------------------------------------------------


static inline idx_t           binarySearch

  ( idx_t                       itemID,
    const idx_t* JEM_RESTRICT   itemIDs,
    const idx_t* JEM_RESTRICT   iperm,
    idx_t                       itemCount )

{
  idx_t  i = 0;
  idx_t  j = itemCount;

  while ( i < j )
  {
    idx_t  k   = i + ((j - i) >> 1);
    idx_t  iid = itemIDs[iperm[k]];

    if      ( itemID > iid )
    {
      i = k + 1;
    }
    else if ( itemID < iid )
    {
      j = k;
    }
    else
    {
      return iperm[k];
    }
  }

  return -1;
}


//=======================================================================
//   class HeapItemMap
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


HeapItemMap::HeapItemMap ()
{
  nextID_ = 0;
  sorted_ = 0;
}


HeapItemMap::HeapItemMap ( const Self& rhs ) :

  itemIDs_ ( rhs.itemIDs_ ),
  iperm_   ( rhs.iperm_   ),
  nextID_  ( rhs.nextID_  ),
  sorted_  ( rhs.sorted_  )

{}


HeapItemMap::~HeapItemMap ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void HeapItemMap::readFrom ( ObjectInput& in )
{
  decode ( in, itemIDs_, nextID_ );

  iperm_.resize ( itemIDs_.size() );

  iperm_  = -1;
  sorted_ =  0;
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void HeapItemMap::writeTo ( ObjectOutput& out ) const
{
  encode ( out, itemIDs_, nextID_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> HeapItemMap::clone () const
{
  return newInstance<Self> ( * this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t HeapItemMap::size () const
{
  return itemIDs_.size ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void HeapItemMap::clear ()
{
  itemIDs_.clear ();
  iperm_  .clear ();

  nextID_ = 0;
  sorted_ = 0;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void HeapItemMap::reserve ( idx_t n )
{
  itemIDs_.reserve ( n );
  iperm_  .reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void HeapItemMap::trimToSize ()
{
  itemIDs_.trimToSize ();
  iperm_  .trimToSize ();
}


//-----------------------------------------------------------------------
//   addItem
//-----------------------------------------------------------------------


idx_t HeapItemMap::addItem ()
{
  using jem::maxOf;

  const idx_t  iitem  = itemIDs_.size ();
  const idx_t  itemID = nextID_;

  if ( nextID_ == maxOf( nextID_ ) )
  {
    throw ItemIDException (
      JEM_FUNC,
      "failed to find a free item ID",
      maxOf ( nextID_ )
    );
  }

  if ( sorted_ == iitem )
  {
    sorted_++;
  }

  nextID_++;

  itemIDs_.pushBack ( itemID );
  iperm_  .pushBack ( iitem  );

  return iitem;
}


idx_t HeapItemMap::addItem ( idx_t itemID )
{
  const idx_t  itemCount = itemIDs_.size ();
  const idx_t  iitem     = itemCount;


  if ( itemID >= nextID_ )
  {
    nextID_ = itemID + 1;

    if ( sorted_ == itemCount )
    {
      sorted_++;
    }
  }
  else
  {
    bool  exists = false;


    if ( (double) sorted_ < (0.9 * (double) itemCount) )
    {
      sortIDs_ ();
    }

    for ( idx_t i = itemCount - 1; i >= sorted_; i-- )
    {
      if ( itemIDs_[i] == itemID )
      {
        exists = true;
        break;
      }
    }

    if ( ! exists )
    {
      exists = ( binarySearch( itemID,
                               itemIDs_.addr (),
                               iperm_  .addr (),
                               sorted_ ) >= 0 );
    }

    if ( exists )
    {
      throw ItemIDException (
        JEM_FUNC,
        String::format ( "item ID %d is already in use", itemID ),
        itemID
      );
    }
  }

  itemIDs_.pushBack ( itemID );
  iperm_  .pushBack ( iitem  );

  return iitem;
}


//-----------------------------------------------------------------------
//   addItems
//-----------------------------------------------------------------------


idx_t HeapItemMap::addItems ( idx_t count )
{
  using jem::maxOf;

  JEM_PRECHECK2 ( count >= 0, "invalid number of items to add" );

  const idx_t  iitem  = itemIDs_.size ();
  const idx_t  itemID = nextID_;

  if ( nextID_ > (maxOf( nextID_ ) - count) )
  {
    throw ItemIDException (
      JEM_FUNC,
      "failed to find a free item ID",
      maxOf ( nextID_ )
    );
  }

  if ( sorted_ == iitem )
  {
    sorted_ += count;
  }

  nextID_ += count;

  for ( idx_t i = 0; i < count; i++ )
  {
    itemIDs_.pushBack ( itemID + i );
    iperm_  .pushBack ( iitem  + i );
  }

  return iitem;
}


//-----------------------------------------------------------------------
//   reorderItems
//-----------------------------------------------------------------------


void HeapItemMap::reorderItems ( const Reordering& reord )
{
  JEM_PRECHECK2 ( reord.oldSize() == size(),
                  "invalid Reordering" );

  const idx_t*  JEM_RESTRICT  rmap    = reord.rmapPtr ();
  idx_t*        JEM_RESTRICT  itemIDs = itemIDs_.addr ();

  const idx_t   newSize = reord.newSize ();


  if ( reord.isSimple() )
  {
    for ( idx_t i = reord.start(); i < newSize; i++ )
    {
      itemIDs[i] = itemIDs[rmap[i]];
    }

    itemIDs_.resize ( newSize );
    iperm_  .resize ( newSize );
  }
  else
  {
    IdxBuffer_&  newIDs  = iperm_;

    newIDs.resize ( newSize );

    for ( idx_t i = 0; i < newSize; i++ )
    {
      newIDs[i] = itemIDs[rmap[i]];
    }

    itemIDs_.resize ( newSize );
    itemIDs_.swap   ( newIDs  );
  }

  iperm_  = -1;
  sorted_ =  0;

  itemIDs = itemIDs_.addr ();

  idx_t   j = 0;

  for ( idx_t i = 0; i < newSize; i++ )
  {
    if ( itemIDs[i] >= j )
    {
      j = itemIDs[i] + 1;
    }
  }

  nextID_ = j;
}


//-----------------------------------------------------------------------
//   findItem
//-----------------------------------------------------------------------


idx_t HeapItemMap::findItem ( idx_t itemID ) const
{
  const idx_t  itemCount = itemIDs_.size ();

  if ( sorted_ < itemCount )
  {
    const_cast<Self*>(this)->sortIDs_ ();
  }

  return binarySearch ( itemID,
                        itemIDs_.addr (),
                        iperm_  .addr (),
                        itemCount );
}


//-----------------------------------------------------------------------
//   findItems
//-----------------------------------------------------------------------


idx_t HeapItemMap::findItems

  ( const IdxVector&  iitems,
    const IdxVector&  itemIDs ) const

{
  JEM_PRECHECK2 ( iitems.size() >= itemIDs.size(),
                  "output array is too small" );

  const idx_t  itemCount = itemIDs_.size ();
  const idx_t  n         = itemIDs .size ();


  if ( sorted_ < itemCount )
  {
    const_cast<Self*>(this)->sortIDs_ ();
  }

  idx_t  k = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  iitem = binarySearch ( itemIDs[i],
                                  itemIDs_.addr (),
                                  iperm_  .addr (),
                                  itemCount );

    iitems[i]  = iitem;

    if ( iitem >= 0 )
    {
      k++;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getItemID
//-----------------------------------------------------------------------


idx_t HeapItemMap::getItemID ( idx_t iitem ) const
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


void HeapItemMap::getItemIDs

  ( const IdxVector&  itemIDs,
    const IdxVector&  iitems ) const

{
  JEM_PRECHECK2 ( itemIDs.size() >= iitems.size(),
                  "output array is too small" );

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


//-----------------------------------------------------------------------
//   sortIDs_
//-----------------------------------------------------------------------


void HeapItemMap::sortIDs_ ()
{
  const idx_t  itemCount = itemIDs_.size ();

  iperm_.toArray() = iarray ( itemCount );

  jem::sort ( iperm_.toArray(), itemIDs_.toArray() );

  sorted_ = itemCount;
}


JIVE_END_PACKAGE ( util )
