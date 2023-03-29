
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
#include <jem/base/IllegalOperationException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/SimpleItemMap.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::SimpleItemMap );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;
using jem::IllegalOperationException;


//=======================================================================
//   class SimpleItemMap
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SimpleItemMap::SimpleItemMap () :

  firstID_   ( 0 ),
  itemCount_ ( 0 )

{}


SimpleItemMap::SimpleItemMap ( const Self& rhs ) :

  firstID_   ( rhs.firstID_   ),
  itemCount_ ( rhs.itemCount_ )

{}


SimpleItemMap::~SimpleItemMap ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void SimpleItemMap::readFrom ( ObjectInput& in )
{
  decode ( in, firstID_, itemCount_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void SimpleItemMap::writeTo ( ObjectOutput& out ) const
{
  encode ( out, firstID_, itemCount_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> SimpleItemMap::clone () const
{
  return newInstance<Self> ( * this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t SimpleItemMap::size () const
{
  return itemCount_;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SimpleItemMap::clear ()
{
  firstID_   = 0;
  itemCount_ = 0;
}


//-----------------------------------------------------------------------
//   addItem
//-----------------------------------------------------------------------


idx_t SimpleItemMap::addItem ()
{
  const idx_t  iitem = itemCount_;

  itemCount_++;

  return iitem;
}


idx_t SimpleItemMap::addItem ( idx_t itemID )
{
  const idx_t  iitem = itemCount_;

  if ( itemCount_ == 0 )
  {
    firstID_ = itemID;
  }
  else if ( itemID - firstID_ != itemCount_ )
  {
    throw ItemIDException (
      JEM_FUNC,
      String::format (
        "illegal item ID: %d; should be %d "
        "(or try another storage mode)",
        itemID,
        itemCount_ + firstID_
      ),
      itemID
    );
  }

  itemCount_++;

  return iitem;
}


//-----------------------------------------------------------------------
//   addItems
//-----------------------------------------------------------------------


idx_t SimpleItemMap::addItems ( idx_t count )
{
  JEM_PRECHECK2 ( count >= 0, "invalid number of items to add" );

  const idx_t  iitem = itemCount_;

  itemCount_ += count;

  return iitem;
}


//-----------------------------------------------------------------------
//   reorderItems
//-----------------------------------------------------------------------


void SimpleItemMap::reorderItems ( const Reordering& )
{
  throw IllegalOperationException (
    JEM_FUNC,
    "unsupported operation: reorderItems"
  );
}


//-----------------------------------------------------------------------
//   findItem
//-----------------------------------------------------------------------


idx_t SimpleItemMap::findItem ( idx_t itemID ) const
{
  idx_t  iitem = itemID - firstID_;

  if ( iitem >= 0 && iitem < itemCount_ )
  {
    return iitem;
  }
  else
  {
    return -1_idx;
  }
}


//-----------------------------------------------------------------------
//   findItems
//-----------------------------------------------------------------------


idx_t SimpleItemMap::findItems

  ( const IdxVector&  iitems,
    const IdxVector&  itemIDs ) const

{
  JEM_PRECHECK2 ( iitems.size() >= itemIDs.size(),
                  "output array is too small" );

  const idx_t  n = itemIDs.size ();

  idx_t        k = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  iitem = itemIDs[i] - firstID_;

    if ( iitem >= 0 && iitem < itemCount_ )
    {
      iitems[i] = iitem;
      k++;
    }
    else
    {
      iitems[i] = -1;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getItemID
//-----------------------------------------------------------------------


idx_t SimpleItemMap::getItemID ( idx_t iitem ) const
{
  return (iitem + firstID_);
}


//-----------------------------------------------------------------------
//   getItemIDs
//-----------------------------------------------------------------------


void SimpleItemMap::getItemIDs

  ( const IdxVector&  itemIDs,
    const IdxVector&  iitems ) const

{
  JEM_PRECHECK2 ( itemIDs.size() >= iitems.size(),
                  "output array is too small" );

  const idx_t  n = iitems.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    itemIDs[i] = iitems[i] + firstID_;
  }
}


JIVE_END_PACKAGE ( util )
