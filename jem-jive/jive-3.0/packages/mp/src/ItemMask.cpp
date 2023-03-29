
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/Reordering.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/ItemMask.h>


JEM_DEFINE_SERIABLE_CLASS( jive::mp::ItemMask );


JIVE_BEGIN_PACKAGE( mp )


using jem::newInstance;


//=======================================================================
//   class ItemMask
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  ItemMask::NEW_VALUES_ = 1 << 0;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ItemMask::ItemMask ()
{}


ItemMask::ItemMask ( const Ref<ItemSet>& items ) :

  items_ ( items )

{
  JEM_PRECHECK ( items );

  connect_ ();
}


ItemMask::ItemMask ( const Self& rhs ) :

  items_ ( rhs.items_        ),
  mask_  ( rhs.mask_.clone() )

{
  connect_ ();
}


ItemMask::~ItemMask ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void ItemMask::resetEvents ()
{
  events_.reset ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ItemMask::readFrom ( ObjectInput& in )
{
  decode ( in, items_, mask_ );

  events_.reset ();

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ItemMask::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items_, mask_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> ItemMask::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t ItemMask::size () const
{
  return items_->size ();
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* ItemMask::getItems () const
{
  return items_.get ();
}


//-----------------------------------------------------------------------
//   toVector
//-----------------------------------------------------------------------


BoolVector ItemMask::toVector () const
{
  BoolVector values ( items_->size() );

  getAllValues ( values );

  return values;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ItemMask::clear ()
{
  mask_.ref ( BoolVector() );

  events_.reset ();

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


bool ItemMask::getValue ( idx_t iitem ) const
{
  if ( checkItemIndex_( iitem ) )
  {
    return mask_[iitem];
  }
  else
  {
    return true;
  }
}


//-----------------------------------------------------------------------
//   getSomeValues
//-----------------------------------------------------------------------


void ItemMask::getSomeValues

  ( const BoolVector&  values,
    const IdxVector&   iitems ) const

{
  JEM_PRECHECK ( values.size() >= iitems.size() );

  const idx_t  n = iitems.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  iitem = iitems[i];

    if ( checkItemIndex_( iitem ) )
    {
      values[i] = mask_[iitem];
    }
    else
    {
      values[i] = true;
    }
  }
}


//-----------------------------------------------------------------------
//   getAllValues
//-----------------------------------------------------------------------


idx_t ItemMask::getAllValues ( const BoolVector& values ) const
{
  const idx_t  itemCount = items_->size ();
  const idx_t  maskSize  = mask_  .size ();


  if ( values.size() < itemCount )
  {
    util::sizeError ( JEM_FUNC, "mask vector",
                      values.size(), itemCount );
  }

  if ( maskSize < itemCount )
  {
    values[slice(BEGIN,maskSize)]     = mask_;
    values[slice(maskSize,itemCount)] = true;
  }
  else
  {
    values[slice(BEGIN,itemCount)] = mask_[slice(BEGIN,itemCount)];
  }

  return itemCount;
}


//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


void ItemMask::setValue ( idx_t iitem, bool value )
{
  checkItemIndex_ ( iitem );

  mask_[iitem] = value;

  events_.emit ( NEW_VALUES_, this );
}


//-----------------------------------------------------------------------
//   setSomeValues
//-----------------------------------------------------------------------


void ItemMask::setSomeValues

  ( const IdxVector&   iitems,
    const BoolVector&  values )

{
  JEM_PRECHECK ( values.size() >= iitems.size() );

  const idx_t  n = iitems.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  iitem = iitems[i];

    checkItemIndex_ ( iitem );

    mask_[iitem] = values[i];
  }

  events_.emit ( NEW_VALUES_, this );
}


//-----------------------------------------------------------------------
//   setAllValues
//-----------------------------------------------------------------------


void ItemMask::setAllValues ( const BoolVector& values )
{
  const idx_t  itemCount = items_->size ();

  if ( values.size() < itemCount )
  {
    util::sizeError ( JEM_FUNC, "mask vector",
                      values.size(), itemCount );
  }

  if ( mask_.size() != itemCount )
  {
    mask_.resize ( itemCount );
  }

  mask_ = values[slice(BEGIN,itemCount)];

  events_.emit ( NEW_VALUES_, this );
}


//-----------------------------------------------------------------------
//   getItemsFor
//-----------------------------------------------------------------------


IdxVector ItemMask::getItemsFor ( bool value ) const
{
  const idx_t  itemCount = items_->size ();
  const idx_t  maskSize  = mask_  .size ();

  IdxVector    iitems;

  idx_t        n = 0;


  for ( idx_t iitem = 0; iitem < maskSize; iitem++ )
  {
    if ( mask_[iitem] == value )
    {
      n++;
    }
  }

  if ( value == true )
  {
    n += itemCount - maskSize;
  }

  iitems.resize ( n );

  n = 0;

  for ( idx_t iitem = 0; iitem < maskSize; iitem++ )
  {
    if ( mask_[iitem] == value )
    {
      iitems[n++] = iitem;
    }
  }

  if ( value == true )
  {
    for ( idx_t iitem = maskSize; iitem < itemCount; iitem++ )
    {
      iitems[n++] = iitem;
    }
  }

  return iitems;
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void ItemMask::store ( const Properties& globdat ) const
{
  Ref<Self> self ( const_cast<Self*>( this ) );

  Globdat::storeFor ( Globdat::ITEM_MASK, self, items_, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<ItemMask> ItemMask::find

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<Self>  mask;

  Globdat::findFor ( mask, Globdat::ITEM_MASK, items, globdat );

  return mask;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<ItemMask> ItemMask::get

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat ( Globdat::getDataFor( items, globdat ) );
  Ref<Self>   mask;

  if ( ! itemdat.find( mask, Globdat::ITEM_MASK ) )
  {
    mask = newInstance<Self> ( items );

    itemdat.set ( Globdat::ITEM_MASK, mask );
  }

  return mask;
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void ItemMask::emitEvents_ ( int events )
{
  if ( events & NEW_VALUES_ )
  {
    newValuesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   resizeMask_
//-----------------------------------------------------------------------


idx_t ItemMask::resizeMask_ ()
{
  const idx_t  oldSize = mask_  .size ();
  const idx_t  newSize = items_->size ();

  BoolVector   tmp ( newSize );

  if ( newSize < oldSize )
  {
    tmp = mask_[slice(BEGIN,newSize)];
  }
  else
  {
    tmp[slice(BEGIN,oldSize)]   = mask_;
    tmp[slice(oldSize,newSize)] = true;
  }

  mask_.swap ( tmp );

  return newSize;
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void ItemMask::connect_ ()
{
  using jem::util::connect;

  connect ( items_->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( items_->newOrderEvent, this, & Self::itemsReordered_ );

  items_->resetEvents ();
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void ItemMask::itemsResized_ ( idx_t itemCount )
{
  if ( itemCount == 0 )
  {
    clear ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void ItemMask::itemsReordered_ ( const Reordering& reord )
{
  util::reorder ( mask_, reord, true );
}


//-----------------------------------------------------------------------
//   checkItemIndex_
//-----------------------------------------------------------------------


inline void ItemMask::checkItemIndex_ ( idx_t iitem )
{
  if ( iitem < 0 || iitem >= mask_.size() )
  {
    checkIndexAgain_ ( iitem );
  }
}


inline bool ItemMask::checkItemIndex_ ( idx_t iitem ) const
{
  if ( iitem < 0 )
  {
    util::indexError ( CLASS_NAME, items_->getItemName (),
                       iitem, items_->size () );
  }

  return (iitem < mask_.size());
}


//-----------------------------------------------------------------------
//   checkIndexAgain_
//-----------------------------------------------------------------------


void ItemMask::checkIndexAgain_ ( idx_t iitem )
{
  const idx_t  itemCount = items_->size ();

  if ( iitem < 0 || iitem >= itemCount )
  {
    util::indexError ( CLASS_NAME, items_->getItemName (),
                       iitem, itemCount );
  }

  if ( iitem >= mask_.size() )
  {
    resizeMask_ ();
  }
}


JIVE_END_PACKAGE( mp )
