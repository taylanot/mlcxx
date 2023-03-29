
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
#include <jem/base/Array.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/BitSet.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/Reordering.h>
#include <jive/util/ColoredItemGroup.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::ColoredItemGroup );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;
using jem::util::BitSet;


//=======================================================================
//   class ColoredItemGroup
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ColoredItemGroup::ColoredItemGroup () :

  imask_ ( 0 ),
  bmask_ ( 0 )

{}


ColoredItemGroup::ColoredItemGroup

  ( const IdxVector&     colors,
    const Ref<ItemSet>&  items ) :

    items_ ( items ),
    imask_ (     0 ),
    bmask_ (     0 )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );
  JEM_PRECHECK2 ( colors.size() == items->size(),
                  "color array size mismatch" );

  const idx_t  itemCount = colors.size ();

  IdxVector    iitems    ( itemCount );

  for ( idx_t iitem = 0; iitem < itemCount; iitem++ )
  {
    iitems[iitem] = iitem;
  }

  init_ ( colors, iitems );

  connect_ ();
}


ColoredItemGroup::ColoredItemGroup

  ( const IdxVector&     colors,
    const IdxVector&     iitems,
    const Ref<ItemSet>&  items ) :

    items_ ( items ),
    imask_ (     0 ),
    bmask_ (     0 )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );
  JEM_PRECHECK2 ( colors.size() == iitems.size(),
                  "color array size mismatch" );

  const idx_t  itemCount  = items->size ();
  const idx_t  iitemCount = iitems.size ();

  for ( idx_t i = 0; i < iitemCount; i++ )
  {
    idx_t  iitem = iitems[i];

    if ( iitem < 0 || iitem >= itemCount )
    {
      indexError ( CLASS_NAME,
                   items->getItemName(), iitem, itemCount );
    }
  }

  init_    ( colors, iitems );
  connect_ ();
}


ColoredItemGroup::~ColoredItemGroup ()
{
  freeMask_ ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ColoredItemGroup::readFrom ( ObjectInput& in )
{
  freeMask_ ();
  decode    ( in, items_, iitems_, listOffsets_, listColors_ );
  connect_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ColoredItemGroup::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items_, iitems_, listOffsets_, listColors_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> ColoredItemGroup::clone () const
{
  return newInstance<Self> ( iitems_, items_ );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t ColoredItemGroup::size () const
{
  return iitems_.size ();
}


//-----------------------------------------------------------------------
//   getIndices
//-----------------------------------------------------------------------


IdxVector ColoredItemGroup::getIndices () const
{
  return iitems_;
}


//-----------------------------------------------------------------------
//   getIndex
//-----------------------------------------------------------------------


idx_t ColoredItemGroup::getIndex ( idx_t i ) const
{
  if ( i < 0 || i >= iitems_.size() )
  {
    indexError ( JEM_FUNC, "item", i, iitems_.size() );
  }

  return iitems_[i];
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* ColoredItemGroup::getItems () const
{
  return items_.get ();
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool ColoredItemGroup::contains ( idx_t iitem ) const
{
  JEM_PRECHECK2 ( iitem >= 0, "invalid item index" );

 again:

  if      ( bmask_ )
  {
    return bmask_->get ( iitem );
  }
  else if ( imask_ )
  {
    return ( binarySearch( iitem, *imask_ ) >= 0 );
  }
  else
  {
    initMask_ ();

    goto again;

    return false;
  }
}


//-----------------------------------------------------------------------
//   filter
//-----------------------------------------------------------------------


void ColoredItemGroup::filter ( const IdxVector& iitems ) const
{
  const idx_t  n = iitems.size ();

 again:

  if ( bmask_ )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      if ( ! bmask_->get( iitems[i] ) )
      {
        iitems[i] = -iitems[i] - 1;
      }
    }
  }
  else if ( imask_ )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      if ( binarySearch( iitems[i], *imask_ ) < 0 )
      {
        iitems[i] = -iitems[i] - 1;
      }
    }
  }
  else
  {
    initMask_ ();

    goto again;
  }
}


//-----------------------------------------------------------------------
//   listCount
//-----------------------------------------------------------------------


idx_t ColoredItemGroup::listCount () const
{
  return listColors_.size ();
}


//-----------------------------------------------------------------------
//   getList
//-----------------------------------------------------------------------


IdxVector ColoredItemGroup::getList ( idx_t ilist ) const
{
  checkListIndex_ ( ilist, listColors_.size() );

  const idx_t  first = listOffsets_[ilist];
  const idx_t  last  = listOffsets_[ilist + 1];

  return iitems_[slice(first,last)];
}


//-----------------------------------------------------------------------
//   getListOffset
//-----------------------------------------------------------------------


idx_t ColoredItemGroup::getListOffset ( idx_t ilist ) const
{
  checkListIndex_ ( ilist, listOffsets_.size() );

  return listOffsets_[ilist];
}


//-----------------------------------------------------------------------
//   getListColor
//-----------------------------------------------------------------------


idx_t ColoredItemGroup::getListColor ( idx_t ilist ) const
{
  checkListIndex_ ( ilist, listColors_.size() );

  return listColors_[ilist];
}


//-----------------------------------------------------------------------
//   initMask_
//-----------------------------------------------------------------------


void ColoredItemGroup::initMask_ () const
{
  Self*        self       = const_cast<Self*> ( this );
  const idx_t  itemCount  = items_->size ();
  const idx_t  iitemCount = iitems_.size ();


  if ( bmask_ || imask_ )
  {
    self->freeMask_ ();
  }

  if ( 32_idx * iitemCount <= itemCount )
  {
    self->imask_ = new IdxVector ( iitemCount );

    *imask_ = iitems_;

    sort ( *imask_ );
  }
  else
  {
    self->bmask_ = new BitSet ( itemCount );

    for ( idx_t i = 0; i < iitemCount; i++ )
    {
      bmask_->set ( iitems_[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   freeMask_
//-----------------------------------------------------------------------


void ColoredItemGroup::freeMask_ ()
{
  if ( imask_ )
  {
    delete imask_;
    imask_ = 0;
  }

  if ( bmask_ )
  {
    delete bmask_;
    bmask_ = 0;
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void ColoredItemGroup::connect_ ()
{
  using jem::util::connect;

  connect ( items_->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( items_->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void ColoredItemGroup::init_

  ( const IdxVector&  colors,
    const IdxVector&  iitems )

{
  const idx_t  iitemCount = iitems.size ();


  iitems_.resize ( iitemCount );

  listOffsets_.ref ( sortColors( iitems_, colors ) );

  idx_t  listCount = listOffsets_.size() - 1;

  listColors_.resize ( listCount );

  for ( idx_t i = 0; i < listCount; i++ )
  {
    listColors_[i] = colors[iitems_[listOffsets_[i]]];
  }

  for ( idx_t i = 0; i < iitemCount; i++ )
  {
    iitems_[i] = iitems[iitems_[i]];
  }
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void ColoredItemGroup::itemsResized_ ( idx_t itemCount )
{
  if ( itemCount == 0 )
  {
    const idx_t  n = iitems_.size ();

    iitems_.ref ( IdxVector() );

    listOffsets_ = 0;

    freeMask_ ();

    if ( n > 0 )
    {
      newSizeEvent.emit ( itemCount, * this );
    }
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void ColoredItemGroup::itemsReordered_ ( const Reordering& reord )
{
  const idx_t*  JEM_RESTRICT  iperm = reord.fmapPtr ();

  const idx_t   iitemCount = iitems_.size ();

  idx_t         n = 0;


  for ( idx_t i = 0; i < iitemCount; i++ )
  {
    if ( iperm[iitems_[i]] >= 0 )
    {
      n++;
    }
  }

  if ( n == iitemCount )
  {
    for ( idx_t i = 0; i < iitemCount; i++ )
    {
      iitems_[i] = iperm[iitems_[i]];
    }
  }
  else
  {
    const idx_t  listCount = listColors_.size ();

    IdxVector    jperm  ( iitemCount );
    IdxVector    jitems ( n );


    freeMask_ ();

    // Reorder the item indices.

    n = 0;

    for ( idx_t i = 0; i < listCount; i++ )
    {
      idx_t  first = listOffsets_[i];
      idx_t  last  = listOffsets_[i + 1];

      listOffsets_[i] = n;

      for ( idx_t j = first; j < last; j++ )
      {
        idx_t  k = iperm[iitems_[j]];

        if ( k >= 0 )
        {
          jperm [j] = n;
          jitems[n] = k;
          n++;
        }
        else
        {
          jperm[j] = -1;
        }
      }
    }

    listOffsets_[listCount] = n;

    iitems_.swap ( jitems );

    // Remove all empty lists

    n = 0;

    for ( idx_t i = 0; i < listCount; i++ )
    {
      if ( listOffsets_[i + 1] > listOffsets_[i] )
      {
        listOffsets_[n + 1] = listOffsets_[i + 1];
        listColors_ [n]     = listColors_ [i];
        n++;
      }
    }

    if ( n != listCount )
    {
      listOffsets_.ref ( listOffsets_[slice(BEGIN,n + 1)] );
      listColors_ .ref ( listColors_ [slice(BEGIN,n)] );
    }

    newOrderEvent.emit ( Reordering( jperm ),
                         ITEMS_REORDERED, *this );
    newSizeEvent .emit ( iitems_.size(),  *this );
  }
}


//-----------------------------------------------------------------------
//   checkListIndex_
//-----------------------------------------------------------------------


inline void ColoredItemGroup::checkListIndex_

  ( idx_t  ilist,
    idx_t  ubound )

{
  if ( ilist < 0 || ilist >= ubound )
  {
    indexError ( CLASS_NAME,
                 "colored item list", ilist, ubound );
  }
}


JIVE_END_PACKAGE( util )
