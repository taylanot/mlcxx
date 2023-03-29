
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
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/BitSet.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/Reordering.h>
#include <jive/util/ArrayItemGroup.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::ArrayItemGroup );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;
using jem::util::BitSet;


//=======================================================================
//   class ArrayItemGroup
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ArrayItemGroup::ArrayItemGroup () :

  imask_ ( 0 ),
  bmask_ ( 0 )

{}


ArrayItemGroup::ArrayItemGroup

  ( const Ref<ItemSet>& items ) :

    items_ ( items ),
    imask_ (     0 ),
    bmask_ (     0 )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  connect_ ();
}


ArrayItemGroup::ArrayItemGroup

  ( const IdxVector&     iitems,
    const Ref<ItemSet>&  items ) :

    items_  (  items ),
    iitems_ ( iitems ),
    imask_  (      0 ),
    bmask_  (      0 )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  checkIndices_ ( iitems );

  connect_ ();
}


ArrayItemGroup::~ArrayItemGroup ()
{
  freeMask_ ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ArrayItemGroup::readFrom ( ObjectInput& in )
{
  freeMask_ ();
  decode    ( in, items_, iitems_ );
  connect_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ArrayItemGroup::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items_, iitems_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> ArrayItemGroup::clone () const
{
  return newInstance<Self> ( iitems_, items_ );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t ArrayItemGroup::size () const
{
  return iitems_.size();
}


//-----------------------------------------------------------------------
//   getIndices
//-----------------------------------------------------------------------


IdxVector ArrayItemGroup::getIndices () const
{
  return iitems_;
}


//-----------------------------------------------------------------------
//   getIndex
//-----------------------------------------------------------------------


idx_t ArrayItemGroup::getIndex ( idx_t i ) const
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


ItemSet* ArrayItemGroup::getItems () const
{
  return items_.get ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ArrayItemGroup::clear ()
{
  if ( iitems_.size() > 0 )
  {
    freeMask_ ();

    iitems_.resize ( 0 );

    newSizeEvent.emit ( 0, *this );
  }
}


//-----------------------------------------------------------------------
//   append
//-----------------------------------------------------------------------


void ArrayItemGroup::append ( const IdxVector&  iitems )
{
  if ( iitems.size() > 0 )
  {
    const idx_t  n = iitems_.size ();

    checkIndices_ ( iitems );
    freeMask_     ();

    iitems_.reshape ( n + iitems.size() );

    iitems_[slice(n,END)] = iitems;

    newSizeEvent.emit ( iitems_.size(), *this );
  }
}


//-----------------------------------------------------------------------
//   reorder
//-----------------------------------------------------------------------


void ArrayItemGroup::reorder ( const Reordering& reord )
{
  JEM_ASSERT2 ( reord.oldSize() == size(),
                "Reordering has an invalid size" );

  freeMask_     ();
  util::reorder ( iitems_, reord );

  newOrderEvent.emit ( reord, GROUP_REORDERED, *this );
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool ArrayItemGroup::contains ( idx_t iitem ) const
{
  JEM_PRECHECK2 ( iitem >= 0, "invalid item index" );

 again:

  if      ( bmask_ )
  {
    return bmask_->get ( iitem );
  }
  else if ( imask_ )
  {
    return ( jem::binarySearch( iitem, *imask_ ) >= 0 );
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


void ArrayItemGroup::filter ( const IdxVector& iitems ) const
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
      if ( jem::binarySearch( iitems[i], *imask_ ) < 0 )
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
//   initMask_
//-----------------------------------------------------------------------


void ArrayItemGroup::initMask_ () const
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

    jem::sort ( *imask_ );
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


void ArrayItemGroup::freeMask_ ()
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


void ArrayItemGroup::connect_ ()
{
  using jem::util::connect;

  connect ( items_->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( items_->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   checkIndices_
//-----------------------------------------------------------------------


void ArrayItemGroup::checkIndices_ ( const IdxVector& iitems ) const
{
  const idx_t  itemCount  = items_->size ();
  const idx_t  iitemCount = iitems .size ();

  for ( idx_t i = 0; i < iitemCount; i++ )
  {
    idx_t  iitem = iitems[i];

    if ( iitem < 0 || iitem >= itemCount )
    {
      indexError ( JEM_FUNC,
                   items_->getItemName(), iitem, itemCount );
    }
  }
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void ArrayItemGroup::itemsResized_ ( idx_t itemCount )
{
  if ( itemCount == 0 )
  {
    const idx_t  n = iitems_.size ();

    iitems_.ref ( IdxVector() );

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


void ArrayItemGroup::itemsReordered_ ( const Reordering& reord )
{
  const idx_t*  JEM_RESTRICT  iperm = reord.fmapPtr ();

  const idx_t   n = iitems_.size ();

  idx_t         k = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( iperm[iitems_[i]] >= 0 )
    {
      k++;
    }
  }

  if ( k == n )
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      iitems_[i] = iperm[iitems_[i]];
    }
  }
  else
  {
    IdxVector  jperm  ( n );
    IdxVector  jitems ( k );

    k = 0;

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  j = iperm[iitems_[i]];

      if ( j >= 0 )
      {
        jperm [i] = k;
        jitems[k] = j;
        k++;
      }
      else
      {
        jperm[i] = -1;
      }
    }

    freeMask_ ();

    iitems_.swap ( jitems );

    newOrderEvent.emit ( Reordering( jperm ),
                         ITEMS_REORDERED, * this );
    newSizeEvent .emit ( k,               * this );
  }
}


JIVE_END_PACKAGE( util )
