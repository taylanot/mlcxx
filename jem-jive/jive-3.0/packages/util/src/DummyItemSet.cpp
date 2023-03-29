
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
#include <jem/io/PrintWriter.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/utilities.h>
#include <jive/util/Reordering.h>
#include <jive/util/SimpleItemMap.h>
#include <jive/util/DummyItemSet.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::DummyItemSet );


using jem::checkedCast;
using jem::newInstance;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class DummyItemSet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  DummyItemSet::NEW_SIZE_ = 1 << 0;


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


DummyItemSet::DummyItemSet ()
{}


DummyItemSet::DummyItemSet

  ( const String&         name,
    const String&         item,
    const Ref<XItemMap>&  map ) :

    Super      ( name ),
    map_       (  map ),
    itemName_  ( item ),
    itemCount_ (    0 )

{
  if ( map_ )
  {
    map_->clear ();
  }
  else
  {
    map_ = newInstance<SimpleItemMap> ();
  }
}


DummyItemSet::DummyItemSet ( const Self& rhs ) :

  Super      ( rhs.myName_    ),
  map_       ( rhs.map_       ),
  itemName_  ( rhs.itemName_  ),
  events_    ( rhs.events_    ),
  itemCount_ ( rhs.itemCount_ )

{
  map_ = checkedCast<XItemMap> ( map_->clone() );
}


DummyItemSet::~DummyItemSet ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void DummyItemSet::resetEvents ()
{
  events_.reset ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void DummyItemSet::readFrom ( ObjectInput& in )
{
  decode ( in, map_, myName_, itemName_ );

  itemCount_ = map_->size ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void DummyItemSet::writeTo ( ObjectOutput& out ) const
{
  encode ( out, map_, myName_, itemName_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> DummyItemSet::clone () const
{
  return newInstance<Self> ( * this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t DummyItemSet::size () const
{
  return itemCount_;
}


//-----------------------------------------------------------------------
//   getItemName
//-----------------------------------------------------------------------


String DummyItemSet::getItemName () const
{
  return itemName_;
}


//-----------------------------------------------------------------------
//   getItemMap
//-----------------------------------------------------------------------


ItemMap* DummyItemSet::getItemMap () const
{
  return map_.get ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void DummyItemSet::clear ()
{
  if ( itemCount_ > 0 )
  {
    map_  ->clear ();
    events_.reset ();

    itemCount_ = 0;

    newSizeEvent.emit ( 0, *this );
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void DummyItemSet::reserve ( idx_t n )
{
  map_->reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void DummyItemSet::trimToSize ()
{
  map_->trimToSize ();
}


//-----------------------------------------------------------------------
//   addItem
//-----------------------------------------------------------------------


idx_t DummyItemSet::addItem ()
{
  const idx_t  iitem = map_->addItem ();

  itemCount_++;

  events_.emit ( NEW_SIZE_, this );

  return iitem;
}


idx_t DummyItemSet::addItem ( idx_t id )
{
  const idx_t  iitem = map_->addItem ( id );

  itemCount_++;

  events_.emit ( NEW_SIZE_, this );

  return iitem;
}


//-----------------------------------------------------------------------
//   addItems
//-----------------------------------------------------------------------


idx_t DummyItemSet::addItems ( idx_t count )
{
  JEM_PRECHECK2 ( count >= 0, "invalid number of items to add" );

  const idx_t  ifirst = map_->addItems ( count );

  itemCount_ += count;

  events_.emit ( NEW_SIZE_, this );

  return ifirst;
}


idx_t DummyItemSet::addItems ( const IdxVector& ids )
{
  const idx_t  ifirst = itemCount_;
  const idx_t  n      = ids.size ();


  if ( n > 1000_idx )
  {
    map_->reserve ( itemCount_ + n );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    map_->addItem ( ids[i] );
  }

  itemCount_ += n;

  events_.emit ( NEW_SIZE_, this );

  return ifirst;
}


//-----------------------------------------------------------------------
//   eraseItems
//-----------------------------------------------------------------------


void DummyItemSet::eraseItems ( const IdxVector& iitems )
{
  reorderItems ( makeEraseReordering( iitems, itemCount_ ) );
}


//-----------------------------------------------------------------------
//   reorderItems
//-----------------------------------------------------------------------


void DummyItemSet::reorderItems ( const Reordering& reord )
{
  map_  ->reorderItems ( reord );
  events_.reset        ();

  itemCount_ = reord.newSize ();

  newOrderEvent.emit ( reord, *this );

  if ( itemCount_ != reord.oldSize() )
  {
    newSizeEvent.emit ( itemCount_, *this );
  }
}


//-----------------------------------------------------------------------
//   printSubset
//-----------------------------------------------------------------------


void DummyItemSet::printSubset

  ( PrintWriter&      out,
    const IdxVector&  iitems ) const

{
  using jem::io::endl;

  const idx_t  n = iitems.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    print ( out, endl, map_->getItemID( iitems[i] ) );
  }
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void DummyItemSet::emitEvents_ ( int events )
{
  if ( events & NEW_SIZE_ )
  {
    newSizeEvent.emit ( itemCount_, *this );
  }
}


JIVE_END_PACKAGE( util )
