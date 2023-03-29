
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
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/AllItemGroup.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::AllItemGroup );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;


//=======================================================================
//   class AllItemGroup
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


AllItemGroup::AllItemGroup ()
{}


AllItemGroup::AllItemGroup ( const Ref<ItemSet>& items ) :

  items_ ( items )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  connect_ ();
}


AllItemGroup::~AllItemGroup ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void AllItemGroup::resetEvents ()
{
  items_->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void AllItemGroup::readFrom ( ObjectInput& in )
{
  using jem::checkedCast;

  items_ = checkedCast<ItemSet> ( in.decodeObject() );

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void AllItemGroup::writeTo ( ObjectOutput& out ) const
{
  out.encodeObject ( items_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> AllItemGroup::clone () const
{
  return newInstance<Self> ( items_ );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t AllItemGroup::size () const
{
  return items_->size ();
}


//-----------------------------------------------------------------------
//   getIndices
//----------------------------------------------------------------------


IdxVector AllItemGroup::getIndices () const
{
  const idx_t  n = items_->size ();

  IdxVector    iitems ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    iitems[i] = i;
  }

  return iitems;
}


//-----------------------------------------------------------------------
//   getIndex
//----------------------------------------------------------------------


idx_t AllItemGroup::getIndex ( idx_t i ) const
{
  return i;
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* AllItemGroup::getItems () const
{
  return items_.get ();
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


bool AllItemGroup::contains ( idx_t iitem ) const
{
  return true;
}


//-----------------------------------------------------------------------
//   filter
//-----------------------------------------------------------------------


void AllItemGroup::filter ( const IdxVector& iitems ) const
{
  return;
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void AllItemGroup::connect_ ()
{
  using jem::util::connect;

  connect ( items_->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( items_->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void AllItemGroup::itemsResized_ ( idx_t itemCount )
{
  newSizeEvent.emit ( itemCount, * this );
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void AllItemGroup::itemsReordered_ ( const Reordering& reord )
{
  newOrderEvent.emit ( reord, ITEMS_REORDERED, * this );
}


JIVE_END_PACKAGE( util )
