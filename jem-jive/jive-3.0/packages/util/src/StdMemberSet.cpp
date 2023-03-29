
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
#include <jem/util/Flex.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/util/EventMask.h>
#include <jive/util/Reordering.h>
#include <jive/util/HashItemMap.h>
#include <jive/util/StdMemberSet.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::StdMemberSet );


JIVE_BEGIN_PACKAGE( util )


using jem::checkedCast;
using jem::newInstance;
using jem::util::Flex;


//=======================================================================
//   class StdMemberSet::Data_
//=======================================================================


class StdMemberSet::Data_ : public jem::Collectable
{
 public:

  typedef Data_           Self;

  static const int        NEW_SIZE = 1 << 0;


  inline                  Data_   ();

                          Data_

    ( const String&         mbrName,
      const Ref<ItemSet>&   items );

  void                    readFrom

    ( ObjectInput&          in );

  void                    writeTo

    ( ObjectOutput&         out )      const;

  inline void             checkMemberIndex

    ( idx_t                 imbr )     const;

  inline void             checkItemIndex

    ( idx_t                 iitem );

  inline void             checkLocalIndex

    ( idx_t                 ilocal )   const;

  inline void             updateItemCount

    ( idx_t                 itemCount );

  inline void             addMember

    ( idx_t                 iitem,
      idx_t                 ilocal );


 public:

  Ref<ItemSet>            items;
  String                  mbrName;
  Flex<idx_t>             members;
  EventMask               events;

  idx_t                   mbrCount;


 private:

  void                    checkItemIndex_

    ( idx_t                 iitem );


 private:

  // This is only an estimate.

  idx_t                   itemCount_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline StdMemberSet::Data_::Data_ ()
{
  mbrCount = -1_idx;
}


StdMemberSet::Data_::Data_

  ( const String&        mname,
    const Ref<ItemSet>&  iset ) :

    items      (  iset ),
    mbrName    ( mname ),
    mbrCount   (     0 ),
    itemCount_ (     0 )

{
  if ( mbrName.size() == 0 )
  {
    mbrName = items->getItemName() + " member";
  }

  itemCount_ = items->size ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdMemberSet::Data_::readFrom ( ObjectInput& in )
{
  decode ( in, items, mbrName, members );

  mbrCount   = members.size() / 2;
  itemCount_ = items->size ();

  if ( 2 * mbrCount != members.size() )
  {
    jem::io::decodeError ( JEM_FUNC );
  }

  events.reset ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdMemberSet::Data_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items, mbrName, members );
}


//-----------------------------------------------------------------------
//   checkMemberIndex
//-----------------------------------------------------------------------


inline void
  StdMemberSet::Data_::checkMemberIndex ( idx_t imbr ) const
{
  if ( imbr < 0 || imbr >= mbrCount )
  {
    indexError ( CLASS_NAME, mbrName, imbr, mbrCount );
  }
}


//-----------------------------------------------------------------------
//   checkItemIndex
//-----------------------------------------------------------------------


inline void StdMemberSet::Data_::checkItemIndex ( idx_t iitem )
{
  if ( iitem < 0 || iitem >= itemCount_ )
  {
    checkItemIndex_ ( iitem );
  }
}


//-----------------------------------------------------------------------
//   checkLocalIndex
//-----------------------------------------------------------------------


inline void
  StdMemberSet::Data_::checkLocalIndex ( idx_t ilocal ) const
{
  if ( ilocal < 0 )
  {
    indexError ( CLASS_NAME, "local " + items->getItemName(),
                 ilocal, jem::maxOf( ilocal ) );
  }
}


//-----------------------------------------------------------------------
//   updateItemCount
//-----------------------------------------------------------------------


inline void StdMemberSet::Data_::updateItemCount ( idx_t n )
{
  itemCount_ = n;
}


//-----------------------------------------------------------------------
//   addMember
//-----------------------------------------------------------------------


inline void StdMemberSet::Data_::addMember

  ( idx_t  iitem,
    idx_t  ilocal )

{
  checkItemIndex   ( iitem );
  checkLocalIndex  ( ilocal );

  members.pushBack ( iitem );
  members.pushBack ( ilocal );

  mbrCount++;
}


//-----------------------------------------------------------------------
//   checkItemIndex_
//-----------------------------------------------------------------------


void StdMemberSet::Data_::checkItemIndex_ ( idx_t iitem )
{
  itemCount_ = items->size ();

  if ( iitem < 0 || iitem >= itemCount_ )
  {
    indexError ( CLASS_NAME,
                 items->getItemName(), iitem, itemCount_ );
  }
}


//=======================================================================
//   class StdMemberSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdMemberSet::StdMemberSet ()
{}


StdMemberSet::StdMemberSet

  ( const String&         name,
    const String&         mbrName,
    const Ref<ItemSet>&   items,
    const Ref<XItemMap>&  map ) :

    Super ( name ),
    map_  (  map )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  data_ = newInstance<Data_> ( mbrName, items );

  if ( map_ )
  {
    map_->clear ();
  }
  else
  {
    map_ = newInstance<HashItemMap> ();
  }

  connect_ ();
}


StdMemberSet::StdMemberSet ( const Self& rhs ) :

  Super ( rhs.myName_ )

{
  data_ = newInstance<Data_>       ( * rhs.data_ );
  map_  = checkedCast<HashItemMap> (   rhs.map_->clone() );

  connect_ ();
}


StdMemberSet::~StdMemberSet ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void StdMemberSet::resetEvents ()
{
  data_->events.reset ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdMemberSet::readFrom ( ObjectInput& in )
{
  decode ( in, myName_ );

  data_ = newInstance<Data_> ();

  data_->readFrom ( in );

  decode   ( in, map_ );
  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdMemberSet::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_ );
  data_->writeTo (  out );
  encode ( out, map_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> StdMemberSet::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t StdMemberSet::size () const
{
  return data_->mbrCount;
}


//-----------------------------------------------------------------------
//   getItemName
//-----------------------------------------------------------------------


String StdMemberSet::getItemName () const
{
  return data_->mbrName;
}


//-----------------------------------------------------------------------
//   getItemMap
//-----------------------------------------------------------------------


ItemMap* StdMemberSet::getItemMap () const
{
  return map_.get ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void StdMemberSet::clear ()
{
  Data_&  d = * data_;

  if ( d.mbrCount > 0 )
  {
    map_    ->clear ();
    d.members.clear ();
    d.events .reset ();

    d.mbrCount = 0;

    newSizeEvent.emit ( 0, *this );
    changeEvent .emit (    *this );
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void StdMemberSet::reserve ( idx_t n )
{
  data_->members.reserve ( 2 * n );
  map_         ->reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void StdMemberSet::trimToSize ()
{
  data_->members.trimToSize ();
  map_         ->trimToSize ();
}


//-----------------------------------------------------------------------
//   addMember
//-----------------------------------------------------------------------


idx_t StdMemberSet::addMember

  ( idx_t  iitem,
    idx_t  ilocal )

{
  Data_&       d    = * data_;

  const idx_t  imbr = map_->addItem ();

  d.addMember   ( iitem, ilocal );
  d.events.emit ( Data_::NEW_SIZE, this );

  return imbr;
}


idx_t StdMemberSet::addMember

  ( idx_t  id,
    idx_t  iitem,
    idx_t  ilocal )

{
  Data_&       d    = * data_;

  const idx_t  imbr = map_->addItem ( id );

  d.addMember   ( iitem, ilocal );
  d.events.emit ( Data_::NEW_SIZE, this );

  return imbr;
}


//-----------------------------------------------------------------------
//   addMembers
//-----------------------------------------------------------------------


idx_t StdMemberSet::addMembers

  ( const IdxVector&  iitems,
    const IdxVector&  ilocals )

{
  JEM_PRECHECK2 ( iitems.size() == ilocals.size(),
                  "Array size mismatch" );

  Data_&       d = * data_;

  const idx_t  n = iitems.size ();

  idx_t        k = d.members.size ();


  d.members.pushBack ( -1_idx, 2 * n );

  for ( idx_t i = 0; i < n; i++, k += 2 )
  {
    idx_t  iitem  = iitems [i];
    idx_t  ilocal = ilocals[i];

    d.checkItemIndex  ( iitem  );
    d.checkLocalIndex ( ilocal );

    d.members[k + 0] = iitem;
    d.members[k + 1] = ilocal;
  }

  d.mbrCount += n;

  k = map_->addItems ( n );

  d.events.emit ( Data_::NEW_SIZE, this );

  return k;
}


//-----------------------------------------------------------------------
//   reorderMembers
//-----------------------------------------------------------------------


void StdMemberSet::reorderMembers ( const Reordering& reord )
{
  JEM_PRECHECK2 ( reord.oldSize() == size(),
                  "invalid Reordering" );

  Data_&        d = * data_;

  const idx_t*  JEM_RESTRICT  rmap    = reord.rmapPtr  ();
  idx_t*        JEM_RESTRICT  members = d.members.addr ();

  const idx_t   start   = reord.start   ();
  const idx_t   newSize = reord.newSize ();


  map_->reorderItems ( reord );

  if ( reord.isSimple() )
  {
    for ( idx_t imbr = start; imbr < newSize; imbr++ )
    {
      idx_t  i = 2 * imbr;
      idx_t  j = 2 * rmap[imbr];

      members[i + 0] = members[j + 0];
      members[i + 1] = members[j + 1];
    }

    d.members.resize ( 2 * newSize );
  }
  else
  {
    Flex<idx_t>  tmpMembers ( 2 * newSize );

    idx_t* JEM_RESTRICT  newMembers = tmpMembers.addr ();


    if ( start > 0 )
    {
      std::memcpy ( newMembers, members,
                    (size_t) (2 * start) * sizeof(idx_t) );

      newMembers += (2 * start);
    }

    for ( idx_t imbr = start; imbr < newSize;
          imbr++, newMembers += 2 )
    {
      idx_t  j = 2 * rmap[imbr];

      newMembers[0] = members[j + 0];
      newMembers[1] = members[j + 1];
    }

    d.members.swap ( tmpMembers );
  }

  d.events.reset ();

  d.mbrCount = newSize;

  newOrderEvent.emit ( reord, *this );

  if ( newSize != reord.oldSize() )
  {
    newSizeEvent.emit ( newSize, *this );
  }

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getMember
//-----------------------------------------------------------------------


void StdMemberSet::getMember

  ( idx_t&  iitem,
    idx_t&  ilocal,
    idx_t   imbr ) const

{
  const Data_&  d = * data_;

  d.checkMemberIndex ( imbr );

  iitem  = d.members[2 * imbr];
  ilocal = d.members[2 * imbr + 1];
}


//-----------------------------------------------------------------------
//   getCompoundItems
//-----------------------------------------------------------------------


ItemSet* StdMemberSet::getCompoundItems () const
{
  return data_->items.get ();
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void StdMemberSet::connect_ ()
{
  using jem::util::connect;

  Data_&  d = * data_;

  connect ( d.items->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( d.items->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void StdMemberSet::emitEvents_ ( int events )
{
  if ( events & Data_::NEW_SIZE )
  {
    newSizeEvent.emit ( data_->mbrCount, *this );
  }

  if ( events )
  {
    changeEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void StdMemberSet::itemsResized_ ( idx_t itemCount )
{
  data_->updateItemCount ( itemCount );

  if ( itemCount == 0 )
  {
    clear ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void StdMemberSet::itemsReordered_ ( const Reordering& reord )
{
  Data_&        d = * data_;

  const idx_t*  JEM_RESTRICT  itemMap = reord.fmapPtr  ();
  idx_t*        JEM_RESTRICT  members = d.members.addr ();

  const idx_t   mbrCount = d.mbrCount;

  IdxVector     mbrMap   ( mbrCount );

  idx_t         jmbr     = 0;


  for ( idx_t imbr = 0; imbr < mbrCount; imbr++ )
  {
    idx_t  iitem = members[2 * imbr];
    idx_t  jitem = itemMap[iitem];

    if ( jitem >= 0 )
    {
      members[2 * jmbr + 0] = jitem;
      members[2 * jmbr + 1] = members[2 * imbr + 1];

      mbrMap[imbr] = jmbr++;
    }
    else
    {
      mbrMap[imbr] = -1;
    }
  }

  if ( jmbr < mbrCount )
  {
    d.events.reset ();

    d.mbrCount = jmbr;

    d.members.resize ( 2 * jmbr );

    newOrderEvent.emit ( Reordering( mbrMap ), *this );
    newSizeEvent .emit ( jmbr,                 *this );
    changeEvent  .emit (                       *this );
  }
}


JIVE_END_PACKAGE( util )
