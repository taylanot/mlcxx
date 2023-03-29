
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
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/util/Event.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/error.h>
#include <jive/util/EventMask.h>
#include <jive/util/Reordering.h>
#include <jive/util/HashItemMap.h>
#include <jive/util/StdGroupSet.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::StdGroupSet );


JIVE_BEGIN_PACKAGE( util )


using jem::checkedCast;
using jem::newInstance;
using jem::numeric::SparseUtils;
using jem::util::ArrayBuffer;


//=======================================================================
//   class StdGroupSet::Data_
//=======================================================================


class StdGroupSet::Data_ : public jem::Collectable
{
 public:

  typedef Data_           Self;

  static const int        NEW_SIZE = 1 << 0;
  static const int        NEW_TOPO = 1 << 1;


  inline                  Data_             ();

                          Data_

    ( const String&         groupName,
      const Ref<ItemSet>&   items );

  void                    readFrom

    ( ObjectInput&          in );

  void                    writeTo

    ( ObjectOutput&         out )              const;

  void                    compress          ();

  void                    compressTo

    ( ArrayBuffer<idx_t>&   offsets,
      ArrayBuffer<idx_t>&   indices )          const;

  inline bool             isCompressed      () const;
  inline idx_t            groupCount        () const;

  inline void             checkGroupIndex

    ( idx_t                 igroup )           const;

  inline void             checkItemIndex

    ( idx_t                 iitem );

  inline void             setItemCount

    ( idx_t                 itemCount );

  inline void             addGroup

    ( const IdxVector&      iitems );

  void                    calcMaxGroupSize  ();


 public:

  Ref<ItemSet>            items;
  String                  groupName;

  // Note: the last entry in groupOffsets is always equal
  // to itemIndices.size().

  ArrayBuffer<idx_t>      groupSizes;
  ArrayBuffer<idx_t>      groupOffsets;
  ArrayBuffer<idx_t>      itemIndices;

  EventMask               events;

  idx_t                   totalSize;
  idx_t                   maxGroupSize;


 private:

  void                    checkItemIndex_

    ( idx_t                 iitem );


 private:

  // This is an estimate only.

  idx_t                   itemCount_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline StdGroupSet::Data_::Data_ ()
{
  totalSize    = -1;
  maxGroupSize = -1;
  itemCount_   = -1;
}


StdGroupSet::Data_::Data_

  ( const String&        gname,
    const Ref<ItemSet>&  iset ) :

    items     (  iset ),
    groupName ( gname )

{
  totalSize    = 0;
  maxGroupSize = 0;

  groupOffsets.pushBack ( 0 );

  if ( groupName.size() == 0 )
  {
    groupName = items->getItemName() + " group";
  }

  itemCount_ = items->size ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdGroupSet::Data_::readFrom ( ObjectInput& in )
{
  decode ( in, items, groupName, groupOffsets, itemIndices );

  idx_t  groupCount = groupOffsets.size() - 1;

  totalSize    = itemIndices .size();
  maxGroupSize = 0;

  groupSizes.resize ( groupCount );

  for ( idx_t i = 0; i < groupCount; i++ )
  {
    idx_t  groupSize = groupOffsets[i + 1] - groupOffsets[i];

    groupSizes[i]  = groupSize;

    if ( groupSize > maxGroupSize )
    {
      maxGroupSize = groupSize;
    }
  }

  itemCount_ = items->size ();

  events.reset ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdGroupSet::Data_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items, groupName );

  if ( isCompressed() )
  {
    encode ( out, groupOffsets, itemIndices );
  }
  else
  {
    ArrayBuffer<idx_t>  offsets, indices;

    compressTo  ( offsets, indices );
    encode ( out, offsets, indices );
  }
}


//-----------------------------------------------------------------------
//   compress
//-----------------------------------------------------------------------


void StdGroupSet::Data_::compress ()
{
  ArrayBuffer<idx_t>  offsets, indices;

  groupSizes.trimToSize ();

  compressTo ( offsets, indices );

  groupOffsets.swap ( offsets );
  itemIndices .swap ( indices );
}


//-----------------------------------------------------------------------
//   compressTo
//-----------------------------------------------------------------------


void StdGroupSet::Data_::compressTo

  ( ArrayBuffer<idx_t>&  offsets,
    ArrayBuffer<idx_t>&  indices ) const

{
  const idx_t* JEM_RESTRICT  mySizes   = groupSizes  .addr ();
  const idx_t* JEM_RESTRICT  myOffsets = groupOffsets.addr ();
  const idx_t* JEM_RESTRICT  myIndices = itemIndices .addr ();

  idx_t*       JEM_RESTRICT  newOffsets;
  idx_t*       JEM_RESTRICT  newIndices;

  const idx_t  groupCount = groupSizes.size ();


  offsets.resize ( groupCount + 1 );
  indices.resize ( totalSize );

  newOffsets = offsets.addr ();
  newIndices = indices.addr ();

  idx_t  k = 0;

  for ( idx_t i = 0; i < groupCount; i++ )
  {
    newOffsets[i] = k;

    idx_t  n = mySizes  [i];
    idx_t  j = myOffsets[i];

    if      ( n == 1_idx )
    {
      newIndices[k] = myIndices[j];
    }
    else if ( n == 2_idx )
    {
      newIndices[k + 0] = myIndices[j + 0];
      newIndices[k + 1] = myIndices[j + 1];
    }
    else if ( n == 3_idx )
    {
      newIndices[k + 0] = myIndices[j + 0];
      newIndices[k + 1] = myIndices[j + 1];
      newIndices[k + 2] = myIndices[j + 2];
    }
    else if ( n >= 4_idx )
    {
      std::memcpy ( &newIndices[k], &myIndices[j],
                    (size_t) n * sizeof(idx_t) );
    }

    k += n;
  }

  newOffsets[groupCount] = k;

  JEM_ASSERT ( k == totalSize );
}


//-----------------------------------------------------------------------
//   isCompressed
//-----------------------------------------------------------------------


inline bool StdGroupSet::Data_::isCompressed () const
{
  return (totalSize == itemIndices.size());
}


//-----------------------------------------------------------------------
//   groupCount
//-----------------------------------------------------------------------


inline idx_t StdGroupSet::Data_::groupCount () const
{
  return groupSizes.size ();
}


//-----------------------------------------------------------------------
//   checkGroupIndex
//-----------------------------------------------------------------------


inline void StdGroupSet::Data_::checkGroupIndex ( idx_t igroup ) const
{
  if ( igroup < 0 || igroup >= groupCount() )
  {
    indexError ( CLASS_NAME, groupName, igroup, groupCount() );
  }
}


//-----------------------------------------------------------------------
//   checkItemIndex
//-----------------------------------------------------------------------


inline void StdGroupSet::Data_::checkItemIndex ( idx_t iitem )
{
  if ( iitem < 0 || iitem >= itemCount_ )
  {
    checkItemIndex_ ( iitem );
  }
}


//-----------------------------------------------------------------------
//   setItemCount
//-----------------------------------------------------------------------


inline void StdGroupSet::Data_::setItemCount ( idx_t n )
{
  itemCount_ = n;
}


//-----------------------------------------------------------------------
//   addGroup
//-----------------------------------------------------------------------


inline void StdGroupSet::Data_::addGroup ( const IdxVector& iitems )
{
  JEM_ASSERT2 ( groupOffsets.back() == itemIndices.size(),
                "invalid group offset array" );

  const idx_t  n = iitems.size ();

  if ( n > 10 )
  {
    idx_t  k = itemIndices.size ();

    itemIndices.pushBack ( -1, n );

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  iitem = iitems[i];

      checkItemIndex ( iitem );

      itemIndices[k + i] = iitem;
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  iitem = iitems[i];

      checkItemIndex ( iitem );

      itemIndices.pushBack ( iitem );
    }
  }

  groupSizes  .pushBack ( n );
  groupOffsets.pushBack ( itemIndices.size() );

  totalSize += n;

  if ( n > maxGroupSize )
  {
    maxGroupSize = n;
  }
}


//-----------------------------------------------------------------------
//   calcMaxGroupSize
//-----------------------------------------------------------------------


void StdGroupSet::Data_::calcMaxGroupSize ()
{
  const idx_t*  JEM_RESTRICT  offsets = groupOffsets.addr ();

  const idx_t   n = groupCount ();

  idx_t         m = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  k = offsets[i + 1] - offsets[i];

    if ( k > m )
    {
      m = k;
    }
  }

  maxGroupSize = m;
}


//-----------------------------------------------------------------------
//   checkItemIndex_
//-----------------------------------------------------------------------


void StdGroupSet::Data_::checkItemIndex_ ( idx_t iitem )
{
  itemCount_ = items->size ();

  if ( iitem < 0 || iitem >= itemCount_ )
  {
    indexError ( CLASS_NAME,
                 items->getItemName(), iitem, itemCount_ );
  }
}


//=======================================================================
//   class StdGroupSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdGroupSet::StdGroupSet ()
{}


StdGroupSet::StdGroupSet

  ( const String&         name,
    const String&         groupName,
    const Ref<ItemSet>&   items,
    const Ref<XItemMap>&  map ) :

    Super ( name ),
    map_  (  map )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  data_ = newInstance<Data_> ( groupName, items );

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


StdGroupSet::StdGroupSet ( const Self& rhs ) :

  Super ( rhs.myName_ )

{
  data_ = newInstance<Data_>    ( * rhs.data_ );
  map_  = checkedCast<XItemMap> (   rhs.map_->clone() );

  connect_ ();
}


StdGroupSet::~StdGroupSet ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void StdGroupSet::resetEvents ()
{
  data_->events.reset ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdGroupSet::readFrom ( ObjectInput& in )
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


void StdGroupSet::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_ );

  data_->writeTo ( out );

  encode ( out, map_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> StdGroupSet::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t StdGroupSet::size () const
{
  return data_->groupCount ();
}


//-----------------------------------------------------------------------
//   getItemName
//-----------------------------------------------------------------------


String StdGroupSet::getItemName () const
{
  return data_->groupName;
}


//-----------------------------------------------------------------------
//   getItemMap
//-----------------------------------------------------------------------


ItemMap* StdGroupSet::getItemMap () const
{
  return map_.get ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void StdGroupSet::clear ()
{
  Data_&  d = * data_;

  if ( d.groupCount() > 0 )
  {
    map_         ->clear    ();
    d.groupSizes  .clear    ();
    d.groupOffsets.clear    ();
    d.groupOffsets.pushBack ( 0 );
    d.itemIndices .clear    ();
    d.events      .reset    ();

    d.totalSize    = 0;
    d.maxGroupSize = 0;

    newSizeEvent.emit ( 0, *this );
    changeEvent .emit (    *this );
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void StdGroupSet::reserve ( idx_t n )
{
  Data_&  d = * data_;

  d.groupSizes  .reserve ( n );
  d.groupOffsets.reserve ( n + 1 );
  d.itemIndices. reserve ( n     );
  map_         ->reserve ( n     );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void StdGroupSet::trimToSize ()
{
  Data_&  d = * data_;

  if ( d.isCompressed() )
  {
    d.groupSizes  .trimToSize ();
    d.groupOffsets.trimToSize ();
    d.itemIndices. trimToSize ();
  }
  else
  {
    d.compress ();
  }

  map_->trimToSize ();
}


//-----------------------------------------------------------------------
//   addGroup
//-----------------------------------------------------------------------


idx_t StdGroupSet::addGroup ( const IdxVector& iitems )
{
  Data_&       d      = * data_;

  const idx_t  igroup = map_->addItem ();

  d.addGroup    ( iitems );
  d.events.emit ( Data_::NEW_SIZE, this );

  return igroup;
}


idx_t StdGroupSet::addGroup

  ( idx_t               id,
    const IdxVector&  iitems )

{
  Data_&       d      = * data_;

  const idx_t  igroup = map_->addItem ( id );

  d.addGroup    ( iitems );
  d.events.emit ( Data_::NEW_SIZE, this );

  return igroup;
}


//-----------------------------------------------------------------------
//   addGroups
//-----------------------------------------------------------------------


idx_t StdGroupSet::addGroups ( const IdxMatrix& groups )
{
  Data_&       d = * data_;

  const idx_t  m = groups.size (0);
  const idx_t  n = groups.size (1);


  if ( m * n == 0 )
  {
    return d.groupCount ();
  }

  if ( m > d.maxGroupSize )
  {
    d.maxGroupSize = m;
  }

  idx_t  k = d.itemIndices.size ();

  d.groupSizes .pushBack (  m, n );
  d.itemIndices.pushBack ( -1, m * n );

  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++, k++ )
    {
      idx_t  iitem = groups(i,j);

      d.checkItemIndex ( iitem );

      d.itemIndices[k] = iitem;
    }

    d.groupOffsets.pushBack ( k );
  }

  k = map_->addItems ( n );

  d.events.emit ( Data_::NEW_SIZE, this );

  return k;
}


//-----------------------------------------------------------------------
//   reorderGroups
//-----------------------------------------------------------------------


void StdGroupSet::reorderGroups ( const Reordering& reord )
{
  JEM_PRECHECK2 ( reord.oldSize() == size(),
                  "invalid Reordering" );

  Data_&        d = * data_;

  const idx_t*  JEM_RESTRICT  rmap = reord.rmapPtr ();

  const idx_t   start   = reord.start   ();
  const idx_t   newSize = reord.newSize ();


  map_->reorderItems ( reord );

  if ( reord.isSimple() && d.isCompressed() )
  {
    idx_t*  JEM_RESTRICT  grsizes = d.groupSizes  .addr ();
    idx_t*  JEM_RESTRICT  offsets = d.groupOffsets.addr ();
    idx_t*  JEM_RESTRICT  indices = d.itemIndices .addr ();

    idx_t  j = offsets[start];

    for ( idx_t igroup = start; igroup < newSize; igroup++ )
    {
      idx_t  jgroup = rmap   [igroup];
      idx_t  i      = offsets[jgroup];
      idx_t  n      = offsets[jgroup + 1];

      grsizes[igroup] = n - i;
      offsets[igroup] = j;

      for ( ; i < n; i++, j++ )
      {
        indices[j] = indices[i];
      }
    }

    offsets[newSize] = j;

    d.groupSizes  .resize ( newSize );
    d.groupOffsets.resize ( newSize + 1 );
    d.itemIndices .resize ( j );
  }
  else
  {
    ArrayBuffer<idx_t>  tmpGrSizes ( newSize );
    ArrayBuffer<idx_t>  tmpOffsets ( newSize + 1 );
    ArrayBuffer<idx_t>  tmpIndices;

    const idx_t*  JEM_RESTRICT  oldGrSizes = d.groupSizes  .addr ();
    const idx_t*  JEM_RESTRICT  oldOffsets = d.groupOffsets.addr ();
    const idx_t*  JEM_RESTRICT  oldIndices = d.itemIndices .addr ();

    idx_t*        JEM_RESTRICT  newGrSizes = tmpGrSizes    .addr ();
    idx_t*        JEM_RESTRICT  newOffsets = tmpOffsets    .addr ();
    idx_t*        JEM_RESTRICT  newIndices;

    idx_t         i, j, n;

    j = 0;

    for ( idx_t igroup = 0; igroup < newSize; igroup++ )
    {
      idx_t  jgroup = rmap[igroup];

      n = oldGrSizes[jgroup];

      newGrSizes[igroup] = n;
      newOffsets[igroup] = j;
      j                 += n;
    }

    newOffsets[newSize] = j;

    tmpIndices.resize ( j );

    newIndices = tmpIndices.addr ();

    for ( idx_t igroup = 0; igroup < newSize; igroup++ )
    {
      idx_t  jgroup = rmap[igroup];

      i = newOffsets[igroup];
      j = oldOffsets[jgroup];
      n = oldGrSizes[jgroup];

      std::memcpy ( newIndices + i, oldIndices + j,
                    (size_t) n * sizeof(idx_t) );
    }

    d.groupSizes  .swap ( tmpGrSizes );
    d.groupOffsets.swap ( tmpOffsets );
    d.itemIndices .swap ( tmpIndices );
  }

  d.totalSize = d.itemIndices.size ();

  d.events.reset     ();
  d.calcMaxGroupSize ();

  newOrderEvent.emit ( reord, *this );

  if ( newSize != reord.oldSize() )
  {
    newSizeEvent.emit ( newSize, *this );
  }

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   setGroupMembers
//-----------------------------------------------------------------------


void StdGroupSet::setGroupMembers

  ( idx_t             igroup,
    const IdxVector&  iitems )

{
  Data_&  d = * data_;

  d.checkGroupIndex ( igroup );

  const idx_t  oldSize = d.groupSizes[igroup];
  const idx_t  newSize = iitems.size ();

  idx_t*  JEM_RESTRICT  idx;


  for ( idx_t i = 0; i < newSize; i++ )
  {
    d.checkItemIndex ( iitems[i] );
  }

  if ( newSize <= oldSize )
  {
    idx_t  i = d.groupOffsets[igroup];

    idx = d.itemIndices.addr ( i );

    if ( newSize < oldSize )
    {
      if ( i + oldSize == d.groupOffsets.back() )
      {
        d.groupOffsets.back() = i + newSize;

        d.itemIndices.popBack( oldSize - newSize );
      }
    }
  }
  else
  {
    idx_t  i = d.groupOffsets[igroup] = d.itemIndices.size ();

    d.itemIndices.pushBack ( -1, newSize );

    d.groupOffsets.back() = i + newSize;

    idx = d.itemIndices.addr ( i );

    if ( newSize > d.maxGroupSize )
    {
      d.maxGroupSize = newSize;
    }
  }

  if ( iitems.isContiguous() )
  {
    std::memcpy ( idx, iitems.addr(),
                  (size_t) newSize * sizeof(idx_t) );
  }
  else
  {
    for ( idx_t i = 0; i < newSize; i++ )
    {
      idx[i] = iitems[i];
    }
  }

  d.groupSizes[igroup] =  newSize;
  d.totalSize         += (newSize - oldSize);

  if ( d.itemIndices.size() > d.totalSize )
  {
    if ( (d.itemIndices.size() - d.totalSize / 2) > d.totalSize )
    {
      d.compress ();
    }
  }

  d.events.emit ( Data_::NEW_TOPO, this );
}


//-----------------------------------------------------------------------
//   getGroupSize
//-----------------------------------------------------------------------


idx_t StdGroupSet::getGroupSize ( idx_t igroup ) const
{
  const Data_&  d = * data_;

  d.checkGroupIndex ( igroup );

  return d.groupSizes[igroup];
}


//-----------------------------------------------------------------------
//   getGroupMembers
//-----------------------------------------------------------------------


idx_t StdGroupSet::getGroupMembers

  ( const IdxVector&  iitems,
    idx_t             igroup ) const

{
  const Data_&  d   = * data_;

  d.checkGroupIndex ( igroup );

  const idx_t   size  = d.groupSizes  [igroup];
  const idx_t   first = d.groupOffsets[igroup];

  const idx_t*  idx   = d.itemIndices.addr ( first );


  if ( iitems.size() < size )
  {
    sizeError ( getContext(), "index vector", iitems.size(), size );
  }

  if ( iitems.isContiguous() )
  {
    idx_t* JEM_RESTRICT  ip = iitems.addr ();

    if      ( size == 1_idx )
    {
      ip[0] = idx[0];
    }
    else if ( size == 2_idx )
    {
      ip[0] = idx[0];
      ip[1] = idx[1];
    }
    else if ( size == 3_idx )
    {
      ip[0] = idx[0];
      ip[1] = idx[1];
      ip[2] = idx[2];
    }
    else if ( size == 4_idx )
    {
      ip[0] = idx[0];
      ip[1] = idx[1];
      ip[2] = idx[2];
      ip[3] = idx[3];
    }
    else if ( size >= 5_idx )
    {
      std::memcpy ( ip, idx, (size_t) size * sizeof(idx_t) );
    }
  }
  else
  {
    if      ( size == 1_idx )
    {
      iitems[0] = idx[0];
    }
    else if ( size == 2_idx )
    {
      iitems[0] = idx[0];
      iitems[1] = idx[1];
    }
    else if ( size == 3_idx )
    {
      iitems[0] = idx[0];
      iitems[1] = idx[1];
      iitems[2] = idx[2];
    }
    else if ( size == 4_idx )
    {
      iitems[0] = idx[0];
      iitems[1] = idx[1];
      iitems[2] = idx[2];
      iitems[3] = idx[3];
    }
    else if ( size >= 5_idx )
    {
      for ( idx_t i = 0; i < size; i++ )
      {
        iitems[i] = idx[i];
      }
    }
  }

  return size;
}


//-----------------------------------------------------------------------
//   getSomeGroupMembers
//-----------------------------------------------------------------------


void StdGroupSet::getSomeGroupMembers

  ( const IdxVector&  iitems,
    const IdxVector&  index,
    idx_t             igroup ) const

{
  JEM_PRECHECK2 ( iitems.size() >= index.size(),
                  "output array is too small" );

  const Data_&  d  = * data_;

  d.checkGroupIndex ( igroup );

  const idx_t   first = d.groupOffsets[igroup];
  const idx_t   n     = index.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    iitems[i] = d.itemIndices[first + index[i]];
  }
}


//-----------------------------------------------------------------------
//   getOneGroupMember
//-----------------------------------------------------------------------


idx_t StdGroupSet::getOneGroupMember

  ( idx_t  index,
    idx_t  igroup ) const

{
  const Data_&  d = * data_;

  d.checkGroupIndex ( igroup );

  return d.itemIndices[d.groupOffsets[igroup] + index];
}


//-----------------------------------------------------------------------
//   getMembersOf
//-----------------------------------------------------------------------


IdxVector StdGroupSet::getMembersOf

  ( const IdxVector&  igroups ) const

{
  const Data_&  d = * data_;
  const idx_t   n = igroups.size ();

  const idx_t*  JEM_RESTRICT  idx = d.itemIndices.addr ();

  IdxVector     iitems;

  idx_t         j, k, m;


  k = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  igroup = igroups[i];

    d.checkGroupIndex ( igroup );

    k += d.groupSizes[igroup];
  }

  iitems.resize ( k );

  j = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  igroup = igroups[i];

    k = d.groupOffsets[igroup];
    m = k + d.groupSizes[igroup];

    for ( ; k < m; k++, j++ )
    {
      iitems[j] = idx[k];
    }
  }

  return iitems;
}


//-----------------------------------------------------------------------
//   maxGroupSize
//-----------------------------------------------------------------------


idx_t StdGroupSet::maxGroupSize () const
{
  return data_->maxGroupSize;
}


//-----------------------------------------------------------------------
//   maxGroupSizeOf
//-----------------------------------------------------------------------


idx_t StdGroupSet::maxGroupSizeOf ( const IdxVector& igroups ) const
{
  const Data_&  d = *data_;
  const idx_t   n = igroups.size ();

  const idx_t*  JEM_RESTRICT  offsets = d.groupOffsets.addr ();

  idx_t         m = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  igroup = igroups[i];

    d.checkGroupIndex ( igroup );

    idx_t  k = offsets[igroup + 1] - offsets[igroup];

    if ( k > m )
    {
      m = k;
    }
  }

  return m;
}


//-----------------------------------------------------------------------
//   toVector
//-----------------------------------------------------------------------


IdxVector StdGroupSet::toVector () const
{
  if ( ! data_->isCompressed() )
  {
    data_->compress ();
  }

  return data_->itemIndices.toArray ();
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


Topology StdGroupSet::toMatrix () const
{
  Data_&  d = * data_;

  if ( ! d.isCompressed() )
  {
    d.compress ();
  }

  return Topology (
    jem::shape ( d.groupCount(), d.items->size() ),
    d.groupOffsets.toArray (),
    d.itemIndices. toArray ()
  );
}


//-----------------------------------------------------------------------
//   getGroupedItems
//-----------------------------------------------------------------------


ItemSet* StdGroupSet::getGroupedItems () const
{
  return data_->items.get ();
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void StdGroupSet::emitEvents_ ( int events )
{
  if ( events & Data_::NEW_SIZE )
  {
    newSizeEvent.emit ( data_->groupCount(), *this );
  }

  if ( events & Data_::NEW_TOPO )
  {
    newTopoEvent.emit ( *this );
  }

  if ( events )
  {
    changeEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void StdGroupSet::connect_ ()
{
  using jem::util::connect;

  Data_&  d = * data_;

  connect ( d.items->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( d.items->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void StdGroupSet::itemsResized_ ( idx_t itemCount )
{
  data_->setItemCount ( itemCount );

  if ( itemCount == 0 )
  {
    clear ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void StdGroupSet::itemsReordered_ ( const Reordering& reord )
{
  Data_&        d = * data_;

  const idx_t*  JEM_RESTRICT  fmap    = reord.fmapPtr       ();
  idx_t*        JEM_RESTRICT  grsizes = d.groupSizes  .addr ();
  idx_t*        JEM_RESTRICT  offsets = d.groupOffsets.addr ();

  const idx_t   groupCount = d.groupCount ();

  int           events     = 0;
  idx_t         j          = 0;


  if ( d.isCompressed() )
  {
    idx_t*  JEM_RESTRICT  indices = d.itemIndices.addr ();

    for ( idx_t igroup = 0; igroup < groupCount; igroup++ )
    {
      idx_t  i = offsets[igroup];
      idx_t  n = offsets[igroup + 1];

      offsets[igroup] = j;

      for ( ; i < n; i++ )
      {
        idx_t  iitem = fmap[indices[i]];

        if ( iitem < 0 )
        {
          events |= Data_::NEW_TOPO;
        }
        else
        {
          indices[j++] = iitem;
        }
      }

      grsizes[igroup] = j - offsets[igroup];
    }
  }
  else
  {
    ArrayBuffer<idx_t>  tmpIndices ( d.totalSize );

    const idx_t*  JEM_RESTRICT  oldIndices = d.itemIndices.addr ();
    idx_t*        JEM_RESTRICT  newIndices = tmpIndices   .addr ();

    for ( idx_t igroup = 0; igroup < groupCount; igroup++ )
    {
      idx_t  i = offsets[igroup];
      idx_t  n = grsizes[igroup] + i;

      offsets[igroup] = j;

      for ( ; i < n; i++ )
      {
        idx_t  iitem = fmap[oldIndices[i]];

        if ( iitem < 0 )
        {
          events |= Data_::NEW_TOPO;
        }
        else
        {
          newIndices[j++] = iitem;
        }
      }

      grsizes[igroup] = j - offsets[igroup];
    }

    d.itemIndices.swap ( tmpIndices );
  }

  offsets[groupCount] = d.totalSize = j;

  d.itemIndices.resize ( d.totalSize );
  d.setItemCount       ( reord.newSize() );

  if ( events )
  {
    d.calcMaxGroupSize ();
    d.events.emit      ( events, this );
  }
}


JIVE_END_PACKAGE( util )
