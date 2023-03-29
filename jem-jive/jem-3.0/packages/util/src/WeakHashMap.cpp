
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */

#include <utility>
#include <jem/base/assert.h>
#include <jem/base/memory.h>
#include <jem/base/Error.h>
#include <jem/base/MemCache.h>
#include <jem/base/WeakRef.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/hash.h>
#include <jem/util/error.h>
#include <jem/util/WeakHashMap.h>


JEM_DEFINE_SERIABLE_CLASS ( jem::util::WeakHashMap );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class WeakHashMap::Node_
//=======================================================================

/*
  Implementation note: a node with a zero "addr" member has not been
  used yet, or has been erased. The member "itemCount_" of the
  WeakHashMap class equals the number of nodes with a non-zero "addr".
*/

class WeakHashMap::Node_
{
 public:

  inline                  Node_   ();


 public:

  idx_t                   next;
  void*                   addr;
  WeakRef<Object>         owner;
  Ref<Object>             value;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline WeakHashMap::Node_::Node_ ()
{
  next = -1_idx;
  addr = nullptr;
}


//=======================================================================
//   class WeakHashMap::Enumerator_
//=======================================================================


class WeakHashMap::Enumerator_ : public MapEnumerator
{
 public:

  typedef WeakHashMap     Map;


  explicit inline         Enumerator_

    ( const Ref<Map>&       map );

  virtual bool            atEnd         () const override;
  virtual void            toNext        ()       override;
  virtual Ref<Object>     getKey        () const override;
  virtual Ref<Object>     getValue      () const override;
  inline void             checkOnMap    () const;


 protected:

  virtual                ~Enumerator_   ();



 private:

  Ref<Map>                map_;
  idx_t                   pos_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline WeakHashMap::Enumerator_::Enumerator_

  ( const Ref<Map>& map ) :

    map_ ( map )

{
  pos_ = 0;

  map_->enumCount_.fetch_add ( 1, std::memory_order_relaxed );
}


WeakHashMap::Enumerator_::~Enumerator_ ()
{
  map_->enumCount_.fetch_sub ( 1, std::memory_order_relaxed );
}


//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


bool WeakHashMap::Enumerator_::atEnd () const
{
  return (pos_ >= map_->lastNode_);
}


//-----------------------------------------------------------------------
//   toNext
//-----------------------------------------------------------------------


void WeakHashMap::Enumerator_::toNext ()
{
  checkOnMap ();

  do
  {
    pos_++;
  }
  while ( pos_ < map_->lastNode_ && ! map_->nodes_[pos_].addr );
}


//-----------------------------------------------------------------------
//   getKey
//-----------------------------------------------------------------------


Ref<Object> WeakHashMap::Enumerator_::getKey () const
{
  checkOnMap ();

  return map_->nodes_[pos_].owner.get ();
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


Ref<Object> WeakHashMap::Enumerator_::getValue () const
{
  checkOnMap ();

  return map_->nodes_[pos_].value;
}


//-----------------------------------------------------------------------
//   checkOnMap
//-----------------------------------------------------------------------


inline void WeakHashMap::Enumerator_::checkOnMap () const
{
  if ( pos_ >= map_->lastNode_ )
  {
    atEndError_ ();
  }
}


//=======================================================================
//   class WeakHashMap
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


WeakHashMap::WeakHashMap () :

  enumCount_ ( 0 )

{
  table_      =  nullptr;
  nodes_      =  nullptr;
  capacity_   =  0;
  lastNode_   =  0;
  itemCount_  =  0;
  tableBits_  = -1;
  writeCount_ =  0;
}


WeakHashMap::~WeakHashMap ()
{
  if ( table_ )
  {
    MemCache::dealloc ( table_, (size_t) hashSize(tableBits_) *
                                sizeof(*table_) );
  }

  if ( nodes_ )
  {
    destroy           ( nodes_, capacity_ );
    MemCache::dealloc ( nodes_, (size_t) capacity_ *
                                sizeof(*nodes_) );
  }

  table_ = nullptr;
  nodes_ = nullptr;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void WeakHashMap::readFrom ( io::ObjectInput& in )
{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  clear   ();
  reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    Ref<Object>  key = in.decodeObject ();

    if ( key )
    {
      Ref<Object>  value = in.decodeObject ();

      insert ( key, value );
    }
  }
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void WeakHashMap::writeTo ( io::ObjectOutput& out ) const
{
  Node_*  nodes = nodes_;

  encode ( out, itemCount_ );

  for ( idx_t i = 0; i < lastNode_; i++ )
  {
    if ( nodes[i].addr )
    {
      Ref<Object>  key = nodes[i].owner.get ();

      encode ( out, key );

      if ( key )
      {
        encode ( out, nodes[i].value );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> WeakHashMap::clone () const
{
  Ref<Self>  map   = newInstance<Self> ();
  Node_*     nodes = nodes_;

  map->reserve ( itemCount_ );

  for ( idx_t i = 0; i < lastNode_; i++ )
  {
    if ( nodes[i].addr )
    {
      Ref<Object>  key = nodes[i].owner.get ();

      if ( key )
      {
        map->insert ( key.clone(), nodes[i].value.clone() );
      }
    }
  }

  return map;
}


//-----------------------------------------------------------------------
//   getMapEnum
//-----------------------------------------------------------------------


Ref<MapEnumerator> WeakHashMap::getMapEnum () const
{
  Ref<Self>  self = const_cast<Self*> ( this );

  return newInstance<Enumerator_> ( self );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Object> WeakHashMap::get ( const Ref<Object>& key ) const
{
  if ( key )
  {
    Node_*  node = find_ ( key );

    if ( node )
    {
      return node->value;
    }
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


bool WeakHashMap::find

  ( Ref<Object>&        value,
    const Ref<Object>&  key ) const

{
  if ( key )
  {
    Node_*  node = find_ ( key );

    if ( node )
    {
      value = node->value;

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void WeakHashMap::insert

  ( const Ref<Object>&  key,
    const Ref<Object>&  value )
{
  checkWritable_ ();

  if ( ! key )
  {
    return;
  }

  Node_*  node = find_ ( key );

  if ( node )
  {
    node->value = value;

    return;
  }

  // Insert a new key-value pair.

  if ( lastNode_ >= capacity_ )
  {
    expand_ ();
  }

  Node_*  nodes = nodes_;
  void*   addr  = key.get ();
  idx_t   tix   = getTableIndex_ ( addr );
  idx_t   i     = table_[tix];

  // Try to re-use a free node.

  while ( i >= 0 )
  {
    node = nodes + i;

    if ( ! node->addr )
    {
      node->addr  = addr;
      node->owner = key;
      node->value = value;

      itemCount_++;

      return;
    }

    i = node->next;
  }

  // Append a new node.

  node = nodes + lastNode_;

  node->addr  = addr;
  node->owner = key;
  node->value = value;
  node->next  = table_[tix];
  table_[tix] = lastNode_;

  lastNode_ ++;
  itemCount_++;
}


void WeakHashMap::insert

  ( Ref<Object>&&  key,
    Ref<Object>&&  value )
{
  insert ( key, value );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void WeakHashMap::erase ( const Ref<Object>& key )
{
  checkWritable_ ();

  if ( key )
  {
    Node_*  node = find_ ( key );

    if ( node )
    {
      node->addr  = nullptr;
      node->owner = nullptr;
      node->value = nullptr;

      itemCount_--;
    }
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void WeakHashMap::clear ()
{
  checkWritable_ ();

  Node_*  nodes = nodes_;
  idx_t   n     = lastNode_;

  for ( idx_t i = 0; i < n; i++ )
  {
    nodes[i].next  = -1;
    nodes[i].addr  = nullptr;
    nodes[i].owner = nullptr;
    nodes[i].value = nullptr;
  }

  if ( table_ )
  {
    n = hashSize ( tableBits_ );

    for ( idx_t i = 0; i < n; i++ )
    {
      table_[i] = -1;
    }
  }

  lastNode_ = itemCount_ = 0;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void WeakHashMap::reserve ( idx_t count )
{
  JEM_PRECHECK2 ( count >= 0, "invalid reserve count" );

  checkWritable_ ();

  if ( count < itemCount_ )
  {
    count = itemCount_;
  }

  if ( count != capacity_ )
  {
    realloc_ ( count );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void WeakHashMap::trimToSize ()
{
  checkWritable_ ();

  if ( writeCount_ > 0 )
  {
    cleanup_ ();
  }

  if ( capacity_ > itemCount_ )
  {
    realloc_ ( itemCount_ );
  }
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


idx_t WeakHashMap::capacity () const
{
  return capacity_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t WeakHashMap::size () const
{
  return itemCount_;
}


//-----------------------------------------------------------------------
//   find_
//-----------------------------------------------------------------------


WeakHashMap::Node_*

  WeakHashMap::find_ ( const Ref<Object>& key ) const

{
  JEM_ASSERT2 ( key, "NULL key" );

  // Note: nodes without an owner can not be garbage collected here
  // because that would not be thread safe.

  Node_*  nodes = nodes_;
  void*   addr  = key.get ();

  if ( lastNode_ < TINY_SIZE_ )
  {
    const int  n = (int) lastNode_;

    for ( int i = 0; i < n; i++ )
    {
      Node_*  node = nodes + i;

      if ( node->addr == addr && node->owner.get() == key )
      {
        return node;
      }
    }
  }
  else
  {
    idx_t  tix = getTableIndex_ ( addr );
    idx_t  i   = table_[tix];

    while ( i >= 0 )
    {
      Node_*  node = nodes + i;

      if ( node->addr == addr && node->owner.get() == key )
      {
        return node;
      }

      i = node->next;
    }
  }

  return (Node_*) 0;
}


//-----------------------------------------------------------------------
//   expand_
//-----------------------------------------------------------------------


void WeakHashMap::expand_ ()
{
  if ( writeCount_ > 0 )
  {
    cleanup_ ();

    if ( lastNode_ < capacity_ )
    {
      return;
    }
  }

  realloc_ ( Super::expand_( capacity_ ) );
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


void WeakHashMap::realloc_ ( idx_t newCap )
{
  if ( newCap < 0 )
  {
    overflowError ( JEM_FUNC, newCap );
  }

  if ( newCap < itemCount_ )
  {
    newCap = itemCount_;
  }

  // Reallocate the hash table if necessary.

  if ( newCap != capacity_ )
  {
    int    newBits = hashBits ( newCap );
    idx_t  newSize = hashSize ( newBits );

    if ( table_ )
    {
      idx_t  oldSize = hashSize ( tableBits_ );

      table_ = (idx_t*) MemCache::realloc

        ( table_, (size_t) oldSize * sizeof(*table_),
                  (size_t) newSize * sizeof(*table_) );
    }
    else
    {
      table_ = (idx_t*)

        MemCache::alloc ( (size_t) newSize * sizeof(*table_) );
    }

    tableBits_ = newBits;
  }

  // Invalidate the hash table.

  if ( table_ )
  {
    idx_t  n  = hashSize ( tableBits_ );

    for ( idx_t i = 0; i < n; i++ )
    {
      table_[i] = -1;
    }
  }

  // Rehash before allocating a new node array so that the hash table
  // is in a consistent state when an exception is thrown. Nodes
  // without an owner are removed.

  Node_*  oldNodes = nodes_;
  idx_t   count    = 0;

  for ( idx_t i = 0; i < lastNode_; i++ )
  {
    Node_*  node = oldNodes + i;

    if ( ! node->addr )
    {
      continue;
    }

    if ( ! node->owner.get() )
    {
      node->addr  = nullptr;
      node->value = nullptr;

      continue;
    }

    Node_*  tail = oldNodes + count;
    idx_t   tix  = getTableIndex_ ( node->addr );

    tail->next   = table_[tix];
    table_[tix]  = count;

    if ( tail != node )
    {
      tail->addr = node->addr;

      tail->owner.swap ( node->owner );
      tail->value.swap ( node->value );
    }

    count++;
  }

  itemCount_ = lastNode_ = count;

  if ( newCap == capacity_ )
  {
    return;
  }

  // Allocate a new node array.

  Node_*  newNodes = (Node_*)

     MemCache::alloc ( (size_t) newCap * sizeof(*newNodes) );

  try
  {
    construct ( newNodes, newCap );
  }
  catch ( ... )
  {
    MemCache::dealloc ( newNodes, (size_t) newCap * sizeof(*newNodes) );
    throw;
  }

  // Copy the nodes to the new array.

  for ( idx_t i = 0; i < lastNode_; i++ )
  {
    Node_*  oldNode = oldNodes + i;
    Node_*  newNode = newNodes + i;

    newNode->next   = oldNode->next;
    newNode->addr   = oldNode->addr;

    newNode->owner.swap ( oldNode->owner );
    newNode->value.swap ( oldNode->value );
  }

  if ( oldNodes )
  {
    destroy           ( oldNodes, capacity_ );
    MemCache::dealloc ( oldNodes, (size_t) capacity_ *
                                  sizeof(*oldNodes) );
  }

  nodes_    = newNodes;
  capacity_ = newCap;
}


//-----------------------------------------------------------------------
//   cleanup_
//-----------------------------------------------------------------------


void WeakHashMap::cleanup_ ()
{
  Node_*  nodes = nodes_;
  idx_t   count = 0;

  for ( idx_t i = 0; i < lastNode_; i++ )
  {
    Node_*  node = nodes + i;

    if ( ! node->addr )
    {
      continue;
    }

    if ( ! node->owner.get() )
    {
      node->addr  = nullptr;
      node->value = nullptr;

      itemCount_--;
    }
    else
    {
      count++;
    }
  }

  if ( count != itemCount_ )
  {
    throw Error ( JEM_FUNC, "inconsistent number of items" );
  }

  if ( lastNode_ > TINY_SIZE_ &&
       lastNode_ > (count + count / 4) )
  {
    realloc_ ( capacity_ );
  }

  writeCount_ = 0;
}



//-----------------------------------------------------------------------
//   checkWritable_
//-----------------------------------------------------------------------


inline void WeakHashMap::checkWritable_ ()
{
  if ( enumCount_.load( std::memory_order_relaxed ) > 0 )
  {
    writeError_ ();
  }

  writeCount_++;

  if ( writeCount_ > (TINY_SIZE_ + itemCount_ / 2) )
  {
    cleanup_ ();
  }
}


//-----------------------------------------------------------------------
//   getTableIndex_
//-----------------------------------------------------------------------


inline idx_t WeakHashMap::getTableIndex_ ( void* addr ) const
{
  return (idx_t) ( hashMask( tableBits_ ) &
                   TypeTraits<void*>::hashValue( addr ) );
}


JEM_END_PACKAGE( util )
