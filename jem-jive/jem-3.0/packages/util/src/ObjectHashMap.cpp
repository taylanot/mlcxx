
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
#include <jem/base/memory.h>
#include <jem/base/MemCache.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/util/hash.h>
#include <jem/util/error.h>
#include <jem/util/ObjectHashMap.h>


JEM_DEFINE_SERIABLE_CLASS( jem::util::ObjectHashMap );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class ObjectHashMap::Node_
//=======================================================================


class ObjectHashMap::Node_
{
 public:

  inline                  Node_   ();


 public:

  idx_t                   next;
  Ref<Object>             key;
  Ref<Object>             value;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ObjectHashMap::Node_::Node_ ()
{
  next = -1_idx;
}


//=======================================================================
//   class ObjectHashMap::Enumerator_
//=======================================================================


class ObjectHashMap::Enumerator_ : public MapEnumerator
{
 public:

  typedef ObjectHashMap   Map;


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


inline ObjectHashMap::Enumerator_::Enumerator_

  ( const Ref<Map>& map ) :

    map_ ( map )

{
  pos_ = 0_idx;

  map_->enumCount_.fetch_add ( 1, std::memory_order_relaxed );
}


ObjectHashMap::Enumerator_::~Enumerator_ ()
{
  map_->enumCount_.fetch_sub ( 1, std::memory_order_relaxed );
}


//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


bool ObjectHashMap::Enumerator_::atEnd () const
{
  return (pos_ >= map_->itemCount_);
}


//-----------------------------------------------------------------------
//   toNext
//-----------------------------------------------------------------------


void ObjectHashMap::Enumerator_::toNext ()
{
  checkOnMap ();

  pos_++;
}


//-----------------------------------------------------------------------
//   getKey
//-----------------------------------------------------------------------


Ref<Object> ObjectHashMap::Enumerator_::getKey () const
{
  checkOnMap ();

  return map_->nodes_[pos_].key;
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


Ref<Object> ObjectHashMap::Enumerator_::getValue () const
{
  checkOnMap ();

  return  map_->nodes_[pos_].value;
}


//-----------------------------------------------------------------------
//   checkOnMap
//-----------------------------------------------------------------------


inline void ObjectHashMap::Enumerator_::checkOnMap () const
{
  if ( pos_ >= map_->itemCount_ )
  {
    atEndError_ ();
  }
}


//=======================================================================
//   class ObjectHashMap
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ObjectHashMap::ObjectHashMap () :

  enumCount_ ( 0 )

{
  table_     =  nullptr;
  nodes_     =  nullptr;
  capacity_  =  0;
  itemCount_ =  0;
  tableBits_ = -1;
}


ObjectHashMap::~ObjectHashMap ()
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


void ObjectHashMap::readFrom ( io::ObjectInput& in )
{
  readFrom_ ( in );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ObjectHashMap::writeTo ( io::ObjectOutput& out ) const
{
  writeTo_ ( out );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> ObjectHashMap::clone () const
{
  Ref<Self>  map = newInstance<Self> ();

  map->clone_ ( *this );

  return map;
}


//-----------------------------------------------------------------------
//   getMapEnum
//-----------------------------------------------------------------------


Ref<MapEnumerator> ObjectHashMap::getMapEnum () const
{
  Ref<Self>  self = const_cast<Self*> ( this );

  return newInstance<Enumerator_> ( self );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Object> ObjectHashMap::get ( const Ref<Object>& key ) const
{
  const Node_*  node = find_ ( key );

  if ( node )
  {
    return node->value;
  }
  else
  {
    return nullptr;
  }
}

//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


bool ObjectHashMap::find

  ( Ref<Object>&        value,
    const Ref<Object>&  key ) const

{
  const Node_*  node = find_ ( key );

  if ( node )
  {
    value = node->value;

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


void ObjectHashMap::insert

  ( const Ref<Object>&  key,
    const Ref<Object>&  value )

{
  checkWritable_ ();

  Node_*  node = find_ ( key );

  if ( node )
  {
    node->value = value;

    return;
  }

  // Insert a new key-value pair.

  if ( itemCount_ >= capacity_ )
  {
    realloc_ ( expand_( capacity_ ) );
  }

  idx_t  tix = getTableIndex_ ( key );

  node = nodes_ + itemCount_;

  node->key   = key;
  node->value = value;
  node->next  = table_[tix];
  table_[tix] = itemCount_;

  itemCount_++;
}


void ObjectHashMap::insert

  ( Ref<Object>&&  key,
    Ref<Object>&&  value )

{
  checkWritable_ ();

  Node_*  node = find_ ( key );

  if ( node )
  {
    node->value = std::move ( value );

    return;
  }

  // Insert a new key-value pair.

  if ( itemCount_ >= capacity_ )
  {
    realloc_ ( expand_( capacity_ ) );
  }

  idx_t  tix = getTableIndex_ ( key );

  node = nodes_ + itemCount_;

  node->key   = std::move ( key );
  node->value = std::move ( value );
  node->next  = table_[tix];
  table_[tix] = itemCount_;

  itemCount_++;
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void ObjectHashMap::erase ( const Ref<Object>& key )
{
  checkWritable_ ();

  Node_*  node = find_ ( key );

  if ( node )
  {
    erase_ ( node );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ObjectHashMap::clear ()
{
  checkWritable_ ();

  Node_*  nodes = nodes_;
  idx_t   n     = itemCount_;

  for ( idx_t i = 0; i < n; i++ )
  {
    nodes[i].next  = -1;
    nodes[i].key   = nullptr;
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

  itemCount_ = 0;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void ObjectHashMap::reserve ( idx_t count )
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


void ObjectHashMap::trimToSize ()
{
  checkWritable_ ();

  if ( capacity_ > itemCount_ )
  {
    realloc_ ( itemCount_ );
  }
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


idx_t ObjectHashMap::capacity () const
{
  return capacity_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t ObjectHashMap::size () const
{
  return itemCount_;
}


//-----------------------------------------------------------------------
//   find_
//-----------------------------------------------------------------------


ObjectHashMap::Node_*

  ObjectHashMap::find_ ( const Ref<Object>& key ) const

{
  Node_*  nodes = nodes_;

  if ( itemCount_ < TINY_SIZE_ )
  {
    const idx_t  n = itemCount_;

    for ( idx_t i = 0; i < n; i++ )
    {
      if ( nodes[i].key.equals( key ) )
      {
        return (nodes + i);
      }
    }
  }
  else
  {
    idx_t  tix = getTableIndex_ ( key );
    idx_t  i   = table_[tix];

    while ( i >= 0 )
    {
      if ( nodes[i].key.equals( key ) )
      {
        return (nodes + i);
      }

      i = nodes[i].next;
    }
  }

  return (Node_*) 0;
}


//-----------------------------------------------------------------------
//   erase_
//-----------------------------------------------------------------------


void ObjectHashMap::erase_ ( Node_* node )
{
  itemCount_--;

  Node_*  tail = nodes_ + itemCount_;

  unlink_ ( node );

  node->key   = nullptr;
  node->value = nullptr;

  if ( tail != node )
  {
    idx_t  tix = unlink_ ( tail );

    node->key  .swap ( tail->key );
    node->value.swap ( tail->value );

    node->next  = table_[tix];
    table_[tix] = (idx_t) (node - nodes_);
  }
}


//-----------------------------------------------------------------------
//   unlink_
//-----------------------------------------------------------------------


idx_t ObjectHashMap::unlink_ ( Node_* node )
{
  idx_t  tix = getTableIndex_ ( node->key );
  idx_t  i   = (idx_t) (node - nodes_);
  idx_t  j   = table_[tix];

  if ( j == i )
  {
    table_[tix] = node->next;
  }
  else
  {
    Node_*  nodes = nodes_;

    while ( nodes[j].next != i )
    {
      j = nodes[j].next;

      JEM_ASSERT ( j >= 0 );
    }

    nodes[j].next = node->next;
  }

  node->next = -1;

  return tix;
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


void ObjectHashMap::realloc_ ( idx_t newCap )
{
  if ( newCap < 0 )
  {
    overflowError ( JEM_FUNC, newCap );
  }

  if ( newCap < itemCount_ )
  {
    newCap = itemCount_;
  }

  // Reallocate the hash table.

  int    newBits = hashBits ( newCap );
  idx_t  newSize = hashSize ( newBits );

  if ( table_ )
  {
    idx_t  oldSize  = hashSize ( tableBits_ );

    table_ = (idx_t*) MemCache::realloc

      ( table_, (size_t) oldSize * sizeof(*table_),
                (size_t) newSize * sizeof(*table_) );
  }
  else
  {
    table_ = (idx_t*) MemCache::alloc

      ( (size_t) newSize * sizeof(*table_) );
  }

  tableBits_ = newBits;

  // Invalidate the hash table.

  for ( idx_t i = 0; i < newSize; i++ )
  {
    table_[i] = -1;
  }

  // Rehash before allocating a new node array so that the hash table
  // is in a consistent state when an exception is thrown. Nodes
  // without an owner are removed.

  Node_*  oldNodes = nodes_;

  for ( idx_t i = 0; i < itemCount_; i++ )
  {
    Node_*  node = oldNodes + i;
    idx_t   tix  = getTableIndex_ ( node->key );

    node->next  = table_[tix];
    table_[tix] = i;
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

  for ( idx_t i = 0; i < itemCount_; i++ )
  {
    Node_*  oldNode = oldNodes + i;
    Node_*  newNode = newNodes + i;

    newNode->next   = oldNode->next;

    newNode->key  .swap ( oldNode->key );
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
//   checkWritable_
//-----------------------------------------------------------------------


inline void ObjectHashMap::checkWritable_ ()
{
  if ( enumCount_.load( std::memory_order_relaxed ) > 0 )
  {
    writeError_ ();
  }
}


//-----------------------------------------------------------------------
//   getTableIndex_
//-----------------------------------------------------------------------


inline idx_t ObjectHashMap::getTableIndex_

  ( const Ref<Object>&  key ) const

{
  return (idx_t) ( hashMask( tableBits_ ) & key.hashValue() );
}


JEM_END_PACKAGE( util )
