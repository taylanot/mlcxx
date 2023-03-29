
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
#include <jem/util/HashDictionary.h>


JEM_DEFINE_SERIABLE_CLASS( jem::util::HashDictionary );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class HashDictionary::Node_
//=======================================================================


class HashDictionary::Node_ : public Item_
{
 public:

  inline                  Node_   ();


 public:

  idx_t                   next;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline HashDictionary::Node_::Node_ ()
{
  hash =  0;
  next = -1;
}


//=======================================================================
//   class HashDictionary::Enumerator_
//=======================================================================


class HashDictionary::Enumerator_ : public DictionaryEnumerator
{
 public:

  explicit inline         Enumerator_

    ( const Ref<HashDict>&  dict );

  virtual bool            atEnd         () const override;
  virtual void            toNext        ()       override;
  virtual String          getKey        () const override;
  virtual Ref<Object>     getValue      () const override;
  inline void             checkOnDict   () const;


 protected:

  virtual                ~Enumerator_   ();


 private:

  Ref<HashDict>           dict_;
  idx_t                   pos_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline HashDictionary::Enumerator_::Enumerator_

  ( const Ref<HashDict>&  dict ) :

    dict_ ( dict )

{
  pos_ = 0;

  dict_->enumCount_.fetch_add ( 1, std::memory_order_relaxed );
}


HashDictionary::Enumerator_::~Enumerator_ ()
{
  dict_->enumCount_.fetch_sub ( 1, std::memory_order_relaxed );
}


//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


bool HashDictionary::Enumerator_::atEnd () const
{
  return (pos_ >= dict_->itemCount_);
}


//-----------------------------------------------------------------------
//   toNext
//-----------------------------------------------------------------------


void HashDictionary::Enumerator_::toNext ()
{
  checkOnDict ();

  pos_++;
}


//-----------------------------------------------------------------------
//   getKey
//-----------------------------------------------------------------------


String HashDictionary::Enumerator_::getKey () const
{
  checkOnDict ();

  if ( pos_ < CACHE_SIZE_ )
  {
    return dict_->cache_[pos_].key;
  }
  else
  {
    return dict_->nodes_[pos_ - CACHE_SIZE_].key;
  }
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


Ref<Object> HashDictionary::Enumerator_::getValue () const
{
  checkOnDict ();

  if ( pos_ < CACHE_SIZE_ )
  {
    return dict_->cache_[pos_].value;
  }
  else
  {
    return dict_->nodes_[pos_ - CACHE_SIZE_].value;
  }
}


//-----------------------------------------------------------------------
//   checkOnDict
//-----------------------------------------------------------------------


inline void HashDictionary::Enumerator_::checkOnDict () const
{
  if ( pos_ >= dict_->itemCount_ )
  {
    atEndError_ ();
  }
}


//=======================================================================
//   class HashDictionary
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


HashDictionary::HashDictionary () :

  enumCount_ ( 0 )

{
  table_     =  nullptr;
  nodes_     =  nullptr;
  capacity_  =  CACHE_SIZE_;
  itemCount_ =  0;
  tableBits_ = -1;
}


HashDictionary::~HashDictionary ()
{
  if ( table_ )
  {
    MemCache::dealloc ( table_, (size_t) hashSize(tableBits_) *
                        sizeof(*table_) );
  }

  if ( nodes_ )
  {
    idx_t  n = capacity_ - CACHE_SIZE_;

    destroy           ( nodes_, n );
    MemCache::dealloc ( nodes_, (size_t) n * sizeof(Node_) );
  }

  table_ = nullptr;
  nodes_ = nullptr;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void HashDictionary::readFrom ( io::ObjectInput& in )
{
  readFrom_ ( in );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void HashDictionary::writeTo ( io::ObjectOutput& out ) const
{
  writeTo_ ( out );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> HashDictionary::clone () const
{
  Ref<Self>  dict = newInstance<Self> ();

  dict->clone_ ( *this );

  return dict;
}


//-----------------------------------------------------------------------
//   getDictEnum
//-----------------------------------------------------------------------


Ref<DictionaryEnumerator> HashDictionary::getDictEnum () const
{
  Ref<Self>  self = const_cast<Self*> ( this );

  return newInstance<Enumerator_> ( self );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Object> HashDictionary::get ( const String& key ) const
{
  const Item_*  item = find_ ( key );

  if ( item )
  {
    return item->value;
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


bool HashDictionary::find

  ( Ref<Object>&   value,
    const String&  key ) const

{
  const Item_*  item = find_ ( key );

  if ( item )
  {
    value = item->value;

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


void HashDictionary::insert

  ( const String&       key,
    const Ref<Object>&  value )

{
  checkWritable_ ();

  Item_*  item = find_ ( key );

  if ( item )
  {
    item->value = value;

    return;
  }

  // Insert a new key-value pair.

  lint   hash = key.hashValue ();
  idx_t  n    = itemCount_;

  if ( n < CACHE_SIZE_ )
  {
    cache_[n].key   = key;
    cache_[n].hash  = hash;
    cache_[n].value = value;
  }
  else
  {
    if ( n >= capacity_ )
    {
      realloc_ ( expand_( capacity_ ) );
    }

    idx_t  tix = getTableIndex_ ( hash );

    n -= CACHE_SIZE_;

    nodes_[n].key   = key;
    nodes_[n].hash  = hash;
    nodes_[n].value = value;
    nodes_[n].next  = table_[tix];
    table_[tix]     = n;
  }

  itemCount_++;
}


void HashDictionary::insert

  ( String&&       key,
    Ref<Object>&&  value )

{
  checkWritable_ ();

  Item_*  item = find_ ( key );

  if ( item )
  {
    item->value = std::move ( value );

    return;
  }

  // Insert a new key-value pair.

  lint   hash = key.hashValue ();
  idx_t  n    = itemCount_;

  if ( n < CACHE_SIZE_ )
  {
    cache_[n].key   = std::move ( key );
    cache_[n].hash  = hash;
    cache_[n].value = std::move ( value );
  }
  else
  {
    if ( n >= capacity_ )
    {
      realloc_ ( expand_( capacity_ ) );
    }

    idx_t  tix = getTableIndex_ ( hash );

    n -= CACHE_SIZE_;

    nodes_[n].key   = std::move ( key );
    nodes_[n].hash  = hash;
    nodes_[n].value = std::move ( value );
    nodes_[n].next  = table_[tix];
    table_[tix]     = n;
  }

  itemCount_++;
}

//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void HashDictionary::erase ( const String& key )
{
  checkWritable_ ();

  Item_*  item = find_ ( key );

  if ( item )
  {
    erase_ ( item );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void HashDictionary::clear ()
{
  checkWritable_ ();

  Item_*  cache = cache_;
  idx_t   n     = min ( itemCount_, (idx_t) CACHE_SIZE_ );

  for ( idx_t i = 0; i < n; i++ )
  {
    cache[i].key   = String ();
    cache[i].value = nullptr;
  }

  if ( itemCount_ > CACHE_SIZE_ )
  {
    Node_*  nodes = nodes_;

    n = itemCount_ - CACHE_SIZE_;

    for ( idx_t i = 0; i < n; i++ )
    {
      nodes[i].next  = -1;
      nodes[i].hash  =  0;
      nodes[i].key   =  String ();
      nodes[i].value =  nullptr;
    }

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


void HashDictionary::reserve ( idx_t count )
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


void HashDictionary::trimToSize ()
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


idx_t HashDictionary::capacity () const
{
  return capacity_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t HashDictionary::size () const
{
  return itemCount_;
}


//-----------------------------------------------------------------------
//   find_
//-----------------------------------------------------------------------


HashDictionary::Item_*

  HashDictionary::find_ ( const String& key ) const

{
  const lint  hash = key.hashValue ();

  Item_*      item = const_cast<Item_*> ( cache_ );
  idx_t       n    = min ( itemCount_, (idx_t) CACHE_SIZE_ );


  for ( idx_t i = 0; i < n; i++, item++ )
  {
    if ( item->hash == hash && key.equals( item->key ) )
    {
      return item;
    }
  }

  if ( itemCount_ > CACHE_SIZE_ )
  {
    Node_*  nodes = nodes_;
    idx_t   tix   = getTableIndex_ ( hash );
    idx_t   i     = table_[tix];

    while ( i >= 0 )
    {
      item = nodes + i;

      if ( item->hash == hash && key.equals( item->key ) )
      {
        return item;
      }

      i = nodes[i].next;
    }
  }

  return (Item_*) 0;
}


//-----------------------------------------------------------------------
//   erase_
//-----------------------------------------------------------------------


void HashDictionary::erase_ ( Item_* item )
{
  item->value = nullptr;

  itemCount_--;

  if ( item >= cache_ && item < (cache_ + CACHE_SIZE_) )
  {
    item->key  = String ();
    item->hash = 0;

    if ( itemCount_ < CACHE_SIZE_ )
    {
      Item_*  tail = cache_ + itemCount_;

      if ( tail != item )
      {
        item->key  .swap ( tail->key );
        item->value.swap ( tail->value );
        jem::swap        ( item->hash, tail->hash );
      }
    }
    else
    {
      Node_*  tail = nodes_ + (itemCount_ - CACHE_SIZE_);

      unlink_ ( tail );

      item->key  .swap ( tail->key );
      item->value.swap ( tail->value );
      jem::swap        ( item->hash, tail->hash );
    }
  }
  else
  {
    Node_*  node = (Node_*) item;

    unlink_ ( node );

    node->key  = String ();
    node->hash = 0;

    if ( itemCount_ >= CACHE_SIZE_ )
    {
      Node_*  tail = nodes_ + (itemCount_ - CACHE_SIZE_);

      if ( tail != node )
      {
        idx_t  tix = unlink_ ( tail );

        node->key  .swap ( tail->key );
        node->value.swap ( tail->value );
        jem::swap        ( item->hash, tail->hash );

        node->next  = table_[tix];
        table_[tix] = (idx_t) (node - nodes_);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   unlink_
//-----------------------------------------------------------------------


idx_t HashDictionary::unlink_ ( Node_* node )
{
  idx_t  tix = getTableIndex_ ( node->hash );
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


void HashDictionary::realloc_ ( idx_t newCap )
{
  if ( newCap < 0 )
  {
    overflowError ( JEM_FUNC, newCap );
  }

  if ( newCap < itemCount_ )
  {
    newCap = itemCount_;
  }

  newCap -= CACHE_SIZE_;

  if ( newCap <= 0 )
  {
    if ( table_ )
    {
      MemCache::dealloc ( table_, (size_t) hashSize(tableBits_) *
                          sizeof(*table_) );

      table_ = nullptr;
    }

    if ( nodes_ )
    {
      idx_t  n = capacity_ - CACHE_SIZE_;

      destroy           ( nodes_, n );
      MemCache::dealloc ( nodes_, (size_t) n * sizeof(Node_) );

      nodes_ = nullptr;
    }

    capacity_  =  CACHE_SIZE_;
    tableBits_ = -1;

    return;
  }

  // Reallocate the hash table.

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
  // is in a consistent state when an exception is thrown.

  Node_*  oldNodes  = nodes_;
  idx_t   nodeCount = itemCount_ - CACHE_SIZE_;

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    Node_*  node = oldNodes + i;
    idx_t   tix  = getTableIndex_ ( node->hash );

    node->next  = table_[tix];
    table_[tix] = i;
  }

  // Allocate a new node array.

  Node_*  newNodes = (Node_*)

    MemCache::alloc ( (size_t) newCap * sizeof(Node_) );

  try
  {
    construct ( newNodes, newCap );
  }
  catch ( ... )
  {
    MemCache::dealloc ( newNodes, (size_t) newCap * sizeof(Node_) );
    throw;
  }

  // Copy the nodes to the new array.

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    Node_*  oldNode = oldNodes + i;
    Node_*  newNode = newNodes + i;

    newNode->next   = oldNode->next;
    newNode->hash   = oldNode->hash;

    newNode->key  .swap ( oldNode->key );
    newNode->value.swap ( oldNode->value );
  }

  if ( oldNodes )
  {
    idx_t  n = capacity_ - CACHE_SIZE_;

    destroy           ( oldNodes, n );
    MemCache::dealloc ( oldNodes, (size_t) n * sizeof(Node_) );
  }

  nodes_    = newNodes;
  capacity_ = newCap + CACHE_SIZE_;
}


//-----------------------------------------------------------------------
//   checkWritable_
//-----------------------------------------------------------------------


inline void HashDictionary::checkWritable_ ()
{
  if ( enumCount_.load( std::memory_order_relaxed ) > 0 )
  {
    writeError_ ();
  }
}


//-----------------------------------------------------------------------
//   getTableIndex_
//-----------------------------------------------------------------------


inline idx_t HashDictionary::getTableIndex_ ( lint hash ) const
{
  return (idx_t) (hashMask( tableBits_ ) & hash);
}


JEM_END_PACKAGE( util )
