
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

#ifndef JEM_UTIL_HASHTABLE_TCC
#define JEM_UTIL_HASHTABLE_TCC

#include <new>
#include <cmath>
#include <utility>
#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/util/error.h>
#include <jem/util/hash.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class HashTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T>

  HashTable<T>::HashTable () :

    allocator_ ( sizeof(Node) )

{
  init_ ( 0 );
}


template <class T>

  HashTable<T>::HashTable ( idx_t cap ) :

    allocator_ ( sizeof(Node) )

{
  JEM_PRECHECK2 ( cap >= 0, "invalid HashTable capacity" );

  init_ ( cap );
}


template <class T>

  HashTable<T>::HashTable ( const HashTable& rhs ) :

    allocator_ ( sizeof(Node) )

{
  init_ ( rhs.capacity() );

  try
  {
    insert ( rhs.begin(), rhs.end() );
  }
  catch ( ... )
  {
    clear_ ();
    throw;
  }
}


template <class T>

  HashTable<T>::HashTable ( HashTable&& rhs ) :

    HashTable ()

{
  swap ( rhs );
}


template <class T>
  template <class InputIterator>

  HashTable<T>::HashTable

  ( InputIterator  first,
    InputIterator  last ) :

    allocator_ ( sizeof(Node) )

{
  init_ ( (idx_t) distance( first, last ) );

  try
  {
    insert ( first, last );
  }
  catch ( ... )
  {
    clear_ ();
    throw;
  }
}


template <class T>
  template <class U>

  HashTable<T>::HashTable

  ( std::initializer_list<U>  list ) :

    HashTable ( list.begin(), list.end() )

{}


template <class T>

  inline HashTable<T>::~HashTable ()

{
  clear_ ();
}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class T>

  inline typename HashTable<T>::Iterator
  HashTable<T>::begin () noexcept

{
  return Iterator ( table_ );
}


template <class T>

  inline typename HashTable<T>::ConstIterator
  HashTable<T>::begin () const noexcept

{
  return ConstIterator ( table_ );
}


template <class T>

  inline typename HashTable<T>::Iterator
  HashTable<T>::end () noexcept

{
  const idx_t  n = hashSize ( tableBits_ );

  return Iterator ( table_ + n, table_[n] );
}


template <class T>

  inline typename HashTable<T>::ConstIterator
  HashTable<T>::end () const noexcept

{
  const idx_t  n = hashSize ( tableBits_ );

  return ConstIterator ( table_ + n, table_[n] );
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T>

  HashTable<T>& HashTable<T>::operator = ( const HashTable& rhs )

{
  if ( this != &rhs )
  {
    clear   ();
    reserve ( rhs.size() );
    insert  ( rhs.begin(), rhs.end() );
  }

  return *this;
}


template <class T>

  inline HashTable<T>&
  HashTable<T>::operator = ( HashTable&& rhs ) noexcept

{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  void HashTable<T>::swap ( HashTable& rhs ) noexcept

{
  allocator_.swap ( rhs.allocator_ );

  jem::swap ( table_,     rhs.table_     );
  jem::swap ( tableBits_, rhs.tableBits_ );
  jem::swap ( size_,      rhs.size_      );
  jem::swap ( capacity_,  rhs.capacity_  );
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


template <class T>

  inline bool HashTable<T>::contains ( const KeyType& key ) const

{
  return (find( key ) != end());
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


template <class T>

  typename HashTable<T>::Iterator
  HashTable<T>::find ( const KeyType& key )

{
  lint    hash  = Traits::hashValue    ( key );
  void**  chain = table_ + tableIndex_ ( hash, tableBits_ );
  Node*   node  = static_cast<Node*>   ( *chain );
  bool    eq;


  while ( node )
  {
    if ( node->hash == hash )
    {
      eq = Traits::equalKeys ( Traits::getKey( node->value ), key );

      if ( eq )
      {
        return Iterator ( chain, node );
      }
    }

    node = node->next;
  }

  return end ();
}


template <class T>

  typename HashTable<T>::ConstIterator
  HashTable<T>::find ( const KeyType& key ) const

{
  lint    hash  = Traits::hashValue    ( key );
  void**  chain = table_ + tableIndex_ ( hash, tableBits_ );
  Node*   node  = static_cast<Node*>   ( *chain );
  bool    eq;


  while ( node )
  {
    if ( node->hash == hash )
    {
      eq = Traits::equalKeys ( Traits::getKey( node->value ), key );

      if ( eq )
      {
        return ConstIterator ( chain, node );
      }
    }

    node = node->next;
  }

  return end ();
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


template <class T>

  typename HashTable<T>::Iterator
  HashTable<T>::insert ( const ValueType& item )

{
  lint    hash  = Traits::hashValue    ( Traits::getKey( item ) );
  void**  chain = table_ + tableIndex_ ( hash, tableBits_ );

  Node*   node;


  erase_ ( chain, hash, Traits::getKey( item ) );

  if ( size_ == capacity_ )
  {
    reserve_ ( 2 * capacity_ + 1 );

    chain = table_ + tableIndex_ ( hash, tableBits_ );
  }

  node       = newNode_ ( item, hash );
  node->next = static_cast<Node*> ( *chain );
  *chain     = node;

  return Iterator ( chain, node );
}


template <class T>

  typename HashTable<T>::Iterator
  HashTable<T>::insert ( ValueType&& item )

{
  lint    hash  = Traits::hashValue    ( Traits::getKey( item ) );
  void**  chain = table_ + tableIndex_ ( hash, tableBits_ );

  Node*   node;


  erase_ ( chain, hash, Traits::getKey( item ) );

  if ( size_ == capacity_ )
  {
    reserve_ ( 2 * capacity_ + 1 );

    chain = table_ + tableIndex_ ( hash, tableBits_ );
  }

  node       = newNode_ ( std::move( item ), hash );
  node->next = static_cast<Node*> ( *chain );
  *chain     = node;

  return Iterator ( chain, node );
}


template <class T>
  template <class InputIterator>

  void HashTable<T>::insert

  ( InputIterator  first,
    InputIterator  last )

{
  while ( first != last )
  {
    insert ( *first );

    ++first;
  }
}


template <class T>
  template <class U>

  inline void HashTable<T>::insert

  ( std::initializer_list<U>  list )

{
  insert ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


template <class T>

  void HashTable<T>::erase ( const KeyType& key )

{
  lint    hash  = Traits::hashValue    ( key );
  void**  chain = table_ + tableIndex_ ( hash, tableBits_ );

  erase_ ( chain, hash, key );
}


template <class T>

  void HashTable<T>::erase ( Iterator pos )

{
  JEM_PRECHECK2 ( pos != end(), "invalid HashTable iterator" );

  void**  chain = pos.getChain ();
  Node*   node  = static_cast<Node*> ( pos.getNode() );


  if ( *chain == node )
  {
    *chain = node->next;
  }
  else
  {
    Node* prev = static_cast<Node*> ( *chain );

    while ( prev->next != node )
    {
      prev = prev->next;
    }

    prev->next = node->next;
  }

  free_ ( node );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class T>

  void HashTable<T>::clear ()

{
  void**       table = table_;
  const idx_t  n     = hashSize ( tableBits_ );

  void*  chain;
  Node*  node;


  for ( idx_t i = 0; i < n; i++ )
  {
    chain = table[i];

    while ( chain )
    {
      node  = static_cast<Node*> ( chain );
      chain = node->next;

      free_ ( node );
    }

    table[i] = nullptr;
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


template <class T>

  void HashTable<T>::reserve ( idx_t cap )

{
  if ( cap > capacity_ )
  {
    reserve_ ( cap );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


template <class T>

  void HashTable<T>::trimToSize ()

{
  if ( size_ < capacity_ )
  {
    reserve_ ( size_ );
  }
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


template <class T>

  inline idx_t HashTable<T>::capacity () const noexcept

{
  return capacity_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T>

  inline idx_t HashTable<T>::size () const noexcept

{
  return size_;
}


//-----------------------------------------------------------------------
//   printStats
//-----------------------------------------------------------------------


template <class T>
  template <class Output>

  void HashTable<T>::printStats ( Output& out ) const

{
  const idx_t  n = hashSize ( tableBits_ );

  Node*        node;

  double       avg;
  double       dev;

  idx_t        chainCount;
  idx_t        minLength;
  idx_t        maxLength;
  idx_t        i, j;


  chainCount = 0;

  for ( i = 0; i < n; i++ )
  {
    if ( table_[i] )
    {
      chainCount++;
    }
  }

  minLength = 0;
  maxLength = 0;
  dev       = 0.0;
  avg       = 0.0;

  if ( chainCount > 0 )
  {
    avg = (double) size_ / (double) chainCount;
  }

  for ( i = 0; i < n; i++ )
  {
    node = static_cast<Node*> ( table_[i] );

    if ( ! node )
    {
      continue;
    }

    j = 0;

    while ( node )
    {
      j++;
      node = node->next;
    }

    if ( minLength == 0 || j < minLength )
    {
      minLength = j;
    }
    if ( j > maxLength )
    {
      maxLength = j;
    }

    dev += ((double) j - avg) * ((double) j - avg);
  }

  if ( chainCount > 0 )
  {
    dev = std::sqrt ( dev ) / (double) chainCount;
  }

  print ( out, "HashTable statistics:" );

  print ( out, "\n  number of items        : ", size_      );
  print ( out, "\n  total number of chains : ", n          );
  print ( out, "\n  number of used chains  : ", chainCount );
  print ( out, "\n  minimum chain length   : ", minLength  );
  print ( out, "\n  maximum chain length   : ", maxLength  );
  print ( out, "\n  average chain length   : ", avg        );
  print ( out, "\n  chain length deviation : ", dev, '\n'  );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T>

  void HashTable<T>::init_ ( idx_t cap )

{
  idx_t  i, n;

  tableBits_ = hashBits ( cap );
  size_      = 0;
  capacity_  = cap;
  n          = hashSize ( tableBits_ );
  table_     = (void**) MemCache::alloc ( (size_t) (n + 1) *
                                          sizeof(*table_) );

  for ( i = 0; i < n; i++ )
  {
    table_[i] = nullptr;
  }

  table_[n] = table_;
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


template <class T>

  inline void HashTable<T>::clear_ ()

{
  clear ();

  if ( table_ )
  {
    const idx_t  n = hashSize ( tableBits_ );

    MemCache::dealloc ( table_, (size_t) (n + 1) *
                                sizeof(*table_) );
  }
}


//-----------------------------------------------------------------------
//   reserve_
//-----------------------------------------------------------------------


template <class T>

  void HashTable<T>::reserve_ ( idx_t newCap )

{
  void**       oldTable = table_;
  const idx_t  oldSize  = hashSize ( tableBits_ );

  void**       newTable;
  int          newBits;
  idx_t        newSize;
  Node*        node;
  idx_t        i, j;


  if ( newCap < 0 )
  {
    overflowError ( JEM_FUNC, newCap );
  }

  if ( newCap < size_ )
  {
    newCap = size_;
  }

  newBits  = hashBits ( newCap  );
  newSize  = hashSize ( newBits );
  newTable =

    (void**) MemCache::alloc ( (size_t) (newSize + 1) *
                               sizeof(*table_) );

  newTable[newSize] = newTable;

  for ( i = 0; i < newSize; i++ )
  {
    newTable[i] = nullptr;
  }

  try
  {
    for ( i = 0; i < oldSize; i++ )
    {
      while ( oldTable[i] )
      {
        node        = static_cast<Node*> ( oldTable[i] );
        j           = tableIndex_ ( node->hash, newBits );
        oldTable[i] = node->next;
        node->next  = static_cast<Node*> ( newTable[j] );
        newTable[j] = node;
      }
    }
  }
  catch ( ... )
  {
    for ( i = 0; i < newSize; i++ )
    {
      while ( newTable[i] )
      {
        node        = static_cast<Node*> ( newTable[i] );
        newTable[i] = node->next;

        free_ ( node );
      }
    }

    MemCache::dealloc ( newTable, (size_t) (newSize + 1) *
                                  sizeof(*table_) );

    throw;
  }

  if ( oldTable )
  {
    MemCache::dealloc ( oldTable, (size_t) (oldSize + 1) *
                                  sizeof(*table_) );
  }

  table_     = newTable;
  tableBits_ = newBits;
  capacity_  = newCap;
}


//-----------------------------------------------------------------------
//   erase_
//-----------------------------------------------------------------------


template <class T>

  inline void HashTable<T>::erase_

  ( void**          chain,
    lint            hash,
    const KeyType&  key )

{
  Node*  node = static_cast<Node*> ( *chain );
  Node*  prev = nullptr;
  bool   eq;


  while ( node )
  {
    if ( node->hash == hash )
    {
      eq = Traits::equalKeys ( Traits::getKey( node->value ), key );

      if ( eq )
      {
        if ( prev )
        {
          prev->next = node->next;
        }
        else
        {
          *chain = node->next;
        }

        free_ ( node );

        return;
      }
    }

    prev = node;
    node = node->next;
  }
}


//-----------------------------------------------------------------------
//   newNode_
//-----------------------------------------------------------------------


template <class T>

  typename HashTable<T>::Node*
  HashTable<T>::newNode_

  ( const ValueType&  value,
    lint              hash )

{
  Node* node = (Node*) allocator_.alloc ();

  try
  {
    new (node) Node ( value, hash );
  }
  catch ( ... )
  {
    allocator_.dealloc ( node );
    throw;
  }

  size_++;
  return node;
}


template <class T>

  typename HashTable<T>::Node*
  HashTable<T>::newNode_

  ( ValueType&&  value,
    lint         hash )

{
  Node* node = (Node*) allocator_.alloc ();

  try
  {
    new (node) Node ( std::move( value ), hash );
  }
  catch ( ... )
  {
    allocator_.dealloc ( node );
    throw;
  }

  size_++;
  return node;
}


//-----------------------------------------------------------------------
//   free_
//-----------------------------------------------------------------------


template <class T>

  inline void HashTable<T>::free_ ( Node* node )

{
  size_--;
  node->~Node ();
  allocator_.dealloc ( node );
}


//-----------------------------------------------------------------------
//   tableIndex_
//-----------------------------------------------------------------------


template <class T>

  inline idx_t HashTable<T>::tableIndex_

  ( lint  hash,
    int   nbits ) noexcept

{
  return (idx_t) (hashMask( nbits ) & hash);
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output, class T>

  void                  print

  ( Output&               out,
    const HashTable<T>&   tbl )

{
  if ( tbl.size() > 0 )
  {
    print         ( out, "[ " );
    printSequence ( out, tbl.begin(), tbl.end(), ", " );
    print         ( out, " ]" );
  }
  else
  {
    print         ( out, "[]" );
  }
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void     swap

  ( HashTable<T>&   lhs,
    HashTable<T>&   rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

#endif
