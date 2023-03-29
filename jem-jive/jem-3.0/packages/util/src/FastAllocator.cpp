
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


#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/util/FastAllocator.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class FastAllocator::Chunk_
//=======================================================================


class FastAllocator::Chunk_
{
 public:

  inline            Chunk_

    ( Chunk_*         nxt,
      size_t          sz );

  inline void*      operator new

    ( size_t         size );

  inline void       operator delete

    ( void*           addr,
      size_t          size );


 public:

  const size_t      size;
  Chunk_*           next;
  byte*             data;

};


//-----------------------------------------------------------------------
//   implementation
//-----------------------------------------------------------------------


inline FastAllocator::Chunk_::Chunk_

  ( Chunk_*  nxt,
    size_t   sz ) :

    size ( sz ),
    next ( nxt ),
    data ( (byte*) MemCache::alloc( sz ) )

{}


//-----------------------------------------------------------------------
//   operator new
//-----------------------------------------------------------------------


inline void* FastAllocator::Chunk_::operator new

  ( size_t  size )

{
  return MemCache::alloc ( size );
}


//-----------------------------------------------------------------------
//   operator delete
//-----------------------------------------------------------------------


inline void FastAllocator::Chunk_::operator delete

  ( void*   addr,
    size_t  size )

{
  MemCache::dealloc ( addr, size );
}


//=======================================================================
//   class FastAllocator
//=======================================================================

//-----------------------------------------------------------------------
//   static data members
//-----------------------------------------------------------------------


const size_t  FastAllocator::MAX_CHUNK_SIZE_ = 16 * 1024;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FastAllocator::FastAllocator ( size_t itemSize ) noexcept :

  itemSize_  ( getItemSize_( itemSize ) ),
  freeList_  ( nullptr ),
  chunkList_ ( nullptr )

{}


FastAllocator::FastAllocator ( FastAllocator&& rhs ) noexcept
{
  itemSize_      = rhs.itemSize_;
  freeList_      = rhs.freeList_;
  chunkList_     = rhs.chunkList_;
  rhs.freeList_  = nullptr;
  rhs.chunkList_ = nullptr;
}


FastAllocator::~FastAllocator ()
{
  Chunk_*  list  = chunkList_;
  Chunk_*  chunk;

  while ( list )
  {
    chunk = list;
    list  = list->next;

    if ( chunk->data )
    {
      MemCache::dealloc ( chunk->data, chunk->size );
    }

    delete chunk;
  }

  freeList_  = nullptr;
  chunkList_ = nullptr;
}


//-----------------------------------------------------------------------
//   alloc
//-----------------------------------------------------------------------


void* FastAllocator::alloc ()
{
  if ( ! freeList_ )
  {
    allocChunk_();
  }

  void*  item = freeList_;

  freeList_   = freeList_->next;

  return item;
}


//-----------------------------------------------------------------------
//   dealloc
//-----------------------------------------------------------------------


void FastAllocator::dealloc ( void* p )
{
  Item_*  item = (Item_*) p;

  item->next = freeList_;
  freeList_  = item;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void FastAllocator::swap ( FastAllocator& rhs ) noexcept
{
  jem::swap ( itemSize_,  rhs.itemSize_ );
  jem::swap ( freeList_,  rhs.freeList_ );
  jem::swap ( chunkList_, rhs.chunkList_ );
}


//-----------------------------------------------------------------------
//   allocChunk_
//-----------------------------------------------------------------------


void FastAllocator::allocChunk_ ()
{
  Chunk_*  chunk;
  Item_*   item;
  size_t   chunkSize;


  if ( chunkList_ )
  {
    chunkSize = 2 * chunkList_->size;

    if ( chunkSize > MAX_CHUNK_SIZE_ )
    {
      chunkSize = chunkList_->size;
    }
  }
  else
  {
    chunkSize = itemSize_;
  }

  chunk      = new Chunk_ ( chunkList_, chunkSize );
  chunkList_ = chunk;

  if ( chunkSize == itemSize_ )
  {
    item       = (Item_*) chunk->data;
    item->next = freeList_;
    freeList_  = item;
  }
  else
  {
    byte*  first = chunk->data;
    byte*  last  = chunk->data + chunkSize;

    while ( first < last )
    {
      item        = (Item_*) first;
      item->next  = freeList_;
      freeList_   = item;
      first      += itemSize_;
    }
  }
}


//-----------------------------------------------------------------------
//   getItemSize_
//-----------------------------------------------------------------------


size_t FastAllocator::getItemSize_ ( size_t itemSize )
{
  if ( itemSize <= sizeof(Item_) )
  {
    return sizeof(Item_);
  }
  else
  {
    return align ( itemSize );
  }
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits<util::FastAllocator>
//=======================================================================


String TypeTraits<util::FastAllocator>::whatis ()
{
  return String ( "a fast allocator" );
}


String TypeTraits<util::FastAllocator>::typeName ()
{
  return String ( "jem::util::FastAllocator" );
}


JEM_END_PACKAGE_BASE
