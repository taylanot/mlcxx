
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


#include <cstdio>
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/util/StdAllocator.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class StdAllocator::Item_
//=======================================================================


class StdAllocator::Item_
{
 public:

  union
  {
    Chunk_*         chunk;
    byte            padding[ALIGNMENT];
  };

  Item_*            next;
  Item_*            prev;


 public:

  inline void       linkTo ( Item_* item );
  inline void       unlink ();

};


//-----------------------------------------------------------------------
//   implementation
//-----------------------------------------------------------------------


inline void StdAllocator::Item_::linkTo ( Item_* item )
{
  next       = item;
  prev       = item->prev;
  next->prev = this;
  prev->next = this;
}


inline void StdAllocator::Item_::unlink ()
{
  next->prev = prev;
  prev->next = next;
}



//=======================================================================
//   class StdAllocator::Chunk_
//=======================================================================


class StdAllocator::Chunk_
{
 public:

  inline            Chunk_  ( size_t sz );
  inline           ~Chunk_  ();

  inline void*      operator new

    ( size_t          size );

  inline void       operator delete

    ( void*           addr,
      size_t          size );


 public:

  const size_t      size;
  int               refcount;
  byte*             data;

};


//-----------------------------------------------------------------------
//   implementation
//-----------------------------------------------------------------------


inline StdAllocator::Chunk_::Chunk_ ( size_t sz ) :

  size     ( sz ),
  refcount ( 0 ),
  data     ( (byte*) MemCache::alloc( sz ) )

{}


inline StdAllocator::Chunk_::~Chunk_ ()
{
  if ( data )
  {
    MemCache::dealloc ( data, size );

    data = nullptr;
  }
}


//-----------------------------------------------------------------------
//   operator new
//-----------------------------------------------------------------------


inline void* StdAllocator::Chunk_::operator new

  ( size_t  size )

{
  return MemCache::alloc ( size );
}


//-----------------------------------------------------------------------
//   operator delete
//-----------------------------------------------------------------------


inline void StdAllocator::Chunk_::operator delete

  ( void*   addr,
    size_t  size )

{
  MemCache::dealloc ( addr, size );
}


//=======================================================================
//   class StdAllocator
//=======================================================================

//-----------------------------------------------------------------------
//   static data members
//-----------------------------------------------------------------------


const size_t  StdAllocator::MAX_CHUNK_SIZE_ = 1024;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdAllocator::StdAllocator ( size_t itemSize ) :

  itemSize_  ( getItemSize_ ( itemSize ) ),
  chunkSize_ ( getItemSize_ ( itemSize ) ),
  freeList_  ( nullptr  )

{
  freeList_ = (Item_*) MemCache::alloc ( sizeof(Item_) );

  freeList_->next = freeList_;
  freeList_->prev = freeList_;
}

StdAllocator::StdAllocator ( StdAllocator&& rhs ) :

  StdAllocator ( rhs.itemSize_ )

{
  swap ( rhs );
}


StdAllocator::~StdAllocator ()
{
  if ( freeList_->next != freeList_ )
  {
    std::fprintf ( stderr,
                   "jem::util::StdAllocator: memory leak detected\n" );
  }

  MemCache::dealloc ( freeList_, sizeof(Item_) );

  freeList_ = nullptr;
}


//-----------------------------------------------------------------------
//   alloc
//-----------------------------------------------------------------------


void* StdAllocator::alloc ()
{
  Item_*  item = freeList_->next;

  if ( item == freeList_ )
  {
    allocChunk_ ();
    item = freeList_->next;
  }

  item->unlink ();
  item->chunk->refcount++;

  return (((byte*) item) + ALIGNMENT);
}


//-----------------------------------------------------------------------
//   dealloc
//-----------------------------------------------------------------------


void StdAllocator::dealloc ( void* p )
{
  Item_*  item = (Item_*) (((byte*) p) - ALIGNMENT);

  item->linkTo ( freeList_ );

  if ( (--item->chunk->refcount) == 0 )
  {
    freeChunk_ ( item->chunk );
  }
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void StdAllocator::swap ( StdAllocator& rhs ) noexcept
{
  jem::swap ( itemSize_,  rhs.itemSize_  );
  jem::swap ( chunkSize_, rhs.chunkSize_ );
  jem::swap ( freeList_,  rhs.freeList_  );
}


//-----------------------------------------------------------------------
//   allocChunk_
//-----------------------------------------------------------------------


void StdAllocator::allocChunk_ ()
{
  Chunk_*  chunk = new Chunk_ ( chunkSize_ );

  if ( chunkSize_ == itemSize_ )
  {
    Item_* item = (Item_*) chunk->data;

    item->chunk = chunk;
    item->linkTo ( freeList_ );
  }
  else
  {
    Item_*  first = (Item_*) chunk->data;
    Item_*  last;
    size_t  offset;

    first->chunk = chunk;
    offset       = itemSize_;

    do
    {
      last        = (Item_*) (chunk->data + offset);
      last->chunk = chunk;
      last->prev  = first;
      first->next = last;
      first       = last;
      offset     += itemSize_;
    }
    while ( offset < chunkSize_ );

    first             = (Item_*) chunk->data;
    first->prev       = freeList_;
    last->next        = freeList_->next;
    first->prev->next = first;
    last->next->prev  = last;
  }

  if ( (2 * chunkSize_) <= MAX_CHUNK_SIZE_ )
  {
    chunkSize_ *= 2;
  }
}


//-----------------------------------------------------------------------
//   freeChunk
//-----------------------------------------------------------------------


void StdAllocator::freeChunk_ ( Chunk_* chunk )
{
  byte*   first = chunk->data;
  byte*   last  = chunk->data + chunk->size - 1;
  Item_*  item;

  while ( first < last )
  {
    item = (Item_*) first;
    item->unlink ();
    first += itemSize_;
  }

  delete chunk;
}


//-----------------------------------------------------------------------
//   getItemSize_
//-----------------------------------------------------------------------


size_t StdAllocator::getItemSize_ ( size_t itemSize )
{
  size_t sz ( align(itemSize) + ALIGNMENT );

  if ( sz < sizeof(Item_) )
  {
    sz = sizeof(Item_);
  }

  return sz;
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits<util::StdAllocator>
//=======================================================================


String TypeTraits<util::StdAllocator>::whatis ()
{
  return String ( "a standard allocator" );
}


String TypeTraits<util::StdAllocator>::typeName ()
{
  return String ( "jem::util::StdAllocator" );
}


JEM_END_PACKAGE_BASE
