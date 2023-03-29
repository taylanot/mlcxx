
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

#ifndef JEM_BASE_ARRAY_DATABLOCK_H
#define JEM_BASE_ARRAY_DATABLOCK_H

#include <jem/base/assert.h>
#include <jem/base/AllocatorUtils.h>
#include <jem/base/MemCache.h>
#include <jem/base/memory.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ArrayDataBlock
//-----------------------------------------------------------------------


template <class T>

  struct ArrayDataBlock

{
 public:

  typedef ArrayDataBlock   Self;


  static Self*             alloc

    ( T**                    ptr,
      idx_t                  size );

  static inline void       dealloc

    ( Self*                  block );

  static inline Self*      realloc

    ( Self*                  block,
      idx_t                  size );

  inline T*                getData () const noexcept;


 public:

  static Self              null;

  size_t                   msize_;
  int                      refcount;
  idx_t                    size;


 private:

  static inline void       destroy_

    ( Self*                  block,
      const True&            fast );

  static void              destroy_

    ( Self*                  block,
      const False&           fast );

  static Self*             realloc_

    ( Self*                  block,
      idx_t                  size,
      const True&            fast );

  static Self*             realloc_

    ( Self*                  block,
      idx_t                  size,
      const False&           fast );

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


template <class T>

  ArrayDataBlock<T> ArrayDataBlock<T>::null = { 0, 1, -1_idx };


//-----------------------------------------------------------------------
//   alloc
//-----------------------------------------------------------------------


template <class T>

  ArrayDataBlock<T>* ArrayDataBlock<T>::alloc

  ( T**    ptr,
    idx_t  size )

{
  JEM_ASSERT2 ( size >= 0, "negative Array allocation size" );

  Self*   block;
  byte*   mem;
  size_t  offset;
  size_t  msize;


  offset = AllocatorUtils::align ( sizeof(Self) );
  msize  = offset + (size_t) size * sizeof(T);
  mem    = (byte*) MemCache::alloc ( msize );
  *ptr   = (T*)   (mem + offset);
  block  = (Self*) mem;

  block->refcount = 1;
  block->msize_   = msize;

  // The block has not yet been initialized, so its size is
  // set to zero.

  block->size = 0;

  return block;
}


//-----------------------------------------------------------------------
//   dealloc
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayDataBlock<T>::dealloc ( Self* block )

{
  block->refcount--;

  if ( block->refcount == 0 )
  {
    if ( block != & null )
    {
      destroy_ ( block, HasTrivialDtor<T>() || IsPOD<T>() );
    }
  }
}


//-----------------------------------------------------------------------
//   realloc
//-----------------------------------------------------------------------


template <class T>

  inline ArrayDataBlock<T>* ArrayDataBlock<T>::realloc

  ( Self*  block,
    idx_t  size )

{
  return realloc_ ( block, size, IsPOD<T>() );
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


template <class T>

  inline T* ArrayDataBlock<T>::getData () const noexcept

{
  return (T*) ( ((byte*) this) +
                AllocatorUtils::align( sizeof(Self) ) );
}


//-----------------------------------------------------------------------
//   destroy_
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayDataBlock<T>::destroy_

  ( Self*        block,
    const True&  fast )

{
  MemCache::dealloc ( block, block->msize_ );
}


template <class T>

  void ArrayDataBlock<T>::destroy_

  ( Self*         block,
    const False&  fast )

{
  destroy ( block->getData(), block->size );
  MemCache::dealloc ( block,  block->msize_ );
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


template <class T>

  ArrayDataBlock<T>* ArrayDataBlock<T>::realloc_

  ( Self*        oldBlock,
    idx_t        newSize,
    const True&  fast )

{
  size_t  msize =

    AllocatorUtils::align ( sizeof(Self) ) +
    (size_t) newSize * sizeof(T);

  Self*   newBlock;

  if ( oldBlock == &null )
  {
    newBlock = (Self*) MemCache::alloc ( msize );
    newBlock->refcount = 1;
  }
  else
  {
    newBlock = (Self*)

      MemCache::realloc ( oldBlock, oldBlock->msize_, msize );
  }

  newBlock->msize_ = msize;
  newBlock->size   = newSize;

  return newBlock;
}


template <class T>

  ArrayDataBlock<T>* ArrayDataBlock<T>::realloc_

  ( Self*         oldBlock,
    idx_t         newSize,
    const False&  fast )

{
  Self*   newBlock;
  T*      oldPtr;
  T*      newPtr;
  idx_t   oldSize;


  oldSize  = oldBlock->size;
  oldPtr   = oldBlock->getData ();
  newBlock = alloc ( &newPtr, newSize );

  try
  {
    if ( newSize <= oldSize )
    {
      construct ( newPtr, oldPtr, oldPtr + newSize );
      newBlock->size = newSize;
    }
    else
    {
      construct ( newPtr, oldPtr, oldPtr + oldSize );
      newBlock->size = oldSize;
      construct ( newPtr + oldSize, (newSize - oldSize) );
      newBlock->size = newSize;
    }
  }
  catch ( ... )
  {
    dealloc ( newBlock );
    throw;
  }

  dealloc ( oldBlock );

  return newBlock;
}


JEM_END_PACKAGE_BASE

#endif
