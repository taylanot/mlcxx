
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

#ifndef JEM_BASE_THREAD_THREADCACHE_H
#define JEM_BASE_THREAD_THREADCACHE_H

#include <cstdlib>
#include <jem/base/utilities.h>
#include <jem/base/AllocatorUtils.h>
#include <jem/base/AutoPointer.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ThreadCache
//-----------------------------------------------------------------------


class ThreadCache : private AllocatorUtils
{
 public:

  typedef ThreadCache   Self;


  static const size_t   FREE_LIST_COUNT = 16;
  static const size_t   MAX_ITEM_SIZE   = ALIGNMENT * FREE_LIST_COUNT;
  static const size_t   DEFAULT_SIZE    = 128 * 1024;
  static const size_t   LARGE_ALLOC     = ALIGNMENT * 64;
  static const byte     CANARY_VALUE    = 123;


  explicit              ThreadCache

    ( size_t              size = DEFAULT_SIZE );

                       ~ThreadCache   ();

  AutoPointer<Self>     clone         ();

  inline void*          alloc

    ( size_t              size,
      ThrowMode           tm = CAN_THROW );

  inline void           dealloc

    ( void*               addr,
      size_t              size )         noexcept;

  void*                 realloc

    ( void*               oldAddr,
      size_t              oldSize,
      size_t              newSize,
      ThrowMode           tm = CAN_THROW );

  void                  setSize

    ( size_t              size );


 private:

                        ThreadCache ( const Self& );
  ThreadCache&          operator =  ( const Self& );

  void*                 alloc_

    ( size_t              size,
      ThrowMode           tm );

  void                  writeError_

    ( const char*         where,
      ThrowMode           tm );

  void                  allocError_

    ( const char*         where,
      size_t              size,
      ThrowMode           tm );


 private:

  struct                Item_;
  class                 ItemList_;
  friend class          ItemList_;

  struct                Item_
  {
    Item_*                next;
  };

  class                 ItemList_
  {
   public:

    inline                ItemList_   ();
                         ~ItemList_   ();

   public:

    Item_*                first;
    int                   length;
    int                   maxLength;

  };


  ItemList_             freeLists_[FREE_LIST_COUNT];
  size_t                cacheSize_;
  byte                  errors_;

};






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ThreadCache::ItemList_
//=======================================================================


ThreadCache::ItemList_::ItemList_ ()
{
  first     = nullptr;
  length    = 0;
  maxLength = 0;
}


//=======================================================================
//   class ThreadCache
//=======================================================================

//-----------------------------------------------------------------------
//   alloc
//-----------------------------------------------------------------------


inline void* ThreadCache::alloc

  ( size_t     size,
    ThrowMode  tm )

{
  if ( size == 0 )
  {
    return nullptr;
  }

  if ( size <= MAX_ITEM_SIZE )
  {
    ItemList_*  list = freeLists_ + ((size - 1) / ALIGNMENT);

    if ( list->first )
    {
      Item_*  item = list->first;

      list->first = item->next;
      list->length--;

      return (void*) item;
    }
    else
    {
      return alloc_ ( align(size), tm );
    }
  }
  else
  {
    size_t  n = align ( size + 1 );
    byte*   p;

    if ( size < LARGE_ALLOC )
    {
      p = (byte*) alloc_ ( n, tm );
    }
    else
    {
      p = (byte*) alloc_ ( n + ALIGNMENT, tm );

      for ( size_t i = 0; i < ALIGNMENT; i++ )
      {
        p[i] = CANARY_VALUE;
      }

      p += ALIGNMENT;
    }

    for ( size_t i = size; i < n; i++ )
    {
      p[i] = CANARY_VALUE;
    }

    return p;
  }
}


//-----------------------------------------------------------------------
//   dealloc
//-----------------------------------------------------------------------


inline void ThreadCache::dealloc

  ( void*   addr,
    size_t  size ) noexcept

{
  if ( size == 0 )
  {
    return;
  }

  if ( size <= MAX_ITEM_SIZE )
  {
    ItemList_*  list = freeLists_ + ((size - 1) / ALIGNMENT);
    Item_*      item = (Item_*) addr;

    if ( list->length == list->maxLength )
    {
      std::free ( item );
    }
    else
    {
      item->next  = list->first;
      list->first = item;
      list->length++;
    }
  }
  else
  {
    size_t  n = align ( size + 1 );
    byte*   p = (byte*) addr;

    for ( size_t i = size; i < n; i++ )
    {
      errors_ = (byte) (errors_ | (p[size] - CANARY_VALUE));
    }

    if ( size >= LARGE_ALLOC )
    {
      p -= ALIGNMENT;

      for ( size_t i = 0; i < ALIGNMENT; i++ )
      {
        errors_ = (byte) (errors_ | (p[i] - CANARY_VALUE));
      }
    }

    std::free ( p );
  }
}


JEM_END_PACKAGE_BASE

#endif
