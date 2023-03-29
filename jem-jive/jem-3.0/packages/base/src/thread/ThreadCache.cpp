
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


#include <cstring>
#include <jem/base/MemoryError.h>
#include <jem/base/OutOfMemoryException.h>
#include <jem/base/thread/ThreadCache.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class ThreadCache::ItemList_
//=======================================================================


ThreadCache::ItemList_::~ItemList_ ()
{
  while ( first )
  {
    Item_*  item = first;

    first = first->next;

    std::free ( item );
  }
}


//=======================================================================
//   class ThreadCache
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ThreadCache::ThreadCache ( size_t size )
{
  errors_ = 0x0;

  setSize ( size );
}


ThreadCache::~ThreadCache ()
{}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


AutoPointer<ThreadCache> ThreadCache::clone ()
{
  return new ThreadCache ( cacheSize_ );
}


//-----------------------------------------------------------------------
//   realloc
//-----------------------------------------------------------------------


void* ThreadCache::realloc

  ( void*      oldAddr,
    size_t     oldSize,
    size_t     newSize,
    ThrowMode  tm )

{
  void*  newAddr;


  if ( oldSize < LARGE_ALLOC || newSize < LARGE_ALLOC )
  {
    newAddr = alloc ( newSize, tm );

    if ( oldSize < newSize )
    {
      std::memcpy ( newAddr, oldAddr, oldSize );
    }
    else
    {
      std::memcpy ( newAddr, oldAddr, newSize );
    }

    dealloc ( oldAddr, oldSize );
  }
  else
  {
    size_t  n = align ( oldSize + 1 );
    byte*   p = (byte*) oldAddr;

    for ( size_t i = oldSize; i < n; i++ )
    {
      errors_ = (byte) (errors_ | (p[i] - CANARY_VALUE));
    }

    p -= ALIGNMENT;

    for ( size_t i = 0; i < ALIGNMENT; i++ )
    {
      errors_ = (byte) (errors_ | (p[i] - CANARY_VALUE));
    }

    if ( errors_ )
    {
      writeError_ ( JEM_FUNC, tm );
    }

    n = align ( newSize + 1 );
    p = (byte*) ::realloc ( p, n + ALIGNMENT );

    if ( ! p )
    {
      allocError_ ( JEM_FUNC, newSize, tm );
    }

    // Make sure that nothing funny happened in realloc().

    for ( size_t i = 0; i < ALIGNMENT; i++ )
    {
      errors_ = (byte) (errors_ | (p[i] - CANARY_VALUE));
    }

    p += ALIGNMENT;

    if ( newSize >= oldSize )
    {
      errors_ = (byte) (errors_ | (p[oldSize] - CANARY_VALUE));
    }

    if ( errors_ )
    {
      writeError_ ( JEM_FUNC, tm );
    }

    for ( size_t i = newSize; i < n; i++ )
    {
      p[i] = CANARY_VALUE;
    }

    newAddr = p;
  }

  return newAddr;
}


//-----------------------------------------------------------------------
//   setSize
//-----------------------------------------------------------------------


void ThreadCache::setSize ( size_t size )
{
  ItemList_*  list = freeLists_ + (FREE_LIST_COUNT - 1);
  Item_*      item;
  size_t      itemSize;


  cacheSize_ = size;

  for ( size_t i = (size_t) FREE_LIST_COUNT; i > 0; i-- )
  {
    itemSize        = i * ALIGNMENT;
    list->maxLength = (int) (size / (i * itemSize));
    size            = size - (size_t) list->maxLength * itemSize;

    while ( list->length > list->maxLength )
    {
      item        = list->first;
      list->first = list->first->next;

      std::free ( item );
      list->length--;
    }

    list--;
  }
}


//-----------------------------------------------------------------------
//   alloc_
//-----------------------------------------------------------------------


void* ThreadCache::alloc_ ( size_t size, ThrowMode tm )
{
  if ( errors_ )
  {
    writeError_ ( JEM_FUNC, tm );
  }

  void*  p = std::malloc  ( size );

  if ( ! p )
  {
    allocError_ ( JEM_FUNC, size, tm );
  }

  return p;
}


//-----------------------------------------------------------------------
//   writeError_
//-----------------------------------------------------------------------


void ThreadCache::writeError_

  ( const char*  where,
    ThrowMode    tm )

{
  errors_ = 0x0;

  if ( tm == NO_THROW )
  {
    jem::abort (
      where, "illegal memory write operation(s) detected"
    );
  }
  else
  {
    throw MemoryError ( where );
  }
}


//-----------------------------------------------------------------------
//   allocError_
//-----------------------------------------------------------------------


void ThreadCache::allocError_

  ( const char*  where,
    size_t       size,
    ThrowMode    tm )

{
  if ( tm == NO_THROW )
  {
    jem::abort   ( where, "out of memory" );
  }
  else
  {
    mallocFailed ( where, (lint) size );
  }
}


JEM_END_PACKAGE_BASE
