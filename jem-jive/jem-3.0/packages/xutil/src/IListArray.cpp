
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
#include <jem/base/limits.h>
#include <jem/base/utilities.h>
#include <jem/base/Error.h>
#include <jem/base/MemCache.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/xutil/IListArray.h>


JEM_BEGIN_PACKAGE( xutil )


//=======================================================================
//   class IListArray
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


IListArray::IListArray ( idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  size_    = 0;
  zlevel_  = 2;
  padding_ = (1_idx << zlevel_) / 2;
  lists_   = 0;

  initHeap_ ();

  if ( n > 0 )
  {
    resize ( n );
  }
}


IListArray::IListArray ( IListArray&& rhs ) :

  IListArray ()

{
  swap ( rhs );
}


IListArray::~IListArray ()
{
  idx_t  n = heap_.size;

  MemCache::dealloc ( lists_, (size_t) size_ * sizeof(*lists_) );
  MemCache::dealloc ( (heap_.addr - 1),
                      (size_t) (n + 2) * sizeof(*heap_.addr) );

  lists_     = 0;
  heap_.addr = 0;
}


//-----------------------------------------------------------------------
//   move operator
//-----------------------------------------------------------------------


IListArray& IListArray::operator = ( IListArray&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void IListArray::swap ( Self& rhs ) noexcept
{
  jem::swap ( size_,    rhs.size_    );
  jem::swap ( zlevel_,  rhs.zlevel_  );
  jem::swap ( padding_, rhs.padding_ );
  jem::swap ( lists_,   rhs.lists_   );
  jem::swap ( heap_,    rhs.heap_    );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void IListArray::clear ()
{
  const idx_t  heapSize  = heap_.size;
  const idx_t  listCount = size_;

  List_*       lists     = lists_;
  idx_t*       heap      = heap_.addr;


  for ( idx_t i = 0; i < heapSize; i++ )
  {
    heap[i] = -1;
  }

  for ( idx_t j = 0; j < listCount; j++ )
  {
    lists[j].size = 0;
  }
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void IListArray::resize ( idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  Heap_&  heap = heap_;

  List_*  lists;
  idx_t   i, k;


  if ( n == size_ )
  {
    return;
  }

  if ( n > size_ )
  {
    lists = (List_*)

      MemCache::realloc ( lists_,
                          (size_t) size_ * sizeof(*lists_),
                          (size_t) n     * sizeof(*lists_) );

    for ( i = size_; i < n; i++ )
    {
      lists[i].begin = 0;
      lists[i].size  = 0;
    }

    lists_ = lists;
  }
  else
  {
    idx_t*  p;
    idx_t   j;


    lists = lists_;

    for ( i = n; i < size_; i++ )
    {
      k = lists[i].size;
      p = heap.addr + lists[i].begin;

      switch ( k )
      {
      case 0:

        break;

      case 1:

        p[0] = -1;

        break;

      case 2:

        p[0] = p[1] = -1;

        break;

      case 3:

        p[0] = p[1] = p[2] = -1;

        break;

      case 4:

        p[0] = p[1] = p[2] = p[3] = -1;

        break;

      default:

        for ( j = 0; j < k; j++ )
        {
          p[j] = -1;
        }
      }
    }

    lists_ = (List_*)

      MemCache::realloc ( lists_,
                          (size_t) size_ * sizeof(*lists_),
                          (size_t) n     * sizeof(*lists_) );
  }

  size_     = n;
  lists     = lists_;
  heap.last = 0;

  for ( i = 0; i < n; i++ )
  {
    if ( (lists[i].begin + lists[i].size) > heap.last )
    {
      heap.last = lists[i].begin + lists[i].size;
    }
  }
}


//-----------------------------------------------------------------------
//   compress
//-----------------------------------------------------------------------


void IListArray::compress ()
{
  idx_t  n = minHeapSize_ ();

  n += n >> (zlevel_ + 1);

  if ( heap_.size > n )
  {
    resizeHeap_ ();
  }
}


//-----------------------------------------------------------------------
//   setZLevel
//-----------------------------------------------------------------------


void IListArray::setZLevel ( int zlevel )
{
  JEM_PRECHECK ( zlevel >= 0 );

  if ( zlevel > 16 )
  {
    zlevel = 16;
  }

  zlevel_  = zlevel;
  padding_ = (1 << zlevel) / 2;
}


//-----------------------------------------------------------------------
//   clearList
//-----------------------------------------------------------------------


void IListArray::clearList ( idx_t ilist )
{
  JEM_ASSERT ( ilist >= 0 && ilist < size() );

  List_&       list = lists_[ilist];
  idx_t*       p    = heap_.addr + list.begin;
  const idx_t  n    = list .size;

  idx_t        i;


  switch ( n )
  {
  case 0:

    break;

  case 1:

    p[0] = -1;

    break;

  case 2:

    p[0] = p[1] = -1;

    break;

  case 3:

    p[0] = p[1] = p[2] = -1;

    break;

  case 4:

    p[0] = p[1] = p[2] = p[3] = -1;

    break;

  default:

    for ( i = 0; i < n; i++ )
    {
      p[i] = -1;
    }
  }

  list.size = 0;
}


//-----------------------------------------------------------------------
//   pruneList
//-----------------------------------------------------------------------


void IListArray::pruneList ( idx_t ilist, const bool* mask )
{
  JEM_ASSERT ( ilist >= 0 && ilist < size() );

  List_&  list = lists_[ilist];
  idx_t*  p    = heap_.addr + list.begin;
  idx_t   n    = list .size;

  idx_t   i, j, k;


  for ( i = j = 0; i < n; i++ )
  {
    k = p[i];

    JEM_ASSERT ( k >= 0 );

    if ( ! mask[k] )
    {
      p[j++] = k;
    }
  }

  list.size = j;

  n -= j;
  p += j;

  switch ( n )
  {
  case 0:

    break;

  case 1:

    p[0] = -1;

    break;

  case 2:

    p[0] = p[1] = -1;

    break;

  case 3:

    p[0] = p[1] = p[2] = -1;

    break;

  case 4:

    p[0] = p[1] = p[2] = p[3] = -1;

    break;

  default:

    for ( i = 0; i < n; i++ )
    {
      p[i] = -1;
    }
  }
}


//-----------------------------------------------------------------------
//   addToList_
//-----------------------------------------------------------------------


void IListArray::addToList_ ( idx_t ilist, idx_t index )
{
  idx_t*  JEM_RESTRICT  src;
  idx_t*  JEM_RESTRICT  dest;

  Heap_&  heap  = heap_;
  List_&  list  = lists_[ilist];

  idx_t   begin = list.begin;
  idx_t   end   = begin + list.size;

  idx_t   i, k, n;


  if ( heap.addr[end] < 0 )
  {
    heap.addr[end++] = index;
    list.size++;

    if ( end > heap.last )
    {
      heap.last = end;
    }

    return;
  }

  // No space; check whether the heap must be resized

  n = list.size + 1;
  k = (n + padding_) >> zlevel_;

  if ( (heap.last + n + k) > heap.size )
  {
    resizeHeap_ ( n + k + (heap.size >> zlevel_) );

    begin = list.begin;

    if ( heap.addr[begin - 1] < 0 )
    {
      heap.addr[--begin] = index;
      list.begin         = begin;
      list.size++;
      return;
    }

    end = begin + list.size;

    if ( heap.addr[end] < 0 )
    {
      heap.addr[end++] = index;
      list.size++;

      if ( end > heap.last )
      {
        heap.last = end;
      }

      return;
    }
  }

  // Move the list

  src        = heap.addr + begin;
  begin      = heap.last + k;
  list.begin = begin;
  list.size  = n;
  heap.last  = begin + n;

  JEM_ASSERT ( heap.last <= heap.size );

  dest = heap.addr + begin;

  n--;

  switch ( n )
  {
  case 0:

    dest[0] = index;

    break;

  case 1:

    dest[0] = src[0];
    dest[1] = index;

    src [0] = -1;

    break;

  case 2:

    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = index;

    src [0] = -1;
    src [1] = -1;

    break;

  case 3:

    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = index;

    src [0] = -1;
    src [1] = -1;
    src [2] = -1;

    break;

  case 4:

    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
    dest[4] = index;

    src [0] = -1;
    src [1] = -1;
    src [2] = -1;
    src [3] = -1;

    break;

  default:

    std::memcpy ( dest, src, (size_t) n * sizeof(idx_t) );

    dest[n] = index;

    for ( i = 0; i < n; i++ )
    {
      src[i] = -1;
    }
  }
}


//-----------------------------------------------------------------------
//   initHeap_
//-----------------------------------------------------------------------


void IListArray::initHeap_ ()
{
  Heap_&  heap = heap_;
  idx_t   n    = 16_idx;

  idx_t   i;


  heap.size = n;
  heap.last = 0;
  heap.addr = (idx_t*)

    MemCache::alloc ( (size_t) (n + 2) * sizeof(*heap.addr) );

  heap.addr[0] = maxOf<idx_t> ();
  heap.addr++;
  heap.addr[n] = maxOf<idx_t> ();

  for ( i = 0; i < n; i++ )
  {
    heap.addr[i] = -1;
  }
}


//-----------------------------------------------------------------------
//   resizeHeap_
//-----------------------------------------------------------------------


void IListArray::resizeHeap_ ( idx_t extra )
{
  const idx_t*  JEM_RESTRICT  src;
  idx_t*        JEM_RESTRICT  dest;

  List_*        lists     = lists_;
  const idx_t*  oldHeap   = heap_.addr;

  const int     zlevel    = zlevel_;
  const idx_t   padding   = padding_;
  const idx_t   listCount = size_;

  idx_t*        newHeap;
  idx_t         newSize;
  idx_t         last;
  idx_t         i, n;


  newSize = n = minHeapSize_ ( extra );

  newHeap     = (idx_t*)

    MemCache::alloc ( (size_t) (n + 2) * sizeof(*newHeap) );

  newHeap[0]  = maxOf<idx_t> ();
  newHeap++;
  newHeap[n]  = maxOf<idx_t> ();

  for ( i = 0; i < newSize; i++ )
  {
    newHeap[i] = -1;
  }

  for ( i = last = 0; i < listCount; i++ )
  {
    List_&  list = lists[i];

    n     = list.size;
    last += (n + padding) >> zlevel;

    JEM_ASSERT ( last + n <= newSize );

    if ( n == 0 )
    {
      list.begin = last;
    }
    else
    {
      src  = oldHeap + list.begin;
      dest = newHeap + last;

      switch ( n )
      {
      case 1:

        dest[0] = src[0];

        break;

      case 2:

        dest[0] = src[0];
        dest[1] = src[1];

        break;

      case 3:

        dest[0] = src[0];
        dest[1] = src[1];
        dest[2] = src[2];

        break;

      case 4:

        dest[0] = src[0];
        dest[1] = src[1];
        dest[2] = src[2];
        dest[3] = src[3];

        break;

      default:

        std::memcpy ( dest, src, (size_t) n * sizeof(idx_t) );
      }

      list.begin = last;
      last      += n;
    }
  }

  n = heap_.size;

  MemCache::dealloc ( (heap_.addr - 1),
                      (size_t) (n + 2) * sizeof(*heap_.addr) );

  heap_.addr = newHeap;
  heap_.size = newSize;
  heap_.last = last;

  if ( (last + extra) > newSize )
  {
    throw jem::Error ( JEM_FUNC, "oops, heap too small" );
  }

  if ( newHeap[-1]      != maxOf<idx_t>() ||
       newHeap[newSize] != maxOf<idx_t>() )
  {
    throw jem::Error ( JEM_FUNC, "oops, corrupt heap" );
  }
}


//-----------------------------------------------------------------------
//   minHeapSize_
//-----------------------------------------------------------------------


idx_t IListArray::minHeapSize_ ( idx_t extra ) const
{
  using jem::max;

  const List_*  lists     = lists_;

  const int     zlevel    = zlevel_;
  const idx_t   padding   = padding_;
  const idx_t   listCount = size_;

  idx_t          k, m, n;
  idx_t          i;


  m = n = 0;

  for ( i = 0; i < listCount; i++ )
  {
    k  = lists[i].size;
    m += k;
    n += k + ((k + padding) >> zlevel);
  }

  n += max ( extra, (m     + padding) >> zlevel );
  n  = max ( n, (listCount + padding) >> zlevel );

  return max ( n, 32_idx );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


void                  decode

  ( DataInput&          in,
    IListArray&         a )

{
  IListArray  tmp;

  uidx_t      sz;
  int         zlevel;
  idx_t       size;
  idx_t       i, j, n;


  tmp.setZLevel ( 1 );

  decode ( in, zlevel, sz );

  size = (idx_t) sz;

  tmp.resize ( size );

  for ( i = 0; i < size; i++ )
  {
    decode ( in, sz );

    n = (idx_t) sz;

    for ( j = 0; j < n; j++ )
    {
      tmp.addToList ( i, 0 );
    }

    in.decode ( tmp.getList( i ), n );
  }

  tmp.setZLevel ( zlevel );
  tmp.compress  ();

  a.swap ( tmp );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


void                  encode

  ( DataOutput&         out,
    const IListArray&   a )

{
  const idx_t   size = a.size ();

  const idx_t*  ip;
  idx_t         i, n;


  encode ( out, a.getZLevel(), size );

  for ( i = 0; i < size; i++ )
  {
    ip = a.getList ( n, i );

    encode    ( out, n );
    out.encode ( ip, n );
  }
}


JEM_END_PACKAGE( xutil )
