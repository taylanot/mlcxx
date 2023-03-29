
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

#ifndef JEM_XUTIL_ILISTARRAY_H
#define JEM_XUTIL_ILISTARRAY_H

#include <jem/base/assert.h>
#include <jem/xutil/import.h>


JEM_BEGIN_PACKAGE( xutil )


//-----------------------------------------------------------------------
//   class IListArray
//-----------------------------------------------------------------------


class IListArray
{
 public:

  typedef IListArray    Self;


  explicit              IListArray

    ( idx_t               n = 0 );

                        IListArray

    ( IListArray&&        rhs );

                       ~IListArray    ();

  IListArray&           operator =

    ( IListArray&&        rhs )                noexcept;

  void                  swap

    ( Self&               rhs )                noexcept;

  inline idx_t          size          () const noexcept;
  void                  clear         ();

  void                  resize

    ( idx_t               n );

  void                  compress      ();

  void                  setZLevel

    ( int                 zlevel );

  inline int            getZLevel     () const noexcept;

  inline idx_t          listSize

    ( idx_t               ilist )        const;

  inline idx_t*         getList

    ( idx_t               ilist );

  inline const idx_t*   getList

    ( idx_t               ilist )        const;

  inline idx_t*         getList

    ( idx_t&              n,
      idx_t               ilist );

  inline const idx_t*   getList

    ( idx_t&              n,
      idx_t               ilist )        const;

  inline void           addToList

    ( idx_t               ilist,
      idx_t               index );

  void                  clearList

    ( idx_t               ilist );

  void                  pruneList

    ( idx_t               ilist,
      const bool*         mask );


 private:

                        IListArray

    ( const Self&         rhs );

  Self&                 operator =

    ( const Self&         rhs );

  void                  addToList_

    ( idx_t               ilist,
      idx_t               index );

  void                  initHeap_     ();

  void                  resizeHeap_

    ( idx_t               extra = 0 );

  idx_t                 minHeapSize_

    ( idx_t               extra = 0 )    const;


 private:

  struct                List_
  {
    idx_t                 begin;
    idx_t                 size;
  };

  struct                Heap_
  {
    idx_t                 size;
    idx_t                 last;
    idx_t*                addr;
  };

  idx_t                 size_;
  int                   zlevel_;
  idx_t                 padding_;

  List_*                lists_;
  Heap_                 heap_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void             swap

  ( IListArray&           lhs,
    IListArray&           rhs ) noexcept;

void                    decode

  ( DataInput&            in,
    IListArray&           a );

void                    encode

  ( DataOutput&           out,
    const IListArray&     a );






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class IListArray
//=======================================================================

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t IListArray::size () const noexcept
{
  return size_;
}


//-----------------------------------------------------------------------
//   getZLevel
//-----------------------------------------------------------------------


inline int IListArray::getZLevel () const noexcept
{
  return zlevel_;
}


//-----------------------------------------------------------------------
//   listSize
//-----------------------------------------------------------------------


inline idx_t IListArray::listSize ( idx_t ilist ) const
{
  JEM_ASSERT ( ilist >= 0 && ilist < size() );

  return lists_[ilist].size;
}


//-----------------------------------------------------------------------
//   getList
//-----------------------------------------------------------------------


inline idx_t* IListArray::getList ( idx_t ilist )
{
  JEM_ASSERT ( ilist >= 0 && ilist < size() );

  return (heap_.addr + lists_[ilist].begin);
}


inline const idx_t* IListArray::getList ( idx_t ilist ) const
{
  JEM_ASSERT ( ilist >= 0 && ilist < size() );

  return (heap_.addr + lists_[ilist].begin);
}


inline idx_t* IListArray::getList ( idx_t& n, idx_t ilist )
{
  JEM_ASSERT ( ilist >= 0 && ilist < size() );

  n = lists_[ilist].size;

  return (heap_.addr + lists_[ilist].begin);
}


inline const idx_t* IListArray::getList ( idx_t& n, idx_t ilist ) const
{
  JEM_ASSERT ( ilist >= 0 && ilist < size() );

  n = lists_[ilist].size;

  return (heap_.addr + lists_[ilist].begin);
}


//-----------------------------------------------------------------------
//   addToList
//-----------------------------------------------------------------------


inline void IListArray::addToList ( idx_t ilist, idx_t index )
{
  JEM_ASSERT ( ilist >= 0 && ilist < size() && index >= 0 );

  List_&  list = lists_[ilist];

  if ( heap_.addr[list.begin - 1] < 0 )
  {
    heap_.addr[--list.begin] = index;
    list .size++;
  }
  else
  {
    addToList_ ( ilist, index );
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void     swap

  ( IListArray&   lhs,
    IListArray&   rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( xutil )

#endif
