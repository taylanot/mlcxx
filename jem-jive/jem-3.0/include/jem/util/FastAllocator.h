
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

#ifndef JEM_UTIL_FASTALLOCATOR_H
#define JEM_UTIL_FASTALLOCATOR_H

#include <new>
#include <jem/base/TypeTraits.h>
#include <jem/base/AllocatorUtils.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class FastAllocator
//-----------------------------------------------------------------------


class FastAllocator : private AllocatorUtils
{
 public:

  explicit              FastAllocator

    ( size_t              itemSize )       noexcept;

                        FastAllocator

    ( FastAllocator&&     rhs )            noexcept;

                       ~FastAllocator   ();

   void*                alloc           ();

   void                 dealloc

    ( void*               p );

  void                  swap

    ( FastAllocator&      rhs )            noexcept;


 private:

                        FastAllocator   ( const FastAllocator& );
  FastAllocator&        operator =      ( const FastAllocator& );

  void                  allocChunk_     ();

  static size_t         getItemSize_

    ( size_t              itemSize );


 private:

  struct                Item_
  {
    Item_*                next;
  };


  class                 Chunk_;
  friend class          Chunk_;

  static const size_t   MAX_CHUNK_SIZE_;

  size_t                itemSize_;
  Item_*                freeList_;
  Chunk_*               chunkList_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void             swap

  ( FastAllocator&        lhs,
    FastAllocator&        rhs ) noexcept;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void swap ( FastAllocator& lhs, FastAllocator& rhs ) noexcept
{
  lhs.swap( rhs );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< util::FastAllocator >
//=======================================================================


template <>

  class TypeTraits<util::FastAllocator> : public BasicTypeTraits

{
 public:

  static String  whatis   ();
  static String  typeName ();

};


JEM_END_PACKAGE_BASE

#endif
