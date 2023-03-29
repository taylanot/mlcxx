
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
#include <jem/util/MallocAllocator.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class MallocAllocator
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MallocAllocator::MallocAllocator ( size_t itemSize ) noexcept :

  itemSize_ ( itemSize )

{}


MallocAllocator::MallocAllocator ( Self&& rhs ) noexcept :

  itemSize_ ( rhs.itemSize_ )

{}


MallocAllocator::~MallocAllocator()
{}


//-----------------------------------------------------------------------
//   alloc
//-----------------------------------------------------------------------


void* MallocAllocator::alloc ()
{
  return MemCache::alloc ( itemSize_ );
}


//-----------------------------------------------------------------------
//   dealloc
//-----------------------------------------------------------------------


void MallocAllocator::dealloc ( void* p )
{
  MemCache::dealloc ( p, itemSize_ );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


void MallocAllocator::swap ( MallocAllocator& rhs ) noexcept
{
  jem::swap ( itemSize_, rhs.itemSize_ );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits<util::MallocAllocator>
//=======================================================================


String TypeTraits<util::MallocAllocator>::whatis ()
{
  return String ( "a malloc-based allocator" );
}


String TypeTraits<util::MallocAllocator>::typeName ()
{
  return String ( "jem::util::MallocAllocator" );
}


JEM_END_PACKAGE_BASE
