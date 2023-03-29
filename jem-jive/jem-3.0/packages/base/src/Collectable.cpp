
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


#include <jem/base/Error.h>
#include <jem/base/WeakProxy.h>
#include <jem/base/Collectable.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Collectable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Collectable::Collectable () :

  refcount_ ( 1 ),
  proxy_    ( nullptr )

{}


Collectable::Collectable ( const Collectable& ) :

  refcount_ ( 1 ),
  proxy_    ( nullptr )

{}


Collectable::~Collectable ()
{}


//-----------------------------------------------------------------------
//   operator delete
//-----------------------------------------------------------------------


void Collectable::operator delete ( void* p, size_t size )
{
  MemCache::dealloc ( p, size );
}


//-----------------------------------------------------------------------
//   kill_
//-----------------------------------------------------------------------


void Collectable::kill_ ( Collectable* c ) noexcept
{
  std::atomic_thread_fence ( std::memory_order_acquire );

  // This is thread safe since the member proxy_ can not be modified
  // once the reference count of this object has dropped to zero.

  if ( c->proxy_ )
  {
    WeakProxy*  proxy = c->proxy_;
    int         refcount;

    proxy->spinlock.lock ();

    proxy->collectable = nullptr;
    proxy->refcount--;
    refcount = proxy->refcount;

    proxy->spinlock.unlock ();

    if ( ! refcount )
    {
      delete proxy;
    }
  }

  delete c;
}


//-----------------------------------------------------------------------
//   disabled operator new
//-----------------------------------------------------------------------


void* Collectable::operator new ( size_t )
{
  static byte  dummy[1];

  throw Error (
    JEM_FUNC,
    "unexpected call of Collectable::operator new"
  );

  // This is to keep g++ happy.

  return dummy;
}


JEM_END_PACKAGE_BASE
