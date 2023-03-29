
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

#ifndef JEM_BASE_COLLECTABLE_H
#define JEM_BASE_COLLECTABLE_H

#include <new>
#include <atomic>
#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>


JEM_BEGIN_PACKAGE_BASE


class WeakRefImp;
class WeakProxy;


//-----------------------------------------------------------------------
//   class Collectable
//-----------------------------------------------------------------------


class Collectable
{
 public:

  friend class            WeakRefImp;


  inline bool             isShared      () const noexcept;

  static inline void      addRef

    ( Collectable&          obj )                noexcept;

  static inline void      addRef

    ( Collectable*          obj )                noexcept;

  static inline void      delRef

    ( Collectable&          obj )                noexcept;

  static inline void      delRef

    ( Collectable*          obj )                noexcept;


 protected:

                          Collectable   ();
                          Collectable   ( const Collectable& );
  virtual                ~Collectable   ();

  void                    operator delete

    ( void*                 ptr,
      size_t                size );


 private:

  Collectable&            operator =    ( const Collectable& );

  static void             kill_

    ( Collectable*          obj )                noexcept;

  void*                   operator new

    ( size_t                size );


 private:

  std::atomic_int         refcount_;
  WeakProxy* volatile     proxy_;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isShared
//-----------------------------------------------------------------------


inline bool Collectable::isShared () const noexcept
{
  return (refcount_.load( std::memory_order_relaxed ) > 1);
}


//-----------------------------------------------------------------------
//   addRef
//-----------------------------------------------------------------------


inline void Collectable::addRef ( Collectable& obj ) noexcept
{
  obj.refcount_.fetch_add ( 1, std::memory_order_relaxed );
}


inline void Collectable::addRef ( Collectable* obj ) noexcept
{
  if ( obj )
  {
    obj->refcount_.fetch_add ( 1, std::memory_order_relaxed );
  }
}


//-----------------------------------------------------------------------
//   delRef
//-----------------------------------------------------------------------


inline void Collectable::delRef ( Collectable& obj ) noexcept
{
  if ( obj.refcount_.fetch_sub( 1, std::memory_order_release ) == 1 )
  {
    kill_ ( &obj );
  }
}


inline void Collectable::delRef ( Collectable* obj ) noexcept
{
  if ( obj &&
       obj->refcount_.fetch_sub( 1, std::memory_order_release ) == 1 )
  {
    kill_ ( obj );
  }
}


JEM_END_PACKAGE_BASE

#endif
