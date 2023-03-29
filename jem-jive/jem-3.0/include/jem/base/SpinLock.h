
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

#ifndef JEM_BASE_SPINLOCK_H
#define JEM_BASE_SPINLOCK_H

#include <atomic>
#include <jem/defines.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class SpinLock
//-----------------------------------------------------------------------


class SpinLock
{
 public:

  inline                SpinLock   () noexcept;

  inline bool           locked     () const noexcept;
  inline bool           tryLock    () noexcept;
  inline void           lock       () noexcept;
  inline void           unlock     () noexcept;


 private:

                        SpinLock   ( const SpinLock& );
  SpinLock&             operator = ( const SpinLock& );

  void                  spin_      ();


 private:

  static const lint     MAX_SPIN_CYCLES_;
  static const lint     MAX_SLEEP_TIME_;

  std::atomic_int       locked_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline SpinLock::SpinLock () noexcept :

  locked_ ( 0 )

{}


//-----------------------------------------------------------------------
//   locked
//-----------------------------------------------------------------------


inline bool SpinLock::locked () const noexcept
{
  return (locked_.load( std::memory_order_relaxed ) != 0);
}


//-----------------------------------------------------------------------
//   tryLock
//-----------------------------------------------------------------------


inline bool SpinLock::tryLock () noexcept
{
  if ( locked_.exchange( 1, std::memory_order_relaxed ) == 0 )
  {
    std::atomic_thread_fence ( std::memory_order_acquire );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   lock
//-----------------------------------------------------------------------


inline void SpinLock::lock () noexcept
{
  if ( locked_.exchange( 1, std::memory_order_relaxed ) == 0 )
  {
    std::atomic_thread_fence ( std::memory_order_acquire );
  }
  else
  {
    spin_ ();
  }
}


//-----------------------------------------------------------------------
//   unlock
//-----------------------------------------------------------------------


inline void SpinLock::unlock () noexcept
{
  locked_.store ( 0, std::memory_order_release );
}


JEM_END_PACKAGE_BASE

#endif
