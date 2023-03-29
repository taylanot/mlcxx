
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

#ifndef JEM_BASE_MUTEX_H
#define JEM_BASE_MUTEX_H

#include <jem/base/assert.h>
#include <jem/base/SpinLock.h>


JEM_BEGIN_PACKAGE_BASE


class Condition;
class FastCondition;
class ThreadNode;


//-----------------------------------------------------------------------
//   class Mutex
//-----------------------------------------------------------------------


class Mutex
{
 public:

  friend class        Condition;
  friend class        FastCondition;


                      Mutex     ();
                     ~Mutex     ();

  inline bool         locked    () const noexcept;
  inline bool         tryLock   ()       noexcept;
  inline void         lock      ();
  inline void         unlock    ();


 private:

  void                lock_     ();
  void                unlock_   ();


 private:

  SpinLock            spinlock_;
  ThreadNode*         queue_;

  bool                locked_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   locked
//-----------------------------------------------------------------------


inline bool Mutex::locked () const noexcept
{
  return locked_;
}


//-----------------------------------------------------------------------
//   tryLock
//-----------------------------------------------------------------------


inline bool Mutex::tryLock () noexcept
{
  bool  result;

  spinlock_.lock ();

  if ( ! locked_ )
  {
    result = locked_ = true;
  }
  else
  {
    result = false;
  }

  spinlock_.unlock ();

  return result;
}


//-----------------------------------------------------------------------
//   lock
//-----------------------------------------------------------------------


inline void Mutex::lock ()
{
  spinlock_.lock ();

  if ( locked_ )
  {
    lock_ ();
  }
  else
  {
    locked_ = true;
  }

  spinlock_.unlock ();
}


//-----------------------------------------------------------------------
//   unlock
//-----------------------------------------------------------------------


void Mutex::unlock ()
{
  JEM_ASSERT_NOTHROW ( locked() );

  spinlock_.lock ();

  if ( queue_ )
  {
    unlock_ ();
  }
  else
  {
    locked_ = false;
  }

  spinlock_.unlock ();
}


JEM_END_PACKAGE_BASE

#endif
