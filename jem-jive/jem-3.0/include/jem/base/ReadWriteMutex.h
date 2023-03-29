
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

#ifndef JEM_BASE_READWRITEMUTEX_H
#define JEM_BASE_READWRITEMUTEX_H

#include <jem/base/SpinLock.h>


JEM_BEGIN_PACKAGE_BASE


class ThreadNode;


//-----------------------------------------------------------------------
//   class ReadWriteMutex
//-----------------------------------------------------------------------


class ReadWriteMutex
{
 public:

                      ReadWriteMutex  ();
                     ~ReadWriteMutex  ();

  inline bool         lockedRead      () const noexcept;
  bool                tryLockRead     ()       noexcept;
  void                lockRead        ();
  void                unlockRead      ();

  inline bool         lockedWrite     () const noexcept;
  bool                tryLockWrite    ()       noexcept;
  void                lockWrite       ();
  void                unlockWrite     ();


 private:

  static const int    MAX_WRITE_COUNT_;

  SpinLock            spinlock_;
  ThreadNode*         readQueue_;
  ThreadNode*         writeQueue_;
  int                 readCount_;
  int                 writeCount_;

};


//-----------------------------------------------------------------------
//   class ReadLock
//-----------------------------------------------------------------------


class ReadLock
{
 public:

  explicit inline     ReadLock ( const ReadWriteMutex&  mutex );
  inline             ~ReadLock ();


 private:

  ReadWriteMutex&     mutex_;

};


//-----------------------------------------------------------------------
//   class WriteLock
//-----------------------------------------------------------------------


class WriteLock
{
 public:

  explicit inline     WriteLock ( const ReadWriteMutex& mutex );
  inline             ~WriteLock ();


 private:

  ReadWriteMutex&     mutex_;

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ReadWriteMutex
//=======================================================================


inline bool ReadWriteMutex::lockedRead () const noexcept
{
  return (readCount_ != 0);
}


inline bool ReadWriteMutex::lockedWrite () const noexcept
{
  return (! readCount_ && writeCount_);
}


//=======================================================================
//   class ReadLock
//=======================================================================


inline ReadLock::ReadLock ( const ReadWriteMutex& mutex ) :

  mutex_ ( const_cast<ReadWriteMutex&>( mutex ) )

{
  mutex_.lockRead ();
}


inline ReadLock::~ReadLock ()
{
  mutex_.unlockRead ();
}


//=======================================================================
//   class WriteLock
//=======================================================================


inline WriteLock::WriteLock ( const ReadWriteMutex& mutex ) :

  mutex_ ( const_cast<ReadWriteMutex&>( mutex ) )

{
  mutex_.lockWrite ();
}


inline WriteLock::~WriteLock ()
{
  mutex_.unlockWrite ();
}


JEM_END_PACKAGE_BASE

#endif
