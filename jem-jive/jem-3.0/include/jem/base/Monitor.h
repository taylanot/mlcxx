
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

#ifndef JEM_BASE_MONITOR_H
#define JEM_BASE_MONITOR_H

#include <jem/base/Mutex.h>
#include <jem/base/FastCondition.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Monitor
//-----------------------------------------------------------------------


class Monitor
{
 public:

  inline              Monitor       ();

  inline bool         locked        () const noexcept;
  inline bool         tryLock       ()       noexcept;
  inline void         lock          ();
  inline void         unlock        ();
  inline void         wait          ();

  inline void         wait

    ( const Time&       timeout );

  inline void         wait

    ( bool&             cancelled );

  inline void         wait

    ( bool&             cancelled,
      const Time&       timeout );

  inline void         waitNoCancel  ();
  inline void         notify        ();
  inline void         notifyAll     ();


 private:

  Mutex               mutex_;
  FastCondition       condition_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline Monitor::Monitor ()
{}


//-----------------------------------------------------------------------
//   locked
//-----------------------------------------------------------------------


inline bool Monitor::locked () const noexcept
{
  return mutex_.locked ();
}


//-----------------------------------------------------------------------
//   tryLock
//-----------------------------------------------------------------------


inline bool Monitor::tryLock () noexcept
{
  return mutex_.tryLock ();
}


//-----------------------------------------------------------------------
//   lock
//-----------------------------------------------------------------------


inline void Monitor::lock ()
{
  mutex_.lock ();
}


//-----------------------------------------------------------------------
//   unlock
//-----------------------------------------------------------------------


inline void Monitor::unlock ()
{
  mutex_.unlock ();
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


inline void Monitor::wait ()
{
  condition_.wait ( mutex_ );
}


inline void Monitor::wait ( const Time& timeout )
{
  condition_.wait ( mutex_, timeout );
}


inline void Monitor::wait ( bool& cancelled )
{
  condition_.wait ( cancelled, mutex_ );
}


inline void Monitor::wait

  ( bool&        cancelled,
    const Time&  timeout )

{
  condition_.wait ( cancelled, mutex_, timeout );
}


//-----------------------------------------------------------------------
//   waitNoCancel
//-----------------------------------------------------------------------


inline void Monitor::waitNoCancel ()
{
  condition_.waitNoCancel ( mutex_ );
}


//-----------------------------------------------------------------------
//   notify
//-----------------------------------------------------------------------


inline void Monitor::notify ()
{
  condition_.notify ( mutex_ );
}


//-----------------------------------------------------------------------
//   notifyAll
//-----------------------------------------------------------------------


inline void Monitor::notifyAll ()
{
  condition_.notifyAll ( mutex_ );
}


JEM_END_PACKAGE_BASE

#endif
