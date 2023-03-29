
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

#ifndef JEM_BASE_NATIVE_POSIXTHREAD_H
#define JEM_BASE_NATIVE_POSIXTHREAD_H

#include <jem/base/Object.h>

extern "C"
{
  #include <pthread.h>
}


JEM_BEGIN_PACKAGE_BASE


class Thread;
class ThreadCache;
class ThreadStart;


//-----------------------------------------------------------------------
//   class PosixThread
//-----------------------------------------------------------------------


class PosixThread : public Object
{
 public:

  typedef PosixThread     Self;
  typedef Object          Super;
  typedef ThreadCache     Cache;


                          PosixThread     ();
                         ~PosixThread     ();

  void                    interrupt       ();
  void                    clearInterrupt  ();

  static void             init            ();

  static void             start

    ( ThreadStart*          start );

  static void             yield

    ( lint                  nsecs )          noexcept;

  static inline Thread*   getThread       () noexcept;

  static inline void      setThread

    ( Thread*               thread );

  static inline Cache*    getCache        ();

  static inline void      setCache

    ( Cache*                cache );


 private:

  static int              sigint_;
  static
    thread_local void*    thread_;
  static
    thread_local void*    cache_;

  pthread_t               tid_;

};


typedef PosixThread       NativeThread;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getThread
//-----------------------------------------------------------------------


inline Thread* PosixThread::getThread () noexcept
{
  return (Thread*) thread_;
}


//-----------------------------------------------------------------------
//   setThread
//-----------------------------------------------------------------------


inline void PosixThread::setThread ( Thread* t )
{
  thread_ = t;
}


//-----------------------------------------------------------------------
//   getCache
//-----------------------------------------------------------------------


inline ThreadCache* PosixThread::getCache ()
{
  return (Cache*) cache_;
}


//-----------------------------------------------------------------------
//   setCache
//-----------------------------------------------------------------------


inline void PosixThread::setCache ( Cache* c )
{
  cache_ = c;
}


JEM_END_PACKAGE_BASE

#endif
