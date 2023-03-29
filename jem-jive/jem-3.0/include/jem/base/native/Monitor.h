
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

#ifndef JEM_BASE_NATIVE_POSIXMONITOR_H
#define JEM_BASE_NATIVE_POSIXMONITOR_H

#include <jem/defines.h>

extern "C"
{
  #include <pthread.h>
}


JEM_BEGIN_PACKAGE_BASE


class Time;


//-----------------------------------------------------------------------
//   class PosixMonitor
//-----------------------------------------------------------------------


class PosixMonitor
{
 public:

                        PosixMonitor  ();
                       ~PosixMonitor  ();

  inline void           lock          ();
  inline void           unlock        ();
  inline void           wait          ();

  void                  wait

    ( const Time&         timeout );

  inline void           notify        ();


 private:

                        PosixMonitor  ( const PosixMonitor& );
  PosixMonitor&         operator =    ( const PosixMonitor& );


 private:

  pthread_mutex_t       mutex_;
  pthread_cond_t        cond_;

};


typedef PosixMonitor    NativeMonitor;





//#######################################################################
//   Implementation
//#######################################################################


inline void PosixMonitor::lock ()
{
  pthread_mutex_lock ( &mutex_ );
}


inline void PosixMonitor::unlock ()
{
  pthread_mutex_unlock ( &mutex_ );
}


inline void PosixMonitor::wait ()
{
  pthread_cond_wait ( &cond_, &mutex_ );
}


inline void PosixMonitor::notify ()
{
  pthread_cond_signal ( &cond_ );
}


JEM_END_PACKAGE_BASE

#endif
