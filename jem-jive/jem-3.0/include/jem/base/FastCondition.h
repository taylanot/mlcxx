
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

#ifndef JEM_BASE_FASTCONDITION_H
#define JEM_BASE_FASTCONDITION_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE_BASE


class Time;
class Mutex;
class ThreadNode;


//-----------------------------------------------------------------------
//   class FastCondition
//-----------------------------------------------------------------------


class FastCondition
{
 public:

                      FastCondition ();
                     ~FastCondition ();

  inline void         wait

    ( const Mutex&      mutex );

  inline void         wait

    ( const Mutex&      mutex,
      const Time&       timeout );

  inline void         wait

    ( bool&             cancelled,
      const Mutex&      mutex );

  inline void         wait

    ( bool&             cancelled,
      const Mutex&      mutex,
      const Time&       timeout );

  inline void         waitNoCancel

    ( const Mutex&      mutex );

  inline void         notify

    ( const Mutex&      mutex );

  inline void         notifyAll

    ( const Mutex&      mutex );


 private:

  void                wait_

    ( Mutex&            mutex,
      bool*             cancelled,
      const Time*       timeout );

  void                waitNoCancel_

    ( Mutex&            mutex );

  void                notify_

    ( Mutex&            mutex );

  void                notifyAll_

    ( Mutex&            mutex );


 private:

  ThreadNode*         queue_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


inline void FastCondition::wait ( const Mutex& mutex )
{
  wait_ ( const_cast<Mutex&>( mutex ), nullptr, nullptr );
}


inline void FastCondition::wait

  ( const Mutex&  mutex,
    const Time&   timeout )

{
  wait_ ( const_cast<Mutex&>( mutex ), nullptr, &timeout );
}


inline void FastCondition::wait

  ( bool&         cancelled,
    const Mutex&  mutex )
{
  cancelled = false;

  wait_ ( const_cast<Mutex&>( mutex ), &cancelled, nullptr );
}


inline void FastCondition::wait

  ( bool&         cancelled,
    const Mutex&  mutex,
    const Time&   timeout )

{
  cancelled = false;

  wait_ ( const_cast<Mutex&>( mutex ), &cancelled, &timeout );
}


//-----------------------------------------------------------------------
//   waitNoCancel
//-----------------------------------------------------------------------


inline void FastCondition::waitNoCancel ( const Mutex& mutex )
{
  waitNoCancel_ ( const_cast<Mutex&>( mutex ) );
}


//-----------------------------------------------------------------------
//   notify
//-----------------------------------------------------------------------


inline void FastCondition::notify ( const Mutex& mutex )
{
  if ( queue_ )
  {
    notify_ ( const_cast<Mutex&>( mutex ) );
  }
}


//-----------------------------------------------------------------------
//   notifyAll
//-----------------------------------------------------------------------


inline void FastCondition::notifyAll ( const Mutex& mutex )
{
  if ( queue_ )
  {
    notifyAll_ ( const_cast<Mutex&>( mutex ) );
  }
}


JEM_END_PACKAGE_BASE

#endif
