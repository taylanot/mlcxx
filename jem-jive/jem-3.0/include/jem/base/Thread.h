
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

#ifndef JEM_BASE_THREAD_H
#define JEM_BASE_THREAD_H

#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE_BASE


class Time;
class ThreadLib;
class MainThread;
class ThreadState;


//-----------------------------------------------------------------------
//   class Thread
//-----------------------------------------------------------------------


class Thread : public Object
{
 public:

  JEM_DECLARE_CLASS     ( Thread, Object );

  friend class            ThreadLib;
  friend class            MainThread;

  static const int        NULL_KEY = -1;


                          Thread        ();

  virtual void            run           () = 0;

  void                    start         ();
  void                    join          ();

  void                    join

    ( const Time&           timeout );

  void                    cancel        ();
  bool                    isCancelled   () const;
  bool                    isAlive       () const;
  static Thread*          getCurrent    ();
  static bool             cancelled     ();

  static bool             allowCancel

    ( bool                  enable  );

  static void             sleep

    ( const Time&           timeout );

  static int              newLocalKey   ();

  static Ref<Object>      getLocal

    ( int                   key );

  static void             setLocal

    ( int                   key,
      const Ref<Object>&    obj );


 protected:

  virtual                ~Thread        ();


 private:

  enum                    MainTag_
  {
                            MAIN_TAG_
  };


  explicit                Thread

    ( MainTag_              tag );

  void                    join_

    ( const Time*           timeout );

  void                    die_          ();


 private:

  ThreadState*            state_;

};


JEM_END_PACKAGE_BASE

#endif
