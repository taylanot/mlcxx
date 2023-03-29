
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

#ifndef JEM_MT_WORKPOOL_H
#define JEM_MT_WORKPOOL_H

#include <jem/util/ObjFlex.h>


namespace jem
{
  class Time;
}


JEM_BEGIN_PACKAGE( mt )


//-----------------------------------------------------------------------
//   class WorkPool
//-----------------------------------------------------------------------


class WorkPool : public Object
{
 public:

  JEM_DECLARE_CLASS     ( WorkPool, Object );

  class                   Job;
  class                   Task;


  explicit                WorkPool

    ( int                   wmax );

  int                     maxWorkers    () const noexcept;
  int                     workerCount   () const noexcept;

  Ref<Job>                newJob

    ( const Ref<Task>&      task );

  void                    waitAll       ();

  bool                    waitAll

    ( const Time&           timeout );

  void                    cancelAll     ();
  void                    killWorkers   ();

  int                     killWorkers

    ( const Time&           timeout );


 protected:

  virtual                ~WorkPool      ();


 private:

  void                    newWorker_    ();


 private:

  class                   Arena_;
  class                   Job_;
  class                   Worker_;

  friend class            Arena_;
  friend class            Job_;
  friend class            Worker_;


  Ref<Arena_>             arena_;
  util::ObjFlex           workers_;
  int                     maxWorkers_;

};


//-----------------------------------------------------------------------
//   class WorkPool::Job
//-----------------------------------------------------------------------


class WorkPool::Job : public Object
{
 public:

  virtual void            start     ()       = 0;

  virtual bool            start

    ( const Time&           timeout )        = 0;

  virtual void            cancel    ()       = 0;
  virtual void            wait      ()       = 0;

  virtual bool            wait

    ( const Time&           timeout )        = 0;

  virtual bool            test      () const = 0;


 protected:

  virtual                ~Job       ();

};


//-----------------------------------------------------------------------
//   class WorkPool::Task
//-----------------------------------------------------------------------


class WorkPool::Task : public Object
{
 public:

  virtual void            run       ()       = 0;


 protected:

  virtual                ~Task      ();

};


JEM_END_PACKAGE( mt )

#endif
