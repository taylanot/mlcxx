
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


#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include <jem/base/Mutex.h>
#include <jem/base/Condition.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/CancellationScope.h>
#include <jem/mt/WorkPool.h>


JEM_DEFINE_CLASS( jem::mt::WorkPool );


JEM_BEGIN_PACKAGE( mt )


//=======================================================================
//   class WorkPool::Arena_
//=======================================================================


class WorkPool::Arena_ : public Collectable
{
 public:

  inline                  Arena_  ();


 public:

  Mutex                   mutex;
  Condition               master;
  Condition               workers;
  Job_*                   job;
  int                     runCount;
  int                     jobCount;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline WorkPool::Arena_::Arena_ ()
{
  job      = 0;
  runCount = 0;
  jobCount = 0;
}


//=======================================================================
//   class WorkPool::Job_
//=======================================================================


class WorkPool::Job_ : public Job
{
 public:

  typedef WorkPool        Pool;
  friend  class           Worker_;


  inline                  Job_

    ( const Ref<Task>&      task,
      const Ref<Pool>&      pool );

  virtual void            start     ()       override;

  virtual bool            start

    ( const Time&           timeout )        override;

  virtual void            cancel    ()       override;
  virtual void            wait      ()       override;

  virtual bool            wait

    ( const Time&           timeout )        override;

  virtual bool            test      () const override;


 private:

  Ref<Task>               task_;
  Ref<Pool>               pool_;
  Condition               master_;
  Thread*                 worker_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline WorkPool::Job_::Job_

  ( const Ref<Task>&  task,
    const Ref<Pool>&  pool ) :

    task_ ( task ),
    pool_ ( pool )

{
  worker_ = 0;
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void WorkPool::Job_::start ()
{
  Pool&    p = * pool_;
  Arena_&  a = * p.arena_;


  a.mutex.lock ();

  if ( worker_ )
  {
    a.mutex.unlock ();
    return;
  }

  try
  {
    a.job = this;

    if ( a.jobCount >= a.runCount && a.runCount < p.maxWorkers_ )
    {
      p.newWorker_ ();
    }

    a.workers.notify ();

    do
    {
      a.master.wait ( a.mutex );
    }
    while ( a.job );
  }
  catch ( ... )
  {
    a.job = 0;
    a.mutex.unlock ();
    throw;
  }

  a.mutex.unlock ();
}


//-----------------------------------------------------------------------
//   start ( timeout )
//-----------------------------------------------------------------------


bool WorkPool::Job_::start ( const Time& timeout )
{
  Pool&    p = * pool_;
  Arena_&  a = * p.arena_;

  bool     r = false;


  a.mutex.lock ();

  if ( worker_ )
  {
    a.mutex.unlock ();
    return r;
  }

  try
  {
    a.job = this;

    if ( a.jobCount >= a.runCount && a.runCount < p.maxWorkers_ )
    {
      p.newWorker_ ();
    }

    a.workers.notify ();
    a.master .wait   ( a.mutex, timeout );
  }
  catch ( ... )
  {
    a.job = 0;
    a.mutex.unlock ();

    throw;
  }

  if ( a.job == 0 )
  {
    r = true;
  }
  else
  {
    a.job = 0;
  }

  a.mutex.unlock ();

  return r;
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------


void WorkPool::Job_::cancel ()
{
  Lock<Mutex>  lock ( pool_->arena_->mutex );

  if ( worker_ )
  {
    worker_->cancel ();
  }
}


//-----------------------------------------------------------------------
//   wait
//-----------------------------------------------------------------------


void WorkPool::Job_::wait ()
{
  Arena_&      a    = * pool_->arena_;

  Lock<Mutex>  lock ( a.mutex );

  while ( worker_ )
  {
    master_.wait ( a.mutex );
  }
}


//-----------------------------------------------------------------------
//   wait ( timeout )
//-----------------------------------------------------------------------


bool WorkPool::Job_::wait ( const Time& timeout )
{
  Arena_&  a = * pool_->arena_;

  bool     r = false;

  {
    Lock<Mutex>  lock ( a.mutex );

    if ( worker_ )
    {
      master_.wait ( a.mutex, timeout );
    }

    r = (worker_ == 0);
  }

  return r;
}


//-----------------------------------------------------------------------
//   test
//-----------------------------------------------------------------------


bool WorkPool::Job_::test () const
{
  return (worker_ == 0);
}


//=======================================================================
//   class WorkPool::Worker_
//=======================================================================


class WorkPool::Worker_ : public Thread
{
 public:

  explicit                Worker_

    ( const Ref<Arena_>&    arena );

  virtual void            run   () override;
  void                    kill  ();


 private:

  Ref<Arena_>             arena_;

  volatile bool           killed_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


WorkPool::Worker_::Worker_ ( const Ref<Arena_>& arena ) :

  arena_ ( arena )

{
  killed_ = false;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void WorkPool::Worker_::run ()
{
  Arena_&    a = * arena_;

  Ref<Job_>  job;


  allowCancel  ( true );

  a.mutex.lock ();

  a.runCount++;

 again:

  try
  {
    while ( ! killed_ )
    {
      while ( ! a.job )
      {
        a.workers.wait ( a.mutex );
      }

      job          = a.job;
      a.job        = 0;
      job->worker_ = this;

      a.jobCount++;

      a.master.notify ();
      a.mutex .unlock ();

      try
      {
        job->task_->run ();
      }
      catch ( ... ) {}

      a.mutex.lock ();

      job->worker_ = 0;

      job->master_.notify ();

      job = nullptr;

      if ( (--a.jobCount) == 0 )
      {
        a.master.notify ();
      }
    }
  }
  catch ( ... )
  {}

  if ( ! killed_ )
  {
    goto again;
  }

  killed_ = false;

  // Don't exit if there is a job pending. Otherwise a deadlock
  // situtation may occur.

  if ( a.job )
  {
    goto again;
  }

  a.runCount--;

  a.mutex.unlock ();
}


//-----------------------------------------------------------------------
//   kill
//-----------------------------------------------------------------------


void WorkPool::Worker_::kill ()
{
  arena_->mutex.lock   ();
  killed_ = true;
  arena_->mutex.unlock ();

  cancel ();
}


//=======================================================================
//   class WorkPool
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


WorkPool::WorkPool ( int wmax )
{
  JEM_PRECHECK ( wmax > 0 );

  arena_      = newInstance<Arena_> ();
  maxWorkers_ = wmax;
}


WorkPool::~WorkPool ()
{
  for ( int i = (int) workers_.size() - 1; i >= 0; i-- )
  {
    workers_.getAs<Worker_>(i)->kill ();
  }
}


//-----------------------------------------------------------------------
//   maxWorkers
//-----------------------------------------------------------------------


int WorkPool::maxWorkers () const noexcept
{
  return maxWorkers_;
}


//-----------------------------------------------------------------------
//   workerCount
//-----------------------------------------------------------------------


int WorkPool::workerCount () const noexcept
{
  return (int) workers_.size ();
}


//-----------------------------------------------------------------------
//   newJob
//-----------------------------------------------------------------------


Ref<WorkPool::Job> WorkPool::newJob ( const Ref<Task>& task )
{
  JEM_PRECHECK ( task );

  return newInstance<Job_> ( task, this );
}


//-----------------------------------------------------------------------
//   waitAll
//-----------------------------------------------------------------------


void WorkPool::waitAll ()
{
  Arena_&      a = * arena_;

  Lock<Mutex>  lock ( a.mutex );

  while ( a.jobCount )
  {
    a.master.wait ( a.mutex );
  }
}


//-----------------------------------------------------------------------
//   waitAll ( timeout )
//-----------------------------------------------------------------------


bool WorkPool::waitAll ( const Time& timeout )
{
  Arena_&      a = * arena_;

  Lock<Mutex>  lock ( a.mutex );

  if ( a.jobCount )
  {
    a.master.wait ( a.mutex, timeout );
  }

  return (a.jobCount == 0);
}


//-----------------------------------------------------------------------
//   cancelAll
//-----------------------------------------------------------------------


void WorkPool::cancelAll ()
{
  for ( int i = (int) workers_.size() - 1; i >= 0; i-- )
  {
    workers_.getAs<Worker_>(i)->cancel ();
  }
}


//-----------------------------------------------------------------------
//   killWorkers
//-----------------------------------------------------------------------


void WorkPool::killWorkers ()
{
  int  i;

  for ( i = (int) workers_.size() - 1; i >= 0; i-- )
  {
    workers_.getAs<Worker_>(i)->kill ();
  }

  i = (int) workers_.size() - 1;

  while ( i >= 0 )
  {
    workers_.getAs<Worker_>(i)->join ();
    workers_.popBack ();
    i--;
  }
}


//-----------------------------------------------------------------------
//   killWorkers ( timeout )
//-----------------------------------------------------------------------


int WorkPool::killWorkers ( const Time& timeout )
{
  Worker_*  w;

  Time      te = Time::now() + timeout;
  Time      dt = timeout;

  int       i, k;


  k = 0;
  i = (int) workers_.size() - 1;

  while ( i >= 0 && dt > Time::zero() )
  {
    w = workers_.getAs<Worker_> ( i );

    w->kill ();
    w->join ( dt );

    if ( w->isAlive() )
    {
      break;
    }

    workers_.popBack ();

    dt = te - Time::now ();

    i--;
    k++;
  }

  return k;
}


//-----------------------------------------------------------------------
//   newWorker_
//-----------------------------------------------------------------------


void WorkPool::newWorker_ ()
{
  Worker_*  w;

  for ( int i = 0; i < (int) workers_.size(); i++ )
  {
    w = workers_.getAs<Worker_> ( i );

    if ( ! w->isAlive() )
    {
      w->start ();
      return;
    }
  }

  if ( workers_.size() < maxWorkers_ )
  {
    Ref<Thread>  t = newInstance<Worker_> ( arena_ );

    t->start ();

    workers_.pushBack ( t.get() );
  }
}


//=======================================================================
//   class WorkPool::Job
//=======================================================================


WorkPool::Job::~Job ()
{}


//=======================================================================
//   class WorkPool::Task
//=======================================================================


WorkPool::Task::~Task ()
{}


JEM_END_PACKAGE( mt )
