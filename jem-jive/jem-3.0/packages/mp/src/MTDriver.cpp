
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


#include <cstring>
#include <cstdlib>
#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include <jem/base/Monitor.h>
#include <jem/base/MemCache.h>
#include <jem/base/CancellationScope.h>
#include <jem/util/ObjFlex.h>
#include <jem/mp/Task.h>
#include <jem/mp/MPException.h>
#include <jem/mp/AbortException.h>
#include <jem/mp/DriverException.h>
#include <jem/mp/UniContext.h>
#include <jem/mp/MTContext.h>
#include <jem/mp/MTDriver.h>


JEM_BEGIN_PACKAGE( mp )


using jem::util::ObjFlex;


//=======================================================================
//   class MTDriver::Result
//=======================================================================


class MTDriver::Result : public Collectable
{
 public:

  explicit inline         Result

    ( int                   threadCount );

  inline void             notifyDone ();

  inline void             notifyAborted

    ( int                   code );


 public:

  Monitor                 monitor;
  int                     busyCount;
  int                     abortCode;
  bool                    aborted;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline MTDriver::Result::Result ( int count ) :

  busyCount ( count ),
  abortCode (     0 ),
  aborted   ( false )

{}


//-----------------------------------------------------------------------
//   notifyDone
//-----------------------------------------------------------------------


inline void MTDriver::Result::notifyDone ()
{
  monitor.lock ();

  if ( (--busyCount) == 0 )
  {
    monitor.notify ();
  }

  monitor.unlock ();
}


//-----------------------------------------------------------------------
//   notifyAborted
//-----------------------------------------------------------------------


inline void MTDriver::Result::notifyAborted ( int code )
{
  monitor.lock ();

  busyCount--;

  if ( ! aborted )
  {
    abortCode = code;
    aborted   = true;

    monitor.notify ();
  }
  else if ( busyCount == 0 )
  {
    monitor.notify ();
  }

  monitor.unlock ();
}


//=======================================================================
//   class MTDriver::Thread
//=======================================================================


class MTDriver::Thread : public jem::Thread
{
 public:

  typedef jem::Thread     Super;


                          Thread

    ( const Ref<Context>&   ctx,
      const Ref<Result>&    result,
      TaskFactory*          factory,
      int                   argc,
      char**                argv );

  virtual void            run    () override;


 protected:

  virtual                ~Thread ();


 private:

  Ref<Context>            context_;
  Ref<Result>             result_;
  TaskFactory*            taskFactory_;

  int                     argc_;
  char**                  argv_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MTDriver::Thread::Thread

  ( const Ref<Context>&  ctx,
    const Ref<Result>&   result,
    TaskFactory*         factory,
    int                  argc,
    char**               argv ) :

    context_     (     ctx ),
    result_      (  result ),
    taskFactory_ ( factory )

{
  // Allocate a private argv array to avoid nasty race conditions.

  argc_ = argc;
  argv_ = (char**) MemCache::alloc ( (size_t) (argc + 1) *
                                     sizeof(char*) );

  std::memcpy ( argv_, argv, (size_t) argc * sizeof(char*) );

  argv_[argc] = 0;
}


MTDriver::Thread::~Thread ()
{
  if ( argv_ )
  {
    MemCache::dealloc ( argv_, (size_t) (argc_ + 1) * sizeof(char*) );

    argv_ = 0;
  }
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void MTDriver::Thread::run ()
{
  JEM_PRECHECK ( context_ );

  try
  {
    Ref<Task> task =

      taskFactory_->newTask ( context_, argc_, argv_ );

    allowCancel ( true );
    task->run   ();

    task = nullptr;

    result_->notifyDone ();
  }
  catch ( const AbortException& ex )
  {
    result_->notifyAborted ( ex.abortCode );
  }
  catch ( ... )
  {
    result_->notifyDone ();
    throw;
  }

  // Free data as soon as possible.

  context_ = nullptr;
  result_  = nullptr;

  MemCache::dealloc ( argv_, (size_t) (argc_ + 1) * sizeof(char*) );

  argv_    = 0;
  argc_    = 0;
}


//=======================================================================
//   class MTDriver
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MTDriver::MTDriver ()
{}


MTDriver::~MTDriver ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void MTDriver::start

  ( TaskFactory&  factory,
    int           argc,
    char**        argv )

{
  scrubArgs ( argc, argv );

  int  n = getThreadCount_ ( argc, argv );

  if ( n > 1 )
  {
    startThreads_ ( factory, n, argc, argv );
  }
  else
  {
    Ref<Context>  ctx  = newInstance<UniContext> ();
    Ref<Task>     task = factory.newTask ( ctx, argc, argv );

    task->run ();
  }
}


//-----------------------------------------------------------------------
//   startThreads_
//-----------------------------------------------------------------------


void MTDriver::startThreads_

  ( TaskFactory&  factory,
    int           n,
    int           argc,
    char**        argv )

{
  ObjFlex         threads;
  Ref<Thread>     thread;
  Ref<MTContext>  context;
  Ref<Result>     result;

  bool            cancelled = false;


  try
  {
    context = newInstance<MTContext> ( n );
  }
  catch ( const MPException& ex )
  {
    throw DriverException ( JEM_FUNC, ex.what() );
  }

  result = newInstance<Result> ( n );

  threads.resize ( n );

  thread = newInstance<Thread> ( context,
                                 result,
                                 & factory,
                                 argc, argv );

  threads.set ( 0, thread );

  for ( int i = 1; i < n; i++ )
  {
    thread = newInstance<Thread> (
      newInstance<MTContext> ( i, *context ),
      result,
      & factory,
      argc, argv
    );

    threads.set ( i, thread );
  }

  for ( int i = 0; i < n; i++ )
  {
    threads.getAs<Thread>(i)->start ();
  }

  result->monitor.lock ();

  while ( ! cancelled )
  {
    if ( result->aborted || result->busyCount == 0 )
    {
      break;
    }

    result->monitor.wait ( cancelled );
  }

  {
    CancellationScope  cancelScope ( false );

    if ( result->busyCount > 0 )
    {
      if ( result->aborted )
      {
        print ( System::err(), "\nCancelling threads ...\n" );
      }

      result->monitor.unlock ();

      for ( int i = 0; i < n; i++ )
      {
        threads.getAs<Thread>(i)->cancel ();
      }

      result->monitor.lock ();

      if ( result->busyCount > 0 && result->aborted )
      {
        // Give all threads two seconds to close down

        print ( System::err(), "Waiting for threads to exit ...\n" );

        result->monitor.wait ( 2_sec );
      }
    }

    result->monitor.unlock ();

    if ( result->aborted )
    {
      for ( int i = 0; i < n; i++ )
      {
        threads.getAs<Thread>(i)->join ( Time( 2.0 / (double) n ) );
      }

      print ( System::err(), "Terminating program.\n\n" );

      std::exit ( result->abortCode );
    }

    for ( int i = 0; i < n; i++ )
    {
      threads.getAs<Thread>(i)->join ();
    }
  }

  if ( cancelled )
  {
    throw CancelledException ( JEM_FUNC, "wait cancelled" );
  }
}


//-----------------------------------------------------------------------
//   getThreadCount_
//-----------------------------------------------------------------------


int MTDriver::getThreadCount_ ( int& argc, char** argv )
{
  char*  end;
  char*  arg;
  long   n;
  int    i;


  for ( i = 1; i < argc; i++ )
  {
    if ( std::strcmp( argv[i], "-np" ) == 0 )
    {
      break;
    }
  }

  if ( i == argc )
  {
    return 1;
  }

  if ( i == argc - 1 )
  {
    throw DriverException (
      JEM_FUNC,
      "missing number of threads"
    );
  }

  arg = argv[i + 1];
  n   = std::strtol ( arg, & end, 0 );

  if ( *arg == '\0' || *end != '\0' || n <= 0 )
  {
    throw DriverException (
      JEM_FUNC,
      String::format (
        "illegal number of threads specified: %s", arg
      )
    );
  }

  if ( n > maxOf<int>() )
  {
    throw DriverException (
      JEM_FUNC,
      String::format (
        "too many threads specified: %d", n
      )
    );
  }

  // Shift the remaining arguments to the left

  for ( i += 2; i < argc; i++ )
  {
    argv[i - 2] = argv[i];
  }

  argc -= 2;

  return (int) n;
}


JEM_END_PACKAGE( mp )
