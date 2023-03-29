
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


#include <jem/mp/config.h>

#ifdef JEM_USE_MPI

#include <jem/base/System.h>
#include <jem/mp/Task.h>
#include <jem/mp/MPException.h>
#include <jem/mp/DriverException.h>
#include <jem/mp/MPIContext.h>
#include <jem/mp/MPIDriver.h>


JEM_BEGIN_PACKAGE( mp )


//=======================================================================
//   class MPIDriver
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MPIDriver::MPIDriver ()
{}


MPIDriver::~MPIDriver ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void MPIDriver::start

  ( TaskFactory&  factory,
    int           argc,
    char**        argv )

{
  Ref<Context>  ctx;
  Ref<Task>     task;
  bool          doAbort;


  doAbort = false;

  try
  {
    ctx = MPIContext::init ( &argc, &argv );
  }
  catch ( const MPException& ex )
  {
    throw DriverException ( JEM_FUNC, ex.what() );
  }

  scrubArgs ( argc, argv );

  try
  {
    task = factory.newTask ( ctx, argc, argv );
    task->run ();
  }
  catch ( const Throwable& ex )
  {
    System::printException (
      System::err      (),
      ex.name          (),
      ex.where         (),
      ex.what          (),
      ex.getStackTrace ()
    );

    doAbort = true;
  }
  catch ( ... )
  {
    System::printException (
      System::err (),
      "unknown",
      "unknown",
      "unknown"
    );

    doAbort = true;
  }

  task = nullptr;
  ctx  = nullptr;

  if ( doAbort )
  {
    print ( System::err(),
            "\nKilling all processes and aborting program.\n" );

    ctx->abort ( 1 );
  }

  // Make sure that MPI is properly shut down; otherwise a program
  // may just "hang". Normally MPI_Finalize is called by the
  // destructor of the MPIContext class. However, if a program has
  // created a circular reference, the destructor may not be called.

  MPIContext::finalize ();
}


JEM_END_PACKAGE( mp )

#endif
