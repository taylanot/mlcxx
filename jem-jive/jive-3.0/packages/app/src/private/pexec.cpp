
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/assert.h>
#include <jem/base/Throwable.h>
#include <jem/base/CancelledException.h>
#include <jem/mp/Context.h>
#include <jem/mp/Task.h>
#include <jem/mp/Driver.h>
#include <jem/mp/DriverFactory.h>
#include <jem/mp/StdDriverFactory.h>
#include <jem/util/Properties.h>
#include <jive/util/ObjectConverter.h>
#include <jive/mp/Globdat.h>
#include <jive/app/Module.h>
#include <jive/app/ProgramArgs.h>
#include <jive/app/Application.h>


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;


//=======================================================================
//   class Application::Task_
//=======================================================================


class Application::Task_ : public jem::mp::Task
{
 public:

                            Task_

    ( int                     argc,
      char**                  argv,
      const Ref<MPContext>&   mpx,
      ModuleConstructor       ctor );


  virtual void              run       () override;


 private:

  int                       argc_;
  char**                    argv_;

  Ref<MPContext>            context_;
  ModuleConstructor         modCtor_;

  Properties                globdat_;
  Properties                conf_;
  Properties                props_;

  String                    phase_;

  int                       myRank_;
  int                       procCount_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


Application::Task_::Task_

  ( int                     argc,
    char**                  argv,
    const Ref<MPContext>&   mpx,
    ModuleConstructor       ctor ) :

    argc_    ( argc ),
    argv_    ( argv ),

    context_ ( mpx  ),
    modCtor_ ( ctor ),

    globdat_ ( "params" ),
    phase_   ( "initialization" )

{
  myRank_    = context_->myRank ();
  procCount_ = context_->size   ();
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void Application::Task_::run ()
{
  using jive::util::ObjConverter;

  try
  {
    Ref<ProgramArgs>  args;
    Ref<Module>       mod;


    if ( procCount_ == 1 )
    {
      loadProperties ( props_, argc_, argv_ );
      initSystem     ( conf_,  props_       );

      args = newInstance<ProgramArgs> ( argc_, argv_ );
    }
    else
    {
      args = loadProperties ( *context_, props_, argc_, argv_ );
      initSystem            ( *context_, conf_,  props_       );
    }

    props_.setConverter ( newInstance<ObjConverter>( globdat_ ) );

    mod = modCtor_ ();

    if ( ! mod )
    {
      return;
    }

    initGlobdat ( conf_, props_, globdat_, context_, args );
    runLoop     ( phase_, *mod, conf_, props_, globdat_ );
  }
  catch ( const jem::CancelledException& )
  {
    throw;
  }
  catch ( const jem::Throwable& ex )
  {
    printError ( phase_,
                 ex.name  (),
                 ex.where (),
                 ex.what  (),
                 ex.getStackTrace () );

    context_->abort ( 1 );
  }
  catch ( ... )
  {
    printError ( phase_, "unknown", "unknown", "unknown" );

    context_->abort ( 1 );
  }
}


//=======================================================================
//   class Application::TaskFactory_
//=======================================================================


class Application::TaskFactory_ : public jem::mp::TaskFactory
{
 public:

  typedef jem::mp::Task     MPTask;


  explicit                  TaskFactory_

    ( ModuleConstructor       ctor );

  virtual Ref<MPTask>       newTask

    ( const Ref<MPContext>&   mpx,
      int                     argc,
      char**                  argv ) override;


 private:

  ModuleConstructor         modCtor_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


Application::TaskFactory_::TaskFactory_ ( ModuleConstructor ctor ) :

  modCtor_ ( ctor )

{}


//-----------------------------------------------------------------------
//   newTask
//-----------------------------------------------------------------------


Ref<jem::mp::Task> Application::TaskFactory_::newTask

  ( const Ref<MPContext>&  mpx,
    int                    argc,
    char**                 argv )

{
  return jem::newInstance<Task_> ( argc, argv, mpx, modCtor_ );
}


//=======================================================================
//   class Application
//=======================================================================

//-----------------------------------------------------------------------
//   pexec
//-----------------------------------------------------------------------


int Application::pexec

  ( int                argc,
    char**             argv,
    ModuleConstructor  ctor )

{
  const char*  phase  = "initialization";
  int          result = 1;

  try
  {
    initSigHandlers ();

    jem::mp::StdDriverFactory::install ();

    Ref<jem::mp::Driver> driver =

      jem::mp::DriverFactory::newInstance ( argc, argv, "default" );

    TaskFactory_ taskFactory ( ctor );

    driver->start ( taskFactory, argc, argv );

    result = 0;
  }
  catch ( const jem::Throwable& ex )
  {
    String where = ex.where ();

    if ( ! where.size() && argc > 0 )
    {
      where = argv[0];
    }

    printError ( phase, ex.name(),
                 where, ex.what(), ex.getStackTrace() );
  }
  catch ( ... )
  {
    printError ( phase );
  }

  return result;
}


JIVE_END_PACKAGE( app )
