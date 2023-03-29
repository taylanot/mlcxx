
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


#include <cstdlib>
#include <jem/base/assert.h>
#include <jem/base/System.h>
#include <jem/base/Thread.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/CancelledException.h>
#include <jem/gl/Display.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/mp/Globdat.h>
#include <jive/app/Application.h>
#include <jive/app/ProgramArgs.h>
#include <jive/app/ModuleFactory.h>
#include <jive/gl/Names.h>
#include <jive/gl/DisplayModule.h>


JEM_DEFINE_CLASS( jive::gl::DisplayModule );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;


//=======================================================================
//   class DisplayModule::Thread_
//=======================================================================


class DisplayModule::Thread_ : public jem::Thread
{
 public:

  inline                Thread_

    ( const Ref<Module>&  mod,
      const Properties&   conf,
      const Properties&   props,
      const Properties&   globdat );


  virtual void          run () override;


 private:

  Ref<Module>           module_;

  Properties            conf_;
  Properties            props_;
  Properties            globdat_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline DisplayModule::Thread_::Thread_

  ( const Ref<Module>&  mod,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat ) :

    module_  ( mod     ),
    conf_    ( conf    ),
    props_   ( props   ),
    globdat_ ( globdat )

{}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void DisplayModule::Thread_::run ()
{
  using jive::app::Application;

  String  phase;
  int     retval = 1;

  try
  {
    Application::runLoop ( phase, *module_,
                           conf_,  props_, globdat_ );

    retval = 0;
  }
  catch ( jem::CancelledException& )
  {
    retval = 0;
  }
  catch ( const jem::Throwable& ex )
  {
    Application::printError ( phase,
                              ex.name  (),
                              ex.where (),
                              ex.what  (),
                              ex.getStackTrace () );
  }
  catch ( ... )
  {
    Application::printError ( phase,
                              "unknown", "unknown", "unknown" );
  }

  std::exit ( retval );
}


//=======================================================================
//   class DisplayModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DisplayModule::TYPE_NAME = "Display";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DisplayModule::DisplayModule

  ( const Ref<Module>&  child ) :

    Super  ( "display" ),
    child_ ( child )

{
  JEM_PRECHECK ( child );

  active_ = true;
}


DisplayModule::DisplayModule

  ( const String&       name,
    const Ref<Module>&  child ) :

    Super  ( name  ),
    child_ ( child )

{
  JEM_PRECHECK ( child );

  active_ = true;
}


DisplayModule::~DisplayModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status DisplayModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::io::Writer;
  using jem::gl::Display;
  using jive::app::ProgramArgs;

  Writer&  out = jem::System::debug ( myName_ );

  if ( mp::Globdat::procCount( globdat ) > 1 )
  {
    print ( out, myName_,
            " : running in MP mode; not starting "
            "display loop\n" );

    active_ = false;

    return child_->init ( conf, props, globdat );
  }
  else
  {
    Properties        myProps = props.findProps ( myName_ );
    Properties        myConf  = conf .makeProps ( myName_ );


    Ref<Thread_>      thr     =

      newInstance<Thread_> ( child_, conf, props, globdat );

    Ref<ProgramArgs>  args    = ProgramArgs::get ( globdat );

    bool              ofs     = false;
    int               mode    = 0;


    myProps.find ( ofs, PropNames::OFF_SCREEN );

    if ( ofs )
    {
      mode |= Display::OFF_SCREEN;
    }

    mode = Display::open ( args->argc(), args->argv(), mode );
    ofs  = ((mode & Display::OFF_SCREEN) != 0);

    myConf.set   ( PropNames::OFF_SCREEN, ofs );
    args->commit ();

    print ( out, myName_, " : starting computation thread ...\n" );

    thr->start ();

    print ( out, myName_, " : entering main display loop  ...\n" );

    Display::mainLoop ();

    // These statements should never be reached.

    std::exit ( 0 );

    return DONE;
  }
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status DisplayModule::run ( const Properties& globdat )
{
  if ( active_ )
  {
    return DONE;
  }
  else
  {
    return child_->run ( globdat );
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void DisplayModule::shutdown ( const Properties& globdat )
{
  if ( ! active_ )
  {
    child_->shutdown ( globdat );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void DisplayModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void DisplayModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  child_->getConfig ( conf, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool DisplayModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> DisplayModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::app::newModule;
  using jive::util::joinNames;

  return newInstance<Self> (
    name,
    newModule ( joinNames( name, PropNames::MODULE ),
                conf, props, globdat )
  );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DisplayModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( gl )
