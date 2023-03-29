
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/RuntimeException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/list.h>
#include <jem/io/NullWriter.h>
#include <jem/mp/Context.h>
#include <jem/util/Pattern.h>
#include <jem/util/Tokenizer.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/mp/Globdat.h>
#include <jive/app/typedefs.h>
#include <jive/app/Names.h>
#include <jive/app/Actions.h>
#include <jive/app/Commands.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/DummyModule.h>
#include <jive/app/DebugModule.h>


JEM_DEFINE_CLASS( jive::app::DebugModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::System;
using jem::io::endl;
using jem::io::flush;
using jem::io::indent;
using jem::io::outdent;


//=======================================================================
//   class DebugModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DebugModule::TYPE_NAME = "Debug";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DebugModule::DebugModule

  ( const String&       name,
    const Ref<Module>&  child ) :

    Super  (  name ),
    child_ ( child )

{
  if ( ! child_ )
  {
    child_      = newInstance<DummyModule> ( name );
    myContext_  = getContext ();
    noiseLevel_ = 0;
  }
  else
  {
    myContext_  = child_->getContext ();
    noiseLevel_ = 1;
  }

  if ( ! myName_.size() )
  {
    myName_ = child_->getName ();
  }

  output_ =

    newInstance<PrintWriter> ( & System::debug( myName_ ) );
}


DebugModule::~DebugModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status DebugModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )


{
  PrintWriter&  out = * output_;

  Status        status;


  timer_.reset ();
  timer_.start ();

  if ( noiseLevel_ > 0 )
  {
    print ( out, "Initializing ", myContext_ );

    if ( noiseLevel_ > 1 )
    {
      print ( out, " with data:\n", indent, endl, globdat );
      print ( out, outdent, endl, endl );
    }
    else
    {
      print ( out, endl, endl );
    }

    out.flush ();
  }

  status = child_->init ( conf, props, globdat );

  storeVars_ ( globdat );

  if ( noiseLevel_ > 0 )
  {
    print ( out, "Done initializing ", myContext_ );
    printStatus_ ( status );
  }

  return status;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status DebugModule::run ( const Properties& globdat )
{
  PrintWriter&  out = * output_;

  Status        status;


  timer_.reset ();

  if ( noiseLevel_ > 0 )
  {
    print ( out, "Invoking ", myContext_ );

    if ( noiseLevel_ > 1 )
    {
      print ( out, " with data:\n", indent, endl, globdat );
      print ( out, outdent, endl, endl );
    }
    else
    {
      print ( out, endl, endl );
    }

    out.flush ();
  }

  status = child_->run ( globdat );

  storeVars_ ( globdat );

  if ( noiseLevel_ > 0 )
  {
    print ( out, "Leaving ", myContext_ );
    printStatus_ ( status );
  }

  return status;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void DebugModule::shutdown ( const Properties& globdat )
{
  PrintWriter&  out = * output_;


  timer_.reset ();

  if ( noiseLevel_ > 0 )
  {
    print ( out, "Shutting down ", myContext_ );

    if ( noiseLevel_ > 1 )
    {
      print ( out, " with data:\n", indent, endl, globdat );
      print ( out, outdent, endl, endl );
    }
    else
    {
      print ( out, endl, endl );
    }

    out.flush ();
  }

  child_->shutdown ( globdat );

  if ( noiseLevel_ > 0 )
  {
    print ( out, "Done shutting down ", myContext_,
            " (exec time = ", timer_.toDouble(), " sec.)" );
    print ( out, endl, endl, flush );
  }

  timer_.stop  ();
  timer_.reset ();
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void DebugModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( noiseLevel_, PropNames::NOISE_LEVEL );
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void DebugModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::NOISE_LEVEL, noiseLevel_ );

  child_->getConfig ( conf, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool DebugModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::Globdat;

  bool  result = child_->takeAction ( action, params, globdat );

  if      ( action == Actions::EXE_COMMAND )
  {
    String  cmd, arg;

    params.get ( cmd, ActionParams::COMMAND );
    params.get ( arg, ActionParams::COMMAND_ARG );

    if ( exeCommand_( cmd, arg, globdat ) )
    {
      result = true;
    }
  }
  else if ( action == Actions::GET_COMMANDS )
  {
    Properties  vars = Globdat::getVariables ( globdat );

    params.set ( Commands::SET_VAR,   vars   .listProps() );
    params.set ( Commands::SHOW_DATA, globdat.listProps() );
    params.set ( Commands::TOUCH_VAR, "" );

    result = true;
  }
  else if ( action == Actions::PRINT_HELP )
  {
    Ref<PrintWriter>  out;

    params.get ( out, ActionParams::PRINTER );
    printHelp  ( *out );

    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void DebugModule::printHelp ( PrintWriter& out ) const
{
  using jem::io::beginItem;
  using jem::io::endItem;

  print ( out, beginItem( "set-var <name>=<value>" ) );
  print ( out,   "Set the runtime variable <name> to <value>." );
  print ( out, endItem );

  print ( out, beginItem( "touch-var <name>" ) );
  print ( out,   "Touch the runtime variable <name>. " );
  print ( out,   "If the variable does not exist, it will be " );
  print ( out,   "created with the initial value nullptr. " );
  print ( out,   "Otherwise this command does nothing." );
  print ( out, endItem );

  print ( out, beginItem( "show-data [pattern]" ) );
  print ( out,   "Print all objects in the global database " );
  print ( out,   "`globdat\' of which the name matches " );
  print ( out,   "<pattern>. If <pattern> is omitted, then all " );
  print ( out,   "objects will be printed." );
  print ( out, endItem );
}


//-----------------------------------------------------------------------
//   getNoiseLevel
//-----------------------------------------------------------------------


int DebugModule::getNoiseLevel () const
{
  return noiseLevel_;
}


//-----------------------------------------------------------------------
//   setNoiseLevel
//-----------------------------------------------------------------------


void DebugModule::setNoiseLevel ( int level )
{
  noiseLevel_ = level;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> DebugModule::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  return newInstance<Self> (
    myName,
    newModule ( joinNames( myName, PropNames::MODULE ),
                conf, props, globdat )
  );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DebugModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   storeVars_
//-----------------------------------------------------------------------


void DebugModule::storeVars_ ( const Properties& globdat )
{
  using jive::util::Globdat;

  if ( myName_.size() > 0 &&
       Globdat::hasVariable( myName_, globdat ) )
  {
    Properties  vars = Globdat::getVariables ( myName_, globdat );

    vars.set ( VarNames::RUNTIME, timer_.toDouble() );
  }
}


//-----------------------------------------------------------------------
//   printStatus_
//-----------------------------------------------------------------------


void DebugModule::printStatus_ ( Status status )
{
  PrintWriter& out = * output_;

  print ( out, " (status = " );

  switch ( status )
  {
  case OK :

    print ( out, "OK" );   break;

  case DONE :

    print ( out, "DONE"); break;

  case EXIT :

    print ( out, "EXIT"); break;
  }

  print ( out, "; exec time = ", timer_.toDouble(), " sec.)" );
  print ( out, endl, endl, flush );
}


//-----------------------------------------------------------------------
//   exeCommand_
//-----------------------------------------------------------------------


bool DebugModule::exeCommand_

  ( const String&      cmd,
    const String&      arg,
    const Properties&  globdat )

{
  bool  result = true;

  if      ( cmd == Commands::SET_VAR )
  {
    exeSetVarCmd_   ( arg, globdat );
  }
  else if ( cmd == Commands::TOUCH_VAR )
  {
    exeTouchVarCmd_ ( arg, globdat );
  }
  else if ( cmd == Commands::SHOW_DATA )
  {
    exeShowDataCmd_ ( arg, globdat );
  }
  else
  {
    result = false;
  }

  return result;
}


//-----------------------------------------------------------------------
//   exeSetVarCmd_
//-----------------------------------------------------------------------


void DebugModule::exeSetVarCmd_

  ( const String&      arg,
    const Properties&  globdat )

{
  using jem::util::Tokenizer;
  using jive::util::Globdat;

  Properties      vars = Globdat::getVariables ( globdat );

  Ref<Tokenizer>  tok;

  if ( arg.size() > 0 && arg.back() != ';' )
  {
    tok = newInstance<Tokenizer> ( arg + ';' );
  }
  else
  {
    tok = newInstance<Tokenizer> ( arg );
  }

  vars.parseFrom ( *tok, "command",
                   Properties::PARSE_STRICT );
}


//-----------------------------------------------------------------------
//   exeTouchVarCmd_
//-----------------------------------------------------------------------


void DebugModule::exeTouchVarCmd_

  ( const String&      name,
    const Properties&  globdat )

{
  using jive::util::Globdat;

  if ( ! name.size() )
  {
    throw jem::IllegalInputException (
      Commands::TOUCH_VAR,
      "missing variable name"
    );
  }

  Properties  vars = Globdat::getVariables ( globdat );

  if ( ! vars.contains( name ) )
  {
    vars.set ( name, nullptr );
  }
}


//-----------------------------------------------------------------------
//   exeShowDataCmd_
//-----------------------------------------------------------------------


void DebugModule::exeShowDataCmd_

  ( const String&      arg,
    const Properties&  globdat )

{
  using jem::io::NullWriter;
  using jem::util::Pattern;

  Ref<MPContext>    mpx = mp::Globdat::getMPContext ( globdat );

  Ref<PrintWriter>  pr;


  if ( mpx->myRank() == 0 )
  {
    pr = newInstance<PrintWriter> ( & System::out() );
  }
  else
  {
    pr = newInstance<PrintWriter> ( newInstance<NullWriter>() );
  }

  print ( *pr, endl, indent );

  if ( arg.size() == 0 )
  {
    print ( *pr, globdat );
  }
  else
  {
    StringVector  list = globdat.listProps ();

    Properties    tmp;
    Ref<Object>   obj;

    Pattern::check ( arg );

    for ( idx_t i = 0; i < list.size(); i++ )
    {
      if ( Pattern::matches( arg, list[i] ) )
      {
        if ( globdat.find( obj, list[i] ) )
        {
          tmp.set ( list[i], obj );
        }
      }
    }

    print ( *pr, tmp );
  }

  print ( *pr, endl, flush );
}


JIVE_END_PACKAGE( app )
