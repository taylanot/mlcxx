
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


#include <jem/base/Time.h>
#include <jem/base/Thread.h>
#include <jem/base/Integer.h>
#include <jem/base/Exception.h>
#include <jem/io/list.h>
#include <jem/io/TermInput.h>
#include <jem/util/Properties.h>
#include <jem/util/Tokenizer.h>
#include <jive/util/Globdat.h>
#include <jive/app/Actions.h>
#include <jive/app/ChainModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/Application.h>
#include <jive/app/InfoModule.h>
#include <jive/fem/ShapeModule.h>
#include <jive/fem/InputModule.h>
#include <jive/fem/InitModule.h>
#include <jive/gl/FemViewModule.h>
#include <jive/geom/declare.h>

#include "FemvizModule.h"


using jive::util::Globdat;


//=======================================================================
//   class FemvizModule
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


FemvizModule::FemvizModule ( const String& name ) :

  Super ( name )

{
  state_ = INACTIVE_;
}


FemvizModule::~FemvizModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status FemvizModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::geom::declareShapes;
  using jive::app::ChainModule;
  using jive::app::InfoModule;
  using jive::fem::ShapeModule;
  using jive::fem::InputModule;
  using jive::fem::InitModule;
  using jive::gl::FemViewModule;


  Ref<ChainModule>  chain = newInstance<ChainModule> ();

  Status            stat;


  child_ = nullptr;

  declareShapes    ();

  chain->pushBack  ( newInstance<InputModule> () );
  chain->pushBack  ( newInstance<ShapeModule> ( "" ) );
  chain->pushBack  ( newInstance<InitModule>  () );
  chain->pushBack  ( newInstance<InfoModule>  () );

  chain->pushBack  (
    newInstance<FemViewModule> ( "view", FemViewModule::FG_MODE )
  );

  if ( ! props.contains( "storageMode" ) )
  {
    props.set ( "storageMode", "Compact" );
  }

  chain->configure ( props, globdat );
  chain->getConfig ( conf,  globdat );

  stat   = chain->init ( conf, props, globdat );
  child_ = chain;

  // Let the main loop know that this object will control the time
  // step number.

  globdat.set ( Globdat::TIME_STEP, 0 );

  state_ = STARTING_;

  return stat;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status FemvizModule::run ( const Properties& globdat )
{
  if ( ! child_ )
  {
    notAliveError ( getContext() );
  }

  idx_t  istep = 0;

  globdat.find ( istep, Globdat::TIME_STEP );

  if ( state_ == RUNNING_ )
  {
    istep++;

    globdat.set ( Globdat::TIME_STEP, istep );
  }
  else
  {
    state_ = RUNNING_;
  }

  return child_->run ( globdat );
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void FemvizModule::shutdown ( const Properties& globdat )
{
  if ( child_ )
  {
    child_->shutdown ( globdat );
  }

  state_ = INACTIVE_;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void FemvizModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( child_ )
  {
    child_->configure ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void FemvizModule::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  if ( child_ )
  {
    child_->getConfig ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool FemvizModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::app::Actions;
  using jive::app::ActionParams;

  bool  result = false;


  if ( child_ )
  {
    result = child_->takeAction ( action, params, globdat );
  }

  if      ( action == Actions::EXE_COMMAND )
  {
    String  cmd;
    String  arg;

    params.get ( cmd, ActionParams::COMMAND );
    params.get ( arg, ActionParams::COMMAND_ARG );

    if ( exeCommand_( cmd, arg, globdat ) )
    {
      result = true;
    }
  }
  else if ( action == Actions::PRINT_HELP )
  {
    Ref<PrintWriter>  pr;

    params.get ( pr, ActionParams::PRINTER );
    printHelp  ( *pr );

    result = true;
  }
  else if ( action == Actions::GET_COMMANDS )
  {
    params.set ( "goto",    "1" );
    params.set ( "animate", "[1:10:1]" );

    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void FemvizModule::printHelp ( PrintWriter& out ) const
{
  using jem::io::indent;
  using jem::io::outdent;
  using jem::io::beginItem;
  using jem::io::endItem;


  print ( out, beginItem( "goto <istep>" ) );
  print ( out,   "Set the current time step/iteration " );
  print ( out,   "number to <istep>." );
  print ( out, endItem );

  print ( out, beginItem( "animate <range> [pause]" ) );
  print ( out,   "Repeatedly set the time step/iteration " );
  print ( out,   "according to the values specified by <range>. " );
  print ( out,   "The optional argument [pause] specifies the " );
  print ( out,   "time in seconds that should be paused before " );
  print ( out,   "going to the next time step/iteration number. " );
  print ( out,   "If not specified, [pause] will be set to one. " );
  print ( out,   "The argument <range> should be specified as\n\n" );
  print ( out,   indent );
  print ( out,     "[<from>:<to>]\n\n" );
  print ( out,   outdent );
  print ( out,   "or as\n\n" );
  print ( out,   indent );
  print ( out,     "[<from>:<to>:<stride>]\n\n" );
  print ( out,   outdent );
  print ( out,   "with <from> the initial time step/iteration " );
  print ( out,   "number; <to> the final time step/iteration " );
  print ( out,   "number; and <stride> the amount with which the " );
  print ( out,   "time step/iteration number is to be incremented. " );
  print ( out,   "The latter may be negative. If <stride> is " );
  print ( out,   "omitted, it will be set to one. Examples:\n\n" );
  print ( out,   indent );
  print ( out,     "animate [0:10]\n" );
  print ( out,     "animate [200:100:-20] 0.4\n\n" );
  print ( out,   outdent );
  print ( out,   "The first command will increment the time step/" );
  print ( out,   "iteration number from 0 to (and including) 10 " );
  print ( out,   "with a pause of 1 second between each increment. " );
  print ( out,   "The second command will decrement the time step/" );
  print ( out,   "iteration number from 200 to 100 with steps of " );
  print ( out,   "20, and a pause of 0.4 seconds in between." );
  print ( out, endItem );
}


//-----------------------------------------------------------------------
//   exeCommand_
//-----------------------------------------------------------------------


bool FemvizModule::exeCommand_

  ( const String&      cmd,
    const String&      arg,
    const Properties&  globdat )

{
  if ( cmd == "goto" )
  {
    idx_t  istep = (idx_t) jem::Integer::parse ( arg );

    if ( istep < 0 )
    {
      throw jem::Exception (
        getContext (),
        "negative time step/iteration number: " + arg
      );
    }

    globdat.set ( Globdat::TIME_STEP, istep );

    if ( child_ )
    {
      child_->run ( globdat );
    }

    return true;
  }

  if ( cmd == "animate" )
  {
    animate_ ( arg, globdat );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   animate_
//-----------------------------------------------------------------------


void FemvizModule::animate_

  ( const String&      arg,
    const Properties&  globdat )

{
  using jem::util::Tokenizer;

  Ref<Tokenizer>  tk = newInstance<Tokenizer> ( arg );

  double  pause;

  idx_t   from, to, stride;
  idx_t   istep;
  int     token;
  idx_t   dir;


  from   = to = 0;
  stride = 1;
  pause  = 1.0;

  if ( tk->nextToken() != '[' )
  {
    tk->parseError ( "no range specified" );
  }

  if ( tk->nextToken() != Tokenizer::INTEGER_TOKEN )
  {
    tk->parseError ( "expected the initial time step/"
                     "iteration number" );
  }

  from = (idx_t) tk->getInteger ();

  if ( from < 0 )
  {
    tk->parseError ( "negative time step/iteration number" );
  }

  if ( tk->nextToken() != ':' )
  {
    tk->parseError ( "expected a `:\'" );
  }

  if ( tk->nextToken() != Tokenizer::INTEGER_TOKEN )
  {
    tk->parseError ( "expected the final time step/"
                     "iteration number" );
  }

  to = (idx_t) tk->getInteger ();

  if ( to < 0 )
  {
    tk->parseError ( "negative time step/iteration number" );
  }

  token = tk->nextToken  ();

  if ( token != ']' )
  {
    if ( token != ':' )
    {
      tk->parseError ( "expected a `:\' or `]\'" );
    }

    if ( tk->nextToken() != Tokenizer::INTEGER_TOKEN )
    {
      tk->parseError ( "expected an integer stride" );
    }

    stride = (idx_t) tk->getInteger ();

    if ( stride == 0 )
    {
      tk->parseError ( "zero stride specified" );
    }

    token = tk->nextToken ();
  }

  if ( token != ']' )
  {
    tk->parseError ( "expected a `]\'" );
  }

  token = tk->nextToken ();

  if ( token != Tokenizer::EOF_TOKEN )
  {
    if ( token != Tokenizer::FLOAT_TOKEN &&
         token != Tokenizer::INTEGER_TOKEN )
    {
      tk->parseError ( "invalid pause interval" );
    }

    pause = tk->getFloat  ();
    token = tk->nextToken ();
  }

  if ( token != Tokenizer::EOF_TOKEN )
  {
    tk->parseError ();
  }

  if ( stride == 0 )
  {
    return;
  }

  if ( stride > 0 )
  {
    dir = 1;
  }
  else
  {
    dir = -1;
  }

  istep = from;

  while ( (dir * istep) <= (dir * to) )
  {
    if ( istep != from && pause > 0.0 )
    {
      if ( sleep_( pause ) )
      {
        return;
      }
    }

    globdat.set ( Globdat::TIME_STEP, istep );
    child_->run ( globdat );

    istep += stride;
  }
}


//-----------------------------------------------------------------------
//   sleep_
//-----------------------------------------------------------------------


bool FemvizModule::sleep_ ( double time )
{
  using jem::Time;
  using jem::io::TermInput;
  using jem::io::TermReader;

  TermReader*  tr = TermInput::getReader ();

  double       dt;

  while ( time > 0 )
  {
    if ( tr->poll( Time::zero() ) > 0 )
    {
      return true;
    }

    dt = jem::min ( time, 1.0 );

    Thread::sleep ( Time( dt ) );

    time -= dt;
  }

  return false;
}
