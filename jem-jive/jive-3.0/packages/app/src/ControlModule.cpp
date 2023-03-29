
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


#include <cctype>
#include <jem/base/assert.h>
#include <jem/base/Time.h>
#include <jem/base/Error.h>
#include <jem/base/Thread.h>
#include <jem/base/System.h>
#include <jem/base/Integer.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/RuntimeException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/utilities.h>
#include <jem/io/list.h>
#include <jem/io/File.h>
#include <jem/io/FileReader.h>
#include <jem/io/NullWriter.h>
#include <jem/io/TermInput.h>
#include <jem/io/IOException.h>
#include <jem/mp/utilities.h>
#include <jem/mp/SyncTimer.h>
#include <jem/util/Pattern.h>
#include <jem/util/Tokenizer.h>
#include <jem/util/Properties.h>
#include <jem/util/PropertyException.h>
#include <jem/xutil/SnoopDictionary.h>
#include <jive/util/utilities.h>
#include <jive/util/FuncUtils.h>
#include <jive/util/ObjectConverter.h>
#include <jive/mp/Globdat.h>
#include <jive/app/typedefs.h>
#include <jive/app/Names.h>
#include <jive/app/Actions.h>
#include <jive/app/Commands.h>
#include <jive/app/CmdCompleter.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/DummyModule.h>
#include <jive/app/ControlModule.h>


JEM_DEFINE_CLASS( jive::app::ControlModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::Time;
using jem::System;
using jem::Thread;
using jem::Exception;
using jem::Collectable;
using jem::io::endl;
using jem::io::flush;
using jem::io::indent;
using jem::io::Reader;
using jem::io::Writer;
using jem::io::TermReader;
using jem::io::WordCompleter;
using jem::mp::SyncTimer;
using jive::util::FuncUtils;


//=======================================================================
//   class ControlModule::RunData_
//=======================================================================


class ControlModule::RunData_ : public Collectable
{
 public:

  inline                    RunData_

    ( const String&           ctx,
      const Ref<MPContext>&   mpx );

  inline void               step

    ( Module&                 module,
      const Properties&       globdat );

  void                      setSteps

    ( idx_t                   count,
      Module&                 module,
      const Properties&       globdat );

  inline idx_t              getSteps    () const;

  virtual void              rehash

    ( Module&                 module,
      const Properties&       globdat );

  void                      watchFile

    ( int                     mode,
      const String&           fname );

  virtual void              setCmdFile

    (  const String&          fname );

  inline  String            checkCommand

    ( const String&           prompt,
      Module&                 module,
      const Properties&       globdat );

  virtual String            readCommand

    ( const String&           prompt,
      Module&                 module,
      const Properties&       globdat )          = 0;


 public:

  const String              context;

  Ref<SyncTimer>            cmdTimer;
  Ref<MPContext>            mpContext;


 protected:

  String                    cmdFileName_;
  idx_t                     stepsLeft_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ControlModule::RunData_::RunData_

  ( const String&          ctx,
    const Ref<MPContext>&  mpx ) :

    context   ( ctx ),
    mpContext ( mpx )

{
  cmdTimer   = newInstance<SyncTimer> ( mpx );
  stepsLeft_ = -1_idx;
}


//-----------------------------------------------------------------------
//   step
//-----------------------------------------------------------------------


inline void ControlModule::RunData_::step

  ( Module&            module,
    const Properties&  globdat )

{
  if ( stepsLeft_ > 0 )
  {
    setSteps ( stepsLeft_ - 1, module, globdat );
  }
}


//-----------------------------------------------------------------------
//   setSteps
//-----------------------------------------------------------------------


void ControlModule::RunData_::setSteps

  ( idx_t              count,
    Module&            module,
    const Properties&  globdat )

{
  // Handle the trivial case.

  if ( count == stepsLeft_ )
  {
    return;
  }

  if ( count <= 0 || count > stepsLeft_ )
  {
    // Change the mode of the given module.

    Properties  params;
    String      action;

    if ( count >= 0 && count <= 5 )
    {
      action = Actions::FG;
    }
    else
    {
      action = Actions::BG;
    }

    module.takeAction ( action, params, globdat );
  }

  stepsLeft_ = count;
}


//-----------------------------------------------------------------------
//   getSteps
//-----------------------------------------------------------------------


inline idx_t ControlModule::RunData_::getSteps () const
{
  return stepsLeft_;
}


//-----------------------------------------------------------------------
//   rehash
//-----------------------------------------------------------------------


void ControlModule::RunData_::rehash

  ( Module&            module,
    const Properties&  globdat )

{}


//-----------------------------------------------------------------------
//   watchFile
//-----------------------------------------------------------------------


void ControlModule::RunData_::watchFile

  ( int            mode,
    const String&  fname )

{
  String  newName = fname;

  if ( newName.size() == 0 && (mode & FG_MODE) )
  {
    newName = "-";
  }

  if ( newName != cmdFileName_ )
  {
    setCmdFile ( newName );
  }
}


//-----------------------------------------------------------------------
//   setCmdFile
//-----------------------------------------------------------------------


void ControlModule::RunData_::setCmdFile ( const String& fname )
{
  cmdFileName_ = fname;

  if ( cmdFileName_.size() )
  {
    if ( mpContext->size() == 1 )
    {
      cmdTimer->setInterval ( 0.5 );
    }
    else
    {
      cmdTimer->setInterval ( 2.0 );
    }

    cmdTimer->reset ();
    cmdTimer->start ();
  }
}


//-----------------------------------------------------------------------
//   checkCommand
//-----------------------------------------------------------------------


inline String ControlModule::RunData_::checkCommand

  ( const String&      prompt,
    Module&            module,
    const Properties&  globdat )

{
  if ( cmdFileName_.size() && (stepsLeft_ == 0 || cmdTimer->check()) )
  {
    return readCommand ( prompt, module, globdat );
  }
  else
  {
    return String ();
  }
}


//=======================================================================
//   class ControlModule::RootData_
//=======================================================================


class ControlModule::RootData_ : public RunData_
{
 public:

  typedef RunData_          Super;


  inline                    RootData_

    ( const String&           ctx,
      const Ref<MPContext>&   mpx );

  virtual void              rehash

    ( Module&                 module,
      const Properties&       globdat )  override;

  virtual void              setCmdFile

    ( const String&           fname )    override;

  virtual String            readCommand

    ( const String&           prompt,
      Module&                 module,
      const Properties&       globdat )  override;


 private:

  void                      readCommand_

    ( String&                 cmd,
      const String&           prompt,
      Module&                 module,
      const Properties&       globdat );

  void                      readTerminal_

    ( String&                 cmd,
      const String&           prompt,
      Module&                 module,
      const Properties&       globdat );

  void                      readCmdFile_

    ( String&                 cmd );


 private:

  Time                      cmdFileStamp_;
  Ref<CmdCompleter>         cmdCompleter_;
  Ref<TermReader>           termReader_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ControlModule::RootData_::RootData_

  ( const String&          ctx,
    const Ref<MPContext>&  mpx ) :

    Super ( ctx, mpx )

{
  JEM_PRECHECK ( mpx->myRank() == 0 );

  cmdFileStamp_ = Time::zero ();
}


//-----------------------------------------------------------------------
//   rehash
//-----------------------------------------------------------------------


void ControlModule::RootData_::rehash

  ( Module&            module,
    const Properties&  globdat )

{
  if ( cmdCompleter_ )
  {
    cmdCompleter_->rehash ( module, globdat );
  }
}


//-----------------------------------------------------------------------
//   setCmdFile
//-----------------------------------------------------------------------


void ControlModule::RootData_::setCmdFile ( const String& fname )
{
  using jem::io::File;
  using jem::io::FileInfo;
  using jem::io::TermInput;

  Super::setCmdFile ( fname );

  cmdFileStamp_ = Time::zero ();
  cmdCompleter_ = nullptr;
  termReader_   = nullptr;

  if      ( cmdFileName_ == "-" )
  {
    termReader_ = TermInput::getReader ();
  }
  else if ( cmdFileName_.size() > 0 )
  {
    FileInfo  info = File::getInfo ( cmdFileName_ );

    if ( info.exists() )
    {
      cmdFileStamp_ = info.mtime;
    }
  }
}


//-----------------------------------------------------------------------
//   readCommand
//-----------------------------------------------------------------------


String ControlModule::RootData_::readCommand

  ( const String&      prompt,
    Module&            module,
    const Properties&  globdat )

{
  using jem::mp::SendBuffer;

  String  cmd;
  idx_t   n;


  if ( ! cmdFileName_.size() )
  {
    return String ();
  }

  readCommand_ ( cmd, prompt, module, globdat );

  if ( cmd.size() == 0 && stepsLeft_ == 0 )
  {
    bool  doSleep = (termReader_ == nullptr);

    do
    {
      if ( doSleep )
      {
        Thread::sleep ( 0.5_sec );
      }

      readCommand_ ( cmd, prompt, module, globdat );
    }
    while ( ! cmd.size() );
  }

  n = cmd.size ();

  mpContext->broadcast ( SendBuffer( &n,         1 ) );
  mpContext->broadcast ( SendBuffer( cmd.addr(), n ) );

  return cmd;
}


//-----------------------------------------------------------------------
//   readCommand_
//-----------------------------------------------------------------------


void ControlModule::RootData_::readCommand_

  ( String&            cmd,
    const String&      prompt,
    Module&            module,
    const Properties&  globdat )

{
  using jem::Time;
  using jem::io::File;
  using jem::io::FileInfo;

  if      ( termReader_ )
  {
    if ( stepsLeft_ != 0 && termReader_->poll( Time::zero() ) == 0 )
    {
      return;
    }
    else
    {
      readTerminal_ ( cmd, prompt, module, globdat );
    }
  }
  else if ( cmdFileName_.size() > 0 )
  {
    FileInfo  info = File::getInfo ( cmdFileName_ );

    if ( ! info.exists() )
    {
      cmdFileStamp_ = Time::zero ();

      return;
    }

    Time  mtime = info.mtime;

    if ( mtime <= cmdFileStamp_ )
    {
      return;
    }

    cmdFileStamp_ = mtime;

    readCmdFile_ ( cmd );
  }
}


//-----------------------------------------------------------------------
//   readTerminal_
//-----------------------------------------------------------------------


void ControlModule::RootData_::readTerminal_

  ( String&            cmd,
    const String&      prompt,
    Module&            module,
    const Properties&  globdat )

{
  using jem::io::IOException;

  try
  {
    if ( ! cmdCompleter_ )
    {
      cmdCompleter_ = newInstance<CmdCompleter> ();

      cmdCompleter_->rehash ( module, globdat );
    }

    cmd = termReader_->readLine ( prompt, cmdCompleter_.get() );
  }
  catch ( const IOException& ex )
  {
    print ( System::err (),
            "Error reading command: ", ex.what (),
            "\n    (command ignored)\n\n" );
  }
}


//-----------------------------------------------------------------------
//   readCmdFile_
//-----------------------------------------------------------------------


void ControlModule::RootData_::readCmdFile_ ( String& cmd )
{
  using jem::StringBuffer;
  using jem::io::FileReader;
  using jem::io::IOException;

  Ref<FileReader>  input;
  StringBuffer     buf;
  int              ch;


  buf.reserve ( 64 );

  ch = -1;

  try
  {
    input = newInstance<FileReader> ( cmdFileName_ );

    input->skipWhite ();
    ch = input->read ();

    while ( ch >= 0 && buf.size() < (128 * 1024) )
    {
      buf += (char) ch;
      ch   = input->read ();
    }
  }
  catch ( const IOException& ex )
  {
    print ( System::err (),
            "Error reading command file `",
            cmdFileName_, "\' : ", ex.what (),
            "\n    (command ignored)\n\n" );

    return;
  }

  if ( ch < 0 )
  {
    cmd = buf.toString ();
  }
  else
  {
    print ( System::warn (),
            "Control file `", cmdFileName_,
            "\' is too large; command ignored\n\n" );
  }
}


//=======================================================================
//   class ControlModule::ProcData_
//=======================================================================


class ControlModule::ProcData_ : public RunData_
{
 public:

  typedef RunData_          Super;


  inline                    ProcData_

    ( const String&           ctx,
      const Ref<MPContext>&   mpx );

  virtual String            readCommand

    ( const String&           prompt,
      Module&                 module,
      const Properties&       globdat )  override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ControlModule::ProcData_::ProcData_

  ( const String&          ctx,
    const Ref<MPContext>&  mpx ) :

    Super ( ctx, mpx )

{
  JEM_PRECHECK ( mpx->myRank() > 0 );
}


//-----------------------------------------------------------------------
//   readCommand
//-----------------------------------------------------------------------


String ControlModule::ProcData_::readCommand

  ( const String&      prompt,
    Module&            module,
    const Properties&  globdat )

{
  using jem::StringBuffer;
  using jem::mp::RecvBuffer;

  if ( ! cmdFileName_.size() )
  {
    return String ();
  }

  StringBuffer  buf;
  idx_t         n;

  mpContext->broadcast ( RecvBuffer( &n, 1 ), 0 );
  mpContext->broadcast ( RecvBuffer( buf.expand( n ), n ), 0 );

  return buf.toString ();
}


//=======================================================================
//   class ControlModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ControlModule::TYPE_NAME = "Control";

const int    ControlModule::FG_MODE   = 1 << 0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ControlModule::ControlModule

  ( const String&       name,
    const Ref<Module>&  child,
    int                 mode ) :

    Super  (  name ),
    child_ ( child ),
    mode_  (  mode )

{
  bool  cond;


  if ( ! child_ )
  {
    child_ = newInstance<DummyModule> ( name );
  }

  if ( mode_ & FG_MODE )
  {
    cmdFile_  = "-";
    cond      = false;
  }
  else
  {
    cond      = true;
  }

  prompt_     = ">";
  pauseTime_  = -1.0;
  runCond_    =  FuncUtils::newCond ( cond );
}


ControlModule::~ControlModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status ControlModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<MPContext>  mpx = mp::Globdat::getMPContext ( globdat );

  rundat_ = nullptr;

  if ( mpx->myRank() == 0 )
  {
    rundat_ = newInstance<RootData_> ( getContext(), mpx );
  }
  else
  {
    rundat_ = newInstance<ProcData_> ( getContext(), mpx );
  }

  rundat_->watchFile ( mode_, cmdFile_ );

  return child_->init ( conf, props, globdat );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status ControlModule::run ( const Properties& globdat )
{
  using jem::mp::allmax;

  if ( ! rundat_ )
  {
    return DONE;
  }

  RunData_&  d    = * rundat_;
  Status     stat = OK;


  if ( d.getSteps() < 0 )
  {
    int  cond = 0;

    try
    {
      cond = (int) FuncUtils::evalCond ( *runCond_, globdat );
    }
    catch ( Exception& ex )
    {
      ex.setContext ( d.context );
      throw;
    }

    if ( allmax( *d.mpContext, cond ) == 0 )
    {
      if ( mode_ & FG_MODE )
      {
        d.setSteps ( 0, *child_, globdat );
      }
      else
      {
        return EXIT;
      }
    }
  }

  try
  {
    stat = child_->run ( globdat );

    if ( stat != OK )
    {
      return stat;
    }
  }
  catch ( const Exception& ex )
  {
    if ( mode_ & FG_MODE )
    {
      print ( System::err(), ex, "\n\n" );

      print ( System::out(), "Type `",
              Commands::EXIT, "\' or `",
              Commands::QUIT, "\' to quit, ",
              "or another command to continue.\n\n" );

      d.setSteps ( 0, *child_, globdat );
    }
    else
    {
      throw;
    }
  }

  String  cmd  = d.checkCommand ( prompt_, *this, globdat );

  if ( cmd.size() )
  {
    String  arg;

    while ( true )
    {
      splitCommand_ ( cmd, arg );

      if ( cmdFile_ != "-" )
      {
        print ( System::debug( myName_ ),
                "Executing command `", cmd, "\' ...\n" );
      }

      if ( cmd == Commands::EXIT || cmd == Commands::QUIT )
      {
        return EXIT;
      }

      tryExeCmd_ ( cmd, arg, globdat );

      if ( d.getSteps() != 0 )
      {
        break;
      }

      cmd = d.readCommand ( prompt_, *this, globdat );
    }
  }
  else if ( pauseTime_ > 0.01 )
  {
    Thread::sleep ( Time( pauseTime_ ) );
  }

  d.step ( *child_, globdat );

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void ControlModule::shutdown ( const Properties& globdat )
{
  if ( rundat_ )
  {
    rundat_ = nullptr;

    child_->shutdown ( globdat );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ControlModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    bool        fgMode;


    if ( myProps.find( cmdFile_, PropNames::CMD_FILE ) )
    {
      if ( rundat_ )
      {
        rundat_->watchFile ( mode_, cmdFile_ );
      }
    }

    if ( myProps.find( fgMode, PropNames::FG_MODE ) )
    {
      int  mode;

      if ( fgMode )
      {
        mode = mode_ |  FG_MODE;
      }
      else
      {
        mode = mode_ & ~FG_MODE;
      }

      setMode ( mode );
    }

    myProps.find ( prompt_,    PropNames::PROMPT );
    myProps.find ( pauseTime_, PropNames::PAUSE  );

    FuncUtils::configCond  ( runCond_,   PropNames::RUN_COND,
                             myProps,    globdat );
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ControlModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );
  bool        fgMode = (mode_ & FG_MODE) ? true : false;


  myConf.set ( PropNames::PROMPT,   prompt_    );
  myConf.set ( PropNames::CMD_FILE, cmdFile_   );
  myConf.set ( PropNames::FG_MODE,  fgMode     );
  myConf.set ( PropNames::PAUSE,    pauseTime_ );

  FuncUtils::getConfig ( myConf, runCond_, PropNames::RUN_COND );
  child_   ->getConfig ( conf,   globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool ControlModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  // Handle the actions that are not propagated to the child module.

  if ( action == Actions::FG )
  {
    setMode ( mode_ |  FG_MODE );

    return true;
  }

  if ( action == Actions::BG )
  {
    setMode ( mode_ & ~FG_MODE );

    return true;
  }

  // Handle the other actions.

  bool  result = child_->takeAction ( action, params, globdat );

  if      ( action == Actions::PRINT_HELP )
  {
    Ref<PrintWriter>  pr;

    params.get ( pr, ActionParams::PRINTER );
    printHelp  ( *pr );

    result = true;
  }
  else if ( action == Actions::GET_COMMANDS )
  {
    getCommands_ ( params, globdat );

    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   setMode
//-----------------------------------------------------------------------


void ControlModule::setMode ( int mode )
{
  if ( mode != mode_ )
  {
    mode_ = mode;

    if ( rundat_ )
    {
      rundat_->watchFile ( mode_, cmdFile_ );
    }
  }
}


//-----------------------------------------------------------------------
//   getPrompt
//-----------------------------------------------------------------------


String ControlModule::getPrompt () const
{
  return prompt_;
}


//-----------------------------------------------------------------------
//   setPrompt
//-----------------------------------------------------------------------


void ControlModule::setPrompt ( const String& prompt )
{
  prompt_ = prompt;
}


//-----------------------------------------------------------------------
//   runWhile
//-----------------------------------------------------------------------


void ControlModule::runWhile ( const String& expr )
{
  JEM_PRECHECK ( expr.size() > 0 );

  runCond_ = FuncUtils::newCond ( expr );
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void ControlModule::printHelp ( PrintWriter& out ) const
{
  using jem::io::beginItem;
  using jem::io::endItem;

  print ( out, beginItem( Commands::HELP ) );
  print ( out,   "Print this help." );
  print ( out, endItem );

  print ( out, beginItem( "exit/quit" ) );
  print ( out,   "Stop the program." );
  print ( out, endItem );

  print ( out, beginItem( Commands::STOP ) );
  print ( out,   "Stop the program and wait for the next command." );
  print ( out, endItem );

  print ( out, beginItem( Commands::CONT ) );
  print ( out,   "Resume the execution of the program." );
  print ( out, endItem );

  print ( out, beginItem( Commands::FG ) );
  print ( out,   "Put the program in the foreground mode and wait " );
  print ( out,   "for the next command." );
  print ( out, endItem );

  print ( out, beginItem( Commands::BG ) );
  print ( out,   "Put the program in the background mode and " );
  print ( out,   "resume its execution." );
  print ( out, endItem );

  print ( out, beginItem( "s/step [N]" ) );
  print ( out,   "Execute <N> iterations/time steps. " );
  print ( out,   "If <N> is not specified, one step is executed." );
  print ( out, endItem );

  print ( out, beginItem( "set <name>=<value>" ) );
  print ( out,   "Set the property <name> to <value>." );
  print ( out, endItem );

  print ( out, beginItem( "eval <expr>" ) );
  print ( out,   "Evaluate the expression <expr> and print " );
  print ( out,   "the result. The expression may involve any " );
  print ( out,   "input parameters and runtime variables." );
  print ( out, endItem );

  print ( out, beginItem( "show [pattern]" ) );
  print ( out,   "Show the properties matching <pattern>. " );
  print ( out,   "If <pattern> is not specified, all properties " );
  print ( out,   "are printed." );
  print ( out, endItem );

  print ( out, beginItem( Commands::REHASH ) );
  print ( out,   "Update the internal command completion " );
  print ( out,   "data base." );
  print ( out, endItem );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> ControlModule::makeNew

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


void ControlModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   getCommands_
//-----------------------------------------------------------------------


void ControlModule::getCommands_

  ( const Properties&  cmds,
    const Properties&  globdat )

{
  Properties  conf;

  cmds.set ( Commands::HELP,   "" );
  cmds.set ( Commands::QUIT,   "" );
  cmds.set ( Commands::EXIT,   "" );
  cmds.set ( Commands::STOP,   "" );
  cmds.set ( Commands::CONT,   "" );
  cmds.set ( Commands::FG,     "" );
  cmds.set ( Commands::BG,     "" );
  cmds.set ( Commands::REHASH, "" );
  cmds.set ( Commands::STEP,   "1" );
  cmds.set ( Commands::EVAL,   ""  );

  getConfig ( conf, globdat );

  StringVector  list = conf.listProps ();

  cmds.set ( Commands::SET,  list );
  cmds.set ( Commands::SHOW, list );
}


//-----------------------------------------------------------------------
//   tryExeCmd_
//-----------------------------------------------------------------------


bool ControlModule::tryExeCmd_

  ( const String&      cmd,
    const String&      arg,
    const Properties&  globdat )

{
  using jem::RuntimeException;
  using jem::util::PropertyException;

  bool  result = false;

  try
  {
    exeCommand_ ( cmd, arg, globdat );

    result = true;
  }
  catch ( const PropertyException& ex )
  {
    print ( System::warn(),
            "Error executing command `", cmd,
            "\' : ", ex, "\n\n" );
  }
  catch ( const Exception& ex )
  {
    print ( System::warn(),
            "Error executing command `", cmd, "\' : ",
            ex.what(), "\n\n" );
  }

  return result;
}


//-----------------------------------------------------------------------
//   exeCommand_
//-----------------------------------------------------------------------


void ControlModule::exeCommand_

  ( const String&      cmd,
    const String&      arg,
    const Properties&  globdat )

{
  JEM_PRECHECK ( rundat_ );

  RunData_&  d = * rundat_;


  if      ( cmd == Commands::STOP )
  {
    d.setSteps (  0, *child_, globdat );
  }
  else if ( cmd == Commands::CONT )
  {
    d.setSteps ( -1, *child_, globdat );
  }
  else if ( cmd == Commands::FG )
  {
    setMode    ( mode_ |  FG_MODE );
    d.setSteps (  0, *child_, globdat );
  }
  else if ( cmd == Commands::BG )
  {
    setMode    ( mode_ & ~FG_MODE );
    d.setSteps ( -1, *child_, globdat );
  }
  else if ( cmd == "s" || cmd == Commands::STEP )
  {
    int  n = 1;

    if ( arg.size() )
    {
      if ( ! jem::Integer::parse( n, arg ) || n <= 0 )
      {
        throw Exception (
          cmd,
          String::format ( "invalid number of steps: %s", arg )
        );
      }
    }

    d.setSteps ( n, *child_, globdat );
  }
  else if ( cmd == Commands::SET )
  {
    if ( ! arg.size() )
    {
      throw Exception ( cmd, "no properties to set specified" );
    }

    exeSetCmd_  ( arg, globdat );
  }
  else if ( cmd == Commands::EVAL )
  {
    exeEvalCmd_ ( arg, globdat );
  }
  else if ( cmd == Commands::SHOW )
  {
    exeShowCmd_ ( arg, globdat );
  }
  else if ( cmd == Commands::HELP )
  {
    exeHelpCmd_ ( arg, globdat );
  }
  else
  {
    Properties  params ( "params" );

    bool        result = false;

    params.set ( ActionParams::COMMAND,     cmd );
    params.set ( ActionParams::COMMAND_ARG, arg );

    result = child_->takeAction ( Actions::EXE_COMMAND,
                                  params, globdat );

    if ( ! result )
    {
      print ( System::warn(),
              "Unknown command: `", cmd,
              "\'; try `help\'\n\n" );
    }
  }
}


//-----------------------------------------------------------------------
//   exeSetCmd_
//-----------------------------------------------------------------------


void ControlModule::exeSetCmd_

  ( const String&      arg,
    const Properties&  globdat )

{
  using jem::io::Writer;
  using jem::util::Tokenizer;
  using jem::xutil::SnoopDict;
  using jive::util::ObjConverter;

  Ref<Tokenizer>     tok;

  Properties         props;

  Ref<SnoopDict>     snoop;
  Ref<ObjConverter>  conv;
  StringVector       list;


  if ( arg.size() > 0 && arg.back() != ';' )
  {
    tok = newInstance<Tokenizer> ( arg + ';' );
  }
  else
  {
    tok = newInstance<Tokenizer> ( arg );
  }

  props.parseFrom ( *tok, "command" );

  snoop = newInstance<SnoopDict>    ( props.getContents() );
  conv  = newInstance<ObjConverter> ( globdat );
  props = Properties ( snoop, conv, props.getName() );

  configure ( props, globdat );
  list.ref  ( snoop->listUntouchedObjects() );

  if ( list.size() )
  {
    Writer&  warn = System::warn ();

    print ( warn, "Command `", Commands::SET, "\' : " );

    if ( list.size() == 1 )
    {
      print ( warn, "unknown property: `", list[0], "\'\n\n" );
    }
    else
    {
      jem::sort ( list );

      print ( warn, "unknown properties:\n" );

      for ( idx_t i = 0; i < list.size(); i++ )
      {
        print ( warn, "\n  ", list[i] );
      }

      print ( warn, "\n\n" );
    }
  }
}


//-----------------------------------------------------------------------
//   exeEvalCmd_
//-----------------------------------------------------------------------


void ControlModule::exeEvalCmd_

  ( const String&      arg,
    const Properties&  globdat )

{
  using jem::IllegalInputException;

  if ( arg.size() == 0 )
  {
    throw IllegalInputException (
      getContext (),
      "no expression specified"
    );
  }

  double  val = FuncUtils::evalExpr ( arg, globdat );

  print ( System::out(), "  ", val, endl );
}


//-----------------------------------------------------------------------
//   exeShowCmd_
//-----------------------------------------------------------------------


void ControlModule::exeShowCmd_

  ( const String&      arg,
    const Properties&  globdat )

{
  using jem::io::NullWriter;
  using jem::util::Pattern;

  Ref<MPContext>    mpx = mp::Globdat::getMPContext ( globdat );

  Ref<PrintWriter>  pr;
  Properties        conf;


  if ( mpx->myRank() == 0 )
  {
    pr = newInstance<PrintWriter> ( & System::out() );
  }
  else
  {
    pr = newInstance<PrintWriter> ( newInstance<NullWriter>() );
  }

  getConfig ( conf, globdat );

  print ( *pr, endl, indent );

  if ( arg.size() == 0 )
  {
    conf.printTo ( *pr );
  }
  else
  {
    StringVector  list = conf.listProps ();

    Properties    tmp;
    Ref<Object>   obj;

    Pattern::check ( arg );

    for ( idx_t i = 0; i < list.size(); i++ )
    {
      if ( Pattern::matches( arg, list[i] ) )
      {
        if ( conf.find( obj, list[i] ) )
        {
          tmp.set ( list[i], obj );
        }
      }
    }

    tmp.printTo ( *pr );
  }

  print ( *pr, endl, flush );
}


//-----------------------------------------------------------------------
//   exeHelpCmd_
//-----------------------------------------------------------------------


void ControlModule::exeHelpCmd_

  ( const String&      arg,
    const Properties&  globdat )

{
  using jem::io::beginList;
  using jem::io::endList;
  using jem::io::NullWriter;

  Ref<MPContext>    mpx = mp::Globdat::getMPContext ( globdat );

  Ref<PrintWriter>  pr;

  Properties        params ( "actionParams" );


  if ( mpx->myRank() == 0 )
  {
    pr = newInstance<PrintWriter> ( & System::out() );
  }
  else
  {
    pr = newInstance<PrintWriter> ( newInstance<NullWriter>() );
  }

  params.set ( ActionParams::PRINTER, pr );

  pr->setPageWidth ( 72 );

  print ( *pr, indent, beginList );

  printHelp          ( *pr );
  child_->takeAction ( Actions::PRINT_HELP, params, globdat );

  print ( *pr, endList, flush );
}


//-----------------------------------------------------------------------
//   splitCommand_
//-----------------------------------------------------------------------


void ControlModule::splitCommand_

  ( String&  cmd,
    String&  arg )

{
  String       text = cmd;

  const char*  s    = text.addr ();
  const idx_t  n    = text.size ();

  idx_t        i, j;


  cmd = "";
  arg = "";

  for ( i = 0; i < n &&   std::isspace(s[i]); i++ ) ;

  for ( j = i; j < n && ! std::isspace(s[j]); j++ ) ;

  if ( i == j )
  {
    return;
  }

  if ( i == 0 && j == n )
  {
    cmd = text;
    return;
  }

  cmd = text[slice(i,j)];

  for ( i = j; i < n && std::isspace(s[i]);     i++ ) ;

  for ( j = n; j > i && std::isspace(s[j - 1]); j-- ) ;

  if ( j > i )
  {
    arg = text[slice(i,j)];
  }
}


JIVE_END_PACKAGE( app )
