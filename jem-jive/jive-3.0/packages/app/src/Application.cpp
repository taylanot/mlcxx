
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
#include <jem/base/limits.h>
#include <jem/base/Time.h>
#include <jem/base/System.h>
#include <jem/base/Signals.h>
#include <jem/base/Throwable.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/TermInput.h>
#include <jem/io/File.h>
#include <jem/io/FileFlags.h>
#include <jem/io/TeeWriter.h>
#include <jem/io/NullWriter.h>
#include <jem/io/BufferedWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/SimpleLogger.h>
#include <jem/io/PatternLogger.h>
#include <jem/io/NumberFormatter.h>
#include <jem/io/FileOpenException.h>
#include <jem/io/ArrayInputStream.h>
#include <jem/io/ArrayOutputStream.h>
#include <jem/io/ObjectInputStream.h>
#include <jem/io/ObjectOutputStream.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/mp/UniContext.h>
#include <jem/util/Tokenizer.h>
#include <jem/util/Dictionary.h>
#include <jem/util/Properties.h>
#include <jive/Array.h>
#include <jive/mp/Globdat.h>
#include <jive/util/Globdat.h>
#include <jive/util/Random.h>
#include <jive/util/Runtime.h>
#include <jive/util/ObjectConverter.h>
#include <jive/app/Names.h>
#include <jive/app/Module.h>
#include <jive/app/ProgramArgs.h>
#include <jive/app/Application.h>
#include "private/LogState.h"
#include "private/LogWriter.h"
#include "private/LogFileWriter.h"
#include "private/LogTermReader.h"


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::System;
using jem::IllegalInputException;
using jem::io::Logger;
using jem::io::SimpleLogger;
using jem::io::PatternLogger;
using jem::io::BufferedWriter;
using jem::mp::UniContext;
using jem::mp::RecvBuffer;
using jem::mp::SendBuffer;
using jem::util::mergeAndReplace;


//=======================================================================
//   class Application::Utils_
//=======================================================================


class Application::Utils_
{
 public:

  enum
  {
                            LOG,
                            OUT,
                            WARN,
                            ERR,
                            LOG_TYPE_COUNT
  };

  static const char*        PREFIX_PROPS[LOG_TYPE_COUNT];


  static String             expandString

    ( const String&           str,
      int                     myRank,
      int                     procCount );

  static void               parseArgs

    ( int&                    argc,
      char**                  argv,
      String&                 preProps,
      String&                 postProps,
      String&                 propsFile );

  static void               shiftArgs

    ( int&                    argc,
      char**                  argv );

  static void               readProps

    ( const Properties&       props,
      const String&           fileName );

  static void               addProps

    ( const Properties&       props,
      const String&           xtraProps );

  static Ref<Writer>        openLogFile

    ( MPContext&              mpx,
      const String&           fname );

  static void               initSystem

    ( MPContext&              mpx,
      const Properties&       conf,
      const Properties&       props );

  static void               hangupHandler

    ( int                     signum );


 public:

  static volatile bool      hangup;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Application::Utils_::PREFIX_PROPS[LOG_TYPE_COUNT] =
  {
    "logPrefix",
    "outPrefix",
    "warnPrefix",
    "errPrefix"
  };

volatile bool  Application::Utils_::hangup = false;


//-----------------------------------------------------------------------
//   expandString
//-----------------------------------------------------------------------


String Application::Utils_::expandString

  ( const String&  str,
    int            myRank,
    int            procCount )

{
  using jem::StringBuffer;
  using jem::io::NumberFormatter;

  const idx_t      n = str.size ();

  NumberFormatter  nf;
  StringBuffer     buf;

  int              prec = 1;
  idx_t            i    = 0;


  for ( int iproc = 10; iproc < procCount; iproc *= 10 )
  {
    prec++;
  }

  nf.nformat.setIntegerDigits ( prec );

  while ( i < n )
  {
    idx_t  j = str.find ( "%p", i );

    if ( j < 0 )
    {
      buf += str[slice(i,END)];
      i    = n;
    }
    else
    {
      print ( buf, str[slice(i,j)], nf(myRank) );

      i = j + 2;
    }
  }

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   parseArgs
//-----------------------------------------------------------------------


void Application::Utils_::parseArgs

  ( int&     argc,
    char**   argv,
    String&  preProps,
    String&  postProps,
    String&  propsFile )

{
  using jem::StringBuffer;

  StringBuffer  buf;
  String        arg;


  buf.reserve ( 32 );

  while ( argc > 1 )
  {
    arg = argv[1];

    if ( arg == "-p" || arg == "--props" )
    {
      shiftArgs ( argc, argv );

      if ( argc <= 1 )
      {
        throw IllegalInputException (
          String::format ( "command-line option `%s\'", arg ),
          "missing properties list"
        );
      }

      arg = argv[1];
      arg = arg.stripWhite ();

      shiftArgs ( argc, argv );

      if ( arg.size() )
      {
        buf += arg;

        if ( buf.back() != ';' )
        {
          buf += ';';
        }
      }
    }
    else if ( arg == "--" )
    {
      shiftArgs ( argc, argv );
      break;
    }
    else if ( arg.find( '-' ) == 0 )
    {
      throw IllegalInputException (
        "command line",
        String::format (
          "illegal command-line option: `%s\'", arg
        )
      );
    }
    else
    {
      break;
    }
  }

  if ( buf.size() > 0 )
  {
    preProps = buf.toString ();
    buf.clear ();
  }
  else
  {
    preProps = "";
  }

  if ( argc > 1 )
  {
    propsFile = jem::clone ( argv[1] );
    shiftArgs ( argc, argv );
  }

  while ( argc > 1 )
  {
    arg = argv[1];

    if ( arg == "-p" || arg == "--props" )
    {
      shiftArgs ( argc, argv );

      if ( argc <= 1 )
      {
        throw IllegalInputException (
          String::format ( "command-line option `%s\'", arg ),
          "missing properties list"
        );
      }

      arg = argv[1];
      arg = arg.stripWhite ();

      shiftArgs ( argc, argv );

      if ( arg.size() )
      {
        buf += arg;

        if ( buf.back() != ';' )
        {
          buf += ';';
        }
      }
    }
    else
    {
      break;
    }
  }

  if ( buf.size() > 0 )
  {
    postProps = buf.toString ();
  }
  else
  {
    postProps = "";
  }
}


//-----------------------------------------------------------------------
//   shiftArgs
//-----------------------------------------------------------------------


void Application::Utils_::shiftArgs

  ( int&    argc,
    char**  argv )

{
  for ( int i = 2; i < argc; i++ )
  {
    argv[i - 1] = argv[i];
  }

  if ( argc > 1 )
  {
    argc--;
  }
}


//-----------------------------------------------------------------------
//   readProps
//-----------------------------------------------------------------------


void Application::Utils_::readProps

  ( const Properties&  props,
    const String&      fileName )

{
  using jem::io::File;
  using jem::io::FileOpenException;

  String  fname = fileName;

  if ( ! File::exists( fileName ) )
  {
    fname = fileName + ".pro";

    if ( ! File::exists( fname ) )
    {
      throw IllegalInputException (
        "command line",
        String::format (
          "properties file `%s\' does not exist", fileName
        )
      );
    }
  }

  try
  {
    props.parseFile ( fname, Properties::PARSE_INCLUDE );
  }
  catch ( const FileOpenException& )
  {
    throw IllegalInputException (
      "command line",
      String::format (
        "failed to open the properties file `%s\'", fname
      )
    );
  }
}


//-----------------------------------------------------------------------
//   addProps
//-----------------------------------------------------------------------


void Application::Utils_::addProps

  ( const Properties&  props,
    const String&      xtraProps )

{
  using jem::util::Tokenizer;

  Ref<Tokenizer>  tokenizer = newInstance<Tokenizer> ( xtraProps );

  try
  {
    props.parseFrom ( *tokenizer, "<command line>" );
  }
  catch ( const jem::Exception& ex )
  {
    throw IllegalInputException ( "command-line properties",
                                  ex.what() );
  }
}


//-----------------------------------------------------------------------
//   openLogFile
//-----------------------------------------------------------------------


Ref<Writer> Application::Utils_::openLogFile

  ( MPContext&     mpx,
    const String&  fname )

{
  using jem::io::FileFlags;

  const int    myRank    = mpx.myRank ();
  const int    procCount = mpx.size   ();

  Ref<Writer>  log;
  int          token;


  token = 0;

  if ( myRank > 0 )
  {
    mpx.recv ( RecvBuffer( &token, 1 ), myRank - 1 );
  }

  if ( myRank == 0 )
  {
    log = newInstance<LogFileWriter> ( fname );
  }
  else
  {
    log = newInstance<LogFileWriter> ( fname, FileFlags::APPEND );
  }

  if ( myRank < procCount - 1 )
  {
    mpx.send ( SendBuffer( &token, 1 ), myRank + 1 );
  }

  return newInstance<BufferedWriter> ( log );
}


//-----------------------------------------------------------------------
//   initSystem
//-----------------------------------------------------------------------


void Application::Utils_::initSystem

  ( MPContext&         mpx,
    const Properties&  conf,
    const Properties&  props )

{
  using jem::byte;
  using jem::Time;
  using jem::io::TermInput;
  using jem::io::TeeWriter;
  using jem::io::NullWriter;

  Properties          logProps =

    props.findProps ( PropNames::LOG );

  Properties          logConf  =

    conf .makeProps ( PropNames::LOG );

  const int           myRank    = mpx.myRank ();
  const int           procCount = mpx.size   ();

  Ref<LogState>       logState;
  Ref<Writer>         logFile;
  Ref<LogFileWriter>  console;
  Ref<Writer>         bitBucket;
  Ref<Writer>         output;
  Ref<Logger>         logger;

  String              logPrefixes [LOG_TYPE_COUNT];
  String              logPattern;

  String              prefixName;
  String              fileName;
  String              caseName;

  bool                toConsole;
  int                 logRank;
  int                 logPrio;
  int                 i;


  if ( props.find( caseName, Module::CASE_NAME ) )
  {
    fileName = "-" + caseName + ".log";
  }
  else
  {
    fileName = "-";
  }

  logRank    = 0;
  logPattern = "*.info";

  logProps.find ( fileName,   PropNames::FILE    );
  logProps.find ( logRank,    PropNames::RANK    );
  logProps.find ( logPattern, PropNames::PATTERN );

  if ( fileName.find( "%p" ) >= 0 )
  {
    logRank  = -1;
    fileName = expandString ( fileName, myRank, procCount );
  }

  logConf.set ( PropNames::FILE,    fileName   );
  logConf.set ( PropNames::RANK,    logRank    );
  logConf.set ( PropNames::PATTERN, logPattern );

  if ( fileName.startsWith( '-' ) )
  {
    toConsole = true;
    fileName  = fileName[slice(1,END)];
  }
  else
  {
    toConsole = false;
  }

  logPrio = LOG;

  if ( logRank >= 0 && logRank != myRank && logPrio < WARN )
  {
    logPrio = WARN;
  }

  logPrefixes[WARN] = "*** WARNING: ";
  logPrefixes[ERR]  = "*** ERROR: ";

  for ( i = 0; i < LOG_TYPE_COUNT; i++ )
  {
    logProps.find ( logPrefixes[i], PREFIX_PROPS[i] );

    if ( logPrefixes[i].find( "%p" ) >= 0 )
    {
      logPrefixes[i] =

        expandString ( logPrefixes[i], myRank, procCount );
    }

    logConf.set ( PREFIX_PROPS[i], logPrefixes[i] );
  }

  logState  = newInstance<LogState>      ();
  bitBucket = newInstance<NullWriter>    ();
  console   = newInstance<LogFileWriter> ();
  logFile   = bitBucket;

  if ( fileName.size() > 0 )
  {
    if      ( logRank < 0 )
    {
      logFile = openLogFile ( mpx, fileName );
    }
    else if ( logRank == myRank )
    {
      logFile = newInstance<BufferedWriter> (
        newInstance<LogFileWriter> ( fileName )
      );
    }
  }

  output = logFile;

  if ( toConsole && (myRank == logRank || logRank < 0) )
  {
    if ( logFile != bitBucket )
    {
      output = newInstance<TeeWriter> ( console, logFile );
    }
    else
    {
      output = console;
    }
  }

  if ( logPrio <= LOG && logPattern.size() > 0 )
  {
    try
    {
      logger = newInstance<PatternLogger> (
        newInstance<LogWriter> (
          (byte) LOG, logState,
          output,     logPrefixes[LOG]
        ),
        logPattern
      );
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( logProps.getContext( PropNames::PATTERN ) );
      throw;
    }
  }
  else
  {
    logger = newInstance<SimpleLogger> ( bitBucket );
  }

  System::setLogger ( logger );

  output = logFile;

  if ( myRank == logRank || (toConsole && logRank < 0) )
  {
    if ( logFile != bitBucket )
    {
      output = newInstance<TeeWriter> ( console, logFile );
    }
    else
    {
      output = console;
    }
  }

  if ( logPrio <= OUT )
  {
    System::setOutStream (
      newInstance<LogWriter> ( (byte) OUT, logState,
                               output,     logPrefixes[OUT] )
    );
  }
  else
  {
    System::setOutStream ( bitBucket );
  }

  if ( logFile != bitBucket )
  {
    output = newInstance<TeeWriter> ( console, logFile );
  }
  else
  {
    output = console;
  }

  if ( logPrio <= WARN )
  {
    System::setWarnStream (
      newInstance<LogWriter> ( (byte) WARN, logState,
                               output,      logPrefixes[WARN] )
    );
  }
  else
  {
    System::setWarnStream ( bitBucket );
  }

  System::setErrStream (
    newInstance<LogWriter> ( (byte) ERR, logState,
                             output,     logPrefixes[ERR] )
  );

  TermInput::setReader (
    newInstance<LogTermReader> ( logState, console )
  );


  print ( *logFile, "Log file created on ", Time::now(), ".\n\n" );
}


//-----------------------------------------------------------------------
//   hangupHandler
//-----------------------------------------------------------------------


void Application::Utils_::hangupHandler ( int signum )
{
  if ( hangup )
  {
    std::exit ( 1 );
  }
  else
  {
    hangup = true;
  }
}


//=======================================================================
//   class Application
//=======================================================================

//-----------------------------------------------------------------------
//   exec
//-----------------------------------------------------------------------


int Application::exec

  ( int                argc,
    char**             argv,
    ModuleConstructor  ctor )

{
  using jive::util::ObjConverter;

  String  phase  = "initialization";
  int     result = 1;

  try
  {
    Ref<ProgramArgs>  args;
    Ref<MPContext>    mpx;
    Ref<Module>       mod;

    Properties        globdat ( "globdat" );
    Properties        props;
    Properties        conf;


    initSigHandlers ();
    loadProperties  ( props, argc, argv  );

    args = newInstance<ProgramArgs> ( argc, argv );
    mpx  = newInstance<UniContext>  ();

    initSystem  ( *mpx, conf,  props );
    initGlobdat ( conf, props, globdat, mpx, args );

    props.setConverter ( newInstance<ObjConverter>( globdat ) );

    mod = ctor ();

    if ( ! mod )
    {
      return 0;
    }

    runLoop ( phase, *mod, conf, props, globdat );

    result = 0;
  }
  catch ( const jem::Throwable& ex )
  {
    String  where = ex.where ();

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


//-----------------------------------------------------------------------
//   runLoop
//-----------------------------------------------------------------------


void Application::runLoop

  ( String&            phase,
    Module&            mod,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::maxOf;

  Module::Status  stat;


  phase = "configuration";

  mod.configure ( props, globdat );
  mod.getConfig ( conf,  globdat );

  phase = "initialization";
  stat  = mod.init ( conf, props, globdat );
  phase = "runtime";

  if ( globdat.contains( util::Globdat::TIME_STEP ) )
  {
    while ( stat == Module::OK && ! Utils_::hangup )
    {
      stat = mod.run ( globdat );
    }
  }
  else
  {
    util::Globdat::initStep ( globdat );

    while ( stat == Module::OK && ! Utils_::hangup )
    {
      util::Globdat::advanceStep ( globdat );

      stat = mod.run ( globdat );

      util::Globdat::commitStep  ( globdat );
    }
  }

  phase = "shutdown";

  mod.shutdown ( globdat );
}


//-----------------------------------------------------------------------
//   loadProperties (uni-process case)
//-----------------------------------------------------------------------


void Application::loadProperties

  ( const Properties&  props,
    const String&      fname )

{
  String  caseName;

  if ( fname.size() )
  {
    idx_t  i = fname.rfind ( '.' );

    if ( i >= 0 )
    {
      caseName = fname[slice(BEGIN,i)];
    }
    else
    {
      caseName = fname;
    }

    props.set ( Module::CASE_NAME, caseName );

    Utils_::readProps ( props, fname );
  }
}


void Application::loadProperties

  ( const Properties&  props,
    int&               argc,
    char**             argv )

{
  JEM_PRECHECK ( argc >= 0 );

  String  preProps;
  String  postProps;
  String  fileName;

  Utils_::parseArgs ( argc, argv, preProps, postProps, fileName );

  if ( preProps.size() > 0 )
  {
    Utils_::addProps ( props, preProps );
  }

  loadProperties ( props, fileName );

  if ( postProps.size() > 0 )
  {
    Utils_::addProps ( props, postProps );
  }
}


//-----------------------------------------------------------------------
//   loadProperties (multi-process case)
//-----------------------------------------------------------------------


Ref<ProgramArgs> Application::loadProperties

  ( MPContext&         mpx,
    const Properties&  props,
    int&               argc,
    char**             argv )

{
  using jem::byte;
  using jem::Array;
  using jem::io::ArrayInputStream;
  using jem::io::ArrayOutputStream;
  using jem::io::ObjectInputStream;
  using jem::io::ObjectOutputStream;


  StringVector  args;
  Array<byte>   buf;
  idx_t         size;


  if ( mpx.myRank() == 0 )
  {
    Ref<ArrayOutputStream>   bufStream;
    Ref<ObjectOutputStream>  objStream;
    bufStream = newInstance<ArrayOutputStream>  ();
    objStream = newInstance<ObjectOutputStream> ( bufStream );

    loadProperties ( props, argc, argv );

    args.resize ( argc );

    for ( int i = 0; i < argc; i++ )
    {
      args[i] = argv[i];
    }

    encode ( *objStream, props, args );

    objStream->close ();
    objStream = nullptr;

    buf.ref ( bufStream->toArray() );

    if ( ! buf.isContiguous() )
    {
      buf.ref ( buf.clone() );
    }

    size = buf.size ();

    mpx.broadcast ( SendBuffer( &size, 1 ) );
    mpx.broadcast ( SendBuffer( buf.addr(), size ) );
  }
  else
  {
    Ref<ArrayInputStream>   bufStream;
    Ref<ObjectInputStream>  objStream;
    Properties              tmp;
    String                  dummy[3];


    Utils_::parseArgs ( argc, argv, dummy[0], dummy[1], dummy[2] );

    mpx.broadcast ( RecvBuffer( &size, 1 ), 0 );

    buf.resize ( size );

    mpx.broadcast ( RecvBuffer( buf.addr(), size ), 0 );

    bufStream = newInstance<ArrayInputStream>  ( buf );
    objStream = newInstance<ObjectInputStream> ( bufStream );

    decode ( *objStream, tmp, args );

    mergeAndReplace ( props, tmp );
  }

  return newInstance<ProgramArgs> ( args );
}


//-----------------------------------------------------------------------
//   initGlobdat
//-----------------------------------------------------------------------


void Application::initGlobdat

  ( const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat,
    Ref<MPContext>      mpx,
    Ref<ProgramArgs>    args )

{
  using jem::util::Dict;
  using jive::util::Random;
  using jive::util::Runtime;

  Properties  params;

  util::Globdat::init ( globdat );

  if ( props.find( params, PropNames::PARAMS ) )
  {
    mergeAndReplace ( conf.makeProps( PropNames::PARAMS ),
                      params );
    mergeAndReplace (
      globdat.makeProps( util::Globdat::VARIABLES ), params
    );
  }

  args->store      ( globdat );
  globdat.set      ( mp::Globdat::MP_CONTEXT, mpx );
  Random:: install ( globdat );
  Runtime::install ( globdat );
}


//-----------------------------------------------------------------------
//   initSystem
//-----------------------------------------------------------------------


void Application::initSystem

  ( const Properties&  conf,
    const Properties&  props )

{
  Ref<MPContext>  mpx = newInstance<UniContext> ();

  Utils_::initSystem ( *mpx, conf, props );
}


void Application::initSystem

  ( MPContext&         mpx,
    const Properties&  conf,
    const Properties&  props )

{
  Utils_::initSystem ( mpx, conf, props );
}


//-----------------------------------------------------------------------
//   initSigHandlers
//-----------------------------------------------------------------------


void Application::initSigHandlers ()
{
  using jem::Signals;

  Utils_::hangup = false;

  System::setSigHandler ( Signals::HANGUP,
                          Utils_::hangupHandler );
  System::setSigHandler ( Signals::FP_EXCEPTION,
                          System::abortOnSignal );
  System::setSigHandler ( Signals::MEMORY_ERROR,
                          System::abortOnSignal );
}


//-----------------------------------------------------------------------
//   printError
//-----------------------------------------------------------------------


void Application::printError

  ( const String&  phase,
    const String&  type,
    const String&  where,
    const String&  what,
    const String&  trace )

{
  using jem::io::endl;
  using jem::io::flush;
  using jem::io::indent;
  using jem::io::outdent;

  Ref<PrintWriter>  p   =

    newInstance<PrintWriter> ( & System::err() );

  String            typ = type;
  String            ctx = where;
  String            msg = what;


  if ( ! typ.size() )
  {
    typ = "unknown";
  }

  if ( ! ctx.size() )
  {
    ctx = "unspecified";
  }

  if ( ! msg.size() )
  {
    msg = "no details available";
  }

  // Flush all other output streams

  System::flush ();

  p->setPageWidth   ( 80 );
  p->setIndentWidth ( 2 );

  print ( *p, "\nFatal ", phase, " error", indent, endl, endl );

  p->setIndentWidth ( 10 );

  print ( *p, "Type    : ", indent, typ, outdent, endl );
  print ( *p, "Context : ", indent, ctx, outdent, endl );
  print ( *p, "Message : ", indent, msg, outdent, endl );

  print ( *p, outdent );

  if ( trace.size() )
  {
    p->setIndentWidth ( 2 );

    print ( *p, "\nStack trace:\n\n",
                indent, trace, outdent, endl );
  }

  print ( *p, outdent, endl, flush );

}


JIVE_END_PACKAGE( app )
