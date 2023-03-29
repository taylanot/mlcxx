
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


#include <cstdlib>
#include <jem/version.h>
#include <jem/util/HashMap.h>
#include <jem/io/NullWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileName.h>
#include <jem/io/StdinReader.h>
#include <jem/io/StderrWriter.h>
#include <jem/io/StdoutWriter.h>
#include <jem/io/SimpleLogger.h>
#include <jem/base/utilities.h>
#include <jem/base/CString.h>
#include <jem/base/Thread.h>
#include <jem/base/Clonable.h>
#include <jem/base/Once.h>
#include <jem/base/Exception.h>
#include <jem/base/LogBuffer.h>
#include <jem/base/native/System.h>
#include <jem/base/thread/ThreadLib.h>
#include <jem/base/thread/ThreadLocal.h>
#include <jem/base/Signals.h>
#include <jem/base/System.h>


JEM_BEGIN_PACKAGE_BASE


using jem::io::Reader;
using jem::io::Writer;
using jem::io::SimpleLogger;
using jem::util::HashMap;


//=======================================================================
//   class System::Properties_
//=======================================================================


class System::Properties_ : public HashMap<String,String>
{
 public:

  typedef Properties_     Self;
  typedef ValueType       Pair;


                          Properties_ ();

  static inline Self&     instance    ();


 private:

  static void             initOnce_   ();


 private:

  static Once             once_;
  static Self*            instance_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Once System::Properties_::once_ = JEM_ONCE_INITIALIZER;

System::Properties_*  System::Properties_::instance_ = nullptr;


//-----------------------------------------------------------------------
//   instance
//-----------------------------------------------------------------------


inline System::Properties_& System::Properties_::instance ()
{
  runOnce ( once_, initOnce_ );

  return *instance_;
}


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


System::Properties_::Properties_ ()
{
  SystemInfo  info = NativeSystem::getInfo ();


  // First some info about jem ...

  insert ( Pair( "jem.version", JEM_VERSION ) );
  insert ( Pair( "jem.date",    __DATE__    ) );

  // File system properties

  insert ( Pair( "file.separator",
                 String( io::FileName::fileSeparator() ) ) );
  insert ( Pair( "path.separator",
                 String( io::FileName::pathSeparator() ) ) );

  // Some info about the user ...

  if ( info.username.size() )
  {
    insert ( Pair( "user.name", info.username ) );
  }

  if ( info.userhome.size() )
  {
    insert ( Pair( "user.home", info.userhome ) );
  }

  // Some info about the host and the OS ...

  if ( info.hostname.size() )
  {
    insert ( Pair( "host.name",  info.hostname ) );
  }

  if ( info.totalmem > 0.0 )
  {
    insert ( Pair( "host.totalmem", String( info.totalmem ) ) );
  }

  if ( info.nrprocs > 0 )
  {
    insert ( Pair( "host.nrprocs", String( info.nrprocs ) ) );
  }

  if ( info.osname.size() )
  {
    insert ( Pair( "os.name", info.osname ) );
  }

  if ( info.machine.size() )
  {
    insert ( Pair( "os.machine", info.machine ) );
  }

  if ( info.version.size() )
  {
    insert ( Pair( "os.version", info.version ) );
  }
}


//-----------------------------------------------------------------------
//   initOnce_
//-----------------------------------------------------------------------


void System::Properties_::initOnce_ ()
{
  instance_ = new Properties_ ();
}


//=======================================================================
//   class System::IOStreams_
//=======================================================================


class System::IOStreams_ : public Object,
                           public Clonable
{
 public:

  typedef IOStreams_      Self;
  typedef Object          Super;


  inline                  IOStreams_  ();

  inline                  IOStreams_

    ( const Self&           rhs );

  virtual Ref<Object>     clone       () const override;

  static inline Self&     get         ();
  static inline Self*     find        ();


 public:

  Ref<Reader>             input;
  Ref<Writer>             outputs[LAST_OUTPUT + 1];
  Ref<Logger>             logger;


 protected:

  virtual                ~IOStreams_  ();


 private:

  void                    initIO_     ();

  static Self*            makeNew_    ();

};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


System::IOStreams_::IOStreams_ ()
{
  initIO_ ();
}


inline System::IOStreams_::IOStreams_ ( const Self& rhs )
{
  const char*  names[] = { "err", "warn", "out" };


  try
  {
    input = rhs.input->dup ();
  }
  catch ( const Exception& )
  {
    print ( *rhs.outputs[ERR_STREAM],
            "The input stream jem::System::in can not be "
            "duplicated;\n",
            "creating a default input stream for the new "
            "thread.\n\n" );

    input = nullptr;
  }

  for ( int i = 0; i <= LAST_OUTPUT; i++ )
  {
    try
    {
      outputs[i] = rhs.outputs[i]->dup ();
    }
    catch ( const Exception& )
    {
      print ( *rhs.outputs[ERR_STREAM],
              "The output stream jem::System::", names[i],
              " can not be duplicated;\n",
              "creating a default output stream for the new "
              "thread.\n\n" );

      outputs[i] = nullptr;
    }
  }

  try
  {
    logger = rhs.logger->dup ();
  }
  catch ( const Exception& )
  {
    print ( *rhs.outputs[ERR_STREAM],
            "The output stream jem::System::log can not be "
            "duplicated;\n",
            "creating a default log stream for the new "
            "thread.\n\n" );

    logger = nullptr;
  }

  initIO_ ();
}


System::IOStreams_::~IOStreams_ ()
{}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> System::IOStreams_::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline System::IOStreams_& System::IOStreams_::get ()
{
  ThreadLib::init ();

  Self*  self =

    static_cast<Self*> ( ThreadLib::getLocal()->getSysdata() );

  if ( ! self )
  {
    self = makeNew_ ();
  }

  return *self;
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline System::IOStreams_* System::IOStreams_::find ()
{
  ThreadLib::init ();

  return static_cast<Self*> ( ThreadLib::getLocal()->getSysdata() );
}


//-----------------------------------------------------------------------
//   initIO_
//-----------------------------------------------------------------------


void System::IOStreams_::initIO_ ()
{
  if ( ! outputs[ERR_STREAM] )
  {
    outputs[ERR_STREAM] = newInstance<io::StderrWriter> ();
  }

  if ( ! outputs[WARN_STREAM] )
  {
    outputs[WARN_STREAM] = newInstance<io::StderrWriter> ();
  }

  if ( ! outputs[OUT_STREAM] )
  {
    outputs[OUT_STREAM] = newInstance<io::StdoutWriter> ();
  }

  if ( ! input )
  {
    input = newInstance<io::StdinReader> ();
  }

  if ( ! logger )
  {
    logger = newInstance<SimpleLogger> (
      newInstance<io::NullWriter> ()
    );
  }
}


//-----------------------------------------------------------------------
//   makeNew_
//-----------------------------------------------------------------------


System::IOStreams_* System::IOStreams_::makeNew_ ()
{
  Ref<Self>  tmp = newInstance<Self> ();

  ThreadLib::getLocal()->setSysdata ( tmp );

  return tmp.get ();
}


//=======================================================================
//   class System::Flusher_
//=======================================================================


class System::Flusher_
{
  public:

                      Flusher_ ();
                     ~Flusher_ ();


    static Flusher_   instance;
};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


System::Flusher_ System::Flusher_::instance;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


System::Flusher_::Flusher_ ()
{}


System::Flusher_::~Flusher_ ()
{
  IOStreams_*  ios = IOStreams_::find ();

  if ( ios )
  {
    for ( int i = 0; i <= LAST_OUTPUT; i++ )
    {
      try
      {
        ios->outputs[i]->flush ();
      }
      catch ( ... ) {}
    }

    try
    {
      ios->logger->flush ();
    }
    catch ( ... ) {}
  }
}


//=======================================================================
//   class System
//=======================================================================

//-----------------------------------------------------------------------
//   in
//-----------------------------------------------------------------------


Reader& System::in ()
{
  return *IOStreams_::get().input;
}


//-----------------------------------------------------------------------
//   err
//-----------------------------------------------------------------------


Writer& System::err ()
{
  return *IOStreams_::get().outputs[ERR_STREAM];
}


//-----------------------------------------------------------------------
//   warn
//-----------------------------------------------------------------------


Writer& System::warn ()
{
  return *IOStreams_::get().outputs[WARN_STREAM];
}


//-----------------------------------------------------------------------
//   out
//-----------------------------------------------------------------------


Writer& System::out ()
{
  return *IOStreams_::get().outputs[OUT_STREAM];
}


//-----------------------------------------------------------------------
//   log
//-----------------------------------------------------------------------


Writer& System::log ()
{
  return IOStreams_::get().logger->getWriter ( String() );
}


Writer& System::log ( const String& tag )
{
  return IOStreams_::get().logger->getWriter ( tag );
}



//-----------------------------------------------------------------------
//   info
//-----------------------------------------------------------------------


Writer& System::info ()
{
  return IOStreams_::get().logger->getWriter ( String( ".info" ) );
}


Writer& System::info ( const String& tag )
{
  return IOStreams_::get().logger->getWriter ( tag + ".info" );
}


//-----------------------------------------------------------------------
//   debug
//-----------------------------------------------------------------------


Writer& System::debug ()
{
  return IOStreams_::get().logger->getWriter ( String( ".debug" ) );
}


Writer& System::debug ( const String& tag )
{
  return IOStreams_::get().logger->getWriter ( tag + ".debug" );
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void System::flush ()
{
  IOStreams_*  ios = IOStreams_::find ();

  if ( ios )
  {
    for ( int i = 0; i <= LAST_OUTPUT; i++ )
    {
      ios->outputs[i]->flush ();
    }

    ios->logger->flush ();
  }
}


//-----------------------------------------------------------------------
//   getOutput
//-----------------------------------------------------------------------


Writer& System::getOutput ( OutputID iod )
{
  return *IOStreams_::get().outputs[iod];
}


//-----------------------------------------------------------------------
//   setOutput
//-----------------------------------------------------------------------


void System::setOutput ( OutputID iod, const Ref<Writer>& out )
{
  JEM_PRECHECK2 ( out, "NULL output stream" );

  IOStreams_::get().outputs[iod] = out;
}


//-----------------------------------------------------------------------
//   setInStream
//-----------------------------------------------------------------------


void System::setInStream ( const Ref<Reader>& r )
{
  JEM_PRECHECK2 ( r, "NULL input stream" );

  IOStreams_::get().input = r;
}


//-----------------------------------------------------------------------
//   setErrStream
//-----------------------------------------------------------------------


void System::setErrStream ( const Ref<Writer>& w )
{
  setOutput ( ERR_STREAM, w );
}


//-----------------------------------------------------------------------
//   setWarnStream
//-----------------------------------------------------------------------


void System::setWarnStream ( const Ref<Writer>& w )
{
  setOutput ( WARN_STREAM, w );
}


//-----------------------------------------------------------------------
//   setOutStream
//-----------------------------------------------------------------------


void System::setOutStream ( const Ref<Writer>& w )
{
  setOutput ( OUT_STREAM, w );
}


//-----------------------------------------------------------------------
//   getLogger
//-----------------------------------------------------------------------


Ref<io::Logger> System::getLogger ()
{
  return IOStreams_::get().logger;
}


//-----------------------------------------------------------------------
//   setLogger
//-----------------------------------------------------------------------


Ref<io::Logger> System::setLogger ( const Ref<Logger>& log )
{
  JEM_PRECHECK2 ( log, "NULL logger" );

  IOStreams_&  ios = IOStreams_::get ();
  Ref<Logger>  tmp = ios.logger;

  ios.logger = log;

  return tmp;
}


//-----------------------------------------------------------------------
//   getProperty
//-----------------------------------------------------------------------


String System::getProperty

  ( const String&  key,
    const String&  def )

{
  Properties_&           props = Properties_::instance ();
  Properties_::Iterator  it    = props.find ( key );

  if ( it == props.end() )
  {
    return def;
  }
  else
  {
    return it->second;
  }
}


//-----------------------------------------------------------------------
//   getenv
//-----------------------------------------------------------------------


String System::getenv

  ( const String&  key,
    const String&  def )

{
  const char* str = ::getenv ( makeCString( key ) );

  if ( str )
  {
    return String ( clone( str ) );
  }
  else
  {
    return def;
  }
}


//-----------------------------------------------------------------------
//   exec
//-----------------------------------------------------------------------


int System::exec ( int (*func)() )
{
  JEM_PRECHECK2 ( func != nullptr, "NULL function pointer" );

  int  retval = 0;

  NativeSystem::setSigHandler ( static_cast<int>(Signals::FP_EXCEPTION),
                                abortOnSignal );
  NativeSystem::setSigHandler ( static_cast<int>(Signals::MEMORY_ERROR),
                                abortOnSignal );

  try
  {
    retval = func ();
  }
  catch ( const Throwable& ex )
  {
    printException ( err(), ex );
    printLogTrace  ( debug() );

    retval = 1;
  }
  catch ( ... )
  {
    printException ( err(), "unknown", "unknown", "unknown" );
    printLogTrace  ( debug() );

    retval = 1;
  }

  return retval;
}


int System::exec

  ( int   (*func)( int, char** ),
    int     argc,
    char**  argv )

{
  JEM_PRECHECK2 ( func != nullptr, "NULL function pointer" );

  int  retval = 0;

  NativeSystem::setSigHandler ( static_cast<int>(Signals::FP_EXCEPTION),
                                abortOnSignal );
  NativeSystem::setSigHandler ( static_cast<int>(Signals::MEMORY_ERROR),
                                abortOnSignal );

  try
  {
    retval = func ( argc, argv );
  }
  catch ( const Throwable& ex )
  {
    String  where = ex.where ();

    if ( ! where.size() && argc > 0 )
    {
      where = argv[0];
    }

    printException ( err(), ex.name(),
                     where, ex.what(), ex.getStackTrace() );
    printLogTrace  ( debug() );

    retval = 1;
  }
  catch ( ... )
  {
    printException ( err() );
    printLogTrace  ( debug() );

    retval = 1;
  }

  return retval;
}


//-----------------------------------------------------------------------
//   werror
//-----------------------------------------------------------------------


void System::werror ( const char* str )
{
  if ( str )
  {
    int  k = 0;
    int  c;

    // Determine the length of the string in a robust kind of way.

    while ( k < 1000 )
    {
      c = str[k];

      if ( c == '\0' || c < 0 || c > 127 )
      {
        break;
      }

      k++;
    }

    NativeSystem::werror ( str, k );
  }
}


void System::werror

  ( const char*  str,
    idx_t        len )

{
  if ( str && len >= 0_lint )
  {
    NativeSystem::werror ( str, len );
  }
}


//-----------------------------------------------------------------------
//   getStackTrace
//-----------------------------------------------------------------------


String System::getStackTrace ( int offset )
{
  return NativeSystem::getStackTrace ( offset + 1 );
}


//-----------------------------------------------------------------------
//   dumpStackTrace
//-----------------------------------------------------------------------


void System::dumpStackTrace ( int offset ) noexcept
{
  NativeSystem::dumpStackTrace ( offset + 1 );
}


//-----------------------------------------------------------------------
//   setSigHandler
//-----------------------------------------------------------------------


System::SigHandler System::setSigHandler

  ( int         signum,
    SigHandler  handler )

{
  return NativeSystem::setSigHandler ( signum, handler );
}


//-----------------------------------------------------------------------
//   abortOnSignal
//-----------------------------------------------------------------------


void System::abortOnSignal ( int signum )
{
  static volatile bool  called = false;

  // Avoid recursive signals.

  if ( called )
  {
    std::abort ();
  }

  called = true;

  werror ( "\n*** " );

  switch ( signum )
  {
  case Signals::INTERRUPT:

    werror ( "Program interrupted" );
    break;

  case Signals::HANGUP:

    werror ( "Program received a hangup signal" );
    break;

  case Signals::FP_EXCEPTION:

    werror ( "Floating point exception" );
    break;

  case Signals::MEMORY_ERROR:

    werror ( "Invalid memory reference" );
    break;

  default:

    werror ( "Program received an unknown signal" );
  }

  werror  ( "; stack trace follows ***\n\n" );

  NativeSystem::dumpStackTrace ( 1 );

  werror  ( "\n\n" );
  werror  ( "*** Log trace follows ***\n\n" );

  // This may crash everything, but an abort will follow anyway.

  LogBuffer::printTrace ( System::err() );

  std::abort ();
}


//-----------------------------------------------------------------------
//   printException
//-----------------------------------------------------------------------


void System::printException

  ( Writer&           out,
    const Throwable&  ex )

{
  printException ( out,
                   ex.name          (),
                   ex.where         (),
                   ex.what          (),
                   ex.getStackTrace () );
}


void System::printException

  ( Writer&        out,
    const String&  name,
    const String&  where,
    const String&  what,
    const String&  trace )

{
  using jem::io::endl;
  using jem::io::print;
  using jem::io::flush;
  using jem::io::indent;
  using jem::io::outdent;
  using jem::io::PrintWriter;

  Ref<PrintWriter>  p   = newInstance<PrintWriter> ( & out );

  String            nam = name;
  String            ctx = where;
  String            msg = what;


  if ( ! nam.size() )
  {
    nam = "unknown";
  }

  if ( ! ctx.size() )
  {
    ctx = "unspecified";
  }

  if ( ! msg.size() )
  {
    msg = "no details available";
  }

  System::flush ();

  p->setPageWidth   ( 80 );
  p->setIndentWidth ( 2 );

  print ( *p, "\n*** Uncaught exception ***\n\n", indent );

  p->setIndentWidth ( 10 );

  print ( *p, "Type    : ", indent, nam, outdent, endl );
  print ( *p, "Context : ", indent, ctx, outdent, endl );
  print ( *p, "Message : ", indent, msg, outdent, endl );

  print ( *p, outdent );

  if ( trace.size() )
  {
    p->setIndentWidth ( 2 );

    print ( *p, "\nStack trace:\n\n",
            indent, trace, outdent, endl );
  }

  print ( *p, endl, flush );
}


//-----------------------------------------------------------------------
//   printLogTrace
//-----------------------------------------------------------------------


void System::printLogTrace ( Writer& out )
{
  using jem::io::endl;
  using jem::io::print;
  using jem::io::flush;
  using jem::io::indent;
  using jem::io::outdent;
  using jem::io::PrintWriter;

  String  trace = LogBuffer::getTrace ( 32 );

  if ( trace.size() )
  {
    Ref<PrintWriter>  p = newInstance<PrintWriter> ( & out );

    System::flush ();

    p->setPageWidth   ( 80 );
    p->setIndentWidth ( 2 );

    print ( *p, "\nLog buffer trace:\n\n" );
    print ( *p, indent, trace, outdent, endl );
    print ( *p, endl, flush );
  }
}


JEM_END_PACKAGE_BASE
