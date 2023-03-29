
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


#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/list.h>
#include <jem/io/File.h>
#include <jem/io/FileStream.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/FileOutputStream.h>
#include <jem/io/ArrayInputStream.h>
#include <jem/io/ArrayOutputStream.h>
#include <jem/io/DataInputStream.h>
#include <jem/io/DataOutputStream.h>
#include <jem/io/ObjectInputStream.h>
#include <jem/io/ObjectOutputStream.h>
#include <jem/io/SerializationException.h>
#include <jem/mp/utilities.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/FuncUtils.h>
#include <jive/mp/Globdat.h>
#include <jive/app/import.h>
#include <jive/app/typedefs.h>
#include <jive/app/Names.h>
#include <jive/app/Actions.h>
#include <jive/app/Commands.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/CheckpointModule.h>


JEM_DEFINE_CLASS( jive::app::CheckpointModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::System;
using jem::Exception;
using jem::io::endl;
using jem::io::flush;
using jem::io::File;
using jem::io::Writer;
using jem::io::ArrayInputStream;
using jem::io::ArrayOutputStream;
using jem::io::ObjectInputStream;
using jem::io::ObjectOutputStream;
using jem::io::SerializationException;
using jem::mp::RecvBuffer;
using jem::mp::SendBuffer;
using jive::util::FuncUtils;

typedef jem::Array<byte>   ByteVector;


//=======================================================================
//   class CheckpointModule::Utils_
//=======================================================================


class CheckpointModule::Utils_
{
 public:

  static void             writeData

    ( MPContext&            mpx,
      const ByteVector&     data,
      const String&         fname,
      bool                  sync );

  static void             writeData

    ( MPContext&            mpx,
      const ByteVector&     data );

  static ByteVector       readData

    ( MPContext&            mpx,
      const String&         fname );

  static ByteVector       readData

    ( MPContext&            mpx );

};


//-----------------------------------------------------------------------
//   writeData (root process)
//-----------------------------------------------------------------------


void CheckpointModule::Utils_::writeData

  ( MPContext&         mpx,
    const ByteVector&  data,
    const String&      fname,
    bool               sync )

{
  using jem::io::File;
  using jem::io::FileStream;
  using jem::io::FileFlags;
  using jem::io::FileOutputStream;
  using jem::io::DataOutputStream;

  const int              procCount = mpx.size ();

  Ref<DataOutputStream>  output;
  Ref<FileStream>        file;
  ByteVector             buf;
  int                    rank;
  idx_t                  n;


  file = File::open ( fname, FileFlags::WRITE );

  output = newInstance<DataOutputStream> (
    newInstance<FileOutputStream> ( file )
  );

  encode ( *output, procCount );

  for ( rank = procCount - 1; rank > 0; rank-- )
  {
    mpx.recv ( RecvBuffer( &n, 1 ), rank );

    buf.resize ( n );

    mpx.recv ( RecvBuffer( buf.addr(), n ), rank );
    encode   ( *output, rank, buf );
  }

  encode ( *output, rank, data );

  if ( sync )
  {
    file->sync ();
  }
}


//-----------------------------------------------------------------------
//   writeData (other processes)
//-----------------------------------------------------------------------


void CheckpointModule::Utils_::writeData

  ( MPContext&         mpx,
    const ByteVector&  data )

{
  const idx_t  n = data.size ();

  mpx.send ( SendBuffer( &n, 1 ), 0 );

  if ( data.isContiguous() )
  {
    mpx.send ( SendBuffer( data.addr(), n ), 0 );
  }
  else
  {
    ByteVector tmp ( data.clone() );

    mpx.send ( SendBuffer( tmp.addr(), n ), 0 );
  }
}


//-----------------------------------------------------------------------
//   readData (root process)
//-----------------------------------------------------------------------


ByteVector CheckpointModule::Utils_::readData

  ( MPContext&     mpx,
    const String&  fname )

{
  using jem::io::FileInputStream;
  using jem::io::DataInputStream;

  const int             procCount = mpx.size ();

  Ref<DataInputStream>  input;
  ByteVector            data;
  int                   count;
  int                   rank;
  int                   iproc;
  idx_t                 n;


  input = newInstance<DataInputStream> (
    newInstance<FileInputStream> ( fname )
  );

  decode ( *input, count );

  if ( count != procCount )
  {
    String  err;

    if ( count > 1 )
    {
      err = String::format (
        "dump file `%s\' requires %d processes", fname, count
      );
    }
    else
    {
      err = String::format (
        "dump file `%s\' requires one process", fname
      );
    }

    throw SerializationException ( JEM_FUNC, err );
  }

  for ( iproc = procCount - 1; iproc > 0; iproc-- )
  {
    decode ( *input, rank );

    if ( iproc != rank )
    {
      throw SerializationException (
        JEM_FUNC,
        String::format (
          "dump file `%s\' contains invalid data", fname
        )
      );
    }

    decode ( *input, data );

    n = data.size ();

    mpx.send ( SendBuffer( &n,          1 ), rank );
    mpx.send ( SendBuffer( data.addr(), n ), rank );
  }

  decode ( *input, iproc );

  if ( iproc != 0 )
  {
    throw SerializationException (
      JEM_FUNC,
      String::format (
        "dump file `%s\' contains invalid data", fname
      )
    );
  }

  decode ( *input, data );

  return data;
}


//-----------------------------------------------------------------------
//   readData (other processes)
//-----------------------------------------------------------------------


ByteVector CheckpointModule::Utils_::readData ( MPContext& mpx )
{
  ByteVector  data;
  idx_t       n;


  mpx.recv    ( RecvBuffer( &n, 1 ), 0 );
  data.resize ( n );
  mpx.recv    ( RecvBuffer( data.addr(), n ), 0 );

  return data;
}


//=======================================================================
//   class CheckpointModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  CheckpointModule::TYPE_NAME  = "Checkpoint";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


CheckpointModule::CheckpointModule

  ( const String&       name,
    const Ref<Module>&  inmod ) :

    Super        (  name ),
    inputModule_ ( inmod )

{
  JEM_PRECHECK ( inmod && name.size() > 0 );

  const char*  cond =

    "save    last = 0.0\n"
    "let     dump = (runtime - last) > 1000.0,\n"
    "        last = if( dump, runtime, last )\n"
    "return  dump";

  backups_  = 0;
  sync_     = false;
  dumpCond_ = FuncUtils::newCond ( cond );
}


CheckpointModule::~CheckpointModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status CheckpointModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Status  status = OK;


  mpx_ = mp::Globdat::getMPContext ( globdat );

  if ( fileName_.size() && checkForDump_() )
  {
    try
    {
      restore_ ( globdat );
    }
    catch ( Exception& ex )
    {
      ex.setContext ( getContext() );
      throw;
    }
  }
  else
  {
    status = inputModule_->init ( conf, props, globdat );

    inputModule_->shutdown ( globdat );
  }

  if ( status != EXIT )
  {
    return OK;
  }
  else
  {
    return status;
  }
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status CheckpointModule::run ( const Properties& globdat )
{
  using jem::mp::allmax;

  if ( mpx_ == nullptr || fileName_.size() == 0 )
  {
    return OK;
  }

  int  cond = 0;

  try
  {
    cond = (int) FuncUtils::evalCond ( *dumpCond_, globdat );
  }
  catch ( Exception& ex )
  {
    ex.setContext ( getContext() );
    throw;
  }

  if ( allmax( *mpx_, cond ) )
  {
    dump_ ( globdat );
  }

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void CheckpointModule::shutdown ( const Properties& globdat )
{
  if ( mpx_ && fileName_.size() )
  {
    dump_ ( globdat );
  }

  mpx_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void CheckpointModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( fileName_, PropNames::FILE );
    myProps.find ( sync_,     PropNames::SYNC );
    myProps.find ( backups_,  PropNames::BACKUPS,
                   0,         maxOf( backups_ ) );

    FuncUtils::configCond ( dumpCond_, PropNames::DUMP_COND,
                            myProps,   globdat );
  }

  inputModule_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void CheckpointModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::FILE,    fileName_ );
  myConf.set ( PropNames::BACKUPS, backups_  );
  myConf.set ( PropNames::SYNC,    sync_     );

  FuncUtils   ::getConfig ( myConf, dumpCond_,
                            PropNames::DUMP_COND );
  inputModule_->getConfig ( conf,   globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool CheckpointModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  bool  result = inputModule_->takeAction ( action, params, globdat );

  if      ( action == Actions::EXE_COMMAND )
  {
    String  cmd;

    params.get ( cmd, ActionParams::COMMAND );

    if ( cmd == Commands::DUMP && mpx_ && fileName_.size() )
    {
      dump_ ( globdat );

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
    params.set ( Commands::DUMP, "" );

    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void CheckpointModule::printHelp ( PrintWriter& out ) const
{
  using jem::io::beginItem;
  using jem::io::endItem;

  if ( fileName_.size() )
  {
    print ( out, beginItem( Commands::DUMP ) );
    print ( out, "Save the current program "
            "state to the dump file `", fileName_, "\'." );
    print ( out, endItem );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> CheckpointModule::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> (
    myName,
    newModule (
      util::joinNames ( myName, PropNames::INPUT_MODULE ),
      conf, props, globdat
    )
  );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void CheckpointModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   dump_
//-----------------------------------------------------------------------


void CheckpointModule::dump_ ( const Properties& globdat )
{
  JEM_PRECHECK ( mpx_ );

  if ( ! fileName_.size() )
  {
    return;
  }

  try
  {
    if ( backups_ > 0 )
    {
      makeBackups_ ();
    }

    doDump_ ( globdat );
  }
  catch ( Exception& ex )
  {
    ex.setContext ( getContext() );
    throw;
  }
}


//-----------------------------------------------------------------------
//   doDump_
//-----------------------------------------------------------------------


void CheckpointModule::doDump_ ( const Properties& globdat )
{
  Writer&  info = System::info ( myName_ );

  Ref<ArrayOutputStream>   bufStream;
  Ref<ObjectOutputStream>  objStream;
  ByteVector               buf;


  print ( info, myName_, " : dumping global data to `",
          fileName_, "\' ...\n\n", flush );

  bufStream = newInstance<ArrayOutputStream>  ();
  objStream = newInstance<ObjectOutputStream> ( bufStream );

  encode ( *objStream, globdat );

  objStream->close ();
  objStream = nullptr;

  buf.ref ( bufStream->toArray() );

  if ( mpx_->myRank() == 0 )
  {
    Utils_::writeData ( *mpx_, buf, fileName_, sync_ );
  }
  else
  {
    Utils_::writeData ( *mpx_, buf );
  }
}


//-----------------------------------------------------------------------
//   restore_
//-----------------------------------------------------------------------


void CheckpointModule::restore_ ( const Properties& globdat )
{
  using jem::io::indent;
  using jem::io::outdent;

  JEM_PRECHECK ( mpx_ );

  Writer&  info = System::info ( myName_ );

  Ref<ArrayInputStream>   bufStream;
  Ref<ObjectInputStream>  objStream;
  Properties              indat;
  String                  name;


  if ( ! fileName_.size() )
  {
    return;
  }

  print ( info, myName_ , " : restoring global data from `",
          fileName_, "\' ...\n\n", flush );

  if ( mpx_->myRank() == 0 )
  {
    bufStream = newInstance<ArrayInputStream> (
      Utils_::readData ( *mpx_, fileName_ )
    );
  }
  else
  {
    bufStream = newInstance<ArrayInputStream> (
      Utils_::readData ( *mpx_ )
    );
  }

  objStream = newInstance<ObjectInputStream> ( bufStream );

  try
  {
    decode ( *objStream, indat );
  }
  catch ( const SerializationException& ex )
  {
    Ref<PrintWriter>  pr =

      newInstance<PrintWriter> ( & System::err() );

    pr->setIndentWidth ( 4 );

    print ( *pr, "Error reading `", fileName_, "\' : ", ex.what() );

    print ( *pr, indent, endl, endl,
            "Object stack trace:", endl, endl );

    objStream->printTrace ( * pr );

    print ( *pr, outdent, endl, endl, flush );

    throw;
  }

  jem::util::mergeNoReplace ( globdat, indat );
}



//-----------------------------------------------------------------------
//   makeBackups_
//-----------------------------------------------------------------------


void CheckpointModule::makeBackups_ ()
{
  JEM_PRECHECK ( mpx_ );

  Writer&  debug = System::debug ( myName_ );

  String   fname;
  String   newName;
  int      level;


  if ( backups_ <= 0 || mpx_->myRank() != 0 )
  {
    return;
  }

  fname = String::format ( "%s.%d", fileName_, backups_ );

  if ( File::exists( fname ) )
  {
    print ( debug, myName_, " : deleting dump file `",
            fname, "\' ...\n" );

    File::erase ( fname );
  }

  print ( debug, myName_, " : backing up dump file `",
          fileName_, "\' ...\n\n", flush );

  for ( level = backups_ - 1; level > 0; level-- )
  {
    newName = fname;
    fname   = String::format ( "%s.%d", fileName_, level );

    if ( File::exists( fname ) )
    {
      File::rename ( fname, newName );
    }
  }

  newName = fname;

  if ( File::exists( fileName_ ) )
  {
    File::rename ( fileName_, newName );
  }
}


//-----------------------------------------------------------------------
//   checkForDump_
//-----------------------------------------------------------------------


bool CheckpointModule::checkForDump_ ()
{
  JEM_PRECHECK ( mpx_ );

  if ( ! fileName_.size() )
  {
    return false;
  }

  int  result = 0;

  if ( mpx_->myRank() == 0 )
  {
    result = File::exists ( fileName_ );

    mpx_->broadcast ( SendBuffer( &result, 1 ) );
  }
  else
  {
    mpx_->broadcast ( RecvBuffer( &result, 1 ), 0 );
  }

  return (result != 0);
}


JIVE_END_PACKAGE( app )
