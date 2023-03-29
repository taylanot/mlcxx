
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
#include <jem/base/System.h>
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FilterWriter.h>
#include <jem/util/Timer.h>
#include <jem/util/CPUTimer.h>
#include <jem/util/Properties.h>
#include <jem/util/ObjectConverter.h>
#include <jem/xutil/SnoopDictionary.h>
#include <jive/util/utilities.h>
#include <jive/mp/Globdat.h>
#include <jive/app/import.h>
#include <jive/app/typedefs.h>
#include <jive/app/Names.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/ReportModule.h>


JEM_DEFINE_CLASS( jive::app::ReportModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::System;
using jem::io::FilterWriter;
using jem::util::Timer;
using jem::util::CPUTimer;
using jem::xutil::SnoopDict;


//=======================================================================
//   class ReportModule::RunData_
//=======================================================================


class ReportModule::RunData_ : public jem::Collectable
{
 public:

  inline                  RunData_        ();

  void                    printTime

    ( const String&         phase )          const;

  inline void             checkWarnings   ();


 public:

  Ref<Writer>             wout;

  volatile bool           warned;
  bool                    warnings;

  Timer                   realTimer;
  CPUTimer                cpuTimer;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ReportModule::RunData_ :: RunData_ ()
{
  wout     = & System::warn ();

  warned   = false;
  warnings = false;

  realTimer.start ();
  cpuTimer .start ();
}


//-----------------------------------------------------------------------
//   printTime
//-----------------------------------------------------------------------


void ReportModule::RunData_::printTime

  ( const String&  phase ) const

{
  Writer&  out = System::out ();

  if ( warned )
  {
    print ( System::warn(), phase,
            " phase completed with warnings.\n" );
  }
  else
  {
    print ( out, phase, " phase completed successfully.\n" );
  }

  print ( out, "\n  Elapsed CPU  time : ", cpuTimer, '.' );
  print ( out, "\n  Elapsed real time : ", realTimer, ".\n\n" );

  out.flush ();
}


//-----------------------------------------------------------------------
//   checkWarnings
//-----------------------------------------------------------------------


inline void ReportModule::RunData_::checkWarnings ()
{
  warnings = warnings || warned;
  warned   = false;
}


//=======================================================================
//   class ReportModule::Writer_
//=======================================================================


class ReportModule::Writer_ : public FilterWriter
{
 public:

  typedef Writer_         Self;
  typedef FilterWriter    Super;


  explicit inline         Writer_

    ( const Ref<RunData_>&  rundat );

  inline                  Writer_

    ( const Self&           rhs );

  virtual Ref<Writer>     dup     ()          override;

  virtual void            write

    ( int                   c )               override;

  virtual void            write

    ( int                   c,
      idx_t                 n )               override;

  virtual void            write

    ( const char*           buf,
      idx_t                 n )               override;

  virtual void            writeNoThrow

    ( const char*           buf,
      idx_t                 n )      noexcept override;


 private:

  Ref<RunData_>           rundat_;

};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline ReportModule::Writer_::Writer_

  ( const Ref<RunData_>& rundat ) :

  Super   ( rundat->wout ),
  rundat_ ( rundat )

{}


inline ReportModule::Writer_::Writer_ ( const Self& rhs ) :

  Super   ( rhs ),
  rundat_ ( rhs.rundat_ )

{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> ReportModule::Writer_::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void ReportModule::Writer_::write ( int c )
{
  rundat_->warned = true;

  output_->write ( c );
}


void ReportModule::Writer_::write ( int c, idx_t n )
{
  rundat_->warned = true;

  output_->write ( c, n );
}


void ReportModule::Writer_::write ( const char* buf, idx_t n )
{
  rundat_->warned = true;

  output_->write ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void ReportModule::Writer_::writeNoThrow

  ( const char*  buf,
    idx_t        n ) noexcept

{
  rundat_->warned = true;

  output_->writeNoThrow ( buf, n );
}


//=======================================================================
//   class ReportModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ReportModule::TYPE_NAME = "Report";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ReportModule::ReportModule

  ( const String&       name,
    const Ref<Module>&  child ) :

    Super  (  name ),
    child_ ( child )

{
  JEM_PRECHECK ( child );

  if ( ! myName_.size() )
  {
    myName_ = child->getName ();
  }
}


ReportModule::~ReportModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status ReportModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  rundat_ = nullptr;

  Ref<RunData_>  newdat = newInstance<RunData_> ();
  Status         status = OK;

  System::setWarnStream ( newInstance<Writer_>( newdat ) );

  try
  {
    if ( mp::Globdat::myRank( globdat ) == 0 )
    {
      status = snoopInit_   ( conf, props, globdat );
    }
    else
    {
      status = child_->init ( conf, props, globdat );

      printConfig_ ( conf );
    }

    newdat->printTime ( "Initialization" );
  }
  catch ( const jem::Exception& )
  {
    System::setWarnStream ( newdat->wout );

    printConfig_ ( conf );

    throw;
  }

  rundat_   .swap ( newdat );
  usedProps_.ref  ( StringVector() );

  return status;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status ReportModule::run ( const Properties& globdat )
{
  if ( ! rundat_ )
  {
    return DONE;
  }

  rundat_->checkWarnings ();

  return child_->run ( globdat );
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void ReportModule::shutdown ( const Properties& globdat )
{
  if ( rundat_ )
  {
    Ref<RunData_>  rundat = rundat_;

    rundat_ = nullptr;

    rundat->printTime     ( "Runtime" );
    rundat->checkWarnings ();
    child_->shutdown      ( globdat );
    rundat->printTime     ( "Shutdown" );
    rundat->checkWarnings ();
    System::setWarnStream ( rundat->wout );

    if ( rundat->warnings )
    {
      print ( System::warn(),
              "\nOne or more warnings have been printed.\n\n" );
    }
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ReportModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( rundat_ )
  {
    child_->configure ( props, globdat );
  }
  else
  {
    Ref<SnoopDict>  snoopDict =

      newInstance<SnoopDict> ( props.getContents() );

    Properties      snoopProps ( snoopDict, props.getName() );

    child_->configure ( snoopProps, globdat );

    usedProps_.ref ( snoopDict->listTouchedObjects() );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ReportModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  child_->getConfig ( conf, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool ReportModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> ReportModule::makeNew

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


void ReportModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   snoopInit_
//-----------------------------------------------------------------------


Module::Status ReportModule::snoopInit_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<SnoopDict>  snoopDict =

    newInstance<SnoopDict>   ( props.getContents() );

  Properties      snoopProps ( snoopDict,
                               props.getConverter(),
                               props.getName() );

  StringVector    list;
  Ref<Object>     obj;
  Status          status;


  // Touch the properties that are used by the Application class,
  // just to avoid some annoying warnings.

  list.ref ( conf.listProps() );

  for ( idx_t i = 0; i < list.size(); i++ )
  {
    snoopProps.find ( obj, list[i] );
  }

  snoopProps.find ( obj, CASE_NAME );
  snoopProps.find ( obj, "argv"    );

  // Touch the properties that have been used before.

  for ( idx_t i = 0 ; i < usedProps_.size(); i++ )
  {
    snoopProps.find ( obj, usedProps_[i] );
  }

  status = child_->init ( conf, snoopProps, globdat );

  printConfig_ ( conf );

  list.ref ( snoopDict->listUntouchedObjects() );

  if ( list.size() )
  {
    Writer&  warn = System::warn ();

    if ( list.size() == 1 )
    {
      print ( warn, "\nUnknown property: `", list[0], "\'\n\n" );
    }
    else
    {
      jem::sort ( list );

      print ( warn, "\nUnknown properties:\n" );

      for ( idx_t i = 0; i < list.size(); i++ )
      {
        print ( warn, "\n  ", list[i] );
      }

      print ( warn, "\n\n" );
    }
  }

  return status;
}


//-----------------------------------------------------------------------
//   printConfig_
//-----------------------------------------------------------------------


void ReportModule::printConfig_ ( const Properties& conf )
{
  using jem::io::endl;
  using jem::io::flush;
  using jem::io::indent;
  using jem::io::outdent;

  Ref<PrintWriter>  pr =

    newInstance<PrintWriter> ( & System::info( myName_ ) );

  pr->setPageWidth ( 100 );

  print ( *pr, endl, "Application configuration data:" );
  print ( *pr, endl, indent, endl, conf, outdent, endl );
  print ( *pr, flush );
}


JIVE_END_PACKAGE( app )
