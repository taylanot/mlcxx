
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


#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/FileWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/GzipFileWriter.h>
#include <jem/io/IOException.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Runtime.h>
#include <jive/util/FuncUtils.h>
#include <jive/mp/Globdat.h>
#include <jive/app/Names.h>
#include <jive/app/Actions.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/SampleModule.h>


JEM_DEFINE_CLASS( jive::app::SampleModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::Exception;
using jem::io::IOException;
using jive::util::FuncUtils;


//=======================================================================
//   class SampleModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SampleModule::TYPE_NAME = "Sample";

const int    SampleModule::ACTIVE_   = 1 << 0;
const int    SampleModule::FG_MODE_  = 1 << 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SampleModule::SampleModule ( const String& name ) :

  Super ( name )

{
  fileName_   = "";
  header_     = "";
  separator_  = " ";
  options_    = 0;
  precision_  = 8;
  sampleCond_ = FuncUtils::newCond ();
  tflush_     = 0.0;
  mode_       = 0;
  warnings_   = 0;
}


SampleModule::~SampleModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status SampleModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  if ( output_ )
  {
    output_->close ();

    output_ = nullptr;
  }

  tflush_   = 0.0;
  warnings_ = 0;

  if ( mp::Globdat::myRank( globdat ) == 0 )
  {
    mode_ |=  ACTIVE_;
  }
  else
  {
    mode_ &= ~ACTIVE_;
  }

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status SampleModule::run ( const Properties& globdat )
{
  if ( fileName_.size() == 0 || (mode_ & ACTIVE_) == 0 ||
       dataSets_.size() == 0 )
  {
    return OK;
  }

  try
  {
    bool  cond = FuncUtils::evalCond ( *sampleCond_, globdat );

    if ( ! cond )
    {
      return OK;
    }

    writeData_ ( globdat );
  }
  catch ( Exception& ex )
  {
    ex.setContext ( getContext() );
    throw;
  }

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void SampleModule::shutdown ( const Properties& globdat )
{
  if ( output_ )
  {
    output_->close ();

    output_ = nullptr;
  }

  mode_ &= ~ACTIVE_;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SampleModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::Array;

  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties  myProps = props.findProps ( myName_ );

  String      fname;
  bool        append;


  if ( myProps.find( fname, PropNames::FILE ) )
  {
    setFileName ( fname );
  }

  myProps.find ( header_,    PropNames::HEADER );
  myProps.find ( separator_, PropNames::SEPARATOR );

  if ( myProps.find( append, PropNames::APPEND ) )
  {
    if ( append )
    {
      options_ |=  APPEND;
    }
    else
    {
      options_ &= ~APPEND;
    }
  }

  if ( myProps.find ( precision_, PropNames::PRECISION, 1, 20 ) )
  {
    if ( output_ )
    {
      output_->nformat.setFloatWidth     ( precision_ + 8 );
      output_->nformat.setFractionDigits ( precision_ );
    }
  }

  if ( myProps.contains( PropNames::DATA_SETS ) )
  {
    Array< Ref<Function> >  tmp;

    FuncUtils::configFuncs ( tmp, PropNames::DATA_SETS,
                             myProps, globdat );

    dataSets_.swap ( tmp );

    warnings_ = 0;
  }

  FuncUtils::configCond ( sampleCond_,
                          PropNames::SAMPLE_COND,
                          myProps, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SampleModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );
  bool        append = options_ & APPEND;


  myConf.set ( PropNames::FILE,      fileName_  );
  myConf.set ( PropNames::HEADER,    header_    );
  myConf.set ( PropNames::SEPARATOR, separator_ );
  myConf.set ( PropNames::APPEND,    append     );
  myConf.set ( PropNames::PRECISION, precision_ );

  FuncUtils::getConfig ( myConf, dataSets_,
                         PropNames::DATA_SETS );
  FuncUtils::getConfig ( myConf, sampleCond_,
                         PropNames::SAMPLE_COND );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool SampleModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  bool  result = true;

  if      ( action == Actions::FG )
  {
    mode_ |=  FG_MODE_;
  }
  else if ( action == Actions::BG )
  {
    mode_ &= ~FG_MODE_;
  }
  else
  {
    result = false;
  }

  return result;
}


//-----------------------------------------------------------------------
//   setFileName
//-----------------------------------------------------------------------


void SampleModule::setFileName ( const String& fname )
{
  if ( fname == fileName_ )
  {
    return;
  }

  if ( output_ )
  {
    output_->close ();

    output_ = nullptr;
  }

  fileName_ = fname;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> SampleModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void SampleModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  makeNew );
  ModuleFactory::declare ( CLASS_NAME, makeNew );
}


//-----------------------------------------------------------------------
//   writeData_
//-----------------------------------------------------------------------


void SampleModule::writeData_ ( const Properties& globdat )
{
  using jem::System;
  using jive::util::Runtime;

  JEM_ASSERT ( mode_ & ACTIVE_ );

  const int  MAX_WARNINGS = 20;

  double     x, rt;
  idx_t      i, j;


  if ( ! output_ )
  {
    openOutput_ ();
  }

  for ( i = j = 0; i < dataSets_.size(); i++ )
  {
    Function*  func = dataSets_[i].get ();

    if ( ! func )
    {
      continue;
    }

    if ( j )
    {
      print ( *output_, separator_ );
    }

    j++;

    x = 0.0;

    try
    {
      x = FuncUtils::evalFunc ( *func, globdat );
    }
    catch ( const Exception& ex )
    {
      const int  width = output_->nformat.getFloatWidth ();

      if ( warnings_ < MAX_WARNINGS )
      {
        print ( System::warn(), myName_, " : ", ex.what(), '\n' );

        warnings_++;
      }

      if ( warnings_ == MAX_WARNINGS )
      {
        print ( System::warn(), myName_,
                " : no more warnings will be printed\n" );

        warnings_++;
      }

      if ( width > 4 )
      {
        output_->write ( ' ', width - 4 );
      }

      print ( *output_, "n.a." );

      continue;
    }

    print ( *output_, x );
  }

  print ( *output_, '\n' );

  rt = Runtime::getValue ( globdat );

  if ( (mode_ & FG_MODE_) || (rt - tflush_ > 1.0) )
  {
    output_->flush ();

    tflush_ = rt;
  }
}


//-----------------------------------------------------------------------
//   openOutput_
//-----------------------------------------------------------------------


void SampleModule::openOutput_ ()
{
  JEM_ASSERT ( mode_ & ACTIVE_ );

  using jem::io::FileFlags;
  using jem::io::FileWriter;
  using jem::io::GzipFileWriter;
  using jive::util::isGzipFile;

  if ( output_ )
  {
    return;
  }

  Ref<Writer>  file;

  if ( isGzipFile( fileName_ ) )
  {
    if ( options_ & APPEND )
    {
      throw IOException (
        getContext (),
        "unable to append to a gzip file; unset the "
        "`append\' option"
      );
    }

    file = newInstance<GzipFileWriter> ( fileName_ );
  }
  else
  {
    FileFlags::OpenFlags  flags = 0;

    if ( options_ & APPEND )
    {
      flags |= FileFlags::APPEND;
    }

    file = newInstance<FileWriter> ( fileName_, flags );
  }

  output_ = newInstance<PrintWriter> ( file );

  output_->nformat.setScientific     ();
  output_->nformat.setFloatWidth     ( precision_ + 8 );
  output_->nformat.setFractionDigits ( precision_ );

  if ( header_.size() )
  {
    print ( *output_, header_, '\n' );
  }
}


JIVE_END_PACKAGE( app )
