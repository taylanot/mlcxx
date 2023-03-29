
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


#include <jem/base/ClassTemplate.h>
#include <jem/io/StdTermReader.h>
#include "LogState.h"
#include "LogFileWriter.h"
#include "LogTermReader.h"


JEM_DEFINE_CLASS( jive::app::LogTermReader );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;


//=======================================================================
//   class LogTermReader
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LogTermReader::LogTermReader

  ( Ref<LogState>       state,
    Ref<LogFileWriter>  logger ) :

    state_  ( state  ),
    logger_ ( logger )

{
  JEM_PRECHECK ( state && logger );
}


LogTermReader::~LogTermReader ()
{}


//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t LogTermReader::poll ( const Time& timeout )
{
  if ( ! reader_ )
  {
    initReader_ ();
  }

  return reader_->poll ( timeout );
}


//-----------------------------------------------------------------------
//   readLine
//-----------------------------------------------------------------------


String LogTermReader::readLine

  ( const String&   prompt,
    WordCompleter*  wcomp )

{
  if ( ! reader_ )
  {
    initReader_ ();
  }

  if ( state_->lastID != (byte) 255 )
  {
    logger_->write ( '\n' );

    state_->lastID = (byte) 255;
  }

  logger_->reset ();

  return reader_->readLine ( prompt, wcomp );
}


//-----------------------------------------------------------------------
//   clearHistory
//-----------------------------------------------------------------------


void LogTermReader::clearHistory ()
{
  if ( reader_ )
  {
    reader_->clearHistory ();
  }
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> LogTermReader::clone () const
{
  return newInstance<Self> ( state_, logger_ );
}


//-----------------------------------------------------------------------
//   initReader_
//-----------------------------------------------------------------------


void LogTermReader::initReader_ ()
{
  using jem::io::StdTermReader;

  reader_ = newInstance<StdTermReader> ();
}


JIVE_END_PACKAGE( app )
