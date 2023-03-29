
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
#include "LogState.h"
#include "LogWriter.h"


JIVE_BEGIN_PACKAGE( app )


//=======================================================================
//   class LogWriter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


LogWriter::LogWriter

  ( byte                  myID,
    const Ref<LogState>&  state,
    const Ref<Writer>&    output,
    const String&         prefix ) :

    Super  ( output, prefix ),

    myID_  ( myID  ),
    state_ ( state )

{
  JEM_PRECHECK ( state );
}


LogWriter::LogWriter ( const Self& rhs ) :

  Super  ( rhs        ),
  myID_  ( rhs.myID_  ),
  state_ ( rhs.state_ )

{}


LogWriter::~LogWriter ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> LogWriter::dup ()
{
  return jem::newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void LogWriter::write ( int c )
{
  if ( state_->lastID != myID_ )
  {
    Super::write ( '\n' );

    state_->lastID = myID_;
  }

  Super::write ( c );
}


void LogWriter::write ( int c, idx_t n )
{
  if ( state_->lastID != myID_ )
  {
    Super::write ( '\n' );

    state_->lastID = myID_;
  }

  Super::write ( c, n );
}


void LogWriter::write ( const char* buf, idx_t n )
{
  if ( state_->lastID != myID_ )
  {
    Super::write ( '\n' );

    state_->lastID = myID_;
  }

  Super::write ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void LogWriter::writeNoThrow ( const char* buf, idx_t n ) noexcept
{
  if ( state_->lastID != myID_ )
  {
    Super::writeNoThrow ( "\n", 1 );

    state_->lastID = myID_;
  }

  Super::writeNoThrow ( buf, n );
}


JIVE_END_PACKAGE( app )
