
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
#include <jem/base/System.h>
#include <jem/base/AssertException.h>
#include <jem/base/PrecheckException.h>
#include <jem/base/assert.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   assertFailed_
//-----------------------------------------------------------------------


void            assertFailed_

  ( const char*   expr,
    const char*   mesg,
    const char*   func,
    const char*   file,
    int           line )

{
  String  details;

  if ( mesg )
  {
    details = String::format ( "%s; assertion ( %s ) failed",
                               mesg, expr );
  }
  else
  {
    details = String::format ( "assertion ( %s ) failed", expr );
  }

  throw AssertException (
    String::format ( "%s:%d: %s", file, line, func ),
    details
  );
}


//-----------------------------------------------------------------------
//   assertNoThrowFailed_
//-----------------------------------------------------------------------


void            assertNoThrowFailed_

  ( const char*   expr,
    const char*   mesg,
    const char*   func,
    const char*   file,
    int           line )

{
  // Because this function will be called in emergency situations, it
  // can only call a limited set of other functions.

  char  buf[16];
  int   k;


  k = sizeof(buf);

  if ( line <= 0 )
  {
    buf[--k] = '0';
  }
  else
  {
    while ( k > 0 && line > 0 )
    {
      buf[--k] = (char) ('0' + line % 10);
      line     = line / 10;
    }
  }

  if ( mesg )
  {
    System::werror ( "\n*** Error: " );
    System::werror ( mesg );
  }

  System::werror ( "\n*** Assertion ( " );
  System::werror ( expr );
  System::werror ( " ) failed\n" );
  System::werror ( "\n  Function : " );
  System::werror ( func );
  System::werror ( "\n  File     : " );
  System::werror ( file );
  System::werror ( "\n  Line     : " );
  System::werror ( buf + k, (idx_t) (sizeof(buf) - (size_t) k) );
  System::werror ( "\n" );

  System::dumpStackTrace ( 1 );

  System::werror ( "\nAborting program.\n\n" );

  std::abort     ();
}


//-----------------------------------------------------------------------
//   precheckFailed_
//-----------------------------------------------------------------------


void            precheckFailed_

  ( const char*   expr,
    const char*   mesg,
    const char*   func,
    const char*   file,
    int           line )

{
  String  details;

  if ( mesg )
  {
    details = String::format ( "%s; precheck ( %s ) failed",
                               mesg, expr );
  }
  else
  {
    details = String::format ( "precheck ( %s ) failed", expr );
  }

  throw PrecheckException (
    String::format ( "%s:%d: %s", file, line, func ),
    details
  );
}


JEM_END_PACKAGE_BASE
