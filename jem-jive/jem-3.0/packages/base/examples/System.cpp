
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

/************************************************************************
 *
 *  This example demonstrates how to use the System class.
 *
 ***********************************************************************/

#include <cstdlib>
#include <jem/base/Signals.h>
#include <jem/base/System.h>
#include <jem/base/String.h>
#include <jem/io/Writer.h>
#include <jem/io/Reader.h>
#include <jem/util/Timer.h>

using namespace jem;
using namespace jem::io;
using namespace jem::util;


//-----------------------------------------------------------------------
//   scroll (String)
//-----------------------------------------------------------------------


void              scroll

  ( Writer&         out,
    const String&   str )

{
  const String  s = str.toUpper ();

  Timer    timer;
  double   dt;
  idx_t    i;

  timer.start ();

  i = 1;

  while ( i <= str.size() )
  {
    dt = timer.toDouble ();

    if ( dt > ((double) i) / 16.0 )
    {
      print ( out, s[i - 1] );

      i++;
      out.flush ();
    }
  }

  print ( out, "\n\n" );
}


//-----------------------------------------------------------------------
//   printSysinfo
//-----------------------------------------------------------------------


void printSysinfo ( Writer& out )
{
  const char*  PROPERTIES[] =
  {
    "os.version",
    "os.name",
    "os.machine",
    "host.name",
    "host.totalmem",
    "host.nrprocs",
    "user.home",
    "user.name",
    0
  };

  print ( out, "----- System info ------\n\n" );

  for ( int i = 0; PROPERTIES[i]; i++ )
  {
    String  value = System::getProperty ( PROPERTIES[i],
                                          "unknown" );
    print ( out, "  ", PROPERTIES[i], " = ", value, endl );
  }

  print ( out, endl );
}


//-----------------------------------------------------------------------
//   memoryError
//-----------------------------------------------------------------------


void memoryError ( int signum )
{
  System::werror ( "*** Invalid memory reference; "
                   "stack trace follows ***\n\n" );

  System::dumpStackTrace ();

  System::werror ( "\n" );

  ::exit ( 1 );
}


//-----------------------------------------------------------------------
//   crashme
//-----------------------------------------------------------------------


void crashme ( int level )
{
  volatile int*  p = 0;

  if ( level < 0 )
  {
    scroll ( System::out(), "going down!" );

    *p = 0;
  }
  else
  {
    crashme ( level - 1 );
  }
}


//-----------------------------------------------------------------------
//  run
//-----------------------------------------------------------------------


int run ()
{
  Writer&  out = System::out ();
  Reader&  in  = System::in  ();

  String   name;
  int      number;


  System::setSigHandler ( Signals::MEMORY_ERROR, memoryError );

  printSysinfo ( out );

  scroll ( out, "hello world" );

  print  ( out, "Please enter your name: " );
  read   ( in,  name );
  print  ( out, endl );

  scroll ( out, "hello " + name + "\nhow is life today?" );

  print  ( out, "please enter an integer: " );
  read   ( in,  number );
  print  ( out, endl );

  scroll ( out, "the integer you entered is: " + String(number) );

  crashme ( 10 );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( &run );
}
