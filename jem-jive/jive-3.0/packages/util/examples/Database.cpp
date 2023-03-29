
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

/************************************************************************
 *
 *  This example performs various operations involving the Database
 *  class.
 *
 ***********************************************************************/

#include <jem/base/System.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/FileOutputStream.h>
#include <jem/io/ObjectInputStream.h>
#include <jem/io/ObjectOutputStream.h>
#include <jem/util/Timer.h>
#include <jive/util/StdPointSet.h>
#include <jive/util/StdDatabase.h>


using namespace jem;
using namespace jem::io;

using jem::util::Timer;
using jem::util::Properties;
using jive::Vector;
using jive::IdxVector;
using jive::util::XPointSet;
using jive::util::StdPointSet;
using jive::util::Database;
using jive::util::StdDatabase;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ( int argc, char** argv )
{
  const idx_t             R   = 4;
  const idx_t             N   = 100;

  Writer&                 out = System::out ();

  Database::IntColumn*    icol;
  Database::FloatColumn*  fcol;
  Ref<XPointSet>          points;
  Ref<Database>           dbase;
  Ref<Database>           dbase2;
  Ref<PrintWriter>        printer;

  Timer                   timer;
  IdxVector               ivec;
  Vector                  vec;


  timer.start ();

  points = newInstance<StdPointSet> ( "points",   "point" );
  dbase  = newInstance<StdDatabase> ( "dbase",    points  );

  print ( out, endl );
  print ( out, "Building point set   ... ", flush );

  timer.reset ();

  points->reserve ( N );

  vec.ref ( Vector( R ) );

  for ( idx_t i = 0; i < N; i++ )
  {
    vec = (double) i * 0.2;

    points->addPoint ( vec );
  }

  print ( out, "ready in ", timer, endl );

  timer.reset ();

  print ( out, "Building database    ... ", flush );

  icol = dbase->addIntColumn ( "integer data" );

  ivec.ref ( IdxVector( 11 ) );

  for ( idx_t i = 0; i < N; i += 2 )
  {
    ivec = i;

    icol->setValue  ( i + 0, -i );
    icol->setValues ( i + 1, ivec, ivec.size() );
  }

  fcol = dbase->addFloatColumn ( "floating data" );

  vec.ref ( Vector( 3 ) );

  for ( idx_t i = 0; i < N; i += 2 )
  {
    vec = (double) i * 0.4;

    fcol->setValue  ( i + 0, (double) -i );
    fcol->setValues ( i + 1, vec, vec.size() );
  }

  print ( out, "ready in ", timer, endl );

  printer = newInstance<PrintWriter> (
    newInstance<FileWriter> ( "dbase.out" ),
    128 * 1024
  );

  dbase->printTo ( * printer );

  timer.reset ();

  print ( out, "Deleting some rows   ... ", flush );

  for ( idx_t i = 0; i < N; i += N / 10 )
  {
    fcol->clearRow ( i );
    icol->clearRow ( i );
  }

  print ( out, "ready in ", timer, endl );

  dbase->printTo ( * printer );

  ivec.resize ( N / 10 );

  for ( idx_t i = 0; i < ivec.size(); i++ )
  {
    ivec[i] = 3 * i;
  }

  timer.reset ();

  print ( out, "Deleting some points ... ", flush );

  points->erasePoints ( ivec );

  print ( out, "ready in ", timer, endl );

  dbase->printTo ( * printer );

  timer.reset ();

  print ( out, "Cloning database     ... " );

  dbase2 = checkedCast<Database> ( dbase->clone() );

  print ( out, "ready in ", timer, endl );

  dbase  = nullptr;

  //  dbase2->printTo ( * printer );

  print ( out, endl );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ( int argc, char** argv )
{
  return System::exec ( & run, argc, argv );
}

