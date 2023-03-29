
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
 *  This example demonstrates how to parse various types of data sets
 *  using the DataParser class.
 *
 ***********************************************************************/

#include <jem/base/System.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/FileOutputStream.h>
#include <jem/io/ObjectInputStream.h>
#include <jem/io/ObjectOutputStream.h>
#include <jem/util/Timer.h>
#include <jem/util/Properties.h>
#include <jive/util/Table.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/Printer.h>
#include <jive/util/Database.h>
#include <jive/util/DataParser.h>
#include <jive/util/StdPointSet.h>
#include <jive/util/StdGroupSet.h>


using namespace jem;
using namespace jem::io;

using jem::io::Writer;
using jem::util::Timer;
using jem::util::Properties;
using jive::IdxVector;
using jive::util::ItemSet;
using jive::util::Database;
using jive::util::DataParser;
using jive::util::XPointSet;
using jive::util::StdPointSet;
using jive::util::XGroupSet;
using jive::util::StdGroupSet;
using jive::util::Table;
using jive::util::ItemGroup;
using jive::util::Printer;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ( int argc, char** argv )
{
  Writer&                  out = System::out ();

  Properties               globdat ( "data" );

  Ref<ObjectInputStream>   dumpIn;
  Ref<ObjectOutputStream>  dumpOut;

  Ref<DataParser>          parser;
  Ref<XPointSet>           points;
  Ref<XGroupSet>           groups;

  String                   baseName;
  String                   fileName;

  Timer                    timer;
  IdxVector                ipoints;

  idx_t                    i, n;


  timer.start ();

  if ( argc > 1 )
  {
    fileName = argv[argc - 1];
  }
  else
  {
    fileName = "data.in";
  }

  i = fileName.rfind ( '.' );

  if ( i >= 0 )
  {
    baseName = fileName[slice(BEGIN,i)];
  }
  else
  {
    baseName = fileName;
  }

  points = newInstance<StdPointSet> ( "points",   "point" );
  groups = newInstance<StdGroupSet> ( "segments", "segment", points );

  points->store ( "points",   globdat );
  groups->store ( "segments", globdat );

  parser = newInstance<DataParser> ();

  parser->addPointSetParser    ( points );
  parser->addGroupSetParser    ( groups );

  parser->addItemGroupsParsers ( globdat );
  parser->addTablesParsers     ( globdat );
  parser->addDbasesParsers     ( globdat );

  timer.reset ();

  print ( out, "Parsing input file `", fileName, "\' ...\n\n" );

  parser->parseFile ( fileName, &System::out() );

  print ( out, "\nReady in ", timer, "\n\n" );

  ItemGroup::printAll ( Printer::get(), globdat );
  Table    ::printAll ( Printer::get(), globdat );
  Database ::printAll ( Printer::get(), globdat );

  points = nullptr;
  groups = nullptr;

  Printer::flush ();

  print ( out, "\n\n" );

  fileName = baseName + ".dump";

  dumpOut = newInstance<ObjectOutputStream> (
    newInstance<FileOutputStream> ( fileName )
  );

  timer.reset ();

  print  ( out, "Writing binary data to `", fileName, "\' ...\n" );
  encode ( *dumpOut, globdat );

  dumpOut = nullptr;

  print ( out, "Ready in ", timer, "\n\n" );

  dumpIn = newInstance<ObjectInputStream> (
    newInstance<FileInputStream> ( fileName )
  );

  timer.reset ();

  print ( out, "Reading data from binary file `",
          fileName, "\' ...\n" );

  decode ( *dumpIn, globdat );
  print  ( out, "Ready in ", timer, "\n\n" );

  points = checkedCast<XPointSet> (
    ItemSet::get ( "points", globdat, JEM_FUNC )
  );

  n = points->size() / 2;

  ipoints.resize ( n );

  for ( i = 0; i < n; i++ )
  {
    ipoints[i] = 2 * i;
  }

  timer.reset ();

  print ( out, "Erasing all even-indexed points ...\n" );

  points->erasePoints ( ipoints );

  print ( out, "Ready in ", timer, "\n\n" );

  groups = checkedCast<XGroupSet> (
    ItemSet::get ( "segments", globdat, JEM_FUNC )
  );

  groups->printTo ( Printer::get() );

  ItemGroup::printAll ( Printer::get(), globdat );
  Table    ::printAll ( Printer::get(), globdat );
  Database ::printAll ( Printer::get(), globdat );

  Printer::flush ();

  print ( out, "\n\n" );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ( int argc, char** argv )
{
  return System::exec ( & run, argc, argv );
}
