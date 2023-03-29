
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
 *  This example demonstrates the Properties class.
 *
 ************************************************************************/

#include <jem/base/System.h>
#include <jem/io/Reader.h>
#include <jem/io/Writer.h>
#include <jem/util/Properties.h>


using namespace jem;
using namespace jem::io;
using namespace jem::util;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ( int argc, char** argv )
{
  Writer&     out = System::out ();
  Reader&     in  = System::in  ();

  Properties  persons;
  Properties  person;

  String      name;
  String      nickName;

  double      length;
  int         age;


  if ( argc < 2 )
  {
    print ( System::err(), "\nNo input file specified\n\n" );
    return 1;
  }

  persons.parseFile ( argv[1] );

  print ( out, endl );
  print ( out, persons );
  print ( out, endl );

  while ( true )
  {
    print ( out, "Enter a name : " );
    read  ( in, name );

    if ( name.size() == 0 )
    {
      return 0;
    }

    persons.get ( person, name );

    print ( out, "\nData for `", name, "\' :\n" );

    if ( person.find( nickName, "nickName" ) )
    {
      print ( out, "\n  nick name : ", nickName );
    }

    person.get ( age,    "age",    0,   200 );
    person.get ( length, "length", 0.0, 3.0 );

    print ( out, "\n  age       : ", age,    " years" );
    print ( out, "\n  length    : ", length, " m" );
    print ( out, "\n\n" );
  }

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ( int argc, char** argv )
{
  return jem::System::exec ( & run, argc, argv );
}
