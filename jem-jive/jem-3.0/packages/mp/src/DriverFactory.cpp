
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


#include <jem/base/assert.h>
#include <jem/base/System.h>
#include <jem/util/Singleton.h>
#include <jem/mp/DriverException.h>
#include <jem/mp/DriverFactory.h>


using jem::util::Singleton;


JEM_BEGIN_PACKAGE( mp )


//=======================================================================
//   class DriverFactory
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


DriverFactory::~DriverFactory ()
{}


//-----------------------------------------------------------------------
//   newDriver
//-----------------------------------------------------------------------


Ref<Driver> DriverFactory::newDriver ( const String& ) const
{
  return nullptr;
}


//-----------------------------------------------------------------------
//   newInstance (given a driver name)
//-----------------------------------------------------------------------


Ref<Driver> DriverFactory::newInstance ( const String& name )
{
  Ref<Driver> driver = Singleton<Self>::get()->newDriver ( name );

  if ( ! driver )
  {
    throw DriverException (
      JEM_FUNC,
      String::format ( "unknown driver: `%s\'", name )
    );
  }

  return driver;
}


//-----------------------------------------------------------------------
//   newInstance (given the command-line arguments)
//-----------------------------------------------------------------------


Ref<Driver> DriverFactory::newInstance

  ( int            argc,
    char**         argv,
    const String&  defaultName )

{
  String  name = System::getenv ( "JEM_MP_DRIVER", defaultName );

  for ( int i = 1; i < argc; i++ )
  {
    String  arg = argv[i];

    if ( arg == "--" )
    {
      break;
    }

    if ( arg == "-md" || arg == "--mp-driver" )
    {
      if ( (i + 1) >= argc )
      {
        throw DriverException (
          String::format ( "command line option `%s\'", arg ),
          "missing driver name"
        );
      }

      name = argv[i + 1];

      break;
    }
  }

  if ( ! name.size() )
  {
    throw DriverException (
      "command line options",
      "no message passing driver has been specified"
    );
  }

  return newInstance ( name );
}


//-----------------------------------------------------------------------
//   getFactory
//-----------------------------------------------------------------------


Ref<DriverFactory> DriverFactory::getFactory ()
{
  return Singleton<Self>::get ();
}


//-----------------------------------------------------------------------
//   setFactory
//-----------------------------------------------------------------------


void DriverFactory::setFactory ( const Ref<Self>& factory )
{
  JEM_PRECHECK ( factory );

  Singleton<Self>::set ( factory );
}


JEM_END_PACKAGE( mp )
