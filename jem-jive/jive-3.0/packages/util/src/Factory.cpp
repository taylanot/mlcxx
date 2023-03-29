
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


#include <jem/base/StringBuffer.h>
#include <jem/base/IllegalInputException.h>
#include <jive/util/Factory.h>


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;
using jem::IllegalInputException;


//=======================================================================
//   class Factory
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Factory::TYPE_PROP = "type";


//-----------------------------------------------------------------------
//   noSuchTypeError
//-----------------------------------------------------------------------


void Factory::noSuchTypeError

  ( const String&  context,
    const String&  type,
    const String&  kind )

{
  throw IllegalInputException (
    context,
    String::format ( "undefined %s: `%s\'", kind, type )
  );
}


void Factory::noSuchTypeError

  ( const String&        context,
    const String&        type,
    const String&        kind,
    const StringVector&  knownTypes )

{
  using jem::StringBuffer;

  const idx_t   n = knownTypes.size ();

  StringBuffer  buf;


  buf.reserve ( 128 );

  print ( buf, "unknown ", kind, ": `", type, "\'.\n" );
  print ( buf, "Available types are: " );

  idx_t  i = 0;

  while ( i < n )
  {
    print ( buf, '`', knownTypes[i], '\'' );

    i++;

    if ( i < n )
    {
      print ( buf, ", " );
    }
  }

  throw IllegalInputException ( context, buf.toString() );
}


JIVE_END_PACKAGE( util )
