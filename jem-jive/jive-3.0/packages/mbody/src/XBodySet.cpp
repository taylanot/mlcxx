
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
#include <jem/base/IllegalInputException.h>
#include <jive/util/XItemMap.h>
#include <jive/util/StdGroupSet.h>
#include <jive/mbody/Globdat.h>
#include <jive/mbody/XBodySet.h>


JIVE_BEGIN_PACKAGE( mbody )


using jem::newInstance;
using jive::util::newXItemMap;
using jive::util::StorageMode;


//=======================================================================
//   class XBodySet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  XBodySet::CLASS_NAME_ = "jive::mbody::XBodySet";


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


XBodySet XBodySet::get

  ( BodyKind           kind,
    const Properties&  globdat,
    const String&      context )

{
  XBodySet  bodies =

    toXBodySet ( Super::get( kind, globdat, context ) );

  if ( ! bodies )
  {
    String  msg;

    if ( kind == REGULAR_BODIES )
    {
      msg = "the standard ";
    }
    else
    {
      msg = "the surface ";
    }

    msg = msg + "body set is not editable (not an XBodySet)";

    throw jem::IllegalInputException ( context, msg );
  }

  return bodies;
}


XBodySet XBodySet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  XBodySet  bodies =

    toXBodySet ( Super::get( name, globdat, context ) );

  if ( ! bodies )
  {
    throw jem::IllegalInputException (
      context,
      String::format (
        "the body set `%s\' is not editable (not an XBodySet)",
        name
      )
    );
  }

  return bodies;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newXBodySet
//-----------------------------------------------------------------------


XBodySet              newXBodySet

  ( BodyKind            kind,
    const ElementSet&   elems,
    StorageMode         mode )

{
  JEM_PRECHECK ( elems );

  return XBodySet (
    newInstance<util::StdGroupSet> (
      Globdat::BODIES[kind],
      "body",
      elems.getData (),
      newXItemMap   ( mode )
    )
  );
}


XBodySet              newXBodySet

  ( const String&       name,
    const ElementSet&   elems,
    StorageMode         mode )

{
  JEM_PRECHECK ( elems );

  return XBodySet (
    newInstance<util::StdGroupSet> (
      name,
      "body",
      elems.getData (),
      newXItemMap   ( mode )
    )
  );
}


JIVE_END_PACKAGE( mbody )
