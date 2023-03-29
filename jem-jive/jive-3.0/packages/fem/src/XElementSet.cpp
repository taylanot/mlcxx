
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
#include <jive/fem/Globdat.h>
#include <jive/fem/XElementSet.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class XElementSet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char* XElementSet::CLASS_NAME_ = "jive::fem::XElementSet";


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


XElementSet XElementSet::get

  ( const Properties&  globdat,
    const String&      context )

{
  XElementSet  elems =

    toXElementSet ( Super::get( globdat, context ) );

  if ( ! elems )
  {
    throw jem::IllegalInputException (
      context,
      "the standard element set is not editable (not an XElementSet)"
    );
  }

  return elems;
}


XElementSet XElementSet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  XElementSet  elems =

    toXElementSet ( Super::get( name, globdat, context ) );

  if ( ! elems )
  {
    throw jem::IllegalInputException (
      context,
      String::format (
        "the element set `%s\' is not editable (not an XElementSet)",
        name
      )
    );
  }

  return elems;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newXElementSet
//-----------------------------------------------------------------------


XElementSet           newXElementSet

  ( const NodeSet&      nodes,
    util::StorageMode   mode )

{
  JEM_PRECHECK ( nodes );

  return XElementSet (
    newInstance<util::StdGroupSet> (
      "elements",
      "element",
      nodes.getData     (),
      util::newXItemMap ( mode )
    )
  );
}


XElementSet           newXElementSet

  ( const String&       name,
    const NodeSet&      nodes,
    util::StorageMode   mode )

{
  JEM_PRECHECK ( nodes );

  return XElementSet (
    newInstance<util::StdGroupSet> (
      name,
      "element",
      nodes.getData     (),
      util::newXItemMap ( mode )
    )
  );
}


JIVE_END_PACKAGE( fem )
