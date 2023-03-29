
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


#include <jem/base/IllegalInputException.h>
#include <jive/util/XItemMap.h>
#include <jive/util/StdPointSet.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/XNodeSet.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class XNodeSet
//=======================================================================

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


XNodeSet XNodeSet::get

  ( const Properties&  globdat,
    const String&      context )

{
  XNodeSet  nodes =

    toXNodeSet ( Super::get( globdat, context ) );

  if ( ! nodes )
  {
    throw jem::IllegalInputException (
      context,
      "the standard node set is not editable (not an XNodeSet)"
    );
  }

  return nodes;
}


XNodeSet XNodeSet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  XNodeSet  nodes =

    toXNodeSet ( Super::get( name, globdat, context ) );

  if ( ! nodes )
  {
    throw jem::IllegalInputException (
      context,
      String::format (
        "the node set `%s\' is not editable (not an XNodeSet)",
        name
      )
    );
  }

  return nodes;
}


//=======================================================================
//   related data
//=======================================================================

//-----------------------------------------------------------------------
//   newXNodeSet
//-----------------------------------------------------------------------


XNodeSet newXNodeSet ( util::StorageMode mode )
{
  return XNodeSet (
    newInstance<util::StdPointSet> (
      "nodes",
      "node",
      util::newXItemMap ( mode )
    )
  );
}


XNodeSet              newXNodeSet

  ( const String&       name,
    util::StorageMode   mode )

{
  return XNodeSet (
    newInstance<util::StdPointSet> (
      name,
      "node",
      util::newXItemMap ( mode )
    )
  );
}


JIVE_END_PACKAGE( fem )
