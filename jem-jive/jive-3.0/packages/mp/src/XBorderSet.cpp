
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
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/XItemMap.h>
#include <jive/util/StdGroupSet.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/XBorderSet.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::newXItemMap;
using jive::util::StorageMode;
using jive::util::StdGroupSet;


//=======================================================================
//   class XBorderSet
//=======================================================================

//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


XBorderSet XBorderSet::find

  ( BorderKind           kind,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat = Globdat::getDataFor ( items, globdat );

  Ref<Data>   borders;

  if ( kind & RECV_BORDERS )
  {
    itemdat.find ( borders, Globdat::RECV_BORDERS );
  }

  if ( kind & SEND_BORDERS )
  {
    itemdat.find ( borders, Globdat::SEND_BORDERS );
  }

  return XBorderSet ( borders );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


XBorderSet XBorderSet::get

  ( BorderKind           kind,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat = Globdat::getDataFor ( items, globdat );

  Ref<Data>   borders;

  if ( kind & RECV_BORDERS )
  {
    itemdat.find ( borders, Globdat::RECV_BORDERS );
  }

  if ( kind & SEND_BORDERS )
  {
    itemdat.find ( borders, Globdat::SEND_BORDERS );
  }

  if ( ! borders )
  {
    XBorderSet xborders ( newXBorderSet( kind, items ) );

    xborders.store ( kind, globdat );

    borders = xborders.getData ();
  }

  return XBorderSet ( borders );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newXBorderSet
//-----------------------------------------------------------------------


XBorderSet              newXBorderSet

  ( BorderKind            kind,
    const Ref<ItemSet>&   items,
    StorageMode           mode )

{
  JEM_PRECHECK ( items );

  String  itemName = items->getItemName ();

  String  bdrName;
  String  setName;


  if ( kind == RECV_BORDERS )
  {
    bdrName = String::format ( "recv %s border", itemName );
    setName = "recvBorders";
  }
  else if ( kind == SEND_BORDERS )
  {
    bdrName = String::format ( "send %s border", itemName );
    setName = "sendBorders";
  }
  else
  {
    bdrName = itemName + " border";
    setName = "borders";
  }

  return XBorderSet (
    newInstance<StdGroupSet> (
      joinNames   ( items->getName(), setName ),
      bdrName,
      items,
      newXItemMap ( mode )
    )
  );
}


XBorderSet              newXBorderSet

  ( const String&         name,
    const Ref<ItemSet>&   items,
    StorageMode           mode )

{
  JEM_PRECHECK ( items );

  return XBorderSet (
    newInstance<StdGroupSet> (
      name,
      items->getItemName() + " border",
      items,
      newXItemMap ( mode )
    )
  );
}


JIVE_END_PACKAGE( mp )
