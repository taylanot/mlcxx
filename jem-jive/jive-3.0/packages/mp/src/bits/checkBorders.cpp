
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


#include <jem/base/array/intrinsics.h>
#include <jem/base/System.h>
#include <jem/mp/exchange.h>
#include <jem/mp/Context.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/mp/utilities.h>
#include <jive/mp/InvalidBordersException.h>


JIVE_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   bordersError
//-----------------------------------------------------------------------


static void         bordersError

  ( const BorderSet&  recvBorders,
    const BorderSet&  sendBorders,
    const String&     what )

{
  String  context;

  if ( recvBorders == sendBorders )
  {
    String  name = recvBorders.getName ();

    context = String::format ( "checkBorders ( %s )", name );
  }
  else
  {
    String  name1 = recvBorders.getName ();
    String  name2 = sendBorders.getName ();

    context = String::format ( "checkBorders ( %s, %s )",
                               name1, name2 );
  }

  throw InvalidBordersException ( context, what );
}


//-----------------------------------------------------------------------
//   checkBorders
//-----------------------------------------------------------------------


void                checkBorders

  ( Context&          mpx,
    const BorderSet&  recvBorders,
    const BorderSet&  sendBorders )

{
  using jem::castTo;
  using jem::Array;
  using jem::io::Writer;
  using jem::mp::XPacket;
  using jive::util::ItemMap;

  const ItemSet*  items  = recvBorders.getItems   ();
  const ItemMap*  imap   = items     ->getItemMap ();

  Array<XPacket>  sendPackets ( sendBorders.size() );
  Array<XPacket>  recvPackets;

  IdxVector       itemIDs;
  IdxVector       iitems;

  idx_t           ib, ip;
  idx_t           n;


  if ( recvBorders.getItems() != sendBorders.getItems() )
  {
    String  name1 = recvBorders.getItems()->getName ();
    String  name2 = sendBorders.getItems()->getName ();

    bordersError (
      recvBorders,
      sendBorders,
      String::format (
        "send and receive borders are associated with different "
        "item sets (recv items = `%s\', send items = `%s\')",
        name1,
        name2
      )
    );
  }

  iitems.resize ( sendBorders.maxBorderSize() );

  for ( ib = 0; ib < sendBorders.size(); ib++ )
  {
    XPacket&  xp = sendPackets[ib];

    xp.rank = (int) sendBorders.getBorderID ( ib );

    n = sendBorders.getBorderItems ( iitems, ib );

    itemIDs   .resize ( n );
    xp.intData.resize ( n );

    imap->getItemIDs  ( itemIDs, iitems[slice(BEGIN,n)] );

    xp.intData = castTo<lint> ( itemIDs );
  }

  exchange ( mpx, recvPackets, sendPackets );

  if ( recvPackets.size() != recvBorders.size() )
  {
    bordersError (
      recvBorders,
      sendBorders,
      String::format (
        "expected %d receive borders, but got %d "
        "receive borders",
        recvBorders.size (),
        recvPackets.size ()
      )
    );
  }

  n = recvBorders.maxBorderSize ();

  iitems .resize ( n );
  itemIDs.resize ( n );

  for ( ip = 0; ip < recvPackets.size(); ip++ )
  {
    XPacket&  xp = recvPackets[ip];

    ib = recvBorders.findBorder ( xp.rank );

    if ( ib < 0 )
    {
      bordersError (
        recvBorders,
        sendBorders,
        String::format (
          "send border on process %d has no matching receive "
          "border on process %d",
          xp .rank,
          mpx.myRank ()
        )
      );
    }

    n = recvBorders.getBorderItems ( iitems, ib );

    if ( n != xp.intData.size() )
    {
      bordersError (
        recvBorders,
        sendBorders,
        String::format (
          "send border on process %d has size %d while the "
          "matching receive border on process %d has size %d",
          xp .rank,
          xp .intData.size (),
          mpx.myRank (),
          n
        )
      );
    }

    imap->getItemIDs ( itemIDs[slice(BEGIN,n)],
                       iitems [slice(BEGIN,n)] );

    for ( idx_t i = 0; i < n; i++ )
    {
      if ( itemIDs[i] != xp.intData[i] )
      {
        Writer&  debug = jem::System::debug ( "jive" );
        String   name  = items->getItemName ();

        print ( debug,
                "Send border on process ", xp.rank,
                " is incompatible with the receive border"
                " on process ", mpx.myRank(), "\n\n" );

        print ( debug,
                "  send border item IDs : ",
                xp.intData, "\n" );
        print ( debug,
                "  recv border item IDs : ",
                itemIDs[slice(BEGIN,n)], "\n\n" );

        bordersError (
          recvBorders,
          sendBorders,
          String::format (
            "send border on process %d is incompatible with the "
            "receive border on process %d "
            "(%s IDs do not match)",
            xp .rank,
            mpx.myRank (),
            name
          )
        );
      }
    }
  }
}


JIVE_END_PACKAGE( mp )
