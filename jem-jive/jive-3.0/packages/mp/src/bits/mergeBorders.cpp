
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
#include <jem/base/Array.h>
#include <jem/base/Error.h>
#include <jem/base/RuntimeException.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/mp/RequestList.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/mp/XBorderSet.h>
#include <jive/mp/utilities.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::Error;
using jem::RuntimeException;
using jem::mp::SendBuffer;
using jem::mp::RecvBuffer;
using jem::mp::RequestList;
using jem::numeric::SparseUtils;
using jive::util::ItemMap;

typedef jem::numeric::SparseStructure  MapMatrix;
typedef jem::util::ArrayBuffer<idx_t>  IdxBuffer;


//-----------------------------------------------------------------------
//   getBorderItems
//-----------------------------------------------------------------------


static void         getBorderItems

  ( IdxVector&        offsets,
    IdxVector&        iitems,
    int               myRank,
    const IdxVector&  itemIDs,
    const BorderSet&  borders )

{
  const idx_t  borderCount = borders.size ();

  MapMatrix    biMap       = borders.toMatrix ();
  MapMatrix    ibMap       = biMap.transpose  ();

  IdxVector    biOffsets   = biMap.getRowOffsets    ();
  IdxVector    biIndices   = biMap.getColumnIndices ();
  IdxVector    ibOffsets   = ibMap.getRowOffsets    ();
  IdxVector    ibIndices   = ibMap.getColumnIndices ();

  IdxVector    borderIDs   ( borderCount );

  idx_t        itemID;
  idx_t        iitem;
  idx_t        ilast;
  idx_t        jlast;
  idx_t        ib;
  idx_t        jb;
  idx_t        i, j, k;


  offsets.resize ( borderCount + 1 );

  k = 0;

  for ( ib = 0; ib < borderCount; ib++ )
  {
    offsets[ib] = k;
    ilast       = biOffsets[ib + 1];

    for ( i = biOffsets[ib]; i < ilast; i++ )
    {
      iitem = biIndices[i];
      k    += 2 * (ibOffsets[iitem + 1] - ibOffsets[iitem]) + 2;
    }

    borderIDs[ib] = borders.getBorderID ( ib );
  }

  offsets[borderCount] = k;

  iitems.resize ( k );

  k = 0;

  for ( ib = 0; ib < borderCount; ib++ )
  {
    ilast = biOffsets[ib + 1];

    for ( i = biOffsets[ib]; i < ilast; i++ )
    {
      iitem       = biIndices[i];
      itemID      = itemIDs[iitem];
      jlast       = ibOffsets[iitem + 1];

      iitems[k++] = itemID;
      iitems[k++] = (idx_t) myRank;

      for ( j = ibOffsets[iitem]; j < jlast; j++, k += 2 )
      {
        jb            = ibIndices[j];
        iitems[k + 0] = itemID;
        iitems[k + 1] = borderIDs[jb];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   makeNewBorders
//-----------------------------------------------------------------------


static void             makeNewBorders

  ( const XBorderSet&     newBorders,
    const IdxVector&      borderIDs,
    const IdxVector&      offsets,
    const IdxVector&      iitems,
    const IdxVector&      itemIDs,
    const Ref<ItemSet>&   items )

{
  const idx_t  borderCount = borderIDs.size ();

  IdxVector    jitems;
  IdxVector    jperm;

  idx_t        ib;
  idx_t        i, j, n;


  newBorders.clear   ();
  newBorders.reserve ( borderCount );

  for ( ib = 0; ib < borderCount; ib++ )
  {
    i = offsets[ib];
    j = offsets[ib + 1];
    n = j - i;

    jitems.ref ( iitems[slice(i,j)] );
    jem::sort  ( jitems, itemIDs );

    j = 0;

    for ( i = 1; i < n; i++ )
    {
      if ( jitems[i] != jitems[j] )
      {
        jitems[++j] = jitems[i];
      }
    }

    n = j + 1;

    newBorders.addBorder ( borderIDs[ib], jitems[slice(BEGIN,n)] );
  }

  newBorders.trimToSize ();
}


//-----------------------------------------------------------------------
//   mergeMyBorders
//-----------------------------------------------------------------------


static void           mergeMyBorders

  ( const XBorderSet&   newBorders,
    int                 myRank,
    int                 procCount,
    const IdxVector&    recvItems,
    const IdxVector&    itemIDs,
    const BorderSet&    sendBorders )

{
  const idx_t  sbCount = sendBorders.size ();

  IdxVector    iitems  ( sendBorders.maxBorderSize() );
  IdxVector    bmap    ( procCount );

  IdxBuffer    borderIDs;
  IdxBuffer    borderOffsets;
  IdxVector    borderItems;
  idx_t        borderCount;

  idx_t        rank;
  idx_t        ib;
  idx_t        jb;
  idx_t        i, j, n;


  borderOffsets.reserve ( sbCount + 1 );
  borderIDs    .reserve ( sbCount );

  bmap        = -1_idx;
  borderCount = 0;

  for ( ib = 0; ib < sbCount; ib++ )
  {
    rank = sendBorders.getBorderID ( ib );

    if ( bmap[rank] >= 0 )
    {
      throw Error ( JEM_FUNC, "invalid send borders" );
    }

    bmap[rank] = borderCount++;

    borderIDs    .pushBack ( rank );
    borderOffsets.pushBack ( sendBorders.getBorderSize( ib ) );
  }

  for ( i = 0, n = recvItems.size(); i < n; i += 2 )
  {
    rank = recvItems[i + 1];

    if ( rank == myRank )
    {
      continue;
    }

    if ( bmap[rank] < 0 )
    {
      bmap[rank] = borderCount++;

      borderIDs    .pushBack ( rank );
      borderOffsets.pushBack (    0 );
    }

    borderOffsets[bmap[rank]]++;
  }

  borderOffsets.pushBack  ( 0 );
  SparseUtils::sumOffsets ( borderOffsets.toArray() );
  borderItems.resize      ( borderOffsets[borderCount] );

  for ( ib = 0; ib < sbCount; ib++ )
  {
    rank = sendBorders.getBorderID    ( ib );
    n    = sendBorders.getBorderItems ( iitems, ib );
    jb   = bmap[rank];
    j    = borderOffsets[jb];

    for ( i = 0; i < n; i++, j++ )
    {
      borderItems[j] = iitems[i];
    }

    borderOffsets[jb] = j;
  }

  for ( i = 0, n = recvItems.size(); i < n; i += 2 )
  {
    rank = recvItems[i + 1];

    if ( rank == myRank )
    {
      continue;
    }

    jb = bmap[rank];
    borderItems[borderOffsets[jb]] = recvItems[i];
    borderOffsets[jb]++;
  }

  SparseUtils::shiftOffsets ( borderOffsets.toArray() );

  makeNewBorders ( newBorders,
                   borderIDs    .toArray (),
                   borderOffsets.toArray (),
                   borderItems,
                   itemIDs,
                   sendBorders.getItems () );
}


//-----------------------------------------------------------------------
//   mergeBorders
//-----------------------------------------------------------------------


BorderSet             mergeBorders

  ( Context&            mpx,
    const BorderSet&    recvBorders,
    const BorderSet&    sendBorders )

{
  XBorderSet  newBorders (
    newXBorderSet (
      SEND_RECV_BORDERS,
      sendBorders.getItems()
    )
  );

  mergeBorders ( newBorders, mpx, recvBorders, sendBorders );

  return newBorders;
}


void                  mergeBorders

  ( const XBorderSet&   newBorders,
    Context&            mpx,
    const BorderSet&    recvBorders,
    const BorderSet&    sendBorders )

{
  JEM_PRECHECK ( newBorders  &&
                 recvBorders &&
                 sendBorders );

  JEM_PRECHECK ( newBorders .getItems() == recvBorders.getItems() &&
                 recvBorders.getItems() == sendBorders.getItems() );


  const ItemSet&    items     = * recvBorders.getItems ();
  const ItemMap&    imap      = * items.getItemMap     ();

  const int         myRank    = mpx.myRank       ();
  const idx_t       itemCount = items      .size ();
  const idx_t       rbCount   = recvBorders.size ();
  const idx_t       sbCount   = sendBorders.size ();

  Ref<RequestList>  reqs;

  IdxVector         iitems    ( jem::iarray( itemCount ) );
  IdxVector         itemIDs   ( itemCount );

  IdxVector         sendOffsets;
  IdxVector         sendItems;
  IdxVector         recvOffsets;
  IdxVector         recvItems;

  int               rank;
  idx_t             itemID;
  idx_t             iitem;
  idx_t             ib;
  idx_t             i, n;


  imap.getItemIDs ( itemIDs, iitems );

  reqs = mpx.newRequestList ();

  recvOffsets.resize ( rbCount + 1 );

  recvOffsets[rbCount] = 0;

  for ( ib = 0; ib < rbCount; ib++ )
  {
    rank = (int) recvBorders.getBorderID ( ib );

    reqs->addRequest ( RecvBuffer( & recvOffsets[ib], 1 ), rank );
  }

  reqs->startAll ();

  getBorderItems ( sendOffsets, sendItems,
                   myRank,      itemIDs, sendBorders );

  for ( ib = 0; ib < sbCount; ib++ )
  {
    rank = (int) sendBorders.getBorderID ( ib );
    n    = sendOffsets[ib + 1] - sendOffsets[ib];

    mpx.send ( SendBuffer( &n, 1 ), rank );
  }

  reqs->waitAll ();
  reqs->clear   ();

  SparseUtils::sumOffsets ( recvOffsets );

  recvItems.resize ( recvOffsets[rbCount] );

  for ( ib = 0; ib < rbCount; ib++ )
  {
    rank = (int) recvBorders.getBorderID ( ib );
    n    = recvOffsets[ib + 1] - recvOffsets[ib];

    reqs->addRequest (
      RecvBuffer ( recvItems.addr(recvOffsets[ib]), n ),
      rank
    );
  }

  reqs->startAll ();

  for ( ib = 0; ib < sbCount; ib++ )
  {
    rank = (int) sendBorders.getBorderID ( ib );
    n    = sendOffsets[ib + 1] - sendOffsets[ib];

    mpx.send (
      SendBuffer ( sendItems.addr(sendOffsets[ib]), n ),
      rank
    );
  }

  reqs->waitAll ();

  n      = recvItems.size ();
  itemID = 0;
  iitem  = 0;

  if ( n > 0 )
  {
    itemID = imap.getItemID ( iitem );
  }

  for ( i = 0; i < n; i += 2 )
  {
    if ( recvItems[i] != itemID )
    {
      itemID = recvItems[i];
      iitem  = imap.findItem ( itemID );

      if ( iitem < 0 )
      {
        String itemName = items.getItemName ();

        throw RuntimeException (
          JEM_FUNC,
          String::format (
            "undefined %s: %d (invalid border set?)",
            itemName,
            itemID
          )
        );
      }
    }

    recvItems[i] = iitem;
  }

  sendOffsets.ref ( IdxVector() );
  sendItems  .ref ( IdxVector() );
  recvOffsets.ref ( IdxVector() );

  reqs = nullptr;

  mergeMyBorders ( newBorders,
                   myRank,
                   mpx.size(),
                   recvItems,
                   itemIDs,
                   sendBorders );
}


JIVE_END_PACKAGE( mp )
