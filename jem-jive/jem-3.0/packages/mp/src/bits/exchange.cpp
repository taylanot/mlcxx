
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


#include <jem/mp/Status.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/mp/RequestList.h>
#include <jem/mp/MPException.h>
#include <jem/mp/exchange.h>


JEM_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   constants
//-----------------------------------------------------------------------


static const int  EXCHANGE_SIZE_TAG = 42;
static const int  EXCHANGE_DATA_TAG = 43;


//-----------------------------------------------------------------------
//   sendExchangeData
//-----------------------------------------------------------------------


template <class T>

  static inline void      sendExchangeData

  ( Context&                ctx,
    int                     dest,
    const Array<T>&         data )

{
  if ( data.size() > 0 )
  {
    if ( data.isContiguous() )
    {
      ctx.send ( SendBuffer( data.addr(), data.size() ),
                 dest, EXCHANGE_DATA_TAG );
    }
    else
    {
      Array<T> tmp ( data.size() );

      tmp = data;

      ctx.send ( SendBuffer( tmp.addr(), tmp.size() ),
                 dest, EXCHANGE_DATA_TAG );
    }
  }
}


//-----------------------------------------------------------------------
//   recvExchangeData
//-----------------------------------------------------------------------


template <class T>

  static inline void      recvExchangeData

  ( RequestList&            reqs,
    int                     src,
    const Array<T>&         data )

{
  if ( data.size() > 0 )
  {
    reqs.addRequest ( RecvBuffer( data.addr(), data.size() ),
                      src, EXCHANGE_DATA_TAG );
  }
}


//-----------------------------------------------------------------------
//   getRecvPacketCount
//-----------------------------------------------------------------------


static idx_t              getRecvPacketCount

  ( Context&                ctx,
    const Array<XPacket>&   sendPackets )

{
  const int       procCount  = ctx        .size ();
  const idx_t     spCount    = sendPackets.size ();

  Array<idx_t,2>  ibuf       ( procCount, 2 );

  Array<idx_t>    sendMask   = ibuf(ALL,0);
  Array<idx_t>    recvCounts = ibuf(ALL,1);


  sendMask = 0;

  for ( idx_t ip = 0; ip < spCount; ip++ )
  {
    XPacket&  xpk = sendPackets[ip];

    if ( xpk.hasData() )
    {
      sendMask[xpk.rank]++;
    }
  }

  ctx.allreduce ( RecvBuffer( recvCounts.addr(), procCount ),
                  SendBuffer( sendMask  .addr(), procCount ),
                  SUM );

  return recvCounts[ctx.myRank()];
}


//-----------------------------------------------------------------------
//   exchange
//-----------------------------------------------------------------------


void                      exchange

  ( Context&                ctx,
    Array<XPacket>&         recvPackets,
    const Array<XPacket>&   sendPackets )

{
  const int         procCount = ctx        .size ();
  const idx_t       spCount   = sendPackets.size ();

  Ref<RequestList>  reqs      = ctx.newRequestList ();

  Array<idx_t>      recvSizes;
  idx_t             sendSizes[3];
  Array<Status>     recvStats;
  idx_t             rpCount;


  for ( idx_t ip = 0; ip < spCount; ip++ )
  {
    XPacket&  xpk = sendPackets[ip];

    if ( xpk.rank < 0 || xpk.rank >= procCount )
    {
      throw MPException (
        JEM_FUNC,
        String::format ( "invalid process rank: %d", xpk.rank )
      );
    }
  }

  rpCount = getRecvPacketCount ( ctx, sendPackets );

  if ( recvPackets.size() != rpCount )
  {
    recvPackets.resize ( rpCount );
  }

  recvSizes.resize ( 3 * rpCount );
  recvStats.resize (     rpCount );

  for ( idx_t ip = 0; ip < rpCount; ip++ )
  {
    idx_t  i = 3 * ip;

    reqs->addRequest ( RecvBuffer( recvSizes.addr(i), 3 ),
                       ANY_SOURCE, EXCHANGE_SIZE_TAG );
  }

  reqs->startAll ();

  for ( idx_t ip = 0; ip < spCount; ip++ )
  {
    XPacket&  xpk = sendPackets[ip];

    if ( xpk.hasData() )
    {
      sendSizes[0] = xpk.binData.size ();
      sendSizes[1] = xpk.intData.size ();
      sendSizes[2] = xpk.dblData.size ();

      ctx.send ( SendBuffer( sendSizes, 3 ),
                 xpk.rank, EXCHANGE_SIZE_TAG );
    }
  }

  reqs->waitAll ( recvStats.addr() );
  reqs->clear   ();

  for ( idx_t ip = 0; ip < rpCount; ip++ )
  {
    XPacket&  xpk = recvPackets[ip];
    idx_t     i   = 3_idx * ip;

    xpk.rank = recvStats[ip].source;

    xpk.binData.resize ( recvSizes[i + 0] );
    xpk.intData.resize ( recvSizes[i + 1] );
    xpk.dblData.resize ( recvSizes[i + 2] );

    recvExchangeData ( *reqs, xpk.rank, xpk.binData );
    recvExchangeData ( *reqs, xpk.rank, xpk.intData );
    recvExchangeData ( *reqs, xpk.rank, xpk.dblData );
  }

  reqs->startAll ();

  for ( idx_t  ip = 0; ip < spCount; ip++ )
  {
    XPacket&  xpk = sendPackets[ip];

    sendExchangeData ( ctx, xpk.rank, xpk.binData );
    sendExchangeData ( ctx, xpk.rank, xpk.intData );
    sendExchangeData ( ctx, xpk.rank, xpk.dblData );
  }

  reqs->waitAll ();
}


JEM_END_PACKAGE( mp )
