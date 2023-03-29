
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


#include <jem/io/DataInputStream.h>
#include <jem/io/DataOutputStream.h>
#include <jem/io/ArrayInputStream.h>
#include <jem/io/ArrayOutputStream.h>
#include <jem/io/SerializationException.h>
#include <jem/mp/Context.h>
#include <jem/mp/BcastStream.h>
#include <jive/util/ItemMap.h>
#include <jive/util/XMemberSet.h>
#include <jive/mp/scatter.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::dynamicCast;
using jem::newInstance;
using jem::io::DataInputStream;
using jem::io::DataOutputStream;
using jem::io::ArrayInputStream;
using jem::io::ArrayOutputStream;
using jem::io::SerializationException;
using jem::mp::BcastStream;
using jem::util::ArrayBuffer;
using jive::util::ItemMap;


//-----------------------------------------------------------------------
//   readMembers
//-----------------------------------------------------------------------


static void         readMembers

  ( DataInput&        in,
    XMemberSet&       mbrs )

{
  ItemSet*  items = mbrs  .getCompoundItems ();
  ItemMap*  imap  = items->getItemMap       ();

  idx_t     mbrCount;
  idx_t     mbrID;
  idx_t     itemID;
  idx_t     iitem;
  idx_t     ilocal;


  decode ( in, mbrCount );

  if ( mbrCount < 0 )
  {
    throw SerializationException (
      JEM_FUNC,
      "error reading data stream (negative number of members)"
    );
  }

  for ( idx_t i = 0; i < mbrCount; i++ )
  {
    decode ( in, mbrID, itemID, ilocal );

    iitem = imap->findItem ( itemID );

    if ( iitem >= 0 )
    {
      mbrs.addMember ( mbrID, iitem, ilocal );
    }
  }
}


//-----------------------------------------------------------------------
//   writeMembers
//-----------------------------------------------------------------------


static void         writeMembers

  ( DataOutput&       out,
    const MemberSet&  mbrs )

{
  const idx_t  mbrCount = mbrs.size ();

  ItemSet*     items    = mbrs  .getCompoundItems ();
  ItemMap*     imap     = items->getItemMap       ();
  ItemMap*     mmap     = mbrs  .getItemMap       ();

  idx_t        mbrID;
  idx_t        itemID;
  idx_t        iitem;
  idx_t        ilocal;
  idx_t        imbr;


  encode ( out, mbrCount );

  for ( imbr = 0; imbr < mbrCount; imbr++ )
  {
    mbrs.getMember ( iitem, ilocal, imbr );

    mbrID  = mmap->getItemID ( imbr  );
    itemID = imap->getItemID ( iitem );

    encode ( out, mbrID, itemID, ilocal );
  }
}


//-----------------------------------------------------------------------
//   recvMembers
//-----------------------------------------------------------------------


void                recvMembers

  ( Context&          mpx,
    XMemberSet&       mbrs,
    int               root )

{
  Ref<DataInputStream>  input  =

    newInstance<DataInputStream> (
      newInstance<BcastStream> ( root, & mpx, nullptr )
    );

  readMembers ( *input, mbrs );
}


//-----------------------------------------------------------------------
//   scatterMembers
//-----------------------------------------------------------------------


void                scatterMembers

  ( Context&          mpx,
    XMemberSet&       recvMbrs,
    const MemberSet&  sendMbrs )

{
  Ref<ArrayInputStream>   inBuf;
  Ref<ArrayOutputStream>  outBuf;
  Ref<DataInputStream>    input;
  Ref<DataOutputStream>   output;


  outBuf = newInstance<ArrayOutputStream> ();
  output = newInstance<DataOutputStream>  ( outBuf );

  writeMembers ( *output, sendMbrs );

  inBuf  = newInstance<ArrayInputStream> ( outBuf->toArray() );

  input  = newInstance<DataInputStream> (
    newInstance<BcastStream> ( mpx.myRank(), & mpx, inBuf )
  );

  readMembers ( *input, recvMbrs );
}


JIVE_END_PACKAGE( mp )
