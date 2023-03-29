
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


#include <jem/base/array/utilities.h>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/AllItemGroup.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/mp/scatter.h>


JIVE_BEGIN_PACKAGE( mp )


using jem::dynamicCast;
using jem::newInstance;
using jem::util::ArrayBuffer;
using jive::util::ItemMap;
using jive::util::ItemGroup;
using jive::util::AllItemGroup;
using jive::util::ArrayItemGroup;

typedef ArrayBuffer<idx_t>  IdxBuffer;


//=======================================================================
//   class GroupCodec
//=======================================================================


class GroupCodec : public ScatterCodec
{
 public:

  enum                    GroupTag
  {
                            NIL_GROUP_TAG,
                            ALL_GROUP_TAG,
                            ARRAY_GROUP_TAG
  };


  virtual void            writeObject

    ( DataOutput&           out,
      const Ref<Object>&    obj )   override;

  virtual Ref<Object>     readObject

    ( DataInput&            in,
      const Ref<ItemSet>&   items ) override;

};


//-----------------------------------------------------------------------
//   writeObject
//-----------------------------------------------------------------------


void GroupCodec::writeObject

  ( DataOutput&         out,
    const Ref<Object>&  obj )

{
  Ref<ItemGroup>  group = dynamicCast<ItemGroup> ( obj );


  if      ( ! group )
  {
    encode ( out, (int) NIL_GROUP_TAG );
  }
  else if ( dynamicCast<AllItemGroup*>( group ) )
  {
    encode ( out, (int) ALL_GROUP_TAG );
  }
  else
  {
    IdxVector  itemIDs = makeContiguous ( group->getIDs() );

    encode     ( out, (idx_t) ARRAY_GROUP_TAG, itemIDs.size() );
    out.encode ( itemIDs.addr(), itemIDs.size() );
  }
}


//-----------------------------------------------------------------------
//   readObject
//-----------------------------------------------------------------------


Ref<Object> GroupCodec::readObject

  ( DataInput&           in,
    const Ref<ItemSet>&  items )

{
  Ref<ItemGroup>  group;
  int             tag;


  decode ( in, tag );

  if      ( tag == ALL_GROUP_TAG )
  {
    group = newInstance<AllItemGroup> ( items );
  }
  else if ( tag != NIL_GROUP_TAG )
  {
    const ItemMap*  imap = items->getItemMap ();

    const idx_t     BLOCK_SIZE = 1024;

    IdxVector       itemIDs ( BLOCK_SIZE );
    IdxBuffer       iitems;

    idx_t           iitem;
    idx_t           i, k, n;


    iitems.reserve ( BLOCK_SIZE );

    decode ( in, n );

    while ( n > 0 )
    {
      k  = jem::min ( n, BLOCK_SIZE );
      n -= k;

      in.decode ( itemIDs.addr(), k );

      for ( i = 0; i < k; i++ )
      {
        iitem = imap->findItem ( itemIDs[i] );

        if ( iitem >= 0 )
        {
          iitems.pushBack ( iitem );
        }
      }
    }

    group = newInstance<ArrayItemGroup> ( iitems.toArray(), items );
  }

  return group;
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   recvGroups
//-----------------------------------------------------------------------


void                    recvGroups

  ( Context&              mpx,
    Dictionary&           recvDict,
    const Ref<ItemSet>&   myItems,
    int                   root )

{
  GroupCodec  codec;

  recvObjects ( mpx, codec, recvDict, myItems, root );
}


//-----------------------------------------------------------------------
//   scatterGroups
//-----------------------------------------------------------------------


void                    scatterGroups

  ( Context&              mpx,
    Dictionary&           recvDict,
    Dictionary&           sendDict,
    const Ref<ItemSet>&   myItems )

{
  GroupCodec  codec;

  scatterObjects ( mpx, codec, recvDict, sendDict, myItems );
}


JIVE_END_PACKAGE( mp )
