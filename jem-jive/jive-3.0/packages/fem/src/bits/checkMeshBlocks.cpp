
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


#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/array/utilities.h>
#include <jem/util/SparseArray.h>
#include <jive/mp/ItemMask.h>
#include <jive/fem/InvalidBlocksException.h>
#include <jive/fem/MeshBlock.h>
#include <jive/fem/MeshCutter.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::Array;
using jem::Error;
using jem::util::SparseArray;
using jive::mp::BorderSet;


//-----------------------------------------------------------------------
//   getRecvBorders
//-----------------------------------------------------------------------


static BorderSet        getRecvBorders

  ( const MeshBlock&      block,
    const String&         kind )

{
  if      ( kind == "node" )
  {
    return block.recvNodeBorders;
  }
  else if ( kind == "element" )
  {
    return block.recvElemBorders;
  }
  else
  {
    throw Error ( JEM_FUNC, "illegal recv border kind: " + kind );
  }

  return BorderSet ();
}


//-----------------------------------------------------------------------
//   getSendBorders
//-----------------------------------------------------------------------


static BorderSet        getSendBorders

  ( const MeshBlock&      block,
    const String&         kind )

{
  if      ( kind == "node" )
  {
    return block.sendNodeBorders;
  }
  else if ( kind == "element" )
  {
    return block.sendElemBorders;
  }
  else
  {
    throw Error ( JEM_FUNC, "illegal send border kind: " + kind );
  }

  return BorderSet ();
}


//-----------------------------------------------------------------------
//   checkNodeMasks
//-----------------------------------------------------------------------


static void             checkNodeMasks

  ( const Array
    < Ref<MeshBlock> >&   blocks )

{
  SparseArray<idx_t>  count ( 0 );

  idx_t               n = 0;


  for ( idx_t iblock = 0; iblock < blocks.size(); iblock++ )
  {
    n += blocks[iblock]->nodes.size ();
  }

  count.reserve ( n );

  for ( idx_t iblock = 0; iblock < blocks.size(); iblock++ )
  {
    NodeSet        nodes = blocks[iblock]->nodes;
    Ref<ItemMask>  mask  = blocks[iblock]->nodeMask;

    n = nodes.size ();

    for ( idx_t inode = 0; inode < n; inode++ )
    {
      const idx_t  nodeID = nodes.getNodeID ( inode );

      if ( mask->getValue( inode ) )
      {
        count[nodeID] += 1;
      }
      else
      {
        count[nodeID] += 0;
      }
    }
  }

  SparseArray<idx_t>::Iterator  it  = count.begin ();
  SparseArray<idx_t>::Iterator  end = count.end   ();

  for ( ; it != end; ++it )
  {
    if ( it->value != 1 )
    {
      String  what;

      if ( it->value == 0 )
      {
        what = String::format
          ( "node %d not owned by any mesh block" );
      }
      else
      {
        what = String::format
          ( "node %d owned by more than one mesh block" );
      }

      throw InvalidBlocksException (
        JEM_FUNC,
        String::format ( "invalid node mask: %s", what )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   checkElemMasks
//-----------------------------------------------------------------------


static void             checkElemMasks

  ( const Array
    < Ref<MeshBlock> >&   blocks )

{
  SparseArray<idx_t>  count ( 0 );

  idx_t               n = 0;


  for ( idx_t iblock = 0; iblock < blocks.size(); iblock++ )
  {
    n += blocks[iblock]->elems.size ();
  }

  count.reserve ( n );

  for ( idx_t iblock = 0; iblock < blocks.size(); iblock++ )
  {
    ElementSet     elems = blocks[iblock]->elems;
    Ref<ItemMask>  mask  = blocks[iblock]->elemMask;

    n = elems.size ();

    for ( idx_t ielem = 0; ielem < n; ielem++ )
    {
      const idx_t  elemID = elems.getElemID ( ielem );

      if ( mask->getValue( ielem ) )
      {
        count[elemID] += 1;
      }
      else
      {
        count[elemID] += 0;
      }
    }
  }

  SparseArray<idx_t>::Iterator  it  = count.begin ();
  SparseArray<idx_t>::Iterator  end = count.end   ();

  for ( ; it != end; ++it )
  {
    if ( it->value != 1 )
    {
      String  what;

      if ( it->value == 0 )
      {
        what = String::format
          ( "element %d not owned by any mesh block" );
      }
      else
      {
        what = String::format
          ( "element %d owned by more than one mesh block" );
      }

      throw InvalidBlocksException (
        JEM_FUNC,
        String::format ( "invalid element mask: %s", what )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   checkBlockBorders
//-----------------------------------------------------------------------


static void             checkBlockBorders

  ( const Array
    < Ref<MeshBlock> >&   blocks,
    const String&         kind )

{
  using jem::equal;
  using jem::io::Writer;

  const idx_t  blockCount = blocks.size ();

  for ( idx_t iblock = 0; iblock < blockCount; iblock++ )
  {
    BorderSet    recvBorders =

      getRecvBorders ( *blocks[iblock], kind );

    BorderSet    sendBorders =

      getSendBorders ( *blocks[iblock], kind );

    const idx_t  recvBorderCount = recvBorders.size ();
    const idx_t  sendBorderCount = sendBorders.size ();

    IdxVector    recvIDs;
    IdxVector    sendIDs;

    idx_t        sendSize;
    idx_t        recvSize;
    idx_t        ib, jb;


    for ( ib = 0; ib < recvBorderCount; ib++ )
    {
      idx_t  jblock = recvBorders.getBorderID ( ib );

      if ( jblock < 0 || jblock >= blockCount )
      {
        throw InvalidBlocksException (
          JEM_FUNC,
          String::format (
            "invalid %s recv border ID: %d; valid range is [0,%d)",
            kind,
            jblock,
            blockCount
          )
        );
      }

      BorderSet  sendBorders2 =

        getSendBorders ( *blocks[jblock], kind );

      jb = sendBorders2.findBorder ( iblock );

      if ( jb < 0 )
      {
        throw InvalidBlocksException (
          JEM_FUNC,
          String::format (
            "recv %s border %d of block %d has no matching "
            "send %s border of block %d",
            kind,
            ib,
            iblock,
            kind,
            jblock
          )
        );
      }

      recvSize = recvBorders .getBorderSize ( ib );
      sendSize = sendBorders2.getBorderSize ( jb );

      if ( recvSize != sendSize )
      {
        throw InvalidBlocksException (
          JEM_FUNC,
          String::format (
            "%s border size mismatch: recv border %d of "
            "block %d has size %d while send border %d of "
            "block %d has size %d",
            kind,
            ib,
            iblock,
            recvSize,
            jb,
            jblock,
            sendSize
          )
        );
      }

      recvIDs.resize ( recvSize );
      sendIDs.resize ( sendSize );

      recvBorders .getBorderItemIDs ( recvIDs, ib );
      sendBorders2.getBorderItemIDs ( sendIDs, jb );

      if ( ! equal( recvIDs, sendIDs ) )
      {
        Writer&  debug = jem::System::debug ( "jive" );

        print ( debug, "Receive ", kind, " border ", ib );
        print ( debug, " of block ", iblock,
                " is incompatible with send border ", jb );
        print ( debug, " of block ", jblock, "\n\n" );

        print ( debug, "  recv border IDs : ", recvIDs, "\n" );
        print ( debug, "  send border IDs : ", sendIDs, "\n\n" );

        throw InvalidBlocksException (
          JEM_FUNC,
          String::format (
            "%s border mismatch: %s IDs in recv border %d of "
            "block %d do not match %s IDs in send border %d of "
            "block %d",
            kind,
            kind,
            ib,
            iblock,
            kind,
            jb,
            jblock
          )
        );
      }
    }

    if ( recvBorders == sendBorders )
    {
      continue;
    }

    for ( ib = 0; ib < sendBorderCount; ib++ )
    {
      idx_t  jblock = sendBorders.getBorderID ( ib );

      if ( jblock < 0 || jblock >= blockCount )
      {
        throw InvalidBlocksException (
          JEM_FUNC,
          String::format (
            "invalid %s send border ID: %d; valid range is [0,%d)",
            &kind,
            jblock,
            blockCount
          )
        );
      }

      BorderSet  recvBorders2 =

        getRecvBorders ( *blocks[jblock], kind );

      jb = recvBorders2.findBorder ( iblock );

      if ( jb < 0 )
      {
        throw InvalidBlocksException (
          JEM_FUNC,
          String::format (
            "send %s border %d of block %d has no matching "
            "recv %s border of block %d",
            kind,
            ib,
            iblock,
            kind,
            jblock
          )
        );
      }

      sendSize = sendBorders .getBorderSize ( ib );
      recvSize = recvBorders2.getBorderSize ( jb );

      if ( recvSize != sendSize )
      {
        throw InvalidBlocksException (
          JEM_FUNC,
          String::format (
            "%s border size mismatch: send border %d of "
            "block %d has size %d while recv border %d of "
            "block %d has size %d",
            kind,
            ib,
            iblock,
            sendSize,
            jb,
            jblock,
            recvSize
          )
        );
      }

      sendIDs.resize ( sendSize );
      recvIDs.resize ( recvSize );

      sendBorders .getBorderItemIDs ( sendIDs, ib );
      recvBorders2.getBorderItemIDs ( recvIDs, jb );

      if ( ! equal( recvIDs, sendIDs ) )
      {
        Writer&  debug = jem::System::debug ( "jive" );

        print ( debug, "Send ", kind, " border ", ib );
        print ( debug, " of block ", iblock,
                " is incompatible with receive border ", jb );
        print ( debug, " of block ", jblock, "\n\n" );

        print ( debug, "  send border IDs : ", sendIDs, "\n" );
        print ( debug, "  recv border IDs : ", recvIDs, "\n\n" );

        throw InvalidBlocksException (
          JEM_FUNC,
          String::format (
            "%s border mismatch: %s IDs in send border %d of "
            "block %d do not match %s IDs in recv border %d of "
            "block %d",
            kind,
            kind,
            ib,
            iblock,
            kind,
            jb,
            jblock
          )
        );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   checkMeshBlocks
//-----------------------------------------------------------------------


void                    checkMeshBlocks

  ( const Array
    < Ref<MeshBlock> >&   blocks )

{
  checkNodeMasks    ( blocks );
  checkElemMasks    ( blocks );
  checkBlockBorders ( blocks, "node"    );
  checkBlockBorders ( blocks, "element" );
}


void                    checkMeshBlocks

  ( MeshCutter&           mcutter )

{
  using jive::util::COMPACT_STORAGE;

  const idx_t  blockCount = mcutter.blockCount ();

  Array< Ref<MeshBlock> >  blocks ( blockCount );

  for ( idx_t iblock = 0; iblock < blockCount; iblock++ )
  {
    blocks[iblock] = mcutter.makeBlock ( iblock, COMPACT_STORAGE );
  }

  checkMeshBlocks ( blocks );
}


JIVE_END_PACKAGE( fem )
