
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/util/SparseArray.h>
#include <jem/numeric/sparse/matmul.h>
#include <jem/numeric/sparse/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/numeric/sparse/SparseMatrix.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/mp/ItemMask.h>
#include <jive/fem/MeshBlock.h>
#include <jive/fem/MeshCutter.h>


JEM_DEFINE_CLASS( jive::fem::MeshCutter );


JIVE_BEGIN_PACKAGE( fem )


using jem::iarray;
using jem::newInstance;
using jem::IllegalArgumentException;
using jem::numeric::matmul;
using jem::numeric::toArray;
using jem::numeric::toMatrix;
using jive::util::StorageMode;

typedef jem::numeric::SparseStructure    MapMatrix;
typedef jem::numeric::SparseMatrix<int>  XMapMatrix;
typedef jem::util::SparseArray<int,2>    XMapBuilder;
typedef jem::util::ArrayBuffer<idx_t>    IdxBuffer;


//=======================================================================
//   class MeshCutter::Data_
//=======================================================================


class MeshCutter::Data_ : public Collectable
{
 public:

                          Data_

    ( idx_t                 blockCount,
      idx_t                 overlap,
      const ElementSet&     elems,
      const IdxVector&      elemMap );


 public:

  const idx_t             blockCount;
  const idx_t             overlap;

  String                  blockName;

  NodeSet                 nodes;
  ElementSet              elems;

  IdxVector               nodeOwners;
  IdxVector               elemOwners;

  MapMatrix               nodeBlockMap;
  MapMatrix               elemBlockMap;

  XMapMatrix              blockNodeMap;
  XMapMatrix              blockElemMap;

  IdxVector               iperm;
  IdxVector               jperm;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


MeshCutter::Data_::Data_

  ( idx_t              blockCount,
    idx_t              overlap,
    const ElementSet&  elems,
    const IdxVector&   elemMap ) :

    blockCount ( blockCount       ),
    overlap    ( overlap          ),
    nodes      ( elems.getNodes() ),
    elems      ( elems            )

{
  using jive::util::joinNames;


  const idx_t  nodeCount = nodes.size ();
  const idx_t  elemCount = elems.size ();

  String       meshName  = elems.getName ();

  idx_t        n;


  n = meshName.rfind ( '.' );

  if ( n >= 0 )
  {
    blockName = joinNames ( meshName[slice(BEGIN,n)], "block" );
  }
  else
  {
    blockName = "block";
  }

  nodeOwners.resize ( nodeCount );
  elemOwners.resize ( elemCount );

  nodeOwners = -1_idx;
  elemOwners = elemMap;

  n = jem::max ( blockCount, nodeCount );
  n = jem::max ( n,          elemCount );

  iperm.resize ( n );
  jperm.resize ( n );

  iperm = -1_idx;
  jperm = -1_idx;
}


//=======================================================================
//   class MeshCutter::Mesh_
//=======================================================================


class MeshCutter::Mesh_
{
 public:

  explicit                Mesh_

    ( const ElementSet&     elems );

  inline MapMatrix        getElemGraph () const;


 public:

  idx_t                   nodeCount;
  idx_t                   elemCount;

  IdxVector               elemOffsets;
  IdxVector               nodeIndices;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


MeshCutter::Mesh_::Mesh_ ( const ElementSet& elems )
{
  MapMatrix topo = elems.toMatrix ();

  nodeCount = topo.size(1);
  elemCount = topo.size(0);

  elemOffsets.ref ( topo.getRowOffsets()    );
  nodeIndices.ref ( topo.getColumnIndices() );
}


//-----------------------------------------------------------------------
//   getElemGraph
//-----------------------------------------------------------------------


inline MapMatrix MeshCutter::Mesh_::getElemGraph () const
{
  MapMatrix topo ( jem::shape ( elemCount, nodeCount ),
                   elemOffsets,
                   nodeIndices );

  return matmul ( topo, topo.transpose() );
}


//=======================================================================
//   class MeshCutter::Utils_
//=======================================================================


class MeshCutter::Utils_
{
 public:

  static const int        NULL_TAG;
  static const int        RECV_TAG;
  static const int        SEND_TAG;


  static void             setNodeOwners

    ( const IdxVector&      nodeOwners,
      const IdxVector&      elemOwners,
      const Mesh_&          mesh,
      idx_t                 blockCount );

  static void             getBlockElemMap

    ( XMapMatrix&           beMap,
      const IdxVector&      elemOwners,
      idx_t                 blockCount );

  static void             expandBlocks

    ( XMapMatrix&           beMap,
      const IdxVector&      elemOwners,
      const Mesh_&          mesh,
      idx_t                 overlap );

  static void             getBlockNodeMap

    ( XMapMatrix&           bnMap,
      const MapMatrix&      beMap,
      const IdxVector&      nodeOwners,
      const Mesh_&          mesh );

  static void             addConstraints

    ( XMapMatrix&           biMap,
      const String&         itemName,
      const IdxVector&      itemOwners,
      const MapMatrix&      itemCons );

  static void             sortBlockItemMap

    ( const XMapMatrix&     biMap,
      const ItemSet&        items );

  static void             getRecvBorders

    ( idx_t                 iblock,
      const XBorderSet&     borders,
      const IdxVector&      iperm,
      const IdxVector&      jperm,
      const XMapMatrix&     biMap,
      const IdxVector&      itemOwners );

  static void             getSendBorders

    ( idx_t                 iblock,
      const XBorderSet&     borders,
      const IdxVector&      iperm,
      const IdxVector&      jperm,
      const XMapMatrix&     biMap,
      const MapMatrix&      ibMap,
      bool                  overlap );

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  MeshCutter::Utils_::NULL_TAG = 0x0;
const int  MeshCutter::Utils_::RECV_TAG = 0x1;
const int  MeshCutter::Utils_::SEND_TAG = 0x2;


//-----------------------------------------------------------------------
//   setNodeOwners
//-----------------------------------------------------------------------


void MeshCutter::Utils_::setNodeOwners

  ( const IdxVector&  nodeOwners,
    const IdxVector&  elemOwners,
    const Mesh_&      mesh,
    idx_t             blockCount )

{
  idx_t  iblock;
  idx_t  ielem;
  idx_t  inode;
  idx_t  i, n;


  nodeOwners = -1_idx;

  for ( ielem = 0; ielem < mesh.elemCount; ielem++ )
  {
    iblock = elemOwners[ielem];
    n      = mesh.elemOffsets[ielem + 1];

    for ( i = mesh.elemOffsets[ielem]; i < n; i++ )
    {
      inode             = mesh.nodeIndices[i];
      nodeOwners[inode] = iblock;
    }
  }

  iblock = 0;

  for ( inode = 0; inode < mesh.nodeCount; inode++ )
  {
    if ( nodeOwners[inode] < 0 )
    {
      nodeOwners[inode] = iblock;
      iblock            = (iblock + 1) % blockCount;
    }
  }
}


//-----------------------------------------------------------------------
//   getBlockElemMap
//-----------------------------------------------------------------------


void MeshCutter::Utils_::getBlockElemMap

  ( XMapMatrix&       beMap,
    const IdxVector&  elemOwners,
    idx_t             blockCount )

{
  using jem::numeric::SparseUtils;


  const idx_t  elemCount = elemOwners.size ();

  IdxVector    beOffsets ( blockCount + 1 );
  IdxVector    beIndices ( elemCount );
  IntVector    elemTags  ( elemCount );

  idx_t        iblock;
  idx_t        ielem;


  beOffsets = 0_idx;

  for ( ielem = 0; ielem < elemCount; ielem++ )
  {
    beOffsets[elemOwners[ielem]]++;
  }

  SparseUtils::sumOffsets ( beOffsets );

  for ( ielem = 0; ielem < elemCount; ielem++ )
  {
    iblock = elemOwners[ielem];
    beIndices[beOffsets[iblock]] = ielem;
    beOffsets[iblock]++;
  }

  SparseUtils::shiftOffsets ( beOffsets );

  elemTags = NULL_TAG;

  beMap = XMapMatrix (
    MapMatrix (
      jem::shape ( blockCount, elemCount ),
      beOffsets,
      beIndices
    ),
    elemTags
  );
}


//-----------------------------------------------------------------------
//   expandBlocks
//-----------------------------------------------------------------------


void MeshCutter::Utils_::expandBlocks

  ( XMapMatrix&       beMap,
    const IdxVector&  elemOwners,
    const Mesh_&      mesh,
    idx_t             overlap )

{
  const idx_t  blockCount = beMap.size(0);

  MapMatrix    elemGraph  = mesh.getElemGraph          ();
  IdxVector    egOffsets  = elemGraph.getRowOffsets    ();
  IdxVector    egIndices  = elemGraph.getColumnIndices ();

  IdxVector    beOffsets  = beMap.getRowOffsets        ();
  IdxVector    beIndices  = beMap.getColumnIndices     ();

  IdxVector    ielems     ( mesh.elemCount );
  IdxVector    jelems     ( mesh.elemCount );

  XMapBuilder  xmap;

  idx_t        ielemCount;
  idx_t        jelemCount;
  idx_t        ilevel;
  idx_t        iblock;
  idx_t        jblock;
  idx_t        ielem;
  idx_t        jelem;
  idx_t        i, j, n;


  toArray ( xmap, beMap );

  for ( iblock = 0; iblock < blockCount; iblock++ )
  {
    i = beOffsets[iblock];
    j = beOffsets[iblock + 1];
    n = j - i;

    ielems[slice(BEGIN,n)] = beIndices[slice(i,j)];
    ielemCount             = n;

    for ( ilevel = 0; ilevel < overlap; ilevel++ )
    {
      if ( ielemCount == 0 )
      {
        break;
      }

      jelemCount = 0;

      for ( i = 0; i < ielemCount; i++ )
      {
        ielem = ielems[i];
        n     = egOffsets[ielem + 1];

        for ( j = egOffsets[ielem]; j < n; j++ )
        {
          jelem = egIndices[j];

          if ( ! xmap.find(iblock,jelem) )
          {
            jblock = elemOwners[jelem];

            xmap(iblock,jelem)   = RECV_TAG;
            xmap(jblock,jelem)   = SEND_TAG;

            jelems[jelemCount++] = jelem;
          }
        }
      }

      jem::swap ( ielems,     jelems     );
      jem::swap ( ielemCount, jelemCount );
    }
  }

  beMap = XMapMatrix ();
  beMap = toMatrix   ( xmap );
}


//-----------------------------------------------------------------------
//   getBlockNodeMap
//-----------------------------------------------------------------------


void MeshCutter::Utils_::getBlockNodeMap

  ( XMapMatrix&       bnMap,
    const MapMatrix&  beMap,
    const IdxVector&  nodeOwners,
    const Mesh_&      mesh )

{
  const idx_t  blockCount = beMap.size(0);

  IdxVector    beOffsets  = beMap.getRowOffsets    ();
  IdxVector    beIndices  = beMap.getColumnIndices ();

  XMapBuilder  xmap;

  idx_t        inode;
  idx_t        ielem;
  idx_t        iblock;
  idx_t        jblock;
  idx_t        i, j, m, n;



  xmap.reshape ( jem::shape( blockCount, mesh.nodeCount ) );
  xmap.reserve ( (idx_t) (1.25 * (double) mesh.nodeCount) );

  for ( inode = 0; inode < mesh.nodeCount; inode++ )
  {
    iblock = nodeOwners[inode];
    xmap(iblock,inode) = NULL_TAG;
  }

  for ( iblock = 0; iblock < blockCount; iblock++ )
  {
    m = beOffsets[iblock + 1];

    for ( i = beOffsets[iblock]; i < m; i++ )
    {
      ielem = beIndices[i];
      n     = mesh.elemOffsets[ielem + 1];

      for ( j = mesh.elemOffsets[ielem]; j < n; j++ )
      {
        inode  = mesh.nodeIndices[j];
        jblock = nodeOwners[inode];

        if ( jblock != iblock )
        {
          xmap(iblock,inode) = RECV_TAG;
          xmap(jblock,inode) = SEND_TAG;
        }
      }
    }
  }

  bnMap = toMatrix ( xmap );
}


//-----------------------------------------------------------------------
//   addConstraints
//-----------------------------------------------------------------------


void MeshCutter::Utils_::addConstraints

  ( XMapMatrix&       biMap,
    const String&     itemName,
    const IdxVector&  itemOwners,
    const MapMatrix&  itemCons )

{
  const int    MAX_ITER   = 20;

  const idx_t  blockCount = biMap   .size(0);
  const idx_t  conCount   = itemCons.size(0);

  IdxVector    conOffsets = itemCons.getRowOffsets    ();
  IdxVector    conIndices = itemCons.getColumnIndices ();

  BoolVector   bmask      ( blockCount );
  IdxVector    jblocks    ( blockCount );

  MapMatrix    ibMap;
  IdxVector    ibOffsets;
  IdxVector    ibIndices;

  XMapBuilder  xmap;

  idx_t        iblock;
  idx_t        jblock;
  idx_t        iitem;
  int          iiter;
  idx_t        icon;
  idx_t        i, j, k, m, n;


  toArray ( xmap, biMap );

  bmask = false;
  iiter = 0;

  while ( true )
  {
    ibMap = biMap.getStructure().transpose ();

    ibOffsets.ref ( ibMap.getRowOffsets    () );
    ibIndices.ref ( ibMap.getColumnIndices () );

    for ( icon = 0; icon < conCount; icon++ )
    {
      k = 0;
      m = conOffsets[icon + 1];

      for ( i = conOffsets[icon]; i < m; i++ )
      {
        iitem = conIndices[i];
        n     = ibOffsets[iitem + 1];

        for ( j = ibOffsets[iitem]; j < n; j++ )
        {
          jblock = ibIndices[j];

          if ( ! bmask[jblock] )
          {
            jblocks[k++]  = jblock;
            bmask[jblock] = true;
          }
        }
      }

      for ( i = conOffsets[icon]; i < m; i++ )
      {
        iitem  = conIndices[i];
        iblock = itemOwners[iitem];

        for ( j = 0; j < k; j++ )
        {
          jblock = jblocks[j];

          if ( jblock != iblock )
          {
            xmap(jblock,iitem) = RECV_TAG;
            xmap(iblock,iitem) = SEND_TAG;
          }
        }
      }

      for ( j = 0; j < k; j++ )
      {
        bmask[jblocks[j]] = false;
      }
    }

    ibOffsets.ref ( IdxVector() );
    ibIndices.ref ( IdxVector() );

    m     = biMap.nonZeroCount ();

    biMap = XMapMatrix ();
    biMap = toMatrix   ( xmap );

    n     = biMap.nonZeroCount ();

    if ( m == n )
    {
      break;
    }

    iiter++;

    if ( iiter > MAX_ITER )
    {
      throw Exception (
        String::format (
          "unable to resolve %s contstraints within %d iterations",
          itemName,
          iiter
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   sortBlockItemMap
//-----------------------------------------------------------------------


void MeshCutter::Utils_::sortBlockItemMap

  ( const XMapMatrix&  biMap,
    const ItemSet&     items )

{
  using jem::sort;

  const idx_t  blockCount = biMap.size(0);
  const idx_t  itemCount  = biMap.size(1);

  IdxVector    offsets    = biMap.getRowOffsets    ();
  IdxVector    indices    = biMap.getColumnIndices ();
  IntVector    values     = biMap.getValues        ();

  IdxVector    itemIDs    ( itemCount );
  IntVector    ibuffer    ( itemCount );
  IdxVector    iitems     ( itemCount );
  IntVector    ivals;

  idx_t        iblock;
  idx_t        i, j;


  iitems = iarray ( itemCount );

  items.getItemMap()->getItemIDs ( itemIDs, iitems );

  for ( iblock = 0; iblock < blockCount; iblock++ )
  {
    i = offsets[iblock];
    j = offsets[iblock + 1];

    iitems.ref ( indices[slice(i,j)] );
    ivals .ref ( values [slice(i,j)] );

    ibuffer[iitems] = ivals;

    sort ( iitems,  itemIDs );

    ivals = ibuffer[iitems];
  }
}


//-----------------------------------------------------------------------
//   getRecvBorders
//-----------------------------------------------------------------------


void MeshCutter::Utils_::getRecvBorders

  ( idx_t              iblock,
    const XBorderSet&  borders,
    const IdxVector&   iperm,
    const IdxVector&   jperm,
    const XMapMatrix&  biMap,
    const IdxVector&   itemOwners )

{
  using jem::sort;
  using jem::numeric::SparseUtils;

  IdxVector    iitems     = biMap.getColumnIndices ( iblock );
  IntVector    itemTags   = biMap.getValues        ( iblock );

  const idx_t  iitemCount = iitems.size ();

  IdxBuffer    borderOffsets;
  IdxBuffer    borderIDs;
  IdxVector    borderItems;
  IdxVector    borderPerm;

  idx_t        borderCount;
  idx_t        iborder;
  idx_t        jblock;
  idx_t        iitem;
  idx_t        ib;
  idx_t        i, j, k;


  borderOffsets.reserve ( 64 );
  borderIDs    .reserve ( 64 );

  borderCount = 0;

  for ( i = 0; i < iitemCount; i++ )
  {
    iitem        = iitems[i];
    iperm[iitem] = i;

    if ( itemTags[i] != RECV_TAG )
    {
      continue;
    }

    jblock = itemOwners[iitem];

    if ( jblock == iblock )
    {
      throw jem::Error ( JEM_FUNC, "invalid RECV tag" );
    }

    iborder = jperm[jblock];

    if ( iborder < 0 )
    {
      iborder       = borderCount++;
      jperm[jblock] = iborder;

      borderIDs    .pushBack ( jblock );
      borderOffsets.pushBack ( 0 );
    }

    borderOffsets[iborder]++;
  }

  borderOffsets.pushBack  ( 0 );

  SparseUtils::sumOffsets ( borderOffsets.toArray() );

  borderItems.resize ( borderOffsets[borderCount] );

  for ( i = 0; i < iitemCount; i++ )
  {
    iitem = iitems[i];

    if ( itemTags[i] != RECV_TAG )
    {
      continue;
    }

    jblock  = itemOwners[iitem];
    iborder = jperm[jblock];
    borderItems[borderOffsets[iborder]] = iitem;

    borderOffsets[iborder]++;
  }

  SparseUtils::shiftOffsets ( borderOffsets.toArray() );

  borderPerm.resize  ( borderCount );
  borders   .reserve ( borderCount );

  borderPerm = iarray ( borderCount );

  sort ( borderPerm, borderIDs.toArray() );

  for ( ib = 0; ib < borderCount; ib++ )
  {
    iborder = borderPerm[ib];
    i       = borderOffsets[iborder];
    j       = borderOffsets[iborder + 1];
    jblock  = borderIDs[iborder];

    sort    ( borderItems[slice(i,j)] );

    for ( k = i; k < j; k++ )
    {
      borderItems[k] = iperm[borderItems[k]];
    }

    borders.addBorder ( jblock, borderItems[slice(i,j)] );

    jperm[jblock] = -1;
  }

  iperm[iitems] = -1;

  borders.trimToSize ();
}


//-----------------------------------------------------------------------
//   getSendBorders
//-----------------------------------------------------------------------


void MeshCutter::Utils_::getSendBorders

  ( idx_t              iblock,
    const XBorderSet&  borders,
    const IdxVector&   iperm,
    const IdxVector&   jperm,
    const XMapMatrix&  biMap,
    const MapMatrix&   ibMap,
    bool               overlap )

{
  using jem::sort;
  using jem::numeric::SparseUtils;

  IdxVector    iitems     = biMap.getColumnIndices ( iblock );
  IntVector    itemTags   = biMap.getValues        ( iblock );

  const idx_t  iitemCount = iitems.size ();

  IdxVector    ibOffsets  = ibMap.getRowOffsets    ();
  IdxVector    ibIndices  = ibMap.getColumnIndices ();

  IdxBuffer    borderOffsets;
  IdxBuffer    borderIDs;
  IdxVector    borderItems;
  IdxVector    borderPerm;

  idx_t        borderCount;
  int          tagMask;
  idx_t        iborder;
  idx_t        jblock;
  idx_t        iitem;
  idx_t        ib;
  idx_t        i, j, k, n;


  borderOffsets.reserve ( 64 );
  borderIDs    .reserve ( 64 );

  borderCount = 0;

  if ( overlap )
  {
    tagMask = SEND_TAG;
  }
  else
  {
    tagMask = (SEND_TAG | RECV_TAG);
  }

  for ( i = 0; i < iitemCount; i++ )
  {
    iitem        = iitems[i];
    iperm[iitem] = i;

    if ( ! (tagMask & itemTags[i]) )
    {
      continue;
    }

    n = ibOffsets[iitem + 1];

    for ( j = ibOffsets[iitem]; j < n; j++ )
    {
      jblock = ibIndices[j];

      if ( jblock == iblock )
      {
        continue;
      }

      iborder = jperm[jblock];

      if ( iborder < 0 )
      {
        iborder       = borderCount++;
        jperm[jblock] = iborder;

        borderIDs    .pushBack ( jblock );
        borderOffsets.pushBack ( 0 );
      }

      borderOffsets[iborder]++;
    }
  }

  borderOffsets.pushBack  ( 0 );

  SparseUtils::sumOffsets ( borderOffsets.toArray() );

  borderItems.resize ( borderOffsets[borderCount] );

  for ( i = 0; i < iitemCount; i++ )
  {
    iitem = iitems[i];

    if ( ! (tagMask & itemTags[i]) )
    {
      continue;
    }

    n = ibOffsets[iitem + 1];

    for ( j = ibOffsets[iitem]; j < n; j++ )
    {
      jblock = ibIndices[j];

      if ( jblock == iblock )
      {
        continue;
      }

      iborder = jperm[jblock];
      borderItems[borderOffsets[iborder]] = iitem;

      borderOffsets[iborder]++;
    }
  }

  SparseUtils::shiftOffsets ( borderOffsets.toArray() );

  borders   .reserve  ( borderCount );
  borderPerm.resize   ( borderCount );

  borderPerm = iarray ( borderCount );

  for ( ib = 0; ib < borderCount; ib++ )
  {
    iborder = borderPerm[ib];
    i       = borderOffsets[iborder];
    j       = borderOffsets[iborder + 1];
    jblock  = borderIDs[iborder];

    sort ( borderItems[slice(i,j)] );

    for ( k = i; k < j; k++ )
    {
      borderItems[k] = iperm[borderItems[k]];
    }

    borders.addBorder ( jblock, borderItems[slice(i,j)] );

    jperm[jblock] = -1;
  }

  iperm[iitems] = -1;

  borders.trimToSize ();
}


//=======================================================================
//   class MeshCutter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MeshCutter::MeshCutter

  ( const Params&  p )

{
  init_ ( p );
}


MeshCutter::MeshCutter

  ( const ElementSet&  elems,
    const IdxVector&   elemMap,
    idx_t              overlap )

{
  init_ ( Params( elems, elemMap, overlap ) );
}



MeshCutter::~MeshCutter ()
{}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void MeshCutter::getInfo ( Info& info ) const
{
  using jem::min;
  using jem::max;


  const Data_&         d = *data_;

  const idx_t  nodeCount = d.nodes.size ();
  const idx_t  elemCount = d.elems.size ();

  IdxVector    bnOffsets = d.blockNodeMap.getRowOffsets    ();
  IdxVector    bnIndices = d.blockNodeMap.getColumnIndices ();
  IntVector    nodeTags  = d.blockNodeMap.getValues        ();

  IdxVector    beOffsets = d.blockElemMap.getRowOffsets    ();
  IdxVector    beIndices = d.blockElemMap.getColumnIndices ();
  IntVector    elemTags  = d.blockElemMap.getValues        ();

  IdxVector    mask;

  idx_t        inodeCount;
  idx_t        ielemCount;
  idx_t        iblock;
  idx_t        i, n;


  info.blockCount   = d.blockCount;
  info.overlap      = d.overlap;

  info.minNodeCount = 0;
  info.maxNodeCount = 0;
  info.minElemCount = 0;
  info.maxElemCount = 0;

  for ( iblock = 0; iblock < d.blockCount; iblock++ )
  {
    inodeCount = bnOffsets[iblock + 1] - bnOffsets[iblock];
    ielemCount = beOffsets[iblock + 1] - beOffsets[iblock];

    if ( iblock == 0 )
    {
      info.minNodeCount = info.maxNodeCount = inodeCount;
      info.minElemCount = info.maxElemCount = ielemCount;
    }
    else
    {
      info.minNodeCount = min ( info.minNodeCount, inodeCount );
      info.maxNodeCount = max ( info.maxNodeCount, inodeCount );
      info.minElemCount = min ( info.minElemCount, ielemCount );
      info.maxElemCount = max ( info.maxElemCount, ielemCount );
    }
  }

  mask.resize ( nodeCount );

  mask = 0;
  n    = nodeTags.size ();

  for ( i = 0; i < n; i++ )
  {
    if ( nodeTags[i] != Utils_::NULL_TAG )
    {
      mask[bnIndices[i]] = 1;
    }
  }

  if ( nodeCount == 0 )
  {
    info.borderNodeCount = 0;
  }
  else
  {
    info.borderNodeCount = sum ( mask );
  }

  mask.resize ( elemCount );

  mask = 0;
  n    = elemTags.size ();

  for ( i = 0; i < n; i++ )
  {
    if ( elemTags[i] != Utils_::NULL_TAG )
    {
      mask[beIndices[i]] = 1;
    }
  }

  if ( elemCount == 0 )
  {
    info.borderElemCount = 0;
  }
  else
  {
    info.borderElemCount = sum ( mask );
  }
}


//-----------------------------------------------------------------------
//   blockCount
//-----------------------------------------------------------------------


idx_t MeshCutter::blockCount () const
{
  return data_->blockCount;
}


//-----------------------------------------------------------------------
//   makeBlock
//-----------------------------------------------------------------------


Ref<MeshBlock> MeshCutter::makeBlock

  ( idx_t        iblock,
    StorageMode  mode )

{
  const Data_&    d    = *data_;
  const String    name = d.blockName + String( iblock );

  Ref<MeshBlock>  b;


  if ( d.overlap == 0 )
  {
    b = newInstance<MeshBlock> ( name,
                                 MeshBlock::NO_OVERLAP, mode );

    getBlockElems   ( iblock,
                      b->nodes,
                      b->elems,
                      b->nodeMask,
                      b->elemMask );

    getBlockBorders ( iblock,
                      b->sendNodeBorders );
  }
  else
  {
    b = newInstance<MeshBlock> ( name, mode );

    getBlockElems   ( iblock,
                      b->nodes,
                      b->elems,
                      b->nodeMask,
                      b->elemMask );

    getBlockBorders ( iblock,
                      b->recvNodeBorders,
                      b->sendNodeBorders,
                      b->recvElemBorders,
                      b->sendElemBorders );
  }

  return b;
}


//-----------------------------------------------------------------------
//   getBlockElems
//-----------------------------------------------------------------------


void MeshCutter::getBlockElems

  ( idx_t                 iblock,
    const XNodeSet&       nodes,
    const XElementSet&    elems,
    const Ref<ItemMask>&  nodeMask,
    const Ref<ItemMask>&  elemMask )

{
  JEM_PRECHECK ( iblock >= 0 && nodes && elems );

  nodes.clear ();
  elems.clear ();

  if ( nodeMask )
  {
    nodeMask->clear ();
  }

  if ( elemMask )
  {
    elemMask->clear ();
  }

  if ( iblock < data_->blockCount )
  {
    getBlockElems_ ( iblock,
                     nodes,
                     elems,
                     nodeMask.get (),
                     elemMask.get () );
  }
}


//-----------------------------------------------------------------------
//   getBlockBorders
//-----------------------------------------------------------------------


void MeshCutter::getBlockBorders

  ( idx_t              iblock,
    const XBorderSet&  borders )

{
  JEM_PRECHECK ( iblock >= 0 && borders );

  const Data_& d = *data_;


  borders.clear ();

  if ( iblock < d.blockCount )
  {
    Utils_::getSendBorders ( iblock,
                             borders,
                             d.iperm,
                             d.jperm,
                             d.blockNodeMap,
                             d.nodeBlockMap,
                             false );
  }
}


void MeshCutter::getBlockBorders

  ( idx_t              iblock,
    const XBorderSet&  recvNodeBorders,
    const XBorderSet&  sendNodeBorders,
    const XBorderSet&  recvElemBorders,
    const XBorderSet&  sendElemBorders )

{
  JEM_PRECHECK ( iblock >= 0     &&
                 recvNodeBorders &&
                 sendNodeBorders &&
                 recvElemBorders &&
                 sendElemBorders );

  const Data_& d = *data_;


  recvNodeBorders.clear ();
  sendNodeBorders.clear ();
  recvElemBorders.clear ();
  sendElemBorders.clear ();

  if ( iblock < d.blockCount )
  {
    Utils_::getSendBorders ( iblock,
                             sendNodeBorders,
                             d.iperm,
                             d.jperm,
                             d.blockNodeMap,
                             d.nodeBlockMap,
                             true );

    Utils_::getSendBorders ( iblock,
                             sendElemBorders,
                             d.iperm,
                             d.jperm,
                             d.blockElemMap,
                             d.elemBlockMap,
                             true );

    Utils_::getRecvBorders ( iblock,
                             recvNodeBorders,
                             d.iperm,
                             d.jperm,
                             d.blockNodeMap,
                             d.nodeOwners );

    Utils_::getRecvBorders ( iblock,
                             recvElemBorders,
                             d.iperm,
                             d.jperm,
                             d.blockElemMap,
                             d.elemOwners );
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void MeshCutter::init_ ( const Params& p )
{
  checkParams_ ( p );

  Mesh_  mesh  ( p.elems );

  idx_t  blockCount;


  if ( mesh.elemCount > 0 )
  {
    blockCount = 1 + max( p.elemMap );
  }
  else
  {
    blockCount = 1;
  }

  data_ = newInstance<Data_> ( blockCount, p.overlap,
                               p.elems,    p.elemMap );

  Data_& d = * data_;


  Utils_::setNodeOwners      (  d.nodeOwners,
                                d.elemOwners,
                                mesh,
                                blockCount );

  Utils_::getBlockElemMap    (  d.blockElemMap,
                                d.elemOwners,
                                blockCount );

  if ( p.options & EXPAND_FIRST )
  {
    if ( p.overlap > 0 )
    {
      Utils_::expandBlocks   (  d.blockElemMap,
                                d.elemOwners,
                                mesh,
                                p.overlap );
    }
  }
  else
  {
    if ( p.elemCons.size(0) > 0 )
    {
      Utils_::addConstraints (  d.blockElemMap,
                                "element",
                                d.elemOwners,
                                p.elemCons );
    }

    if ( p.overlap > 0 )
    {
      Utils_::expandBlocks   (  d.blockElemMap,
                                d.elemOwners,
                                mesh,
                                p.overlap );
    }
  }

  if ( p.elemCons.size(0) > 0 )
  {
    Utils_::addConstraints   (  d.blockElemMap,
                                "element",
                                d.elemOwners,
                                p.elemCons );
  }

  Utils_::getBlockNodeMap    (  d.blockNodeMap,
                                d.blockElemMap.getStructure(),
                                d.nodeOwners,
                                mesh );

  if ( p.nodeCons.size(0) > 0 )
  {
    Utils_::addConstraints   (  d.blockNodeMap,
                                "node",
                                d.nodeOwners,
                                p.nodeCons );
  }

  Utils_::sortBlockItemMap   (  d.blockNodeMap,
                               *d.nodes.getData() );

  Utils_::sortBlockItemMap   (  d.blockElemMap,
                               *d.elems.getData() );

  d.nodeBlockMap = d.blockNodeMap.getStructure().transpose ();
  d.elemBlockMap = d.blockElemMap.getStructure().transpose ();
}


//-----------------------------------------------------------------------
//   getBlockElems_
//-----------------------------------------------------------------------


void MeshCutter::getBlockElems_

  ( idx_t               iblock,
    const XNodeSet&     nodes,
    const XElementSet&  elems,
    ItemMask*           nodeMask,
    ItemMask*           elemMask )

{
  Data_&       d      = *data_;

  IdxVector    inodes = d.blockNodeMap.getColumnIndices ( iblock );
  IdxVector    ielems = d.blockElemMap.getColumnIndices ( iblock );

  const idx_t  inodeCount = inodes.size ();
  const idx_t  ielemCount = ielems.size ();

  IdxVector    jnodes  ( d.elems.maxElemNodeCount() );
  Vector       coords  ( d.nodes.rank()             );

  IdxVector    nodeIDs;
  IdxVector    elemIDs;

  idx_t        inode;
  idx_t        ielem;
  idx_t        i, j, n;


  nodes  .reserve ( inodeCount );
  nodeIDs.resize  ( inodeCount );

  d.nodes.getData()->getItemMap()->getItemIDs ( nodeIDs, inodes );

  for ( i = 0; i < inodeCount; i++ )
  {
    inode = inodes[i];
    d.iperm[inode] = i;

    d.nodes.getNodeCoords ( coords,     inode  );
    nodes  .addNode       ( nodeIDs[i], coords );
  }

  // This is to make sure that the nodes in the block
  // have the same rank as the nodes in the original mesh.

  if ( inodeCount == 0 && d.nodes.size() > 0 )
  {
    d.nodes.getNodeCoords ( coords, 0 );
    nodes  .addNode       ( d.nodes.getNodeID(0), coords );
  }

  nodes.trimToSize ();

  if ( nodeMask )
  {
    for ( i = 0; i < inodeCount; i++ )
    {
      if ( d.nodeOwners[inodes[i]] != iblock )
      {
        nodeMask->setValue ( i, false );
      }
    }
  }

  // Node IDs no longer needed; transfer memory to the
  // elemIDs array.

  nodeIDs.swap    ( elemIDs    );

  elems  .reserve ( ielemCount );
  elemIDs.resize  ( ielemCount );

  d.elems.getData()->getItemMap()->getItemIDs ( elemIDs, ielems );

  for ( i = 0; i < ielemCount; i++ )
  {
    ielem = ielems[i];
    n     = d.elems.getElemNodes ( jnodes, ielem );

    for ( j = 0; j < n; j++ )
    {
      jnodes[j] = d.iperm[jnodes[j]];
    }

    elems.addElement ( elemIDs[i], jnodes[slice(BEGIN,n)] );
  }

  elems.trimToSize ();

  if ( elemMask )
  {
    for ( i = 0; i < ielemCount; i++ )
    {
      if ( d.elemOwners[ielems[i]] != iblock )
      {
        elemMask->setValue ( i, false );
      }
    }
  }

  d.iperm[inodes] = -1;
}


//-----------------------------------------------------------------------
//   checkParams_
//-----------------------------------------------------------------------


void MeshCutter::checkParams_ ( const Params& p )
{
  const idx_t  elemCount = p.elems.size ();

  idx_t        ielem;
  idx_t        iblock;


  if ( p.elemMap.size() != elemCount )
  {
    throw IllegalArgumentException (
      CLASS_NAME,
      "invalid element map (wrong size)"
    );
  }

  for ( ielem = 0; ielem < elemCount; ielem++ )
  {
    iblock = p.elemMap[ielem];

    if ( iblock < 0 )
    {
      throw IllegalArgumentException (
        CLASS_NAME,
        "invalid element map (negative block index)"
      );
    }
  }

  if ( p.overlap < 0 )
  {
    throw IllegalArgumentException (
      CLASS_NAME,
      String::format ( "invalid overlap: %d", p.overlap )
    );
  }

  if ( p.nodeCons.size(0) > 0 )
  {
    const idx_t  nodeCount = p.elems.getNodes().size();

    if ( p.nodeCons.size(1) != nodeCount || ! p.nodeCons.isValid() )
    {
      throw IllegalArgumentException (
        CLASS_NAME,
        "invalid node constraints"
      );
    }
  }

  if ( p.elemCons.size(0) > 0 )
  {
    if ( p.elemCons.size(1) != elemCount || ! p.elemCons.isValid() )
    {
      throw IllegalArgumentException (
        CLASS_NAME,
        "invalid element constraints"
      );
    }

    if ( p.overlap == 0 )
    {
      throw IllegalArgumentException (
        CLASS_NAME,
        "element constraints require non-zero overlap"
      );
    }
  }
}


//=======================================================================
//   class MeshCutter::Info
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


MeshCutter::Info::Info ()
{
  blockCount      = 0;
  overlap         = 0;
  minNodeCount    = 0;
  maxNodeCount    = 0;
  borderNodeCount = 0;

  minElemCount    = 0;
  maxElemCount    = 0;
  borderElemCount = 0;
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


#undef  PRINT_FIELD
#define PRINT_FIELD( name, value )              \
                                                \
  print ( out, endl, name, dots, value )


void MeshCutter::Info::printTo ( PrintWriter& out ) const
{
  using jem::io::endl;
  using jem::io::dots;

  PRINT_FIELD ( "blocks",          blockCount      );
  PRINT_FIELD ( "overlap",         overlap         );
  PRINT_FIELD ( "border nodes",    borderNodeCount );
  PRINT_FIELD ( "min nodes/block", minNodeCount    );
  PRINT_FIELD ( "max nodes/block", maxNodeCount    );

  if ( overlap > 0 )
  {
    PRINT_FIELD ( "border elements", borderElemCount );
  }

  PRINT_FIELD ( "min elements/block", minElemCount );
  PRINT_FIELD ( "max elements/block", maxElemCount );
}


#undef PRINT_FIELD


//=======================================================================
//   class MeshCutter::Params
//=======================================================================


MeshCutter::Params::Params

  ( const ElementSet&  e ) :

    elems ( e )

{
  JEM_PRECHECK ( elems );

  elemMap.resize ( elems.size() );

  elemMap = 0;
  overlap = 0;
  options = 0;
}


MeshCutter::Params::Params

  ( const ElementSet&  e,
    const IdxVector&   map,
    idx_t              ovp,
    Options            opt ) :

    elems   (   e ),
    elemMap ( map ),
    overlap ( ovp ),
    options ( opt )

{
  JEM_PRECHECK ( elems );
}


//=======================================================================
//   class MeshCutter::Exception
//=======================================================================


MeshCutter::Exception::Exception ( const String& what ) :

  Super ( MeshCutter::CLASS_NAME, what )

{}


String MeshCutter::Exception::name () const
{
  return "jive::fem::MeshCutter::Exception";
}


JIVE_END_PACKAGE( fem )
