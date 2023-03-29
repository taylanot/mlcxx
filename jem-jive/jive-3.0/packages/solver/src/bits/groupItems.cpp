
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


#include <jem/base/limits.h>
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/numeric/sparse/matmul.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/numeric/sparse/SparseFilters.h>
#include <jive/util/utilities.h>
#include <jive/util/DofSpace.h>
#include <jive/util/GroupSet.h>
#include <jive/graph/WGraph.h>
#include <jive/graph/Partitioner.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/SparseMatrixExtension.h>
#include <jive/solver/utilities.h>


JIVE_BEGIN_PACKAGE( solver )


using jive::util::GroupSet;
using jive::graph::WGraph;
using jive::algebra::SparseMatrixExt;


//=======================================================================
//   private helper functions
//=======================================================================


static IdxVector          getDofItemMap

  ( const DofSpace&         dofs );

static void               makeSymmetric

  ( IdxVector&              xadj,
    IdxVector&              adjncy,
    IdxVector&              edgew,
    const IdxVector&        mask );

static void               groupItems

  ( const IdxVector&        itemMap,
    idx_t                   partCount );

static void               groupItems

  ( const String&           name,
    const IdxVector&        itemMap,
    idx_t                   partCount,
    const GroupSet&         items );

static void               groupItems

  ( const String&           name,
    const IdxVector&        itemMap,
    idx_t                   partCount,
    const SparseMatrixExt&  matrix,
    const DofSpace&         dofs );

static void               groupItems

  ( const String&           name,
    const IdxVector&        itemMap,
    idx_t                   partCount,
    const WGraph&           itemGraph );


//-----------------------------------------------------------------------
//   getDofItemMap
//-----------------------------------------------------------------------


static IdxVector          getDofItemMap

  ( const DofSpace&         dofs )

{
  using jem::iarray;

  const idx_t  dofCount = dofs.dofCount ();

  IdxVector    idofs    ( dofCount );
  IdxVector    iitems   ( dofCount );
  IdxVector    jtypes   ( dofCount );


  idofs = iarray ( dofCount );

  dofs.decodeDofIndices ( iitems, jtypes, idofs );

  return iitems;
}


//-----------------------------------------------------------------------
//   makeSymmetric
//-----------------------------------------------------------------------


static void               makeSymmetric

  ( IdxVector&              xadj,
    IdxVector&              adjncy,
    IdxVector&              edgew,
    const IdxVector&        mask )

{
  using jem::numeric::SparseUtils;

  const idx_t  nodeCount = xadj.size() - 1;

  IdxVector    accu      ( nodeCount );
  IdxVector    xadj2     ( nodeCount + 1 );
  IdxVector    adjncy2;
  IdxVector    edgew2;

  idx_t        nnz;


  // Count the maximum number of entries in each row.

  xadj2 = 0_idx;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    idx_t  begin = xadj[inode];
    idx_t  end   = xadj[inode + 1];

    xadj2[inode] += (end - begin);

    for ( idx_t j = begin; j < end; j++ )
    {
      idx_t  jnode = adjncy[j];

      if ( jnode != inode )
      {
        xadj2[jnode]++;
      }
    }
  }

  SparseUtils::sumOffsets ( xadj2 );

  nnz = xadj2[nodeCount];

  adjncy2.resize ( nnz );
  edgew2 .resize ( nnz );

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    idx_t  begin = xadj [inode];
    idx_t  end   = xadj [inode + 1];
    idx_t  ipos  = xadj2[inode];

    for ( idx_t j = begin; j < end; j++, ipos++ )
    {
      idx_t  jnode = adjncy[j];

      adjncy2[ipos] = jnode;
      edgew2 [ipos] = edgew[j];

      if ( jnode != inode )
      {
        idx_t  jpos = xadj2[jnode];

        adjncy2[jpos] = inode;
        edgew2 [jpos] = edgew[j];
        xadj2 [jnode] = jpos + 1;
      }
    }

    xadj2[inode] = ipos;
  }

  SparseUtils::shiftOffsets ( xadj2 );

  // Merge duplicate entries per row.

  nnz  = 0_idx;
  mask = -1_idx;
  accu = 0_idx;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    idx_t  begin = xadj2[inode];
    idx_t  end   = xadj2[inode + 1];

    xadj2[inode] = nnz;
    mask [inode] = inode;

    for ( idx_t j = begin; j < end; j++ )
    {
      idx_t  jnode = adjncy2[j];

      if ( mask[jnode] != inode )
      {
        mask[jnode]    = inode;
        adjncy2[nnz++] = jnode;
      }

      accu[jnode] += edgew2[j];
    }

    begin = xadj2[inode];
    end   = nnz;

    for ( idx_t j = begin; j < end; j++ )
    {
      idx_t  jnode = adjncy2[j];

      edgew2[j]   = 1 + accu[jnode];
      accu[jnode] = 0;
    }

    accu[inode] = 0;
  }

  xadj2[nodeCount] = nnz;

  if ( nnz < adjncy2.size() )
  {
    adjncy2.reshape ( nnz );
    edgew2 .reshape ( nnz );
  }

  xadj  .swap ( xadj2 );
  adjncy.swap ( adjncy2 );
  edgew .swap ( edgew2 );
}


//-----------------------------------------------------------------------
//   groupItems (trivial case)
//-----------------------------------------------------------------------


static void               groupItems

  ( const IdxVector&        itemMap,
    idx_t                   partCount )

{
  const idx_t  itemCount = itemMap.size ();

  idx_t        partSize  = itemCount / partCount;
  idx_t        ipart     = 0;


  for ( idx_t i = 0; i < itemCount; i++ )
  {
    if ( partSize <= 0 )
    {
      ipart++;

      if ( ipart < partCount )
      {
        partSize = (itemCount - i) / (partCount - ipart);
      }
    }
    else
    {
      partSize--;
    }

    itemMap[i] = ipart;
  }
}


//-----------------------------------------------------------------------
//   groupItems (with a GroupSet)
//-----------------------------------------------------------------------


static void               groupItems

  ( const String&           name,
    const IdxVector&        itemMap,
    idx_t                   partCount,
    const GroupSet&         items )

{
  using jem::numeric::matmul;
  using jem::numeric::SparseFilters;

  SparseIdxMatrix  itemTopo ( items.toMatrix() );

  itemTopo.getValues() = 1;

  itemTopo = matmul ( itemTopo, itemTopo.transpose(),
                      ! SparseFilters<idx_t>::diagonal() );

  WGraph  itemGraph ( itemTopo.getRowOffsets(),
                      itemTopo.getColumnIndices() );

  groupItems ( name, itemMap, partCount, itemGraph );
}


//-----------------------------------------------------------------------
//   groupItems (with a sparse matrix)
//-----------------------------------------------------------------------


static void               groupItems

  ( const String&           name,
    const IdxVector&        itemMap,
    idx_t                   partCount,
    const SparseMatrixExt&  matrix,
    const DofSpace&         dofs )

{
  using jem::isTiny;
  using jem::numeric::SparseUtils;

  const idx_t   dofCount    = dofs.dofCount ();
  const idx_t   itemCount   = itemMap.size  ();

  SparseMatrix  spMatrix    = matrix  .toSparseMatrix   ();
  IdxVector     rowOffsets  = spMatrix.getRowOffsets    ();
  IdxVector     colIndices  = spMatrix.getColumnIndices ();
  Vector        matValues   = spMatrix.getValues        ();

  IdxVector     dofItemMap  = getDofItemMap ( dofs );

  Vector        accu        ( itemCount );
  IdxVector     mask        ( itemCount );
  IdxVector     xadj        ( itemCount + 1 );
  IdxVector     adjncy      ( colIndices.size() );
  Vector        edgev       ( colIndices.size() );
  IdxVector     edgew       ( colIndices.size() );

  idx_t         nnz;


  // Count the maximum number of entries per row in the item
  // connectivity graph.

  xadj = 0_idx;

  for ( idx_t idof = 0; idof < dofCount; idof++ )
  {
    idx_t  iitem = dofItemMap[idof];

    xadj[iitem] += (rowOffsets[idof + 1] -
                    rowOffsets[idof]);
  }

  SparseUtils::sumOffsets ( xadj );

  // Construct the initial connectivity graph that may contain
  // duplicate entries per row.

  for ( idx_t idof = 0; idof < dofCount; idof++ )
  {
    idx_t  iitem = dofItemMap[idof];
    idx_t  begin = rowOffsets[idof];
    idx_t  end   = rowOffsets[idof + 1];
    idx_t  ipos  = xadj[iitem];

    for ( idx_t j = begin; j < end; j++ )
    {
      adjncy[ipos] = dofItemMap[colIndices[j]];
      edgev [ipos] = matValues [j];
      ipos++;
    }

    xadj[iitem] = ipos;
  }

  SparseUtils::shiftOffsets ( xadj );

  // Construct the non-symmetric connectivity graph by merging
  // duplicate entries per row.

  accu = 0.0;
  mask = -1_idx;
  nnz  = 0_idx;

  for ( idx_t iitem = 0; iitem < itemCount; iitem++ )
  {
    idx_t  begin = xadj[iitem];
    idx_t  end   = xadj[iitem + 1];

    xadj[iitem] = nnz;
    mask[iitem] = iitem;

    for ( idx_t j = begin; j < end; j++ )
    {
      idx_t  jitem = adjncy[j];

      if ( mask[jitem] != iitem )
      {
        mask[jitem] = iitem;
        adjncy[nnz] = jitem;
        nnz++;
      }

      accu[jitem] += std::fabs ( edgev[j] );
    }

    // Determine the maximum row value and reset the accumulator.

    begin = xadj[iitem];
    end   = nnz;

    double  s = accu[iitem];

    accu[iitem] = 0.0;

    for ( idx_t j = begin; j < end; j++ )
    {
      idx_t     jitem = adjncy[j];
      double  value = accu[jitem];

      edgev[j]    = value;
      accu[jitem] = 0.0;

      if ( value > s )
      {
        s = value;
      }
    }

    // Compute the edge weights. Zero weights will be set to 1 later
    // when the connectivity graph is made symmetric.

    if ( ! isTiny( s ) )
    {
      s = 9.0 / s;
    }

    for ( idx_t j = begin; j < end; j++ )
    {
      edgew[j] = (idx_t) (0.5 + s * edgev[j]);
    }
  }

  xadj[itemCount] = nnz;

  if ( nnz < adjncy.size() )
  {
    adjncy.reshape ( nnz );
    edgew .reshape ( nnz );
  }

  // Free memory that is no longer needed.

  edgev.ref ( Vector() );
  accu .ref ( Vector() );

  makeSymmetric ( xadj, adjncy, edgew, mask );

  mask = 1_idx;

  WGraph  itemGraph ( xadj, adjncy, mask, edgew );

  groupItems ( name, itemMap, partCount, itemGraph );
}


//-----------------------------------------------------------------------
//   groupItems (with a graph)
//-----------------------------------------------------------------------


static void               groupItems

  ( const String&           name,
    const IdxVector&        itemMap,
    idx_t                   partCount,
    const WGraph&           itemGraph )

{
  using jive::util::joinNames;
  using jive::graph::Partitioner;
  using jive::graph::newPartitioner;

  Ref<Partitioner>  parter =

    newPartitioner ( joinNames( name, "partitioner" ),
                     Properties(), Properties(), partCount );

  parter->partition ( itemMap, itemGraph, partCount );
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   groupItems
//-----------------------------------------------------------------------


void                      groupItems

  ( const String&           name,
    const IdxVector&        itemMap,
    idx_t                   partCount,
    const ItemSet&          items,
    Ref<DofSpace>           dofs,
    Ref<AbstractMatrix>     matrix )

{
  using jem::System;

  JEM_PRECHECK ( itemMap.size() == items.size() && partCount > 0 );

  // Handle the trivial case.

  if ( partCount == 1 )
  {
    itemMap = 0;

    return;
  }

  print ( System::debug( name ), name,
          " : grouping the ", items.getItemsName(),
          " into ", partCount, " sub sets ...\n" );

  // Try to use the matrix topology.

  if ( matrix &&
       dofs   &&  dofs->getItems() == &items )
  {
    AbstractMatrix::Shape  sh = matrix->shape ();

    if ( sh[0] == sh[1] && sh[0] == dofs->dofCount() )
    {
      SparseMatrixExt*  smx =

        matrix->getExtension<SparseMatrixExt> ();

      if ( smx )
      {
        print ( System::debug( name ), name,
                " : using the matrix topology.\n" );

        groupItems ( name, itemMap, partCount, *smx, *dofs );

        return;
      }
    }
  }

  // Try to use the topology of the item set itself.

  const GroupSet*  groups =

    dynamic_cast<const GroupSet*> ( &items );

  if ( groups )
  {
    print ( System::debug( name ), name,
            " : using the ", items.getItemName(),
            " set topology.\n" );

    groupItems ( name, itemMap, partCount, *groups );

    return;
  }

  // No topology available; fall back to a trivial grouping.

  print ( System::debug( name ), name,
          " : no topology available.\n" );

  groupItems ( itemMap, partCount );
}


JIVE_END_PACKAGE( solver )
