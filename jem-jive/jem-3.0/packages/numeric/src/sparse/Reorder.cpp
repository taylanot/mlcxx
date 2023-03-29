
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


#include <jem/base/assert.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/SparseStructure.h>
#include <jem/numeric/sparse/Reorder.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class Reorder
//=======================================================================

//-----------------------------------------------------------------------
//   automa
//-----------------------------------------------------------------------


void Reorder::automa

  ( const Array<idx_t>&     iperm,
    const SparseStructure&  mstruct )

{
  JEM_PRECHECK2 ( mstruct.size(0) == mstruct.size(1),
                  "non-square SparseStructure" );

  idx_t  nnz  = mstruct.nonZeroCount ();
  idx_t  size = mstruct.size (0);

  if      ( ! nnz )
  {
    iperm = iarray ( size );
  }
  else if ( ! mstruct.isSymmetric() )
  {
    colamd ( iperm, mstruct );
  }
  else
  {
    // Estimate the relative amount of fill-ins that would be
    // obtained with RCM and choose the reordering algorithm
    // on that measure.

    idx_t  fill = calcFill_ ( mstruct );

    // This is a simple heuristic based on several tests.

    if ( (fill / nnz) > 16 )
    {
      nd  ( iperm, mstruct );
    }
    else
    {
      rcm ( iperm, mstruct );
    }
  }
}


//-----------------------------------------------------------------------
//   superReorder
//-----------------------------------------------------------------------


void Reorder::superReorder

  ( const Array<idx_t>&     iperm,
    const SparseStructure&  mstruc,
    Algorithm               reorder )

{
  JEM_PRECHECK2 ( reorder, "invalid reorder algorithm" );

  SparseStructure  superStruc;
  Array<idx_t>     superNodes;
  Array<idx_t>     superPerm;


  mergeSuperNodes  ( superNodes, superStruc, mstruc );
  superPerm.resize ( superStruc.size(0) );
  reorder          ( superPerm,  superStruc );
  expandSuperPerm  ( iperm, superPerm, superNodes );
}


//-----------------------------------------------------------------------
//   mergeSuperNodes
//-----------------------------------------------------------------------


void Reorder::mergeSuperNodes

  ( Array<idx_t>&           superNodes,
    SparseStructure&        superStruc,
    const SparseStructure&  inputStruc )

{
  JEM_PRECHECK2 ( inputStruc.isValid(), "invalid SparseStructure" );
  JEM_PRECHECK2 ( inputStruc.size(0) == inputStruc.size(1),
                  "non-square SparseStructure" );


  const idx_t    nodeCount   = inputStruc.size(0);

  Array<idx_t>   nodeOffsets = inputStruc.getRowOffsets    ();
  Array<idx_t>   nodeIndices = inputStruc.getColumnIndices ();

  Array<idx_t>   superOffsets;
  Array<idx_t>   superIndices;

  Array<idx_t>   superMask;
  Array<idx_t>   nodeMap;

  Array<idx_t>   inodes;
  Array<idx_t>   jnodes;

  idx_t          superCount;
  idx_t          inode;
  idx_t          isup;
  idx_t          jsup;
  idx_t          i, j, k;


  superNodes.resize ( nodeCount + 1 );
  nodeMap   .resize ( nodeCount );

  superNodes[0] = 0;
  superCount    = 0;

  if ( nodeCount > 0 )
  {
    i = nodeOffsets[0];
    j = nodeOffsets[1];

    inodes.ref ( nodeIndices[slice(i,j)] );

    nodeMap[0] = 0;
    superCount++;
  }

  for ( inode = 1; inode < nodeCount; inode++ )
  {
    i = nodeOffsets[inode];
    j = nodeOffsets[inode + 1];

    jnodes.ref ( nodeIndices[slice(i,j)] );

    if ( ! equal( inodes, jnodes ) )
    {
      superNodes[superCount++] = inode;
      inodes.swap ( jnodes );
    }

    nodeMap[inode] = superCount - 1;
  }

  superNodes[superCount] = nodeCount;

  if ( superCount == nodeCount )
  {
    superStruc = inputStruc;
    return;
  }

  superNodes  .reshape ( superCount + 1 );
  superOffsets.resize  ( superCount + 1 );
  superMask   .resize  ( superCount );

  superMask = -1_idx;
  k         = 0;

  for ( isup = 0; isup < superCount; isup++ )
  {
    inode = superNodes[isup];
    j     = nodeOffsets[inode + 1];

    for ( i = nodeOffsets[inode]; i < j; i++ )
    {
      jsup = nodeMap[nodeIndices[i]];

      if ( superMask[jsup] != isup )
      {
        superMask[jsup] = isup;
        k++;
      }
    }
  }

  superIndices.resize ( k );

  superMask = -1_idx;
  k         = 0;

  for ( isup = 0; isup < superCount; isup++ )
  {
    inode = superNodes[isup];
    j     = nodeOffsets[inode + 1];

    superOffsets[isup] = k;

    for ( i = nodeOffsets[inode]; i < j; i++ )
    {
      jsup = nodeMap[nodeIndices[i]];

      if ( superMask[jsup] != isup )
      {
        superMask[jsup]   = isup;
        superIndices[k++] = jsup;
      }
    }
  }

  superOffsets[superCount] = k;

  superStruc = SparseStructure ( shape( superCount, superCount ),
                                 superOffsets, superIndices );
}


//-----------------------------------------------------------------------
//   expandSuperPerm
//-----------------------------------------------------------------------


void Reorder::expandSuperPerm

  ( const Array<idx_t>&  nodePerm,
    const Array<idx_t>&  superPerm,
    const Array<idx_t>&  superNodes )

{
  JEM_PRECHECK2 ( nodePerm .size() == superNodes.back() &&
                  superPerm.size() == superNodes.size() - 1,
                  "Array size mismatch" );

  const idx_t  superCount = superPerm.size ();

  idx_t        inode;
  idx_t        jnode;
  idx_t        isup;
  idx_t        jsup;
  idx_t        j, n;


  inode = 0;

  for ( isup = 0; isup < superCount; isup++ )
  {
    jsup  = superPerm[isup];
    jnode = superNodes[jsup];
    n     = superNodes[jsup + 1] - jnode;

    for ( j = 0; j < n; j++ )
    {
      nodePerm[inode++] = jnode + j;
    }
  }
}


//-----------------------------------------------------------------------
//   calcFill_
//-----------------------------------------------------------------------


idx_t Reorder::calcFill_ ( const SparseStructure& mstruct )
{
  const idx_t   size   = mstruct.size (0);

  Array<idx_t>  xadj   = mstruct.getRowOffsets    ();
  Array<idx_t>  adjncy = mstruct.getColumnIndices ();

  Array<idx_t>  inodes ( size );
  Array<idx_t>  jnodes ( size );
  Array<idx_t>  ilist  ( size );
  Array<idx_t>  imap   ( size );
  Array<idx_t>  mask   ( size );

  idx_t         iiter  = 0;
  idx_t         ilast  = size;
  idx_t         fill   = 0;

  idx_t         icount;
  idx_t         jcount;
  idx_t         count;
  idx_t         width;
  idx_t         depth;
  idx_t         inode;
  idx_t         jnode;
  idx_t         ipos;
  idx_t         n;


  if ( ! size )
  {
    return fill;
  }

  // All nodes that have not yet been visited are stored in a list
  // so that a graph consisting of many disjoint sub-graphs can be
  // processed without performance degradation.

  for ( inode = 0; inode < size; inode++ )
  {
    ilist[inode] =  inode;
    imap [inode] =  inode;
    mask [inode] = -1;
  }

  while ( ilast )
  {
    inodes[0]   = inode = ilist[0];
    mask[inode] = iiter++;
    icount      = 1;

    // Do a level-set traversal to find a fringe node.

    while ( icount )
    {
      jcount = 0;

      for ( idx_t i = 0; i < icount; i++ )
      {
        inode = inodes[i];
        n     = xadj[inode + 1];

        for ( idx_t j = xadj[inode]; j < n; j++ )
        {
          jnode = adjncy[j];

          if ( mask[jnode] != iiter )
          {
            mask  [jnode]    = iiter;
            jnodes[jcount++] = jnode;
          }
        }
      }

      jem::swap ( inodes, jnodes );
      jem::swap ( icount, jcount );
    }

    // Do another level-set traversal, starting at the fringe
    // node. Remove all visited nodes from the free list.

    JEM_PRECHECK ( jcount > 0 );

    inodes[0]    = inode = jnodes[jcount - 1];
    icount       = 1;
    count        = 1;
    depth        = 0;
    jnode        = ilist[--ilast];
    ipos         = imap[inode];
    ilist[ipos]  = jnode;
    imap[jnode]  = ipos;
    imap[inode]  = -1;
    ilist[ilast] = -1;

    while ( icount )
    {
      depth++;

      jcount = 0;

      for ( idx_t i = 0; i < icount; i++ )
      {
        inode = inodes[i];
        n     = xadj[inode + 1];

        for ( idx_t j = xadj[inode]; j < n; j++ )
        {
          jnode = adjncy[j];

          if ( imap[jnode] >= 0 )
          {
            inode            = ilist[--ilast];
            ipos             = imap[jnode];
            ilist [ipos]     = inode;
            imap  [inode]    = ipos;
            imap  [jnode]    = -1;
            ilist [ilast]    = -1;
            jnodes[jcount++] = jnode;
          }
        }
      }

      count += jcount;

      jem::swap ( inodes, jnodes );
      jem::swap ( icount, jcount );
    }

    width = count / depth;
    fill += 2 * width * (count - width);
  }

  return fill;
}


JEM_END_PACKAGE( numeric )
