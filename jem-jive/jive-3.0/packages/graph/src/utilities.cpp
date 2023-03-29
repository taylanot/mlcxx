
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
#include <jem/base/array/operators.h>
#include <jive/util/Random.h>
#include <jive/graph/Graph.h>
#include <jive/graph/utilities.h>


JIVE_BEGIN_PACKAGE( graph )


//=======================================================================
//   private functions
//=======================================================================


static idx_t         findFringeNode_

  ( const BoolVector&  visited,
    const Graph&       gr,
    idx_t              seed )

{
  JEM_ASSERT ( seed >= 0 && seed < gr.nodeCount() );

  const idx_t  nodeCount = gr.nodeCount ();

  IdxMatrix    ibuf      ( nodeCount, 2 );

  IdxVector    inodes    ( ibuf[0] );
  IdxVector    jnodes    ( ibuf[1] );

  idx_t        inodeCount;
  idx_t        jnodeCount;
  idx_t        inode;
  idx_t        jnode;
  idx_t        i, j, n;


  inodes[0]     = seed;
  visited[seed] = true;
  inodeCount    = 1;

  while ( inodeCount > 0 )
  {
    jnodeCount = 0;

    for ( i = 0; i < inodeCount; i++ )
    {
      inode = inodes[i];
      n     = gr.xadj[inode + 1];

      for ( j = gr.xadj[inode]; j < n; j++ )
      {
        jnode = gr.adjncy[j];

        if ( ! visited[jnode] )
        {
          jnodes[jnodeCount++] = jnode;
          visited[jnode]       = true;
        }
      }
    }

    jem::swap ( inodes,     jnodes     );
    jem::swap ( inodeCount, jnodeCount );
  }

  if ( jnodeCount > 0 )
  {
    return jnodes[jnodeCount - 1];
  }
  else
  {
    return -1_idx;
  }
}


//=======================================================================
//   public functions
//=======================================================================


void                randomize

  ( const IdxVector&  iperm,
    Random&           rand )

{
  const idx_t  n = iperm.size ();

  idx_t        i, j, k;


  if ( n <= 4 )
  {
    return;
  }

  for ( i = 0; i < n; i += 16 )
  {
    j = rand.next ( n - 4 );
    k = rand.next ( n - 4 );

    jem::swap ( iperm[j + 0], iperm[k + 0] );
    jem::swap ( iperm[j + 1], iperm[k + 1] );
    jem::swap ( iperm[j + 2], iperm[k + 2] );
    jem::swap ( iperm[j + 3], iperm[k + 3] );
  }
}


//-----------------------------------------------------------------------
//   findFringeNode (given a graph and seed node)
//-----------------------------------------------------------------------


idx_t           findFringeNode

  ( const Graph&  gr,
    idx_t         seed )

{
  const idx_t  nodeCount = gr.nodeCount ();

  if ( nodeCount < 1 )
  {
    return -1_idx;
  }
  else
  {
    JEM_PRECHECK ( seed >= 0 && seed < gr.nodeCount() );

    BoolVector    visited ( nodeCount );

    visited = false;

    return findFringeNode_ ( visited, gr, seed );
  }
}


//-----------------------------------------------------------------------
//   findFringeNode (given a graph and mask vector)
//-----------------------------------------------------------------------


idx_t                findFringeNode

  ( const Graph&       gr,
    const BoolVector&  mask )

{
  JEM_PRECHECK ( mask.size() == gr.nodeCount() );

  const idx_t nodeCount = gr.nodeCount ();

  if ( nodeCount < 1 )
  {
    return -1_idx;
  }
  else
  {
    BoolVector  visited ( nodeCount );
    idx_t       seed;

    for ( seed = 0; seed < nodeCount; seed++ )
    {
      if ( mask[seed] )
      {
        break;
      }
    }

    if ( seed < nodeCount )
    {
      visited = ! mask;

      return findFringeNode_ ( visited, gr, seed );
    }
    else
    {
      return -1_idx;
    }
  }
}


//-----------------------------------------------------------------------
//   findFringeNode (given a graph, mask vector and seed node)
//-----------------------------------------------------------------------


idx_t                findFringeNode

  ( const Graph&       gr,
    const BoolVector&  mask,
    idx_t              seed )

{
  JEM_PRECHECK ( seed        >= 0              &&
                 seed        <  gr.nodeCount() &&
                 mask.size() == gr.nodeCount() );

  BoolVector  visited ( ! mask );

  return findFringeNode_ ( visited, gr, seed );
}


JIVE_END_PACKAGE( graph )
