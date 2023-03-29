
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
#include <jive/util/error.h>
#include <jive/graph/Graph.h>


JIVE_BEGIN_PACKAGE( graph )


using jem::iarray;


//=======================================================================
//   class Graph
//=======================================================================

//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


bool Graph::isValid () const
{
  const idx_t  nodeCount = this->nodeCount ();
  const idx_t  edgeCount = this->edgeCount ();


  if ( nodeCount < 0 )
  {
    return false;
  }

  if ( xadj[0] != 0 || xadj[nodeCount] != edgeCount )
  {
    return false;
  }

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    if ( xadj[i + 1] < xadj[i] )
    {
      return false;
    }

    if ( xadj[i] < 0 || xadj[i] > edgeCount )
    {
      return false;
    }
  }

  for ( idx_t i = 0; i < edgeCount; i++ )
  {
    if ( adjncy[i] < 0 || adjncy[i] >= nodeCount )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   maxNodeEdgeCount
//-----------------------------------------------------------------------


idx_t Graph::maxNodeEdgeCount () const
{
  const idx_t  nodeCount = this->nodeCount ();

  idx_t        maxCount = 0;


  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    idx_t  count = xadj[i + 1] - xadj[i];

    if ( count > maxCount )
    {
      maxCount = count;
    }
  }

  return maxCount;
}


//-----------------------------------------------------------------------
//   getPart
//-----------------------------------------------------------------------


Graph Graph::getPart ( const IdxVector& inodes ) const
{
  JEM_PRECHECK ( isValid() );

  const idx_t  nodeCount  = this->nodeCount ();
  const idx_t  inodeCount = inodes.size     ();

  IdxVector    nmap       ( nodeCount );

  Graph        part;
  idx_t        inode;
  idx_t        jnode;
  idx_t        i, j, k, n;


  n            =  0;
  nmap         = -1;
  nmap[inodes] = iarray ( inodeCount );

  for ( i = 0; i < inodeCount; i++ )
  {
    inode = inodes[i];

    if ( inode < 0 || inode >= nodeCount )
    {
      util::indexError ( JEM_FUNC, "node", inode, nodeCount );
    }

    k = xadj[inode + 1];

    for ( j = xadj[inode]; inode < k; j++ )
    {
      if ( nmap[adjncy[j]] >= 0 )
      {
        n++;
      }
    }
  }

  part.xadj  .resize ( inodeCount + 1 );
  part.adjncy.resize ( n );

  n = 0;

  for ( i = 0; i < inodeCount; i++ )
  {
    part.xadj[i] = n;
    inode        = inodes[i];
    k            = xadj[inode + 1];

    for ( j = xadj[inode]; inode < k; j++ )
    {
      jnode = nmap[adjncy[j]];

      if ( jnode >= 0 )
      {
        part.adjncy[n++] = jnode;
      }
    }
  }

  part.xadj[inodeCount] = n;

  return part;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   toGraph
//-----------------------------------------------------------------------


Graph toGraph ( const SparseStruct& s )
{
  JEM_PRECHECK ( s.size(0) == s.size(1) );

  return Graph ( s.getRowOffsets(), s.getColumnIndices() );
}


JIVE_END_PACKAGE( graph )
