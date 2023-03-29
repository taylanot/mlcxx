
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
#include <jive/graph/WGraph.h>


JIVE_BEGIN_PACKAGE( graph )


using jem::iarray;


//=======================================================================
//   class WGraph
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


WGraph::WGraph

  ( const IdxVector&  xadj,
    const IdxVector&  adjncy ) :

    Super ( xadj, adjncy )

{
  nodeWeights.resize ( nodeCount() );
  edgeWeights.resize ( edgeCount() );

  nodeWeights = 1;
  edgeWeights = 1;
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


bool WGraph::isValid () const
{
  const idx_t  nodeCount = this->nodeCount ();
  const idx_t  edgeCount = this->edgeCount ();


  if ( ! Super::isValid() )
  {
    return false;
  }

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    if ( nodeWeights[i] <= 0 )
    {
      return false;
    }
  }

  for ( idx_t i = 0; i < edgeCount; i++ )
  {
    if ( edgeWeights[i] <= 0 )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   weight
//-----------------------------------------------------------------------


idx_t WGraph::weight () const
{
  const idx_t  nodeCount = this->nodeCount ();

  idx_t        w = 0;


  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    w += nodeWeights[i];
  }

  return w;
}


//-----------------------------------------------------------------------
//   nodeDegree
//-----------------------------------------------------------------------


idx_t WGraph::nodeDegree ( idx_t inode ) const
{
  const idx_t  n = xadj[inode + 1];

  idx_t        d = 0;


  for ( idx_t i = xadj[inode]; i < n; i++ )
  {
    d += edgeWeights[i];
  }

  return d;
}


//-----------------------------------------------------------------------
//   maxNodeDegree
//-----------------------------------------------------------------------


idx_t WGraph::maxNodeDegree () const
{
  const idx_t  nodeCount = this->nodeCount ();

  idx_t        dmax      = 0;


  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    idx_t  n = xadj[inode + 1];
    idx_t  d = 0;

    for ( idx_t i = xadj[inode]; i < n; i++ )
    {
      d += edgeWeights[i];
    }

    if ( d > dmax )
    {
      dmax = d;
    }
  }

  return dmax;
}


//-----------------------------------------------------------------------
//   getPart
//-----------------------------------------------------------------------


WGraph WGraph::getPart ( const IdxVector& inodes ) const
{
  JEM_PRECHECK ( isValid() );

  const idx_t  nodeCount  = this->nodeCount ();
  const idx_t  inodeCount = inodes.size     ();

  IdxVector    nmap ( nodeCount );

  WGraph       part;

  idx_t        inode;
  idx_t        jnode;
  idx_t        i, j, k, n;


  n            = 0;
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

    for ( j = xadj[inode]; j < k; j++ )
    {
      if ( nmap[adjncy[j]] >= 0 )
      {
        n++;
      }
    }
  }

  part.xadj       .resize ( inodeCount + 1 );
  part.adjncy     .resize ( n );
  part.nodeWeights.resize ( inodeCount );
  part.edgeWeights.resize ( n );

  part.nodeWeights = nodeWeights[inodes];

  n = 0;

  for ( i = 0; i < inodeCount; i++ )
  {
    part.xadj[i] = n;
    inode        = inodes[i];
    k            = xadj[inode + 1];

    for ( j = xadj[inode]; j < k; j++ )
    {
      jnode = nmap[adjncy[j]];

      if ( jnode >= 0 )
      {
        part.adjncy[n]      = jnode;
        part.edgeWeights[n] = edgeWeights[j];
        n++;
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


WGraph                    toGraph

  ( const SparseIdxMatrix&  matrix,
    const IdxVector&        nodeWeights )

{
  JEM_PRECHECK ( matrix.size(0) == matrix.size(1) &&
                 matrix.size(0) == nodeWeights.size() );

  return WGraph ( matrix.getRowOffsets    (),
                  matrix.getColumnIndices (),
                  nodeWeights,
                  matrix.getValues        () );
}


JIVE_END_PACKAGE( graph )
