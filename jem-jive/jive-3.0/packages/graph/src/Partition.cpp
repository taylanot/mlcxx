
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
#include <jem/base/RuntimeException.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jive/util/error.h>
#include <jive/graph/Partition.h>


JIVE_BEGIN_PACKAGE( graph )


//=======================================================================
//   class Partition
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Partition::Partition

  ( const WGraph&  gr,
    idx_t          partCount ) :

    graph_ ( gr )

{
  const idx_t  nodeCount = gr.nodeCount ();

  JEM_PRECHECK ( gr.isValid() && partCount >= 0 );

  nodeMap_    .resize ( nodeCount );
  partWeights_.resize ( partCount );

  nodeMap_     = -1;
  partWeights_ =  0;
  graphWeight_ =  gr.weight        ();
  maxDegree_   =  gr.maxNodeDegree ();
}


Partition::~Partition ()
{}


//-----------------------------------------------------------------------
//   setPartCount
//-----------------------------------------------------------------------


void Partition::setPartCount ( idx_t pcount )
{
  JEM_PRECHECK ( pcount >= 0 );

  if ( pcount != partWeights_.size() )
  {
    partWeights_.resize ( pcount );
  }

  nodeMap_     = -1;
  partWeights_ =  0;
}


//-----------------------------------------------------------------------
//   mapNodes
//-----------------------------------------------------------------------


void Partition::mapNodes ( const IdxVector&  nmap )
{
  JEM_PRECHECK ( nmap.size() == nodeCount() );

  const idx_t  nodeCount = this->nodeCount ();
  const idx_t  partCount = this->partCount ();

  partWeights_ = 0_idx;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    idx_t  ipart = nmap[inode];

    if ( ipart < 0 || ipart >= partCount )
    {
      util::indexError ( JEM_FUNC, "partition", ipart, partCount );
    }

    nodeMap_[inode]      = ipart;
    partWeights_[ipart] += graph_.nodeWeights[inode];
  }
}


//-----------------------------------------------------------------------
//   mapUnmappedNodes
//-----------------------------------------------------------------------


void Partition::mapUnmappedNodes ( idx_t ipart )
{
  JEM_PRECHECK ( ipart >= 0 && ipart < partCount() );

  const idx_t  nodeCount = this->nodeCount ();


  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    if ( nodeMap_[inode] < 0 )
    {
      mapNode ( inode, ipart );
    }
  }
}


//-----------------------------------------------------------------------
//   unmapNodes
//-----------------------------------------------------------------------


void Partition::unmapNodes ()
{
  nodeMap_     = -1_idx;
  partWeights_ = 0_idx;
}


//-----------------------------------------------------------------------
//   getPart
//-----------------------------------------------------------------------


WGraph Partition::getPart

  ( const IdxVector&  inodes,
    idx_t             ipart ) const

{
  JEM_PRECHECK ( ipart         >= 0           &&
                 ipart         <  partCount() &&
                 inodes.size() >= nodeCount() );

  const idx_t  nodeCount = this->nodeCount ();

  idx_t        n         = 0;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    if ( nodeMap_[inode] == ipart )
    {
      inodes[n++] = inode;
    }
  }

  return graph_.getPart ( inodes[slice(BEGIN,n)] );
}


//-----------------------------------------------------------------------
//   getSuperGraph
//-----------------------------------------------------------------------


WGraph Partition::getSuperGraph () const
{
  using jem::numeric::SparseUtils;

  const idx_t  nodeCount   = this->nodeCount ();
  const idx_t  partCount   = this->partCount ();
  const idx_t  edgeCount   = this->edgeCount ();

  IdxVector    xadj        ( partCount + 1 );
  IdxVector    inodes      ( nodeCount );
  IdxVector    adjncy      ( edgeCount );
  IdxVector    edgeWeights ( edgeCount );
  IdxVector    accu        ( partCount );

  idx_t        inode;
  idx_t        ipart;
  idx_t        jpart;

  idx_t        i, j, k, m, n;


  xadj = 0;

  for ( inode = 0; inode < nodeCount; inode++ )
  {
    ipart = nodeMap_[inode];

    if ( ipart < 0 )
    {
      throw jem::RuntimeException (
        JEM_FUNC,
        String::format ( "unmapped node: %d", inode )
      );
    }

    xadj[ipart]++;
  }

  SparseUtils::sumOffsets ( xadj );

  for ( inode = 0; inode < nodeCount; inode++ )
  {
    ipart                 = nodeMap_[inode];
    inodes[xadj[ipart]++] = inode;
  }

  SparseUtils::shiftOffsets ( xadj );

  accu = -1_idx;
  k    = 0;

  for ( ipart = 0; ipart < partCount; ipart++ )
  {
    i           = xadj[ipart];
    m           = xadj[ipart + 1];
    xadj[ipart] = k;

    // Make sure that a super node is not a neighbor of itself

    accu[ipart] = 0;

    for ( ; i < m; i++ )
    {
      inode = inodes[i];
      j     = graph_.xadj[inode];
      n     = graph_.xadj[inode + 1];

      for ( ; j < n; j++ )
      {
        jpart = nodeMap_[graph_.adjncy[j]];

        if ( accu[jpart] < 0 )
        {
          adjncy[k++]  = jpart;
          accu[jpart]  = graph_.edgeWeights[j];
        }
        else
        {
          accu[jpart] += graph_.edgeWeights[j];
        }
      }
    }

    for ( i = xadj[ipart]; i < k; i++ )
    {
      jpart          =  adjncy[i];
      edgeWeights[i] =  accu[jpart];
      accu[jpart]    = -1;
    }

    accu[ipart] = -1;
  }

  xadj[partCount] = k;

  if ( k < edgeCount )
  {
    adjncy     .reshape ( k );
    edgeWeights.reshape ( k );
  }

  return WGraph ( xadj, adjncy, partWeights_.clone(), edgeWeights );
}


//-----------------------------------------------------------------------
//   mapFromSuper
//-----------------------------------------------------------------------


void Partition::mapFromSuper ( const Partition& super )
{
  JEM_PRECHECK ( partCount() == super.nodeCount() );

  const idx_t  nodeCount = this->nodeCount ();

  if ( partWeights_.size() != super.partCount() )
  {
    partWeights_.resize ( super.partCount() );
  }

  partWeights_ = super.partWeights_;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    nodeMap_[inode] = super.nodeMap_[nodeMap_[inode]];
  }
}


JIVE_END_PACKAGE( graph )
