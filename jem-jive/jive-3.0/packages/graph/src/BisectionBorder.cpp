
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
#include <jive/graph/WGraph.h>
#include <jive/graph/PQueue.h>
#include <jive/graph/Partition.h>
#include <jive/graph/BisectionBorder.h>


JIVE_BEGIN_PACKAGE( graph )


//=======================================================================
//   class BisectionBorder
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


BisectionBorder::BisectionBorder ( const Ref<Partition>& p ) :

  partition_  ( p ),
  bnodeCount_ ( 0 ),
  cutWeight_  ( 0 )

{
  JEM_PRECHECK ( p && p->partCount() == 2 );

  const idx_t  nodeCount = p->nodeCount ();

  WGraph       gr        = p->getGraph  ();

  Node_*       node;

  idx_t        idegree;
  idx_t        edegree;
  idx_t        ewgt;
  idx_t        inode;
  idx_t        jnode;
  idx_t        ipart;
  idx_t        jpart;
  idx_t        ilast;
  idx_t        i;


  nodes_  .resize ( nodeCount );
  ibnodes_.resize ( nodeCount );

  for ( inode = 0; inode < nodeCount; inode++ )
  {
    ilast   = gr.xadj[inode + 1];
    ipart   = p->nodeMap ( inode );
    idegree = 0;
    edegree = 0;

    for ( i = gr.xadj[inode]; i < ilast; i++ )
    {
      jnode = gr.adjncy[i];
      jpart = p->nodeMap ( jnode );
      ewgt  = gr.edgeWeights[i];

      if ( jpart == ipart )
      {
        idegree += ewgt;
      }
      else
      {
        edegree += ewgt;
      }
    }

    node          = nodes_.addr ( inode );
    node->idegree = idegree;
    node->edegree = edegree;

    if ( edegree > 0 )
    {
      node->index = bnodeCount_;
      ibnodes_[bnodeCount_++] = inode;
    }
    else
    {
      node->index = -1;
    }

    cutWeight_ += edegree;
  }

  cutWeight_ = cutWeight_ / 2;
}


//-----------------------------------------------------------------------
//   getBorderNodes
//-----------------------------------------------------------------------


idx_t BisectionBorder::getBorderNodes ( const IdxVector& inodes ) const
{
  JEM_PRECHECK ( inodes.size() >= size() );

  const idx_t  n = bnodeCount_;

  inodes[slice(BEGIN,n)] = ibnodes_[slice(BEGIN,n)];

  return n;
}


//-----------------------------------------------------------------------
//   remapNode (without a priority queue)
//-----------------------------------------------------------------------


void BisectionBorder::remapNode ( idx_t inode )
{
  Partition&     p  = *partition_;
  const WGraph&  gr = p.getGraph ();

  Node_*         node;

  idx_t          ipart;
  idx_t          jpart;
  idx_t          npart;
  idx_t          jnode;
  idx_t          ilast;
  idx_t          ewgt;
  idx_t          i;


  node  = nodes_.addr ( inode );
  ipart = p.nodeMap   ( inode );
  npart = (ipart + 1) % 2;

  // Update the node `inode'.

  p.remapNode ( inode, npart );

  cutWeight_ += (node->idegree - node->edegree);

  jem::swap ( node->idegree, node->edegree );

  if ( node->edegree == 0 && node->index >= 0 )
  {
    removeBorderNode_ ( inode );
  }

  if ( node->edegree > 0 && node->index < 0 )
  {
    addBorderNode_    ( inode );
  }

  // Update all neighbors of the node `inode'.

  ilast = gr.xadj[inode + 1];

  for ( i = gr.xadj[inode]; i < ilast; i++ )
  {
    ewgt  = gr.edgeWeights[i];
    jnode = gr.adjncy[i];
    jpart = p.nodeMap   ( jnode );
    node  = nodes_.addr ( jnode );

    if      ( jpart == ipart )
    {
      node->idegree -= ewgt;
      node->edegree += ewgt;

      if ( node->edegree > 0 && node->index < 0 )
      {
        addBorderNode_ ( jnode );
      }
    }
    else if ( jpart == npart )
    {
      node->idegree += ewgt;
      node->edegree -= ewgt;

      if ( node->edegree == 0 && node->index >= 0 )
      {
        removeBorderNode_ ( jnode );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   remapNode (with two priority queues)
//-----------------------------------------------------------------------


void BisectionBorder::remapNode

  ( PQueue&            q0,
    PQueue&            q1,
    idx_t              inode,
    const BoolVector&  mask )

{
  Partition&     p  = *partition_;
  const WGraph&  gr = p.getGraph ();

  PQueue*        q[2];
  Node_*         node;

  idx_t          ipart;
  idx_t          jpart;
  idx_t          npart;
  idx_t          jnode;
  idx_t          ilast;
  idx_t          ewgt;
  idx_t          i;


  q[0] = &q0;
  q[1] = &q1;

  node  = nodes_.addr ( inode );
  ipart = p.nodeMap   ( inode );
  npart = (ipart + 1) % 2;

  // Update the node `inode'.

  p.remapNode ( inode, npart );

  cutWeight_ += (node->idegree - node->edegree);

  jem::swap ( node->idegree, node->edegree );

  if ( node->edegree == 0 && node->index >= 0 )
  {
    removeBorderNode_ ( inode );
  }

  if ( node->edegree > 0 && node->index < 0 )
  {
    addBorderNode_ ( inode );
  }

  // Update all neighbors of the node `inode'.

  ilast = gr.xadj[inode + 1];

  for ( i = gr.xadj[inode]; i < ilast; i++ )
  {
    ewgt  = gr.edgeWeights[i];
    jnode = gr.adjncy[i];
    jpart = p.nodeMap   ( jnode );
    node  = nodes_.addr ( jnode );

    if ( jpart == ipart )
    {
      node->idegree -= ewgt;
      node->edegree += ewgt;

      if ( node->edegree > 0 && node->index < 0 )
      {
        addBorderNode_ ( jnode );

        if ( mask[jnode] )
        {
          q[jpart]->insert ( jnode, node->edegree - node->idegree );
        }
      }
      else if ( mask[jnode] )
      {
        q[jpart]->update ( jnode, node->edegree - node->idegree );
      }
    }
    else if ( jpart == npart )
    {
      node->idegree += ewgt;
      node->edegree -= ewgt;

      if ( node->edegree == 0 && node->index >= 0 )
      {
        removeBorderNode_ ( jnode );

        if ( mask[jnode] )
        {
          q[jpart]->remove ( jnode );
        }
      }
      else if ( mask[jnode] )
      {
        q[jpart]->update ( jnode, node->edegree - node->idegree );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   remapNodes
//-----------------------------------------------------------------------


void BisectionBorder::remapNodes ( const IdxVector& nmap )
{
  Partition&   p         = *partition_;
  const idx_t  nodeCount = p.nodeCount ();

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    if ( nmap[inode] != p.nodeMap( inode ) )
    {
      remapNode ( inode );
    }
  }
}


//-----------------------------------------------------------------------
//   addBorderNode_
//-----------------------------------------------------------------------


inline void BisectionBorder::addBorderNode_ ( idx_t inode )
{
  nodes_[inode].index     = bnodeCount_;
  ibnodes_[bnodeCount_++] = inode;
}


//-----------------------------------------------------------------------
//   removeBorderNode_
//-----------------------------------------------------------------------


inline void BisectionBorder::removeBorderNode_ ( idx_t inode )
{
  bnodeCount_--;

  const idx_t  i = nodes_[inode].index;
  const idx_t  j = ibnodes_[bnodeCount_];

  nodes_[j].index     =  i;
  ibnodes_[i]         =  j;
  nodes_[inode].index = -1;
}


JIVE_END_PACKAGE( graph )
