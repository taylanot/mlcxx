
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
#include <jive/graph/WGraph.h>
#include <jive/graph/PQueue.h>
#include <jive/graph/Partition.h>
#include <jive/graph/PartitionBorder.h>


JIVE_BEGIN_PACKAGE( graph )


//=======================================================================
//   class PartitionBorder
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


PartitionBorder::PartitionBorder ( const Ref<Partition>& p ) :

  partition_  ( p ),
  bnodeCount_ ( 0 ),
  cutWeight_  ( 0 )

{
  JEM_PRECHECK ( p );

  const idx_t  nodeCount = p->nodeCount ();
  const idx_t  partCount = p->partCount ();

  WGraph       gr        = p->getGraph  ();

  IdxVector    count     ( partCount );

  Edge*        edges;
  Node_*       node;

  idx_t        idegree;
  idx_t        edegree;
  idx_t        inode;
  idx_t        jnode;
  idx_t        ipart;
  idx_t        jpart;
  idx_t        ifirst;
  idx_t        ilast;
  idx_t        ie, ne;
  idx_t        i;


  edges_  .resize ( gr.edgeCount() );
  nodes_  .resize ( nodeCount );
  ibnodes_.resize ( nodeCount );

  count = -1;

  for ( inode = 0; inode < nodeCount; inode++ )
  {
    ifirst       = gr.xadj[inode];
    ilast        = gr.xadj[inode + 1];
    ipart        = p->nodeMap ( inode );
    count[ipart] = 0;
    edges        = edges_.addr ( ifirst );
    ne           = 0;

    for ( i = ifirst; i < ilast; i++ )
    {
      jnode = gr.adjncy[i];
      jpart = p->nodeMap ( jnode );

      if ( count[jpart] < 0 )
      {
        edges[ne++].ipart = jpart;
        count[jpart]      = gr.edgeWeights[i];
      }
      else
      {
        count[jpart] += gr.edgeWeights[i];
      }
    }

    idegree      = count[ipart];
    edegree      = 0;
    count[ipart] = -1;

    for ( ie = 0; ie < ne; ie++ )
    {
      jpart             = edges[ie].ipart;
      edges[ie].weight  = count[jpart];
      edegree          += count[jpart];
      count[jpart]      = -1;
    }

    node            = nodes_.addr ( inode );
    node->idegree   = idegree;
    node->edegree   = edegree;
    node->edgeCount = ne;
    node->edges     = edges;

    if ( ne > 0 )
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


idx_t PartitionBorder::getBorderNodes ( const IdxVector& inodes ) const
{
  JEM_PRECHECK ( inodes.size() >= size() );

  const idx_t  n = bnodeCount_;

  inodes[slice(BEGIN,n)] = ibnodes_[slice(BEGIN,n)];

  return n;
}


//-----------------------------------------------------------------------
//   remapNode (without a priority queue)
//-----------------------------------------------------------------------


void PartitionBorder::remapNode

  ( idx_t  inode,
    idx_t  npart )

{
  Partition&     p  = *partition_;
  const WGraph&  gr = p.getGraph ();

  Edge*          edges;
  Edge*          edge;
  Node_*         node;

  idx_t          delta;
  idx_t          ipart;
  idx_t          jpart;
  idx_t          jnode;
  idx_t          ilast;
  idx_t          ewgt;
  idx_t          ie;
  idx_t          i;


  ipart = p.nodeMap   ( inode );
  node  = nodes_.addr ( inode );
  edges = node->edges;

  // Update the node `inode'.

  p.remapNode ( inode, npart );

  ie = findEdge_ ( *node, npart );

  if ( ie < 0 )
  {
    delta          = node->idegree;
    node->edegree += delta;
    node->idegree  = 0;

    if ( node->edegree > 0 && node->index < 0 )
    {
      addBorderNode_ ( inode );
    }
  }
  else
  {
    edge           = edges + ie;
    delta          = node->idegree - edge->weight;
    edge->ipart    = ipart;
    node->edegree += delta;

    jem::swap ( node->idegree, edge->weight );

    if ( edge->weight == 0 )
    {
      node->edgeCount--;
      *edge = edges[node->edgeCount];
    }

    if ( node->edegree == 0 && node->index >= 0 )
    {
      removeBorderNode_ ( inode );
    }
  }

  cutWeight_ += delta;

  // Update all neighbors of the node `inode'.

  ilast = gr.xadj[inode + 1];

  for ( i = gr.xadj[inode]; i < ilast; i++ )
  {
    ewgt  = gr.edgeWeights[i];
    jnode = gr.adjncy[i];
    jpart = p.nodeMap   ( jnode );
    node  = nodes_.addr ( jnode );
    edges = node->edges;

    if ( jpart == ipart )
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

    if ( jpart != ipart )
    {
      ie = findEdge_ ( *node, ipart );

      if ( ie >= 0 )
      {
        edge          = edges + ie;
        edge->weight -= ewgt;

        if ( edge->weight == 0 )
        {
          node->edgeCount--;
          *edge = edges[node->edgeCount];
        }
      }
    }

    if ( jpart != npart )
    {
      ie = findEdge_ ( *node, npart );

      if ( ie >= 0 )
      {
        edges[ie].weight += ewgt;
      }
      else
      {
        edges[node->edgeCount].ipart  = npart;
        edges[node->edgeCount].weight = ewgt;
        node->edgeCount++;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   remapNode (with a priority queue)
//-----------------------------------------------------------------------


void PartitionBorder::remapNode

  ( PQueue&            q,
    idx_t              inode,
    idx_t              npart,
    const BoolVector&  mask )

{
  Partition&     p  = *partition_;
  const WGraph&  gr = p.getGraph ();

  Edge*          edges;
  Edge*          edge;
  Node_*         node;

  idx_t          delta;
  idx_t          ipart;
  idx_t          jpart;
  idx_t          jnode;
  idx_t          ilast;
  idx_t          ewgt;
  idx_t          ie;
  idx_t          i;


  ipart = p.nodeMap   ( inode );
  node  = nodes_.addr ( inode );
  edges = node->edges;

  // Update the node `inode'.

  p.remapNode ( inode, npart );

  ie = findEdge_ ( *node, npart );

  if ( ie < 0 )
  {
    delta          = node->idegree;
    node->edegree += delta;
    node->idegree  = 0;

    if ( node->edegree > 0 && node->index < 0 )
    {
      addBorderNode_ ( inode );
    }
  }
  else
  {
    edge           = edges + ie;
    delta          = node->idegree - edge->weight;
    edge->ipart    = ipart;
    node->edegree += delta;

    jem::swap ( node->idegree, edge->weight );

    if ( edge->weight == 0 )
    {
      node->edgeCount--;
      *edge = edges[node->edgeCount];
    }

    if ( node->edegree == 0 && node->index >= 0 )
    {
      removeBorderNode_ ( inode );
    }
  }

  cutWeight_ += delta;

  // Update all neighbors of the node `inode'.

  ilast = gr.xadj[inode + 1];

  for ( i = gr.xadj[inode]; i < ilast; i++ )
  {
    ewgt  = gr.edgeWeights[i];
    jnode = gr.adjncy[i];
    jpart = p.nodeMap   ( jnode );
    node  = nodes_.addr ( jnode );
    edges = node->edges;

    if ( jpart == ipart )
    {
      node->idegree -= ewgt;
      node->edegree += ewgt;

      if ( node->edegree > 0 && node->index < 0 )
      {
        addBorderNode_ ( jnode );

        if ( mask[jnode] )
        {
          q.insert ( jnode, node->edegree - node->idegree );
        }
      }
      else if ( mask[jnode] )
      {
        q.update ( jnode, node->edegree - node->idegree );
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
          q.remove ( jnode );
        }
      }
      else if ( mask[jnode] )
      {
        q.update ( jnode, node->edegree - node->idegree );
      }
    }

    if ( jpart != ipart )
    {
      ie = findEdge_ ( *node, ipart );

      if ( ie >= 0 )
      {
        edge          = edges + ie;
        edge->weight -= ewgt;

        if ( edge->weight == 0 )
        {
          node->edgeCount--;
          *edge = edges[node->edgeCount];
        }
      }
    }

    if ( jpart != npart )
    {
      ie = findEdge_ ( *node, npart );

      if ( ie >= 0 )
      {
        edges[ie].weight += ewgt;
      }
      else
      {
        edges[node->edgeCount].ipart  = npart;
        edges[node->edgeCount].weight = ewgt;
        node->edgeCount++;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   remapNodes
//-----------------------------------------------------------------------


void PartitionBorder::remapNodes ( const IdxVector& nmap )
{
  Partition&   p         = *partition_;
  const idx_t  nodeCount = p.nodeCount ();


  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    idx_t  ipart = nmap[inode];

    if ( ipart != p.nodeMap( inode ) )
    {
      remapNode ( inode, ipart );
    }
  }
}


//-----------------------------------------------------------------------
//   findEdge_
//-----------------------------------------------------------------------


inline idx_t PartitionBorder::findEdge_

  ( const Node_&  node,
    idx_t         ipart )

{
  if ( node.edgeCount > 0 )
  {
    if ( node.edges[0].ipart == ipart )
    {
      return 0;
    }

    for ( idx_t i = 1; i < node.edgeCount; i++ )
    {
      if ( node.edges[i].ipart == ipart )
      {
        return i;
      }
    }
  }

  return -1_idx;
}


//-----------------------------------------------------------------------
//   addBorderNode_
//-----------------------------------------------------------------------


inline void PartitionBorder::addBorderNode_ ( idx_t inode )
{
  nodes_[inode].index     = bnodeCount_;
  ibnodes_[bnodeCount_++] = inode;
}


//-----------------------------------------------------------------------
//   removeBorderNode_
//-----------------------------------------------------------------------


inline void PartitionBorder::removeBorderNode_ ( idx_t inode )
{
  bnodeCount_--;

  const idx_t  i = nodes_[inode].index;
  const idx_t  j = ibnodes_[bnodeCount_];

  nodes_[j].index     =  i;
  ibnodes_[i]         =  j;
  nodes_[inode].index = -1;
}


JIVE_END_PACKAGE( graph )
