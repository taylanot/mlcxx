
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Random.h>
#include <jive/graph/utilities.h>
#include <jive/graph/Names.h>
#include <jive/graph/Partition.h>
#include <jive/graph/PartitionControl.h>
#include <jive/graph/PartitionerFactory.h>
#include <jive/graph/GreedyOptimizer.h>
#include <jive/graph/GreedyPartitioner.h>


JEM_DEFINE_CLASS( jive::graph::GreedyPartitioner );


JIVE_BEGIN_PACKAGE( graph )


using jem::newInstance;
using jive::util::joinNames;


//=======================================================================
//   class GreedyPartitioner
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GreedyPartitioner::TYPE_NAME = "Greedy";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GreedyPartitioner::GreedyPartitioner

  ( const String&          name,
    const Ref<Random>&     rand,
    const Ref<Optimizer>&  optimizer ) :

    Super      (      name ),
    rand_      (      rand ),
    optimizer_ ( optimizer )

{
  if ( ! rand_ )
  {
    rand_ = newInstance<Random> ();
  }

  if ( ! optimizer_ )
  {
    optimizer_ = newInstance<GreedyOptimizer> (
      joinNames ( name, PropNames::OPTIMIZER )
    );
  }
}


GreedyPartitioner::~GreedyPartitioner ()
{}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void GreedyPartitioner::configure ( const Properties& props )
{
  optimizer_->configure ( props );
}


void GreedyPartitioner::getConfig ( const Properties& props ) const
{
  optimizer_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Partitioner> GreedyPartitioner::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Optimizer>  optimizer;

  String  name = joinNames ( myName, PropNames::OPTIMIZER );


  if ( props.contains( name ) )
  {
    optimizer = newOptimizer ( name, conf, props );
  }
  else
  {
    conf.makeProps(name).set ( PropNames::TYPE,
                               GreedyOptimizer::TYPE_NAME );

    optimizer = GreedyOptimizer::makeNew ( name,  conf,
                                           props, globdat );
  }

  return newInstance<Self> ( myName, Random::get( globdat ),
                             optimizer );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void GreedyPartitioner::declare ()
{
  PartitionerFactory::declare ( TYPE_NAME,  & makeNew );
  PartitionerFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   bisect_
//-----------------------------------------------------------------------


void GreedyPartitioner::bisect_

  ( Partition&      p,
    const Control&  ctrl )

{
  using jem::System;

  JEM_PRECHECK ( p.partCount() == 2 );

  const idx_t   nodeCount = p.nodeCount ();

  const WGraph  gr        = p.getGraph  ();

  IntMatrix     ibuf      ( nodeCount, 2 );

  IdxVector     inodes    ( ibuf[0] );
  IdxVector     jnodes    ( ibuf[1] );

  bool          done;

  idx_t         scatterCount;
  idx_t         nodesLeft;
  idx_t         inodeCount;
  idx_t         jnodeCount;

  idx_t         wtarget;

  idx_t         inode;
  idx_t         jnode;
  idx_t         ipart;

  idx_t         i, j, n;


  // First construct the smallest partition

  if ( ctrl.targetWeight(0) <= ctrl.targetWeight(1) )
  {
    ipart = 0;
  }
  else
  {
    ipart = 1;
  }

  wtarget      = ctrl.targetWeight ( ipart );
  scatterCount = 0;
  nodesLeft    = nodeCount - 1;
  inodeCount   = 0;
  inode        = findFringeNode ( gr, rand_->next( nodeCount ) );

  if ( inode < 0 )
  {
    return;
  }

  p.mapNode ( inode, ipart );

  inodes[inodeCount++] = inode;
  nodesLeft--;

  done = (p.partWeight(ipart) >= wtarget || nodesLeft <= 0);

  while ( ! done )
  {
    jnodeCount = 0;

    for ( i = 0; i < inodeCount && ! done; i++ )
    {
      inode = inodes[i];
      n     = gr.xadj[inode + 1];

      for ( j = gr.xadj[inode]; j < n && ! done; j++ )
      {
        jnode = gr.adjncy[j];

        if ( p.notMapped( jnode ) )
        {
          p.mapNode ( jnode, ipart );

          jnodes[jnodeCount++] = jnode;
          nodesLeft--;

          done = (p.partWeight(ipart) >= wtarget || nodesLeft <= 0);
        }
      }
    }

    jem::swap ( inodes,     jnodes     );
    jem::swap ( inodeCount, jnodeCount );

    if ( ! done && inodeCount == 0 )
    {
      scatterCount++;

      print ( System::debug(), myName_,
              " : scattered part (", ipart, ")\n" );

      inode = findSeedNode_ ( p );

      if ( inode < 0 )
      {
        break;
      }

      p.mapNode ( inode, ipart );

      inodes[inodeCount++] = inode;
      nodesLeft--;

      done = (p.partWeight(ipart) >= wtarget || nodesLeft <= 0);
    }
  }

  // Map all remaining vertices to the other partition

  p.mapUnmappedNodes ( (ipart + 1) % 2 );

  if ( scatterCount > 0 )
  {
    print ( System::debug(), myName_,
            " : created ", scatterCount, " scattered parts\n" );
  }

  optimizer_->optimize ( p, ctrl );
}


//-----------------------------------------------------------------------
//   partition_
//-----------------------------------------------------------------------


void GreedyPartitioner::partition_

  ( Partition&      p,
    const Control&  ctrl )

{
  using jem::System;

  const idx_t   nodeCount = p.nodeCount ();
  const idx_t   partCount = p.partCount ();

  const WGraph  gr        = p.getGraph  ();

  IntMatrix     ibuf      ( nodeCount, 2 );

  IdxVector     inodes    ( ibuf[0] );
  IdxVector     jnodes    ( ibuf[1] );

  idx_t         wtotal, wrest, wtarget, wmin;

  bool          notDone;

  idx_t         scatterCount;
  idx_t         inodeCount;
  idx_t         jnodeCount;

  idx_t         inode;
  idx_t         jnode;
  idx_t         ipart;

  idx_t         i, j, n;


  wtotal       = p.graphWeight ();
  wrest        = wtotal;
  scatterCount = 0;
  inodeCount   = 0;
  jnodeCount   = 0;

  for ( ipart = 0; ipart < partCount; ipart++ )
  {
    wrest  -= ctrl.targetWeight ( ipart );
    wtarget = wtotal - wrest;

    if ( ipart == partCount - 1 )
    {
      wmin = wtarget;
    }
    else
    {
      wmin = wtotal - wrest - ctrl.maxWeight( ipart + 1 ) +

        ctrl.targetWeight ( ipart + 1 );

      if ( wmin < ctrl.targetWeight( ipart ) )
      {
        wmin = ctrl.targetWeight ( ipart );
      }
    }

    inode = findSeedNode_ ( p, inodes[slice(BEGIN,inodeCount)] );

    if ( inode < 0 )
    {
      break;
    }

    p.mapNode ( inode, ipart );

    inodes[0]  = inode;
    inodeCount = 1;
    notDone    = (p.partWeight( ipart ) < wtarget);

    while ( notDone )
    {
      jnodeCount = 0;

      for ( i = 0; i < inodeCount && notDone; i++ )
      {
        inode = inodes[i];
        n     = gr.xadj[inode + 1];

        for ( j = gr.xadj[inode]; j < n && notDone; j++ )
        {
          jnode = gr.adjncy[j];

          if ( p.notMapped( jnode ) )
          {
            p.mapNode ( jnode, ipart );

            jnodes[jnodeCount++] = jnode;
            notDone = (p.partWeight( ipart ) < wtarget);
          }
        }
      }

      jem::swap ( inodes,     jnodes     );
      jem::swap ( inodeCount, jnodeCount );

      // Check for scattered parts.

      if ( inodeCount == 0 )
      {
        if ( p.partWeight( ipart ) >= wmin )
        {
          break;
        }

        scatterCount++;

        print ( System::debug( myName_ ), myName_,
                " : scattered part (", ipart, ")\n" );

        inode = findSeedNode_ ( p );

        if ( inode < 0 )
        {
          break;
        }

        p.mapNode ( inode, ipart );

        inodes[inodeCount++] = inode;
        notDone = (p.partWeight( ipart ) < wtarget);
      }
    }

    wtotal -= p.partWeight( ipart );
  }

  if ( scatterCount > 0 )
  {
    print ( System::debug( myName_ ), myName_,
            " : created ", scatterCount, " scattered parts\n" );
  }

  optimizer_->optimize ( p, ctrl );
}


//-----------------------------------------------------------------------
//   findSeedNode_ (given a partition)
//-----------------------------------------------------------------------


idx_t GreedyPartitioner::findSeedNode_ ( const Partition& p ) const
{
  const idx_t  nodeCount = p.nodeCount ();

  BoolVector   mask ( nodeCount );


  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    mask[inode] = p.notMapped ( inode );
  }

  return findFringeNode ( p.getGraph(), mask );
}


//-----------------------------------------------------------------------
//   findSeedNode_ (given a partition and a node list)
//-----------------------------------------------------------------------


idx_t GreedyPartitioner::findSeedNode_

  ( const Partition&  p,
    const IdxVector&  inodes ) const

{
  const idx_t    inodeCount = inodes.size ();

  const WGraph&  gr         = p.getGraph ();

  idx_t          edgeCount;
  idx_t          minCount;
  idx_t          inode;
  idx_t          jnode;
  idx_t          seed;
  idx_t          i, j, n;


  seed     = -1;
  minCount = 0;

  for ( i = 0; i < inodeCount; i++ )
  {
    inode = inodes[i];
    n     = gr.xadj[inode + 1];

    for ( j = gr.xadj[inode]; j < n; j++ )
    {
      jnode = gr.adjncy[j];

      if ( p.notMapped( jnode ) )
      {
        edgeCount = gr.nodeEdgeCount ( jnode );

        if ( seed < 0 || edgeCount < minCount )
        {
          seed     = jnode;
          minCount = edgeCount;
        }
      }
    }
  }

  if ( seed < 0 )
  {
    seed = findSeedNode_ ( p );
  }

  return seed;
}


JIVE_END_PACKAGE( graph )
