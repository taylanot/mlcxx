
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/Writer.h>
#include <jem/util/Properties.h>
#include <jive/util/Random.h>
#include <jive/graph/utilities.h>
#include <jive/graph/Names.h>
#include <jive/graph/PQueue.h>
#include <jive/graph/Partition.h>
#include <jive/graph/PartitionControl.h>
#include <jive/graph/PartitionBorder.h>
#include <jive/graph/BisectionBorder.h>
#include <jive/graph/OptimizerFactory.h>
#include <jive/graph/Balancer.h>
#include <jive/graph/GreedyOptimizer.h>


JEM_DEFINE_CLASS( jive::graph::GreedyOptimizer );


JIVE_BEGIN_PACKAGE( graph )


using jem::io::Writer;


//=======================================================================
//   class GreedyOptimizer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GreedyOptimizer::TYPE_NAME  = "Greedy";
const int    GreedyOptimizer::MAX_PASSES = 8;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GreedyOptimizer::GreedyOptimizer

  ( const String&       name,
    const Ref<Random>&  rand ) :

  Super      (       name ),
  rand_      (       rand ),
  maxPasses_ ( MAX_PASSES )

{
  if ( ! rand_ )
  {
    rand_ = jem::newInstance<Random> ();
  }
}


GreedyOptimizer::~GreedyOptimizer ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void GreedyOptimizer::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( maxPasses_,
                   PropNames::MAX_PASSES,
                   0, jem::maxOf( maxPasses_ ) );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void GreedyOptimizer::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::MAX_PASSES, maxPasses_ );
}


//-----------------------------------------------------------------------
//   optimize
//-----------------------------------------------------------------------


void GreedyOptimizer::optimize

  ( Partition&      p,
    const Control&  ctrl )

{
  using jem::io::endl;

  Writer&  debug = jem::System::debug ( myName_ );

  idx_t    cutWeight[2];


  if ( p.partCount() == 2 )
  {
    BisectionBorder border ( &p );

    cutWeight[0] = border.cutWeight ();

    optimize          ( border, *rand_, ctrl,     maxPasses_ );
    Balancer::balance ( border,         ctrl, 2 * maxPasses_ );

    cutWeight[1] = border.cutWeight ();
  }
  else
  {
    PartitionBorder border ( &p );

    cutWeight[0] = border.cutWeight ();

    optimize          ( border, *rand_, ctrl,     maxPasses_ );
    Balancer::balance ( border, *rand_, ctrl, 2 * maxPasses_ );

    cutWeight[1] = border.cutWeight ();
  }

  print ( debug, myName_, " : initial edge-cut weight = ",
          cutWeight[0], endl );

  print ( debug, myName_, " : final   edge-cut weight = ",
          cutWeight[1], endl );
}


//-----------------------------------------------------------------------
//   optimize (bisection)
//-----------------------------------------------------------------------


void GreedyOptimizer::optimize

  ( BisectionBorder&  b,
    Random&           rand,
    const Control&    ctrl,
    int               maxPasses )

{
  typedef BisectionBorder::Node  Node;

  Partition&    p  = * b.getPartition ();
  const WGraph  gr =   p.getGraph     ();

  const idx_t   nodeCount = p.nodeCount ();

  PQueue        queue     ( nodeCount, p.maxNodeDegree() );
  IdxVector     inodes    ( nodeCount );
  BoolVector    mask      ( nodeCount );

  Node          node;

  bool          bcond;

  idx_t         wmin[2];
  idx_t         wmax[2];
  idx_t         wpart[2];
  idx_t         wnode;
  idx_t         gain;

  idx_t         inodeCount;
  idx_t         moveCount;

  int           ipass;
  idx_t         inode;
  idx_t         ipart;
  idx_t         jpart;
  idx_t         i;


  for ( ipart = 0; ipart < 2; ipart++ )
  {
    wmin[ipart]  = ctrl.minWeight  ( ipart );
    wmax[ipart]  = ctrl.maxWeight  ( ipart );
    wpart[ipart] = p   .partWeight ( ipart );
  }

  mask = true;

  for ( ipass = 0; ipass < maxPasses; ipass++ )
  {
    moveCount  = 0;
    inodeCount = b.getBorderNodes ( inodes );

    randomize ( inodes[slice(BEGIN,inodeCount)], rand );

    for ( i = 0; i < inodeCount; i++ )
    {
      inode = inodes[i];

      b.getNode    ( node,  inode );
      queue.insert ( inode, node.edegree - node.idegree );
    }

    while ( queue.size() > 0 )
    {
      inode = queue.popFront ();
      wnode = gr.nodeWeights[inode];
      ipart = p.nodeMap ( inode );
      jpart = (ipart + 1) % 2;

      b.getNode ( node, inode );

      gain  = node.edegree - node.idegree;

      bcond = ( wpart[ipart] - wnode >= wmin[ipart] &&
                wpart[jpart] + wnode <= wmax[jpart] );

      if ( bcond && (gain > 0 || wpart[ipart] > wmax[ipart]) )
      {
        b.remapNode ( queue, inode, mask );

        wpart[ipart] -= wnode;
        wpart[jpart] += wnode;

        moveCount++;
      }
    }

    if ( moveCount == 0 )
    {
      break;
    }
  }
}


//-----------------------------------------------------------------------
//   optimize (partition)
//-----------------------------------------------------------------------


void GreedyOptimizer::optimize

  ( PartitionBorder&  b,
    Random&           rand,
    const Control&    ctrl,
    int               maxPasses )

{
  typedef PartitionBorder::Node  Node;

  Partition&    p  = * b.getPartition ();
  const WGraph  gr =   p.getGraph     ();

  const idx_t   nodeCount = p.nodeCount ();

  PQueue        queue     ( nodeCount, p.maxNodeDegree() );
  IdxVector     inodes    ( nodeCount );
  BoolVector    mask      ( nodeCount );

  Node          node;

  double        delta;
  bool          bcond;
  idx_t         maxGain;
  idx_t         gain;
  idx_t         wnode;
  idx_t         wpart;

  idx_t         inodeCount;
  idx_t         moveCount;

  int           ipass;
  idx_t         inode;
  idx_t         ipart;
  idx_t         jpart;
  idx_t         kpart;
  idx_t         i, ie;


  mask = true;

  for ( ipass = 0; ipass < maxPasses; ipass++ )
  {
    moveCount  = 0;
    inodeCount = b.getBorderNodes ( inodes );

    randomize ( inodes[slice(BEGIN,inodeCount)], rand );

    for ( i = 0; i < inodeCount; i++ )
    {
      inode = inodes[i];

      b.getNode    ( node, inode );
      queue.insert ( inode, node.edegree - node.idegree );
    }

    while ( queue.size() > 0 )
    {
      inode = queue.popFront ();
      ipart = p.nodeMap(inode);
      wnode = gr.nodeWeights[inode];
      wpart = p.partWeight(ipart);

      if ( wpart - wnode < ctrl.minWeight(ipart) )
      {
        continue;
      }

      b.getNode ( node, inode );

      maxGain =  0;
      jpart   = -1;

      if ( wpart > ctrl.maxWeight(ipart) )
      {
        // Optimize for load balance.

        for ( ie = 0; ie < node.edgeCount; ie++ )
        {
          kpart = node.edges[ie].ipart;

          delta = ( (double) ctrl.targetWeight(kpart) *
                    (double) wpart                    -
                    (double) ctrl.targetWeight(ipart) *
                    (double) (p.partWeight(kpart) + wnode) );

          if ( delta > 0.0 )
          {
            gain = node.edges[ie].weight - node.idegree;

            if ( jpart < 0 || gain > maxGain )
            {
              jpart   = kpart;
              maxGain = gain;
            }
          }
        }
      }
      else if ( node.edegree > node.idegree )
      {
        // Optimize for the edge-cut weight.

        for ( ie = 0; ie < node.edgeCount; ie++ )
        {
          kpart = node.edges[ie].ipart;
          gain  = node.edges[ie].weight - node.idegree;

          bcond = ( p.partWeight(kpart) + wnode <=
                    ctrl.maxWeight(kpart) );

          if ( gain > maxGain && bcond )
          {
            jpart   = kpart;
            maxGain = gain;
          }
        }
      }

      if ( jpart >= 0 )
      {
        b.remapNode ( queue, inode, jpart, mask );
        moveCount++;
      }
    }

    if ( moveCount == 0 )
    {
      break;
    }
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Optimizer> GreedyOptimizer::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return jem::newInstance<Self> ( name, Random::get( globdat ) );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void GreedyOptimizer::declare ()
{
  OptimizerFactory::declare ( TYPE_NAME,  & makeNew );
  OptimizerFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( graph )
