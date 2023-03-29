
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
#include <jem/numeric/utilities.h>
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
#include <jive/graph/FMOptimizer.h>


JEM_DEFINE_CLASS( jive::graph::FMOptimizer );


JIVE_BEGIN_PACKAGE( graph )


using jem::io::Writer;


//=======================================================================
//   class FMOptimizer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  FMOptimizer::TYPE_NAME    = "FM";
const int    FMOptimizer::MAX_PASSES   = 8;
const int    FMOptimizer::MAX_ROLLBACK = 50;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


FMOptimizer::FMOptimizer

  ( const String&       name,
    const Ref<Random>&  rand ) :

  Super        (         name ),
  rand_        (         rand ),
  maxPasses_   (   MAX_PASSES ),
  maxRollback_ ( MAX_ROLLBACK )

{
  if ( ! rand_ )
  {
    rand_ = jem::newInstance<Random> ();
  }
}


FMOptimizer::~FMOptimizer ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void FMOptimizer::configure ( const Properties& props )
{
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( maxPasses_,
                   PropNames::MAX_PASSES,
                   0, maxOf( maxPasses_ ) );
    myProps.find ( maxRollback_,
                   PropNames::MAX_ROLLBACK,
                   0, maxOf( maxRollback_ ) );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void FMOptimizer::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set  ( PropNames::MAX_PASSES,   maxPasses_   );
  myProps.set  ( PropNames::MAX_ROLLBACK, maxRollback_ );
}


//-----------------------------------------------------------------------
//   optimize
//-----------------------------------------------------------------------


void FMOptimizer::optimize

  ( Partition&      p,
    const Control&  ctrl )

{
  using jem::io::endl;

  Writer&  debug = jem::System::debug ( myName_ );

  idx_t    cutWeight[2];


  if ( p.partCount() == 2 )
  {
    BisectionBorder  border ( &p );

    cutWeight[0] = border.cutWeight ();

    Balancer::balance ( border, ctrl, 2 * maxPasses_ );
    optimize          ( border, *rand_, ctrl, maxPasses_, maxRollback_ );

    cutWeight[1] = border.cutWeight ();
  }
  else
  {
    PartitionBorder  border ( &p );

    cutWeight[0] = border.cutWeight ();

    Balancer::balance ( border, *rand_, ctrl, 2 * maxPasses_ );
    optimize          ( border, *rand_, ctrl, maxPasses_, maxRollback_ );

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


void FMOptimizer::optimize

  ( BisectionBorder&  b,
    Random&           rand,
    const Control&    ctrl,
    int               maxPasses,
    int               maxRollback )

{
  using jem::numeric::abs;
  typedef BisectionBorder::Node  Node;

  Partition&    p  = * b.getPartition ();
  const WGraph  gr =   p.getGraph     ();

  const idx_t   nodeCount = p   .nodeCount    ();
  const idx_t   wtarget0  = ctrl.targetWeight (0);

  BoolVector    mask      ( nodeCount );
  IdxVector     inodes    ( nodeCount );
  IdxVector     imoves    ( nodeCount );

  PQueue        queue[2];
  Node          node;

  idx_t         ub,    minUb,  maxUb;
  idx_t         wcut,  iniCut, minCut;
  idx_t         wnode;
  idx_t         gain;

  idx_t         inodeCount;
  idx_t         moveCount;
  int           rollCount;

  int           ipass;
  idx_t         inode;
  idx_t         ipart;
  idx_t         i;


  queue[0].init ( nodeCount, p.maxNodeDegree() );
  queue[1].init ( nodeCount, p.maxNodeDegree() );

  mask  = true;

  maxUb = ( 1 + p.graphWeight() / nodeCount +
            abs ( p.partWeight(0) - wtarget0 ) );

  for ( ipass = 0; ipass < maxPasses; ipass++ )
  {
    minUb      = abs ( p.partWeight(0) - wtarget0 );
    iniCut     = b.cutWeight ();
    minCut     = iniCut;
    wcut       = iniCut;
    rollCount  = 0;
    moveCount  = 0;
    inodeCount = b.getBorderNodes ( inodes );

    queue[0].clear ();
    queue[1].clear ();

    randomize ( inodes[slice(BEGIN,inodeCount)], rand );

    for ( i = 0; i < inodeCount; i++ )
    {
      inode = inodes[i];
      ipart = p.nodeMap(inode);

      b.getNode           ( node,  inode );
      queue[ipart].insert ( inode, node.edegree - node.idegree );
    }

    for ( i = 0; i < nodeCount; i++ )
    {
      if ( p.partWeight(0) >= wtarget0 )
      {
        ipart = 0;
      }
      else
      {
        ipart = 1;
      }

      if ( queue[ipart].size() == 0 )
      {
        break;
      }

      inode = queue[ipart].popFront ();

      wnode = gr.nodeWeights[inode];

      b.getNode ( node, inode );

      gain        = node.edegree - node.idegree;
      wcut       -= gain;
      mask[inode] = false;

      imoves[moveCount++] = inode;

      if ( ipart == 0 )
      {
        ub = abs ( p.partWeight(0) - wnode - wtarget0 );
      }
      else
      {
        ub = abs ( p.partWeight(0) + wnode - wtarget0 );
      }

      if ( (wcut <  minCut && ub <= maxUb) ||
           (wcut <= minCut && ub <  minUb) )
      {
        minCut    = wcut;
        minUb     = ub;
        rollCount = 0;
      }
      else
      {
        rollCount++;
      }

      b.remapNode ( queue[0], queue[1], inode, mask );

      if ( rollCount >= maxRollback )
      {
        break;
      }
    }

    for ( i = 0; i < moveCount; i++ )
    {
      mask[imoves[i]] = true;
    }

    for ( i = 0; i < rollCount; i++ )
    {
      inode = imoves[--moveCount];

      b.remapNode ( inode );
    }

    if ( moveCount == 0 || minCut == iniCut )
    {
      break;
    }
  }
}


//-----------------------------------------------------------------------
//   optimize (partition)
//-----------------------------------------------------------------------


void FMOptimizer::optimize

  ( PartitionBorder&  b,
    Random&           rand,
    const Control&    ctrl,
    int               maxPasses,
    int               maxRollback )

{
  typedef PartitionBorder::Node  Node;


  Partition&    p  = * b.getPartition ();
  const WGraph  gr =   p.getGraph     ();

  const idx_t   nodeCount = p.nodeCount ();

  BoolVector    mask      ( nodeCount );
  IdxVector     inodes    ( nodeCount );
  IdxVector     imoves    ( nodeCount );
  IdxVector     iparts    ( nodeCount );

  PQueue        queue     ( nodeCount, p.maxNodeDegree() );
  Node          node;

  bool          bcond;
  idx_t         wcut, iniCut, minCut;
  idx_t         gain, maxGain;
  idx_t         wnode;
  idx_t         wpart;

  idx_t         inodeCount;
  idx_t         moveCount;
  int           rollCount;

  int           ipass;
  idx_t         inode;
  idx_t         ipart;
  idx_t         jpart;
  idx_t         kpart;
  idx_t         i, ie;


  mask = true;

  for ( ipass = 0; ipass < maxPasses; ipass++ )
  {
    iniCut     = b.cutWeight ();
    minCut     = iniCut;
    wcut       = iniCut;
    rollCount  = 0;
    moveCount  = 0;
    inodeCount = b.getBorderNodes ( inodes );

    queue.clear ();

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

      for ( ie = 0; ie < node.edgeCount; ie++ )
      {
        kpart = node.edges[ie].ipart;
        gain  = node.edges[ie].weight - node.idegree;

        bcond = ( p.partWeight(kpart) + wnode <=
                  ctrl.maxWeight(kpart) );

        if ( bcond && (jpart < 0 || gain > maxGain) )
        {
          jpart   = kpart;
          maxGain = gain;
        }
      }

      if ( jpart < 0 )
      {
        continue;
      }

      wcut       -= maxGain;
      mask[inode] = false;

      imoves[moveCount] = inode;
      iparts[moveCount] = ipart;

      moveCount++;

      if ( wcut < minCut )
      {
        minCut   = wcut;
        rollCount = 0;
      }
      else
      {
        rollCount++;
      }

      b.remapNode ( queue, inode, jpart, mask );

      if ( rollCount >= maxRollback )
      {
        break;
      }
    }

    for ( i = 0; i < moveCount; i++ )
    {
      mask[imoves[i]] = true;
    }

    for ( i = 0; i < rollCount; i++ )
    {
      moveCount--;
      b.remapNode ( imoves[moveCount], iparts[moveCount] );
    }

    if ( moveCount == 0 || minCut == iniCut )
    {
      break;
    }
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Optimizer> FMOptimizer::makeNew

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


void FMOptimizer::declare ()
{
  OptimizerFactory::declare ( TYPE_NAME,  & makeNew );
  OptimizerFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( graph )
