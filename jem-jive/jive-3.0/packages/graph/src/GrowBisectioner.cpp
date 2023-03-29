
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Random.h>
#include <jive/graph/utilities.h>
#include <jive/graph/Names.h>
#include <jive/graph/PQueue.h>
#include <jive/graph/Partition.h>
#include <jive/graph/PartitionControl.h>
#include <jive/graph/BisectionBorder.h>
#include <jive/graph/PartitionerFactory.h>
#include <jive/graph/FMOptimizer.h>
#include <jive/graph/GrowBisectioner.h>


JEM_DEFINE_CLASS( jive::graph::GrowBisectioner );


JIVE_BEGIN_PACKAGE( graph )


using jem::newInstance;
using jive::util::joinNames;


//=======================================================================
//   class GrowBisectioner
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GrowBisectioner::TYPE_NAME  = "GB";
const int    GrowBisectioner::MAX_PASSES = 4;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GrowBisectioner::GrowBisectioner

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
    optimizer_ = newInstance<FMOptimizer> (
      joinNames  ( name, PropNames::OPTIMIZER )
    );
  }
}


GrowBisectioner::~GrowBisectioner ()
{}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void GrowBisectioner::configure ( const Properties& props )
{
  optimizer_->configure ( props );
}


void GrowBisectioner::getConfig ( const Properties& props ) const
{
  optimizer_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Partitioner> GrowBisectioner::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Optimizer>  optimizer;

  String  name = joinNames ( myName, PropNames::OPTIMIZER );


  if ( props.contains( name ) )
  {
    optimizer = newOptimizer ( name, conf, props, globdat );
  }
  else
  {
    conf.makeProps(name).set ( PropNames::TYPE,
                               FMOptimizer::TYPE_NAME );

    optimizer = FMOptimizer::makeNew ( name,  conf,
                                       props, globdat );
  }

  return newInstance<Self> ( myName, Random::get( globdat ),
                             optimizer );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void GrowBisectioner::declare ()
{
  PartitionerFactory::declare ( TYPE_NAME,  & makeNew );
  PartitionerFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   bisect_
//-----------------------------------------------------------------------


void GrowBisectioner::bisect_

  ( Partition&      p,
    const Control&  ctrl )

{
  using jem::System;

  JEM_PRECHECK ( p.partCount() == 2 );

  const idx_t  nodeCount = p.nodeCount ();

  IdxVector    nodeMap   ( nodeCount );

  idx_t        cutWgt;
  idx_t        minCut;

  idx_t        inode;
  idx_t        ipart;
  idx_t        jpart;
  int          ipass;


  // Select the smallest partition to build

  if ( ctrl.targetWeight(0) <= ctrl.targetWeight(1) )
  {
    ipart = 0;
  }
  else
  {
    ipart = 1;
  }

  jpart   = (ipart + 1) % 2;
  nodeMap = -1;
  ipass   =  0;
  minCut  =  0;

  do
  {
    // Map all nodes to the largest part

    p.unmapNodes ();

    for ( inode = 0; inode < nodeCount; inode++ )
    {
      p.mapNode ( inode, jpart );
    }

    inode = findFringeNode ( p.getGraph(), rand_->next( nodeCount ) );

    if ( inode < 0 )
    {
      return;
    }

    cutWgt = growPart_ ( p, ipart, inode, ctrl );

    if ( ipass == 0 || cutWgt < minCut )
    {
      minCut = cutWgt;

      for ( inode = 0; inode < nodeCount; inode++ )
      {
        nodeMap[inode] = p.nodeMap ( inode );
      }
    }

    print ( System::debug( myName_ ), myName_,
            " : pass ", ipass, ", cut weight = ", cutWgt, '\n' );

    ipass++;
  }
  while ( ipass < MAX_PASSES );

  p.mapNodes ( nodeMap );

  optimizer_->optimize ( p, ctrl );
}


//-----------------------------------------------------------------------
//   partition_
//-----------------------------------------------------------------------


void GrowBisectioner::partition_

  ( Partition&      p,
    const Control&  ctrl )

{
  throw jem::IllegalInputException (
    getContext (),
    String::format (
      "illegal number of partitions: %d; must be two",
      p.partCount ()
    )
  );
}


//-----------------------------------------------------------------------
//   growPart_
//-----------------------------------------------------------------------


idx_t GrowBisectioner::growPart_

  ( Partition&      p,
    idx_t           ipart,
    idx_t           seed,
    const Control&  ctrl )

{
  typedef BisectionBorder::Node  Node;

  const WGraph     gr        = p   .getGraph     ();
  const idx_t      nodeCount = p   .nodeCount    ();
  const idx_t      wtarget   = ctrl.targetWeight ( ipart );

  BisectionBorder  border    ( &p );

  PQueue           queue     ( nodeCount, p.maxNodeDegree() );
  IdxVector        inodes    ( nodeCount );
  BoolVector       mask      ( nodeCount );

  Node             node;

  idx_t            inodeCount;
  idx_t            moveCount;
  idx_t            maxMoves;

  idx_t            jpart;
  idx_t            inode;
  idx_t            i;


  mask       =  true;
  maxMoves   =  nodeCount - 1;
  moveCount  =  0;
  jpart      = (ipart + 1) % 2;
  mask[seed] = false;

  border.remapNode ( seed );
  moveCount++;

  while ( moveCount < maxMoves && p.partWeight( ipart ) < wtarget )
  {
    inodeCount = border.getBorderNodes ( inodes );

    for ( i = 0; i < inodeCount; i++ )
    {
      inode = inodes[i];

      if ( p.nodeMap( inode ) == jpart )
      {
        border.getNode ( node,  inode );
        queue .insert  ( inode, node.edegree - node.idegree );
      }
    }

    while ( queue.size()          > 0        &&
            moveCount             < maxMoves &&
            p.partWeight( ipart ) < wtarget )
    {
      inode = queue.popFront ();

      if ( p.nodeMap( inode ) == jpart )
      {
        mask[inode] = false;

        border.remapNode ( queue, inode, mask );
        moveCount++;
      }
    }

    if ( queue.size()          == 0        &&
         moveCount             <  maxMoves &&
         p.partWeight( ipart ) <  wtarget )
    {
      seed       = findFringeNode ( p.getGraph(), mask );
      mask[seed] = false;

      border.remapNode ( seed );
      moveCount++;
    }
  }

  return border.cutWeight ();
}


JIVE_END_PACKAGE( graph )
