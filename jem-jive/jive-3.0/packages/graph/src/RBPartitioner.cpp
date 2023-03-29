
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
#include <jem/base/array/utilities.h>
#include <jem/util/ObjFlex.h>
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
#include <jive/graph/RBPartitioner.h>


JEM_DEFINE_CLASS( jive::graph::RBPartitioner );


JIVE_BEGIN_PACKAGE( graph )


using jem::newInstance;
using jive::util::joinNames;


//=======================================================================
//   class RBPartitioner::Bisection_
//=======================================================================


class RBPartitioner::Bisection_ : public Partition
{
 public:

  typedef Bisection_      Self;
  typedef Partition       Super;


  explicit                Bisection_

    ( const WGraph&         graph );

                          Bisection_

    ( const WGraph&         graph,
      const IdxVector&      inodes );


  void                    getParts

     ( Ref<Self>&           p0,
       Ref<Self>&           p1 )      const;


 public:

  IdxVector               nodeIDs;

};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


RBPartitioner::Bisection_::Bisection_

  ( const WGraph&  graph ) :

    Super   ( graph, 2 ),
    nodeIDs ( graph.nodeCount() )

{
  nodeIDs = jem::iarray ( nodeCount() );
}


RBPartitioner::Bisection_::Bisection_

  ( const WGraph&     graph,
    const IdxVector&  inodes ) :

    Super   ( graph, 2 ),
    nodeIDs ( inodes )

{}


//-----------------------------------------------------------------------
//   getParts
//-----------------------------------------------------------------------


void RBPartitioner::Bisection_::getParts

  ( Ref<Self>&  p0,
    Ref<Self>&  p1 ) const

{
  JEM_PRECHECK ( partCount() == 2 );

  IdxVector  inodes ( nodeCount() );

  WGraph     gr     = getPart ( inodes, 0 );

  idx_t      n      = gr.nodeCount ();


  for ( idx_t i = 0; i < n; i++ )
  {
    inodes[i] = nodeIDs[inodes[i]];
  }

  p0 = newInstance<Self> ( gr, inodes[slice(BEGIN,n)].clone() );

  gr = getPart ( inodes, 1 );

  n  = gr.nodeCount ();

  for ( idx_t i = 0; i < n; i++ )
  {
    inodes[i] = nodeIDs[inodes[i]];
  }

  p1 = newInstance<Self> ( gr, inodes[slice(BEGIN,n)].clone() );
}


//=======================================================================
//   class RBPartitioner
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  RBPartitioner::TYPE_NAME = "RB";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


RBPartitioner::RBPartitioner

  ( const String&            name,
    const Ref<Random>&       rand,
    const Ref<Partitioner>&  parter,
    const Ref<Optimizer>&    optimizer ) :

    Super      ( name      ),
    parter_    ( parter    ),
    optimizer_ ( optimizer )

{
  Ref<Random>  rnd = rand;

  if ( ! rnd )
  {
    rnd = newInstance<Random> ();
  }

  if ( ! parter_ )
  {
    parter_ = newInstance<GreedyPartitioner> (
      joinNames ( myName_, PropNames::PARTITIONER ),
      rnd
    );
  }

  if ( ! optimizer_ )
  {
    optimizer_ = newInstance<GreedyOptimizer> (
      joinNames ( myName_, PropNames::OPTIMIZER ),
      rnd
    );
  }
}


RBPartitioner::~RBPartitioner ()
{}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void RBPartitioner::configure ( const Properties& props )
{
  parter_   ->configure ( props );
  optimizer_->configure ( props );
}


void RBPartitioner::getConfig ( const Properties& props ) const
{
  parter_   ->getConfig ( props );
  optimizer_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Partitioner> RBPartitioner::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Partitioner>  parter;
  Ref<Optimizer>    optimizer;
  String            name;


  name = joinNames ( myName, PropNames::PARTITIONER );

  if ( props.contains( name ) )
  {
    parter = PartitionerFactory::newInstance ( name,  conf,
                                               props, globdat );
  }
  else
  {
    conf.makeProps(name).set ( PropNames::TYPE,
                               GreedyPartitioner::TYPE_NAME );

    parter = GreedyPartitioner::makeNew ( name,  conf,
                                          props, globdat );
  }

  name = joinNames ( myName, PropNames::OPTIMIZER );

  if ( props.contains( name ) )
  {
    optimizer = newOptimizer ( name, conf, props, globdat );
  }
  else
  {
    conf.makeProps(name).set ( PropNames::TYPE,
                               GreedyOptimizer::TYPE_NAME );

    optimizer = GreedyOptimizer::makeNew ( name,  conf,
                                           props, globdat );
  }

  return newInstance<Self> ( myName, Random::get( globdat ),
                             parter, optimizer );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void RBPartitioner::declare ()
{
  PartitionerFactory::declare ( TYPE_NAME,  & makeNew );
  PartitionerFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   bisect_
//-----------------------------------------------------------------------


void RBPartitioner::bisect_

  ( Partition&      part,
    const Control&  ctrl )

{
  JEM_PRECHECK ( part.partCount() == 2 );

  parter_->partition ( part, ctrl );
}


//-----------------------------------------------------------------------
//   partition_
//-----------------------------------------------------------------------


void RBPartitioner::partition_

  ( Partition&      part,
    const Control&  ctrl )

{
  using jem::io::endl;
  using jem::staticCast;
  using jem::System;
  using jem::util::ObjFlex;

  ObjFlex          bList[2];
  ObjFlex          cList[2];

  Ref<Bisection_>  b, b0, b1;
  Ref<Control>     c, c0, c1;

  idx_t            depth;
  idx_t            n;


  for ( int i = 0; i < 2; i++ )
  {
    bList[i].reserve ( 16 );
    cList[i].reserve ( 16 );
  }

  b = newInstance<Bisection_> ( part.getGraph() );

  bList[0].pushBack ( b );
  cList[0].pushBack ( const_cast<Control*>( &ctrl ) );

  depth = 0;

  while ( bList[0].size() < ctrl.partCount() )
  {
    depth++;

    print ( System::debug( myName_ ), myName_,
            " : depth = ", depth, endl );

    bList[1].clear ();
    cList[1].clear ();

    for ( idx_t i = 0; i < bList[0].size(); i++ )
    {
      b = bList[0].getAs<Bisection_> ( i );
      c = cList[0].getAs<Control>    ( i );

      if ( c->partCount() < 2 )
      {
        bList[1].pushBack ( b );
        cList[1].pushBack ( c );
      }
      else
      {
        bisect_     ( *b, c0, c1, *c );
        b->getParts ( b0, b1 );

        bList[1].pushBack ( b0 );
        bList[1].pushBack ( b1 );
        cList[1].pushBack ( c0 );
        cList[1].pushBack ( c1 );
      }

      // Free some memory

      bList[0].set ( i, nullptr );
    }

    bList[0].swap ( bList[1] );
    cList[0].swap ( cList[1] );
  }

  // Merge all bisections into one partition

  for ( idx_t i = 0; i < bList[0].size(); i++ )
  {
    b = bList[0].getAs<Bisection_> ( i );
    n = b->nodeCount ();

    for ( idx_t inode = 0; inode < n; inode++ )
    {
      part.mapNode ( b->nodeIDs[inode], i );
    }
  }

  optimizer_->optimize ( part, ctrl );
}


//-----------------------------------------------------------------------
//   bisect_
//-----------------------------------------------------------------------


void RBPartitioner::bisect_

  ( Partition&      p,
    Ref<Control>&   c0,
    Ref<Control>&   c1,
    const Control&  ctrl ) const

{
  Vector        xwgt [ 2 ];
  IdxVector     iwgt ( 2 );

  Ref<Control>  c;

  double        scale;
  idx_t         offset[2];
  idx_t         m, n;


  m    = ctrl.partCount() / 2;
  n    = ctrl.partCount() - m;
  iwgt = 0;

  for ( idx_t i = 0; i < m; i++ )
  {
    iwgt[0] += ctrl.targetWeight ( i );
  }

  for ( idx_t i = m; i < ctrl.partCount(); i++ )
  {
    iwgt[1] += ctrl.targetWeight ( i );
  }

  c = newInstance<Control> ( iwgt, 0.0 );

  parter_->partition ( p, *c );

  xwgt[0].resize( m );
  xwgt[1].resize( n );

  offset[0] = 0;
  offset[1] = m;

  for ( int i = 0; i < 2; i++ )
  {
    if ( iwgt[i] > 0 )
    {
      scale = 1.0 / (double) iwgt[i];
    }
    else
    {
      scale = 0.0;
    }

    for ( idx_t j = 0; j < xwgt[i].size(); j++ )
    {
      xwgt[i][j] = scale * (double) ctrl.targetWeight ( j + offset[i] );
    }
  }

  c0 = newInstance<Control> ( xwgt[0],
                              p.partWeight( 0 ), ctrl.unbalance() );

  c1 = newInstance<Control> ( xwgt[1],
                              p.partWeight( 1 ), ctrl.unbalance() );
}


JIVE_END_PACKAGE( graph )
