
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
#include <jive/graph/FMOptimizer.h>
#include <jive/graph/GreedyOptimizer.h>
#include <jive/graph/GrowBisectioner.h>
#include <jive/graph/RBPartitioner.h>
#include <jive/graph/MLPartitioner.h>


JEM_DEFINE_CLASS( jive::graph::MLPartitioner );


JIVE_BEGIN_PACKAGE( graph )


using jem::newInstance;
using jive::util::joinNames;


//=======================================================================
//   class MLPartitioner
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MLPartitioner::TYPE_NAME = "ML";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MLPartitioner::MLPartitioner

  ( const String&            name,
    const Ref<Random>&       rand,
    const Ref<Partitioner>&  parter,
    const Ref<Optimizer>&    optimizer ) :

    Super      ( name      ),
    rand_      ( rand      ),
    parter_    ( parter    ),
    optimizer_ ( optimizer )

{
  if ( ! rand_ )
  {
    rand_ = newInstance<Random> ();
  }

  if ( ! parter_ )
  {
    String  pname = joinNames ( myName_, PropNames::PARTITIONER );

    parter_ = newInstance<RBPartitioner> (
      pname,
      rand_,
      newInstance<GrowBisectioner> (
        joinNames ( pname, PropNames::PARTITIONER )
      ),
      newInstance<FMOptimizer> (
        joinNames ( pname, PropNames::OPTIMIZER )
      )
    );
  }

  if ( ! optimizer_ )
  {
    optimizer_ = newInstance<GreedyOptimizer> (
      joinNames ( myName_, PropNames::OPTIMIZER )
    );
  }
}


MLPartitioner::~MLPartitioner ()
{}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void MLPartitioner::configure ( const Properties& props )
{
  parter_   ->configure ( props );
  optimizer_->configure ( props );
}


void MLPartitioner::getConfig ( const Properties& props ) const
{
  parter_   ->getConfig ( props );
  optimizer_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Partitioner> MLPartitioner::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Random>       rand = Random::get ( globdat );

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
    String  pname =

      joinNames ( name, PropNames::PARTITIONER );

    String  oname =

      joinNames ( name, PropNames::OPTIMIZER );

    conf.makeProps(name) .set ( PropNames::TYPE,
                                RBPartitioner::TYPE_NAME );
    conf.makeProps(pname).set ( PropNames::TYPE,
                                GrowBisectioner::TYPE_NAME );
    conf.makeProps(oname).set ( PropNames::TYPE,
                                FMOptimizer::TYPE_NAME );

    parter = newInstance<RBPartitioner> (
      name,
      rand,
      GrowBisectioner::makeNew ( pname, conf, props, globdat ),
      FMOptimizer    ::makeNew ( oname, conf, props, globdat )
    );
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

  return newInstance<Self> ( myName, rand, parter, optimizer );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MLPartitioner::declare ()
{
  PartitionerFactory::declare ( TYPE_NAME,  & makeNew );
  PartitionerFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   partition_
//-----------------------------------------------------------------------


void MLPartitioner::partition_

  ( Partition&      part,
    const Control&  ctrl )

{
  using jem::io::endl;
  using jem::staticCast;
  using jem::System;
  using jem::util::ObjFlex;

  Ref<Partition>  p0;
  Ref<Partition>  p1;

  ObjFlex         levels;


  p0 = & part;

  levels.reserve  ( 16 );
  levels.pushBack ( p0 );

  while ( p0->nodeCount() > 16 * ctrl.partCount() )
  {
    p1.swap ( p0 );

    p0 = coarsen_  ( *p1, *rand_, levels.size(), ctrl );

    levels.pushBack ( p0 );

    print ( System::debug( myName_ ), myName_,
            " : level ",       levels.size(),
            ", graph size = ", p0->nodeCount(), endl );

    if ( p1->nodeCount() < (idx_t) (1.5 * (double) p0->nodeCount()) )
    {
      break;
    }
  }

  print ( System::info( myName_ ), myName_,
          " : level count = ",       levels.size(),
          ", coarsest graph size = ", p0->nodeCount(), endl );

  parter_->partition ( *p0, ctrl );

  // Uncoarsen the graph

  while ( levels.size() > 1 )
  {
    levels.popBack ();

    p1 = static_cast<Partition*> ( levels.back() );

    p1->mapFromSuper ( *p0 );

    optimizer_->optimize ( *p1, ctrl );

    p0.swap ( p1 );
  }
}


//-----------------------------------------------------------------------
//   coarsen_
//-----------------------------------------------------------------------


Ref<Partition> MLPartitioner::coarsen_

  ( Partition&      p,
    Random&         rand,
    idx_t           level,
    const Control&  ctrl )

{
  using jem::iarray;

  const idx_t    nodeCount = p.nodeCount ();

  const WGraph&  gr        = p.getGraph  ();

  IntMatrix      ibuf  ( nodeCount, 2 );

  IdxVector      nmap  ( ibuf[0] );
  IdxVector      iperm ( ibuf[1] );

  double         gain, maxGain;

  idx_t          pcount;
  idx_t          inode;
  idx_t          jnode;
  idx_t          i, j, k, n;


  iperm = iarray ( nodeCount );

  if ( level > 2 || nodeCount <= 128 * ctrl.partCount() )
  {
    // Give little nodes a chance to gain some weight

    jem::sort ( iperm, gr.nodeWeights );
  }
  else
  {
    randomize ( iperm, rand );
  }

  nmap   = -1;
  pcount =  0;

  for ( i = 0; i < nodeCount; i++ )
  {
    inode = iperm[i];

    if ( nmap[inode] >= 0 )
    {
      continue;
    }

    n       =  gr.xadj[inode + 1];
    jnode   = -1;
    maxGain = -1.0;

    for ( j = gr.xadj[inode]; j < n; j++ )
    {
      k = gr.adjncy[j];

      if ( nmap[k] < 0 )
      {
        gain = (double) gr.edgeWeights[j] /
                       ((double) gr.nodeWeights[k] + 1.0);

        if ( gain > maxGain )
        {
          jnode   = k;
          maxGain = gain;
        }
      }
    }

    nmap[inode] = pcount;

    if ( jnode >= 0 )
    {
      nmap[jnode] = pcount;
    }

    pcount++;
  }

  p.setPartCount ( pcount );
  p.mapNodes     ( nmap   );

  return newInstance<Partition> ( p.getSuperGraph() );
}


JIVE_END_PACKAGE( graph )
