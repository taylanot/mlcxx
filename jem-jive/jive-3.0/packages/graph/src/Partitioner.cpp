
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/RuntimeException.h>
#include <jive/graph/Partition.h>
#include <jive/graph/PartitionControl.h>
#include <jive/graph/Partitioner.h>


JEM_DEFINE_CLASS( jive::graph::Partitioner );


JIVE_BEGIN_PACKAGE( graph )


using jem::newInstance;


//=======================================================================
//   class Partitioner
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Partitioner::Partitioner ( const String& name ) :

  Super ( name )

{}


Partitioner::~Partitioner ()
{}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void Partitioner::configure ( const Properties& props )
{}


void Partitioner::getConfig ( const Properties& props ) const
{}


//-----------------------------------------------------------------------
//   partition
//-----------------------------------------------------------------------


void Partitioner::partition

  ( Partition&      part,
    const Control&  ctrl )

{
  const idx_t  nodeCount = part.nodeCount ();
  const idx_t  partCount = ctrl.partCount ();


  part.setPartCount ( partCount );

  // Handle the trivial cases:

  if ( nodeCount == 0 || partCount == 0 )
  {
    return;
  }

  if ( partCount == 1 )
  {
    part.mapUnmappedNodes ( 0 );

    return;
  }

  if ( partCount >= nodeCount )
  {
    for ( idx_t inode = 0; inode < nodeCount; inode++ )
    {
      part.mapNode ( inode, inode );
    }

    return;
  }

  if ( partCount == 2 )
  {
    bisect_    ( part, ctrl );
  }
  else
  {
    partition_ ( part, ctrl );
  }

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    if ( ! part.isMapped( inode ) )
    {
      throw jem::RuntimeException (
        getContext (),
        String::format (
          "graph node %d has not been mapped to a partition", inode
        )
      );
    }
  }
}


void Partitioner::partition

  ( const IdxVector&  nodeMap,
    const WGraph&     graph,
    idx_t             partCount )

{
  JEM_PRECHECK ( partCount >= 0 );

  Ref<Control>  ctrl =

    newInstance<Control> ( partCount, graph.weight() );

  partition ( nodeMap, graph, *ctrl );
}


void Partitioner::partition

  ( const IdxVector&  nodeMap,
    const WGraph&     graph,
    const Control&    ctrl )

{
  JEM_PRECHECK ( graph.nodeCount() == nodeMap.size() );

  const idx_t     nodeCount = graph.nodeCount ();

  Ref<Partition>  part = newInstance<Partition> ( graph );


  partition ( *part, ctrl );

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    nodeMap[inode] = part->nodeMap ( inode );
  }
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Partitioner::getContext () const
{
  return NamedObject::makeContext ( "partitioner", myName_ );
}


//-----------------------------------------------------------------------
//   bisect_
//-----------------------------------------------------------------------


void Partitioner::bisect_

  ( Partition&      part,
    const Control&  ctrl )

{
  partition_ ( part, ctrl );
}


JIVE_END_PACKAGE( graph )
