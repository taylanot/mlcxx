
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

#ifndef JIVE_GRAPH_PARTITIONBORDER_H
#define JIVE_GRAPH_PARTITIONBORDER_H

#include <jem/base/TypeTraits.h>
#include <jem/base/Collectable.h>
#include <jem/util/Flex.h>
#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( graph )


class PQueue;
class Partition;


//-----------------------------------------------------------------------
//   class PartitionBorder
//-----------------------------------------------------------------------


class PartitionBorder
{
 public:

  typedef PartitionBorder   Self;

  struct                    Edge
  {
    idx_t                     ipart;
    idx_t                     weight;
  };


  struct                    Node
  {
    idx_t                     idegree;
    idx_t                     edegree;
    idx_t                     edgeCount;
    const Edge*               edges;
  };


  explicit                  PartitionBorder

    ( const Ref<Partition>&   p );

  inline idx_t              size            () const;
  inline idx_t              cutWeight       () const;
  inline Partition*         getPartition    () const;

  idx_t                     getBorderNodes

    ( const IdxVector&        inodes )         const;

  inline void               getNode

    ( Node&                   node,
      idx_t                   inode )          const;

  void                      remapNode

    ( idx_t                   inode,
      idx_t                   ipart );

  void                      remapNode

    ( PQueue&                 q,
      idx_t                   inode,
      idx_t                   ipart,
      const BoolVector&       mask );

  void                      remapNodes

    ( const IdxVector&        nmap );


 private:

                            PartitionBorder ( const Self& );
  Self&                     operator =      ( const Self& );


 private:

  struct                    Node_
  {
    idx_t                     index;
    idx_t                     idegree;
    idx_t                     edegree;
    idx_t                     edgeCount;
    Edge*                     edges;
  };


  static inline idx_t       findEdge_

    ( const Node_&            node,
      idx_t                   ipart );

  inline void               addBorderNode_

    ( idx_t                   inode );

  inline void               removeBorderNode_

    ( idx_t                   inode );


 private:

  Ref<Partition>            partition_;
  jem::util::Flex<Node_>    nodes_;
  jem::util::Flex<Edge>     edges_;
  IdxVector                 ibnodes_;
  idx_t                     bnodeCount_;
  idx_t                     cutWeight_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t PartitionBorder::size () const
{
  return bnodeCount_;
}


//-----------------------------------------------------------------------
//   cutWeight
//-----------------------------------------------------------------------


inline idx_t PartitionBorder::cutWeight () const
{
  return cutWeight_;
}


//-----------------------------------------------------------------------
//   getPartition
//-----------------------------------------------------------------------


inline Partition* PartitionBorder::getPartition () const
{
  return partition_.get ();
}


//-----------------------------------------------------------------------
//   getNode
//-----------------------------------------------------------------------


inline void PartitionBorder::getNode

  ( Node&  node,
    idx_t  inode ) const

{
  const Node_&  n = nodes_[inode];

  node.edegree   = n.edegree;
  node.idegree   = n.idegree;
  node.edgeCount = n.edgeCount;
  node.edges     = n.edges;
}


JIVE_END_PACKAGE( graph )

#endif
