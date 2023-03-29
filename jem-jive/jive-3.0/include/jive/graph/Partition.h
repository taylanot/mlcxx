
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

#ifndef JIVE_GRAPH_PARTITION_H
#define JIVE_GRAPH_PARTITION_H

#include <jem/base/Object.h>
#include <jive/graph/WGraph.h>


JIVE_BEGIN_PACKAGE( graph )


//-----------------------------------------------------------------------
//   class Partition
//-----------------------------------------------------------------------


class Partition : public Object
{
 public:

  typedef Partition         Self;
  typedef Object            Super;


  explicit                  Partition

    ( const WGraph&           graph,
      idx_t                   partCount = 0 );

  inline idx_t              nodeCount       () const;
  inline idx_t              edgeCount       () const;
  inline idx_t              partCount       () const;
  inline idx_t              graphWeight     () const;
  inline idx_t              maxNodeDegree   () const;
  inline const WGraph&      getGraph        () const;

  void                      setPartCount

    ( idx_t                    pcount );

  inline void               mapNode

    ( idx_t                   inode,
      idx_t                   ipart );

  void                      mapNodes

    ( const IdxVector&        nmap );

  void                      mapUnmappedNodes

    ( idx_t                   ipart );

  inline void               unmapNode

    ( idx_t                   inode );

  void                      unmapNodes      ();

  inline void               remapNode

    ( idx_t                   inode,
      idx_t                   ipart );

  inline bool               isMapped

    ( idx_t                   inode )          const;

  inline bool               notMapped

    ( idx_t                   inode )          const;

  inline idx_t                nodeMap

    ( idx_t                   inode )          const;

  inline idx_t                partWeight

    ( idx_t                   ipart )          const;

  WGraph                    getPart

    ( const IdxVector&        inodes,
      idx_t                   ipart )          const;

  WGraph                    getSuperGraph   () const;

  void                      mapFromSuper

    ( const Partition&        super );


 protected:

  virtual                  ~Partition       ();


 private:

  WGraph                    graph_;

  IdxVector                 nodeMap_;
  IdxVector                 partWeights_;

  idx_t                     graphWeight_;
  idx_t                     maxDegree_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   nodeCount
//-----------------------------------------------------------------------


inline idx_t Partition::nodeCount () const
{
  return graph_.nodeCount ();
}


//-----------------------------------------------------------------------
//   edgeCount
//-----------------------------------------------------------------------


inline idx_t Partition::edgeCount () const
{
  return graph_.edgeCount ();
}


//-----------------------------------------------------------------------
//   partCount
//-----------------------------------------------------------------------


inline idx_t Partition::partCount () const
{
  return partWeights_.size ();
}


//-----------------------------------------------------------------------
//   graphWeight
//-----------------------------------------------------------------------


inline idx_t Partition::graphWeight () const
{
  return graphWeight_;
}


//-----------------------------------------------------------------------
//   maxNodeDegree
//-----------------------------------------------------------------------


inline idx_t Partition::maxNodeDegree () const
{
  return maxDegree_;
}


//-----------------------------------------------------------------------
//   getGraph
//-----------------------------------------------------------------------


inline const WGraph& Partition::getGraph () const
{
  return graph_;
}


//-----------------------------------------------------------------------
//   mapNode
//-----------------------------------------------------------------------


inline void Partition::mapNode

  ( idx_t  inode,
    idx_t  ipart )

{
  JEM_ASSERT ( notMapped( inode ) );

  nodeMap_[inode]      = ipart;
  partWeights_[ipart] += graph_.nodeWeights[inode];
}


//-----------------------------------------------------------------------
//   unmapNode
//-----------------------------------------------------------------------


inline void Partition::unmapNode ( idx_t inode )
{
  JEM_ASSERT ( isMapped( inode ) );

  const idx_t  ipart   = nodeMap_[inode];

  nodeMap_[inode]      = -1;
  partWeights_[ipart] -= graph_.nodeWeights[inode];
}


//-----------------------------------------------------------------------
//   remapNode
//-----------------------------------------------------------------------


inline void Partition::remapNode

  ( idx_t  inode,
    idx_t  ipart )

{
  JEM_ASSERT ( isMapped( inode ) );

  const idx_t  jpart   = nodeMap_[inode];
  const idx_t  w       = graph_.nodeWeights[inode];

  nodeMap_[inode]      = ipart;
  partWeights_[jpart] -= w;
  partWeights_[ipart] += w;
}


//-----------------------------------------------------------------------
//   isMapped
//-----------------------------------------------------------------------


inline bool Partition::isMapped ( idx_t inode ) const
{
  return (nodeMap_[inode] >= 0);
}


//-----------------------------------------------------------------------
//   notMapped
//-----------------------------------------------------------------------


inline bool Partition::notMapped ( idx_t inode ) const
{
  return (nodeMap_[inode] < 0);
}


//-----------------------------------------------------------------------
//   nodeMap
//-----------------------------------------------------------------------


inline idx_t Partition::nodeMap ( idx_t inode ) const
{
  return nodeMap_[inode];
}


//-----------------------------------------------------------------------
//   partWeight
//-----------------------------------------------------------------------


inline idx_t Partition::partWeight ( idx_t ipart ) const
{
  return partWeights_[ipart];
}


JIVE_END_PACKAGE( graph )

#endif
