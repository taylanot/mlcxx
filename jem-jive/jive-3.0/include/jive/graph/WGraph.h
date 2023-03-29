
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

#ifndef JIVE_GRAPH_WGRAPH_H
#define JIVE_GRAPH_WGRAPH_H

#include <jive/graph/Graph.h>


JIVE_BEGIN_PACKAGE( graph )


//-----------------------------------------------------------------------
//   class WGraph
//-----------------------------------------------------------------------


class WGraph : public Graph
{
 public:

  typedef WGraph            Self;
  typedef Graph             Super;


  inline                    WGraph        ();

                            WGraph

    ( const IdxVector&        xadj,
      const IdxVector&        adjncy );

  inline                    WGraph

    ( const IdxVector&        xadj,
      const IdxVector&        adjncy,
      const IdxVector&        nodeWeights,
      const IdxVector&        edgeWeights );

  inline                    WGraph

    ( const WGraph&           rhs );

  inline WGraph&            operator =

    ( const WGraph&           rhs );

  inline void               swap

    ( WGraph&                 rhs );

  bool                      isValid       () const;
  idx_t                     weight        () const;

  idx_t                     nodeDegree

    ( idx_t                   inode )        const;

  idx_t                     maxNodeDegree () const;

  WGraph                    getPart

    ( const IdxVector&        inodes )       const;


 public:

  IdxVector                 nodeWeights;
  IdxVector                 edgeWeights;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void                 swap

  ( WGraph&                   lhs,
    WGraph&                   rhs );

WGraph                      toGraph

  ( const SparseIdxMatrix&    matrix,
    const IdxVector&          nodeWeights );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class WGraph
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline WGraph::WGraph ()
{}


inline WGraph::WGraph

  ( const IdxVector&  xadj,
    const IdxVector&  adjncy,
    const IdxVector&  nweights,
    const IdxVector&  eweights ) :

    Super       ( xadj, adjncy ),
    nodeWeights ( nweights ),
    edgeWeights ( eweights )

{}


inline WGraph::WGraph ( const WGraph& rhs ) :

  Super       ( rhs ),
  nodeWeights ( rhs.nodeWeights ),
  edgeWeights ( rhs.edgeWeights )

{}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


inline WGraph& WGraph::operator = ( const WGraph& rhs )
{
  nodeWeights.ref ( rhs.nodeWeights );
  edgeWeights.ref ( rhs.edgeWeights );

  return static_cast<Self&> ( Super::operator = ( rhs ) );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void WGraph::swap ( WGraph& rhs )
{
  Super::swap ( rhs );

  nodeWeights.swap ( rhs.nodeWeights );
  edgeWeights.swap ( rhs.edgeWeights );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void swap ( WGraph& lhs, WGraph& rhs )
{
  lhs.swap ( rhs );
}


JIVE_END_PACKAGE( graph )

#endif
