
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

#ifndef JIVE_GRAPH_GRAPH_H
#define JIVE_GRAPH_GRAPH_H

#include <jive/Array.h>
#include <jive/SparseMatrix.h>


JIVE_BEGIN_PACKAGE( graph )


//-----------------------------------------------------------------------
//   class Graph
//-----------------------------------------------------------------------


class Graph
{
 public:

  typedef Graph             Self;


  inline                    Graph             ();

  inline                    Graph

    ( const IdxVector&        xadj,
      const IdxVector&        adjncy );

  inline                    Graph

    ( const Graph&            rhs );

  inline Graph&             operator =

    ( const Graph&            rhs );

  inline void               swap

    ( Graph&                  rhs );

  bool                      isValid           () const;
  inline idx_t              nodeCount         () const;
  inline idx_t              edgeCount         () const;

  inline idx_t              nodeEdgeCount

    ( idx_t                   inode )            const;

  idx_t                     maxNodeEdgeCount  () const;

  Graph                     getPart

    ( const IdxVector&        inodes )           const;


 public:

  IdxVector                 xadj;
  IdxVector                 adjncy;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void                 swap

  ( Graph&                    lhs,
    Graph&                    rhs );

Graph                       toGraph

  ( const SparseStruct&       s );






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Graph
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Graph::Graph ()
{}


inline Graph::Graph

  ( const IdxVector&  xad,
    const IdxVector&  adj ) :

    xadj   ( xad ),
    adjncy ( adj )

{}


inline Graph::Graph ( const Graph& rhs ) :

  xadj   (   rhs.xadj ),
  adjncy ( rhs.adjncy )

{}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


inline Graph& Graph::operator = ( const Graph& rhs )
{
  xadj  .ref (   rhs.xadj );
  adjncy.ref ( rhs.adjncy );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void Graph::swap ( Graph& rhs )
{
  xadj  .swap (   rhs.xadj );
  adjncy.swap ( rhs.adjncy );
}


//-----------------------------------------------------------------------
//   nodeCount
//-----------------------------------------------------------------------


inline idx_t Graph::nodeCount () const
{
  return (xadj.size() - 1);
}


//-----------------------------------------------------------------------
//   edgeCount
//-----------------------------------------------------------------------


inline idx_t Graph::edgeCount () const
{
  return adjncy.size ();
}


//-----------------------------------------------------------------------
//   nodeEdgeCount
//-----------------------------------------------------------------------

inline idx_t Graph::nodeEdgeCount ( idx_t inode ) const
{
  return (xadj[inode + 1] - xadj[inode]);
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void swap ( Graph& lhs, Graph& rhs )
{
  lhs.swap ( rhs );
}



JIVE_END_PACKAGE( graph )

#endif
