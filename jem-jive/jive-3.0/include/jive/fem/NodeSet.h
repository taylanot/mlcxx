
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

#ifndef JIVE_FEM_NODESET_H
#define JIVE_FEM_NODESET_H

#include <jive/util/ItemSetSkin.h>
#include <jive/util/PointSet.h>
#include <jive/fem/import.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::util::Event;


//-----------------------------------------------------------------------
//   class NodeSet
//-----------------------------------------------------------------------


class NodeSet : public ItemSetSkin
{
 public:

  typedef NodeSet           Self;
  typedef ItemSetSkin       Super;
  typedef util::PointSet    Data;


  inline                    NodeSet       ()       noexcept;

  explicit inline           NodeSet

    ( const Ref<Data>&        nodes )              noexcept;

  explicit inline           NodeSet

    ( Ref<Data>&&             nodes )              noexcept;

  inline Event
    < Data& >&              newGeomEvent  () const;
  inline Event
    < idx_t, Data& >&       newRankEvent  () const;

  inline Data*              getData       () const noexcept;
  inline idx_t              rank          () const;

  inline String             getNodePath

    ( idx_t                   nodeID )       const;

  inline idx_t              findNode

    ( idx_t                   nodeID )       const;

  inline idx_t              getNodeID

    ( idx_t                   inode )        const;

  inline void               getNodeCoords

    ( const Vector&           coords,
      idx_t                   inode )        const;

  inline void               getCoords

    ( const Matrix&           coords )       const;

  inline void               getSomeCoords

    ( const Matrix&           coords,
      const IdxVector&        inodes )       const;

  inline Matrix             toMatrix      () const;

  inline void               checkRank

    ( const String&           context,
      idx_t                   rank )         const;

  void                      checkRank

    ( const String&           context,
      idx_t                   minRank,
      idx_t                   maxRank )      const;

  void                      store

    ( const Properties&       globdat )      const;

  void                      store

    ( const String&           name,
      const Properties&       globdat )      const;

  static Self               find

    ( const Properties&       globdat );

  static Self               find

    ( const String&           name,
      const Properties&       globdat );

  static Self               get

    ( const Properties&       globdat,
      const String&           context );

  static Self               get

    ( const String&           name,
      const Properties&       globdat,
      const String&           context );


 protected:

  inline Data&              nodes_        () const;

};








//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class NodeSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline NodeSet::NodeSet () noexcept
{}


inline NodeSet::NodeSet ( const Ref<Data>& nodes ) noexcept :

  Super ( nodes )

{}


inline NodeSet::NodeSet ( Ref<Data>&& nodes ) noexcept :

  Super ( std::move( nodes ) )

{}


//-----------------------------------------------------------------------
//   newGeomEvent
//-----------------------------------------------------------------------


inline Event < util::PointSet& >& NodeSet::newGeomEvent () const
{
  return nodes_().newGeomEvent;
}


//-----------------------------------------------------------------------
//   newRankEvent
//-----------------------------------------------------------------------


inline Event < idx_t, util::PointSet& >&

  NodeSet::newRankEvent () const

{
  return nodes_().newRankEvent;
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline NodeSet::Data* NodeSet::getData () const noexcept
{
  return jem::staticCast<Data*> ( data_ );
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


inline idx_t NodeSet::rank () const
{
  return nodes_().rank ();
}


//-----------------------------------------------------------------------
//   getNodePath
//-----------------------------------------------------------------------


inline String NodeSet::getNodePath ( idx_t nodeID ) const
{
  return nodes_().getItemPath ( nodeID );
}


//-----------------------------------------------------------------------
//   findNode
//-----------------------------------------------------------------------


inline idx_t NodeSet::findNode ( idx_t nodeID ) const
{
  return nodes_().findItem ( nodeID );
}


//-----------------------------------------------------------------------
//   getNodeID
//-----------------------------------------------------------------------


inline idx_t NodeSet::getNodeID ( idx_t inode  ) const
{
  return nodes_().getItemID ( inode );
}


//-----------------------------------------------------------------------
//   getNodeCoords
//-----------------------------------------------------------------------


inline void NodeSet::getNodeCoords

  ( const Vector&  coords,
    idx_t          inode  ) const

{
  nodes_().getPointCoords ( coords, inode );
}



//-----------------------------------------------------------------------
//   getCoords
//-----------------------------------------------------------------------


inline void NodeSet::getCoords

  ( const Matrix& coords ) const

{
  nodes_().getCoords ( coords );
}


//-----------------------------------------------------------------------
//   getSomeCoords
//-----------------------------------------------------------------------


inline void NodeSet::getSomeCoords

  ( const Matrix&     coords,
    const IdxVector&  inodes ) const

{
  nodes_().getSomeCoords ( coords, inodes );
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


inline Matrix NodeSet::toMatrix () const
{
  return nodes_().toMatrix ();
}


//-----------------------------------------------------------------------
//   checkRank
//-----------------------------------------------------------------------


inline void NodeSet::checkRank

  ( const String&  context,
    idx_t          rank ) const

{
  checkRank ( context, rank, rank );
}


//-----------------------------------------------------------------------
//   nodes_
//-----------------------------------------------------------------------


inline NodeSet::Data& NodeSet::nodes_ () const
{
  return jem::staticCast<Data&> ( data_ );
}


JIVE_END_PACKAGE( fem )

#endif
