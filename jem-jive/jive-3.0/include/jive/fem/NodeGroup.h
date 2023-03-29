
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

#ifndef JIVE_FEM_NODEGROUP_H
#define JIVE_FEM_NODEGROUP_H

#include <jive/util/ItemGroupSkin.h>
#include <jive/fem/NodeSet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class NodeGroup
//-----------------------------------------------------------------------


class NodeGroup : public ItemGroupSkin
{
 public:

  typedef NodeGroup         Self;
  typedef ItemGroupSkin     Super;


  inline                    NodeGroup   ()       noexcept;

  explicit inline           NodeGroup

    ( const Ref<Data>&        data )             noexcept;

  explicit inline           NodeGroup

    ( Ref<Data>&&             data )             noexcept;

  inline NodeSet            getNodes    () const;

  inline void               getCoords

    ( const Matrix&           coords )     const;

  static inline Self        find

    ( const String&           name,
      const NodeSet&          nodes,
      const Properties&       globdat );

  static inline Self        get

    ( const String&           name,
      const NodeSet&          nodes,
      const Properties&       globdat,
      const String&           context );

  static Self               get

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat,
      const String&           context );

  static bool               isValidData

    ( const Ref<Data>&        data );

  static inline void        checkData

    ( const Ref<Data>&        data,
      const String&           context );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


NodeGroup                   newNodeGroup

  ( const NodeSet&            nodes );

NodeGroup                   newNodeGroup

  ( const SliceAll&           all,
    const NodeSet&            nodes );

NodeGroup                   newNodeGroup

  ( const IdxVector&          inodes,
    const NodeSet&            nodes );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline NodeGroup::NodeGroup () noexcept
{}


inline NodeGroup::NodeGroup ( const Ref<Data>& data ) noexcept :

  Super ( data )

{}

inline NodeGroup::NodeGroup ( Ref<Data>&& data ) noexcept :

  Super ( std::move( data ) )

{}


//-----------------------------------------------------------------------
//   getNodes
//-----------------------------------------------------------------------


inline NodeSet NodeGroup::getNodes () const
{
  return NodeSet (
    static_cast<NodeSet::Data*> ( data_->getItems() )
  );
}


//-----------------------------------------------------------------------
//   getCoords
//-----------------------------------------------------------------------


inline void NodeGroup::getCoords ( const Matrix& coords ) const
{
  getNodes().getSomeCoords ( coords, getIndices() );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline NodeGroup NodeGroup::find

  ( const String&       name,
    const NodeSet&      nodes,
    const Properties&   globdat )

{
  return Self ( Data::find( name, nodes.getData(), globdat ) );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline NodeGroup NodeGroup::get

  ( const String&      name,
    const NodeSet&     nodes,
    const Properties&  globdat,
    const String&      context )

{
  return Self ( Data::get( name, nodes.getData(),
                           globdat, context ) );
}


//-----------------------------------------------------------------------
//   checkData
//-----------------------------------------------------------------------


inline void NodeGroup::checkData

  ( const Ref<Data>&  data,
    const String&     context )

{
  if ( ! isValidData( data ) )
  {
    dataError ( context, "node", data );
  }
}


JIVE_END_PACKAGE( fem )

#endif
