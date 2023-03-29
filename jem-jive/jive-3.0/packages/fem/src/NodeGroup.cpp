
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


#include <jem/base/Class.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/util/AllItemGroup.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/util/EmptyItemGroup.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/NodeGroup.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class NodeGroup
//=======================================================================

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


NodeGroup NodeGroup::get

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const String&      context )

{
  String  name;


  props.get ( name, "nodes" );
  conf .set ( "nodes", name );

  Ref<Data>  data =

    Data::find ( name, Globdat::NODES, globdat );

  if ( data == nullptr || ! isValidData( data ) )
  {
    throw jem::IllegalInputException (
      context,
      String::format (
        "undefined node group: `%s\'", name
      )
    );
  }

  return NodeGroup ( data );
}


//-----------------------------------------------------------------------
//   isValidData
//-----------------------------------------------------------------------


bool NodeGroup::isValidData ( const Ref<Data>& data )
{
  if ( ! data )
  {
    return true;
  }

  jem::Class*  type = data->getItems()->getClass ();

  return NodeSet::Data::getType()->isBaseOf ( type );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newNodeGroup
//-----------------------------------------------------------------------


NodeGroup           newNodeGroup

  ( const NodeSet&    nodes )

{
  using jive::util::EmptyItemGroup;

  JEM_PRECHECK ( nodes );

  return NodeGroup

    ( newInstance<EmptyItemGroup>( nodes.getData() ) );
}


NodeGroup           newNodeGroup

  ( const SliceAll&   all,
    const NodeSet&    nodes )

{
  using jive::util::AllItemGroup;

  JEM_PRECHECK ( nodes );

  return NodeGroup

    ( newInstance<AllItemGroup>( nodes.getData() ) );
}


NodeGroup           newNodeGroup

  ( const IdxVector&  inodes,
    const NodeSet&    nodes )

{
  using jive::util::ArrayItemGroup;

  JEM_PRECHECK ( nodes );

  return NodeGroup

    ( newInstance<ArrayItemGroup>( inodes, nodes.getData() ) );
}


JIVE_END_PACKAGE( fem )
