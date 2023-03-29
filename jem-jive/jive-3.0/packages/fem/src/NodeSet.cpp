
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
#include <jem/base/IllegalInputException.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/NodeSet.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::IllegalInputException;


//=======================================================================
//   class NodeSet
//=======================================================================

//-----------------------------------------------------------------------
//   checkRank
//-----------------------------------------------------------------------


void NodeSet::checkRank

  ( const String&  context,
    idx_t            minRank,
    idx_t            maxRank ) const

{
  const idx_t  r = rank ();

  if ( r < minRank || r > maxRank )
  {
    String  myName = getName ();

    if ( minRank == maxRank )
    {
      throw IllegalInputException (
        context,
        String::format (
          "node set `%s\' has invalid rank: %d; must be %d",
          myName,
          r, minRank
        )
      );
    }
    else
    {
      throw IllegalInputException (
        context,
        String::format (
          "node set `%s\' has invalid rank: %d; "
          "valid range is [%d,%d]",
          myName,
          r, minRank, maxRank
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void NodeSet::store ( const Properties&  globdat ) const
{
  JEM_PRECHECK ( *this );

  data_->store ( Globdat::NODES, globdat );
}


void NodeSet::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  data_->store ( name, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


NodeSet NodeSet::find ( const Properties& globdat )
{
  return find ( Globdat::NODES, globdat );
}


NodeSet NodeSet::find

  ( const String&      name,
    const Properties&  globdat )

{
  using jem::dynamicCast;
  using jive::util::ItemSet;

  Ref<Data>  data =

    dynamicCast<Data> ( ItemSet::find ( name, globdat ) );

  return NodeSet ( data );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


NodeSet NodeSet::get

  ( const Properties&  globdat,
    const String&      context )

{
  NodeSet  nodes = find ( Globdat::NODES, globdat );

  if ( ! nodes )
  {
    throw IllegalInputException (
      context,
      "no nodes have been defined"
    );
  }

  return nodes;
}


NodeSet NodeSet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  NodeSet  nodes = find ( name, globdat );

  if ( ! nodes )
  {
    throw IllegalInputException (
      context,
      "undefined node set: " + name
    );
  }

  return nodes;
}


JIVE_END_PACKAGE( fem )
