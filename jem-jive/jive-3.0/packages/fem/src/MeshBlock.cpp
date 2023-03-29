
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


#include <jem/base/ClassTemplate.h>
#include <jem/io/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/xml/print.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemGroup.h>
#include <jive/mp/ItemMask.h>
#include <jive/fem/MeshBlock.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::MeshBlock );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jive::mp::RECV_BORDERS;
using jive::mp::SEND_BORDERS;
using jive::mp::SEND_RECV_BORDERS;
using jive::mp::newXBorderSet;
using jive::util::joinNames;
using jive::util::StorageMode;


//=======================================================================
//   class MeshBlock
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MeshBlock::MeshBlock ()
{}


MeshBlock::MeshBlock

  ( const String&  name,
    StorageMode    mode ) :

    nodes            (
      newXNodeSet    ( joinNames( name, "nodes" ), mode )
    ),

    elems (
      newXElementSet ( joinNames( name, "elements" ),
                       nodes, mode )
    ),

    recvNodeBorders  (
      newXBorderSet  ( RECV_BORDERS, nodes, mode )
    ),

    sendNodeBorders  (
      newXBorderSet  ( SEND_BORDERS, nodes, mode )
    ),

    recvElemBorders  (
      newXBorderSet  ( RECV_BORDERS, elems, mode )
    ),

    sendElemBorders  (
      newXBorderSet  ( SEND_BORDERS, elems, mode )
    )

{
  nodeMask = newInstance<ItemMask> ( nodes.getData() );
  elemMask = newInstance<ItemMask> ( elems.getData() );
}


MeshBlock::MeshBlock

  ( const String&  name,
    NoOverlapTag   tag,
    StorageMode    mode ) :

    nodes            (
      newXNodeSet    ( joinNames( name, "nodes" ), mode )
    ),

    elems            (
      newXElementSet ( joinNames( name, "elements" ),
                       nodes, mode )
    ),

    recvNodeBorders  (
      newXBorderSet  ( SEND_RECV_BORDERS, nodes, mode )
    ),

    sendNodeBorders  ( recvNodeBorders ),

    recvElemBorders  (
      newXBorderSet  ( SEND_RECV_BORDERS, elems, mode )
    ),

    sendElemBorders  ( recvElemBorders )

{
  nodeMask = newInstance<ItemMask> ( nodes.getData() );
  elemMask = newInstance<ItemMask> ( elems.getData() );
}


MeshBlock::MeshBlock

  ( const XNodeSet&     nodes,
    const XElementSet&  elems,
    StorageMode         mode ) :

    nodes            ( nodes ),
    elems            ( elems ),

    recvNodeBorders  (
      newXBorderSet  ( RECV_BORDERS, nodes, mode )
    ),

    sendNodeBorders  (
      newXBorderSet  ( SEND_BORDERS, nodes, mode )
    ),

    recvElemBorders  (
      newXBorderSet  ( RECV_BORDERS, elems, mode )
    ),

    sendElemBorders  (
      newXBorderSet  ( SEND_BORDERS, elems, mode )
    )

{
  nodeMask = newInstance<ItemMask> ( nodes.getData() );
  elemMask = newInstance<ItemMask> ( elems.getData() );
}


MeshBlock::MeshBlock

  ( const XNodeSet&     nodes,
    const XElementSet&  elems,
    NoOverlapTag        tag,
    StorageMode         mode ) :

    nodes            ( nodes ),
    elems            ( elems ),

    recvNodeBorders  (
      newXBorderSet  ( SEND_RECV_BORDERS, nodes, mode )
    ),

    sendNodeBorders  ( recvNodeBorders ),

    recvElemBorders  (
      newXBorderSet  ( SEND_RECV_BORDERS, elems, mode )
    ),

    sendElemBorders  ( recvElemBorders )

{
  nodeMask = newInstance<ItemMask> ( nodes.getData() );
  elemMask = newInstance<ItemMask> ( elems.getData() );
}


MeshBlock::~MeshBlock ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void MeshBlock::readFrom ( ObjectInput& in )
{
  decode ( in, nodes, elems );
  decode ( in, recvNodeBorders, sendNodeBorders );
  decode ( in, recvElemBorders, sendElemBorders );
  decode ( in, nodeMask, elemMask );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void MeshBlock::writeTo ( ObjectOutput& out ) const
{
  encode ( out, nodes, elems );
  encode ( out, recvNodeBorders, sendNodeBorders );
  encode ( out, recvElemBorders, sendElemBorders );
  encode ( out, nodeMask, elemMask );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void MeshBlock::printTo ( PrintWriter& out ) const
{
  using jem::xml::startTag;
  using jem::xml::endTag;
  using jem::xml::beginStartTag;
  using jem::xml::attribute;

  String  tagName;


  out.pushSettings ();

  out.nformat.setIntegerWidth (
    jem::io::getWidthOf ( jem::max( nodes.size(), elems.size() ) )
  );

  tagName = nodes.getTagName ();

  print ( out, startTag( tagName ), nodes, endTag( tagName ) );

  tagName = elems.getTagName ();

  print ( out, startTag( tagName ), elems, endTag( tagName ) );

  printMask_ ( out, *nodeMask );
  printMask_ ( out, *elemMask );

  tagName = recvNodeBorders.getTagName ();

  print ( out, startTag( tagName ),
          recvNodeBorders, endTag( tagName ) );

  if ( sendNodeBorders != recvNodeBorders )
  {
    tagName = sendNodeBorders.getTagName ();

    print ( out, startTag( tagName ),
            sendNodeBorders, endTag( tagName ) );
  }

  tagName = recvElemBorders.getTagName ();

  print ( out, startTag( tagName ),
          recvElemBorders, endTag( tagName ) );

  if ( sendElemBorders != recvElemBorders )
  {
    tagName = sendElemBorders.getTagName ();

    print ( out, startTag( tagName ),
            sendElemBorders, endTag( tagName ) );
  }

  out.popSettings ();
}


//-----------------------------------------------------------------------
//   printMask_
//-----------------------------------------------------------------------


void MeshBlock::printMask_

  ( PrintWriter&     out,
    const ItemMask&  mask )
{
  using jem::xml::endTag;
  using jem::xml::beginStartTag;
  using jem::xml::attribute;
  using jem::xml::endStartTag;


  ItemSet&  items   = * mask.getItems ();
  String    tagName = ItemGroup::getTagName ( items.getItemName() );


  print ( out, beginStartTag( tagName ),
          attribute( "name", "mask" ), endStartTag );

  ItemGroup::printGroup ( out, items,
                          mask.getItemsFor ( false ) );

  print ( out, endTag( tagName ) );
}


JIVE_END_PACKAGE( fem )
