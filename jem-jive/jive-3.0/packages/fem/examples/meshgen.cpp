
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


/************************************************************************
 *
 *  This example demonstrates how to generate a simple rectangular
 *  Finite Element mesh consisting of quadrilateral elements. The
 *  mesh and a few node groups are written to the standard output
 *  in an XML-based format that can be read by the Jive input module.
 *
 ***********************************************************************/

#include <jem/base/Tuple.h>
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/HashDictionary.h>
#include <jem/xml/print.h>
#include <jive/Array.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>


using namespace jem;

using jem::util::Dictionary;
using jem::util::HashDictionary;
using jem::xml::startTag;
using jem::xml::endTag;
using jive::util::MINIMAL_STORAGE;
using jive::Vector;
using jive::IdxVector;
using jive::util::ItemGroup;
using jive::util::ArrayItemGroup;
using jive::fem::newXNodeSet;
using jive::fem::newXElementSet;
using jive::fem::XNodeSet;
using jive::fem::XElementSet;


//-----------------------------------------------------------------------
//   typedefs & constants
//-----------------------------------------------------------------------


typedef jem::Tuple<idx_t,2>   IdxTuple;
typedef jem::Tuple<double,2>  FloatTuple;
typedef jem::io::PrintWriter  Printer;


static const int              XDIM = 0;
static const int              YDIM = 1;


//-----------------------------------------------------------------------
//   makeMesh
//-----------------------------------------------------------------------


int makeMesh ()
{
  const IdxTuple   elemCount ( 1000, 100 );
  const IdxTuple   nodeCount ( elemCount + 1 );

  XNodeSet         nodes = newXNodeSet    (        MINIMAL_STORAGE );
  XElementSet      elems = newXElementSet ( nodes, MINIMAL_STORAGE );

  Ref<Dictionary>  ngroups;
  Ref<Printer>     prn;

  Vector           coords ( 2 );
  IdxVector        inodes ( 4 );

  FloatTuple       delta;
  String           tagName;

  idx_t            m, n;


  nodes.reserve ( product(nodeCount) );
  elems.reserve ( product(elemCount) );

  delta = 1.0 / castTo<double>( elemCount );

  for ( idx_t j = 0; j < nodeCount[YDIM]; j++ )
  {
    coords[YDIM] = delta[YDIM] * j;

    for ( idx_t i = 0; i < nodeCount[XDIM]; i++ )
    {
      coords[XDIM] = delta[XDIM] * i;

      nodes.addNode ( coords );
    }
  }

  m       = nodeCount[XDIM];
  n       = nodeCount[YDIM];
  ngroups = newInstance<HashDictionary> ();

  ngroups->insert (
    "x0",
    newInstance<ArrayItemGroup> (
      IdxVector ( iarray( slice( 0, m * n, m ) ) ),
      nodes.getData ()
    )
  );

  ngroups->insert (
    "x1",
    newInstance<ArrayItemGroup> (
      IdxVector ( iarray( slice( m - 1, m * n, m ) ) ),
      nodes.getData ()
    )
  );

  ngroups->insert (
    "y0",
    newInstance<ArrayItemGroup> (
      IdxVector ( iarray( slice( 0, m ) ) ),
      nodes.getData ()
    )
  );

  ngroups->insert (
    "y1",
    newInstance<ArrayItemGroup> (
      IdxVector ( iarray( slice( m * (n - 1), m * n ) ) ),
      nodes.getData ()
    )
  );

  for ( idx_t j = 0; j < elemCount[YDIM]; j++ )
  {
    inodes[0] = j * nodeCount[XDIM];
    inodes[1] = inodes[0] + 1;
    inodes[2] = inodes[1] + nodeCount[XDIM];
    inodes[3] = inodes[2] - 1;

    for ( idx_t i = 0; i < elemCount[XDIM]; i++ )
    {
      elems.addElement ( inodes );

      inodes += 1;
    }
  }

  prn     = newInstance<Printer> ( & System::out() );
  tagName = nodes.getTagName ();

  *prn << startTag ( tagName );
  *prn <<   nodes;
  *prn << endTag   ( tagName );

  tagName = elems.getTagName ();

  *prn << startTag ( tagName );
  *prn <<   elems;
  *prn << endTag   ( tagName );

  ItemGroup::printSome ( *prn, *ngroups );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main()
{
  return System::exec ( & makeMesh );
}

