
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


#include <jem/base/System.h>
#include <jive/util/StdMesh.h>
#include <jive/util/MeshParser.h>


using namespace jem;
using namespace jive::util;


int showNode ( int argc, char** argv )
{
  Ref<Mesh>        mesh;
  Ref<MeshParser>  parser;
  int              nodeNumber;
  int              nodeIndex;


  if ( argc < 1 ) {
    System::err() << "Missing mesh file\n";
    return 1;
  }

  mesh   = newInstance<StdMesh>    ();
  parser = newInstance<MeshParser> ();

  parser->parseFile ( argv[1] );

  System::out() << "Please enter a node number: ";
  System::in()  >> nodeNumber;

  nodeIndex = mesh->findNode ( nodeNumber );

  if ( nodeIndex < 0 ) {
    System::err() << "Invalid node number\n";
    return 1;
  }

  System::out() << "Node coordinates: "
                << mesh->getNodeCoordinates ( nodeIndex ) << '\n';

  return 0;
}

int main ( int argc, char** argv )
{
  return System::exec ( & showNode, argc, argv );
}
 
