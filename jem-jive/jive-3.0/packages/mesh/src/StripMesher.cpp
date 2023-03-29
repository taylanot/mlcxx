
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
#include <jem/base/ClassTemplate.h>
#include <jive/mesh/Mesh.h>
#include <jive/mesh/StripMesher.h>


JEM_DEFINE_CLASS( jive::mesh::StripMesher );


JIVE_BEGIN_PACKAGE( mesh )


//======================================================================
//   class StripMesher
//======================================================================

//----------------------------------------------------------------------
//   constructor & destructor
//----------------------------------------------------------------------


StripMesher::StripMesher ( Options options )
{
  options_ = options;
}


StripMesher::~StripMesher ()
{}


//----------------------------------------------------------------------
//   triangulate3
//----------------------------------------------------------------------


void StripMesher::triangulate3

  ( const XElementSet&  elems,
    const Vector&       xouter,
    const IdxVector&    iouter,
    const Vector&       xinner,
    const IdxVector&    iinner,
    Options             xtraOpts )

{
  JEM_PRECHECK ( xouter.size() >  0  &&
                 xinner.size() >  0  &&

                 xouter.size() +
                 xinner.size() >  2  &&

                 xouter.size() == iouter.size() &&
                 xinner.size() == iinner.size() );


  Options  options = options_ | xtraOpts;


  if ( iouter.size() >= iinner.size() )
  {
    triangulate3_ ( elems,
                    xouter, iouter,
                    xinner, iinner,
                    options );
  }
  else
  {
    options = options ^ NUMBER_CLOCKWISE;

    triangulate3_ ( elems,
                    xinner, iinner,
                    xouter, iouter,
                    options );
  }
}


//----------------------------------------------------------------------
//   triangulate6
//----------------------------------------------------------------------


void StripMesher::triangulate6

  ( const XMesh&        mesh,
    idx_t               iright,
    const Vector&       xouter,
    const IdxVector&    iouter,
    idx_t               ileft,
    const Vector&       xinner,
    const IdxVector&    iinner,
    Options             xtraOpts )

{
  JEM_PRECHECK ( ileft         >= 0  &&
                 iright        >= 0  &&

                 xouter.size() >  0  &&
                 xinner.size() >  0  &&

                 xouter.size() +
                 xinner.size() >  2  &&

                 xouter.size() == iouter.size() &&
                 xinner.size() == iinner.size() );


  Options  options = options_ | xtraOpts;

  if ( options & WRAP_AROUND )
  {
    JEM_PRECHECK ( ileft == iright );
  }

  if ( iouter.size() >= iinner.size() )
  {
    triangulate6_ ( mesh,
                    iright, xouter, iouter,
                    ileft,  xinner, iinner,
                    options );
  }
  else
  {
    options = options ^ NUMBER_CLOCKWISE;

    triangulate6_ ( mesh,
                    iright, xinner, iinner,
                    ileft,  xouter, iouter,
                    options );
  }
}


//----------------------------------------------------------------------
//   triangulate3_
//----------------------------------------------------------------------


void StripMesher::triangulate3_

  ( const XElementSet&  elems,
    const Vector&       xouter,
    const IdxVector&    iouter,
    const Vector&       xinner,
    const IdxVector&    iinner,
    Options             options )

{
  const idx_t  lastInner = iinner.size() - 1;
  const idx_t  lastOuter = iouter.size() - 1;

  IdxVector&   elemNodes = elemNodes_;

  double       xmax;

  idx_t        node0, node1, node2;

  idx_t        i, j;


  elemNodes.resize ( 3 );

  if ( options & NUMBER_CLOCKWISE )
  {
    node0 = 0;
    node1 = 2;
    node2 = 1;
  }
  else
  {
    node0 = 0;
    node1 = 1;
    node2 = 2;
  }

  j = 0;

  for ( i = 0;  i < lastInner; i++ )
  {
    xmax = 0.5 * (xinner[i] + xinner[i + 1]);

    while ( j < lastOuter && (xouter[j + 1] <= xmax) )
    {
      elemNodes[node0] = iouter[j];
      elemNodes[node1] = iinner[i];
      elemNodes[node2] = iouter[j + 1];

      elems.addElement ( elemNodes );

      j++;
    }

    elemNodes[node0] = iouter[j];
    elemNodes[node1] = iinner[i];
    elemNodes[node2] = iinner[i + 1];

    elems.addElement ( elemNodes );
  }

  for ( ; j < lastOuter; j++ )
  {
    elemNodes[node0] = iouter[j];
    elemNodes[node1] = iinner[lastInner];
    elemNodes[node2] = iouter[j + 1];

    elems.addElement ( elemNodes );
  }

  if ( options & WRAP_AROUND )
  {
    elemNodes[node0] = iouter[lastOuter];
    elemNodes[node1] = iinner[0];
    elemNodes[node2] = iouter[0];

    elems.addElement ( elemNodes );

    if ( lastInner > 0 )
    {
      elemNodes[node0] = iouter[lastOuter];
      elemNodes[node1] = iinner[lastInner];
      elemNodes[node2] = iinner[0];

      elems.addElement ( elemNodes );
    }
  }
}


//----------------------------------------------------------------------
//   triangulate6_
//----------------------------------------------------------------------


void StripMesher::triangulate6_

  ( const XMesh&        mesh,
    idx_t               iright,
    const Vector&       xouter,
    const IdxVector&    iouter,
    idx_t               ileft,
    const Vector&       xinner,
    const IdxVector&    iinner,
    Options             options )

{
  using jem::max;

  const idx_t  lastInner = iinner.size() - 1;
  const idx_t  lastOuter = iouter.size() - 1;
  const idx_t  endInner  = iinner.size() - 2;
  const idx_t  endOuter  = iouter.size() - 2;

  IdxVector&   elemNodes = elemNodes_;

  double       xmax;

  idx_t        node0, node1, node2;
  idx_t        node3, node4, node5;
  idx_t        imid0, imid1;

  idx_t        i, j;


  nodePos0_.resize ( 2 );
  nodePos1_.resize ( 2 );
  elemNodes.resize ( 6 );

  if ( options & NUMBER_CLOCKWISE )
  {
    node0 = 0;
    node1 = 5;
    node2 = 4;
    node3 = 3;
    node4 = 2;
    node5 = 1;
  }
  else
  {
    node0 = 0;
    node1 = 1;
    node2 = 2;
    node3 = 3;
    node4 = 4;
    node5 = 5;
  }

  j     = 0;
  imid0 = imid1 = ileft;

  for ( i = 0; i < endInner; i += 2 )
  {
    xmax = 0.5 * (xinner[i] + xinner[i + 2]);

    while ( j < endOuter && (xouter[j + 2] <= xmax) )
    {
      imid0 = imid1;
      imid1 = addMidNode_ ( mesh, iinner[i], iouter[j + 2] );

      elemNodes[node0] = iouter[j];
      elemNodes[node1] = imid0;
      elemNodes[node2] = iinner[i];
      elemNodes[node3] = imid1;
      elemNodes[node4] = iouter[j + 2];
      elemNodes[node5] = iouter[j + 1];

      mesh.elems.addElement ( elemNodes );

      j += 2;
    }

    imid0 = imid1;

    if ( j == lastOuter && (i + 2) == lastInner )
    {
      imid1 = iright;
    }
    else
    {
      imid1 = addMidNode_ ( mesh, iinner[i + 2], iouter[j] );
    }

    elemNodes[node0] = iouter[j];
    elemNodes[node1] = imid0;
    elemNodes[node2] = iinner[i];
    elemNodes[node3] = iinner[i + 1];
    elemNodes[node4] = iinner[i + 2];
    elemNodes[node5] = imid1;

    mesh.elems.addElement ( elemNodes );
  }

  for ( ; j < endOuter; j += 2 )
  {
    imid0 = imid1;

    if ( (j + 2) == lastOuter && i == lastInner )
    {
      imid1 = iright;
    }
    else
    {
      imid1 = addMidNode_ ( mesh, iinner[i], iouter[j + 2] );
    }

    elemNodes[node0] = iouter[j];
    elemNodes[node1] = imid0;
    elemNodes[node2] = iinner[i];
    elemNodes[node3] = imid1;
    elemNodes[node4] = iouter[j + 2];
    elemNodes[node5] = iouter[j + 1];

    mesh.elems.addElement ( elemNodes );
  }

  if ( options & WRAP_AROUND )
  {
    if ( lastInner <= 0 )
    {
      elemNodes[node0] = iouter[endOuter];
      elemNodes[node1] = imid1;
      elemNodes[node2] = iinner[0];
      elemNodes[node3] = ileft;
      elemNodes[node4] = iouter[0];
      elemNodes[node5] = iouter[endOuter + 1];

      mesh.elems.addElement ( elemNodes );
    }
    else
    {
      imid0 = imid1;
      imid1 = addMidNode_ ( mesh, iinner[0], iouter[endOuter] );

      elemNodes[node0] = iouter[endOuter];
      elemNodes[node1] = imid1;
      elemNodes[node2] = iinner[0];
      elemNodes[node3] = ileft;
      elemNodes[node4] = iouter[0];
      elemNodes[node5] = iouter[endOuter + 1];

      mesh.elems.addElement ( elemNodes );

      elemNodes[node0] = iouter[endOuter];
      elemNodes[node1] = imid0;
      elemNodes[node2] = iinner[endInner];
      elemNodes[node3] = iinner[endInner + 1];
      elemNodes[node4] = iinner[0];
      elemNodes[node5] = imid1;

      mesh.elems.addElement ( elemNodes );
    }
  }
}


//-----------------------------------------------------------------------
//   addMidNode_
//-----------------------------------------------------------------------


inline idx_t StripMesher::addMidNode_

  ( const XMesh&  mesh,
    idx_t         node0,
    idx_t         node1 )

{
  Vector&  nodePos0 = nodePos0_;
  Vector&  nodePos1 = nodePos1_;

  mesh.nodes.getNodeCoords ( nodePos0_, node0 );
  mesh.nodes.getNodeCoords ( nodePos1_, node1 );

  nodePos1[0] = 0.5 * (nodePos0[0] + nodePos1[0]);
  nodePos1[1] = 0.5 * (nodePos0[1] + nodePos1[1]);

  return mesh.nodes.addNode ( nodePos1 );
}


JIVE_END_PACKAGE( mesh )
