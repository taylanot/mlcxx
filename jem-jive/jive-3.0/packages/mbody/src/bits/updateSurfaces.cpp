
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
#include <jem/base/Array.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/ItemMap.h>
#include <jive/geom/ShapeTable.h>
#include <jive/fem/XElementSet.h>
#include <jive/mbody/XBodySet.h>


using jive::util::Topology;
using jive::util::TopoVector;
using jive::fem::XElementSet;


JIVE_BEGIN_PACKAGE( mbody )


typedef jem::util::ArrayBuffer<idx_t>  IdxBuffer;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   addBoundaryElems_
//-----------------------------------------------------------------------


void                  addBoundaryElems_

  ( IdxBuffer&          elemOffsets,
    IdxBuffer&          nodeIndices,
    const Topology&     elemTopo,
    const IdxVector&    elemNodes )

{
  IdxVector    bndOffsets = elemTopo.getRowOffsets    ();
  IdxVector    bndIndices = elemTopo.getColumnIndices ();

  const idx_t  bndCount   = elemTopo.size(0);

  idx_t        ibnd;
  idx_t        i, n;


  i = 0;

  for ( ibnd = 0; ibnd < bndCount; ibnd++ )
  {
    elemOffsets.pushBack ( elemNodes.size() );

    n = bndOffsets[ibnd + 1];

    for ( ; i < n; i++ )
    {
      nodeIndices.pushBack ( elemNodes[bndIndices[i]] );
    }
  }
}


//-----------------------------------------------------------------------
//   killInternalBounds_
//-----------------------------------------------------------------------


void                  killInternalBounds_

  ( IdxBuffer&          elemOffsets,
    IdxBuffer&          nodeIndices,
    const IdxVector&    bodyOffsets )

{
  using jem::shape;
  using jem::numeric::findSimilarRows;

  const idx_t  bodyCount = bodyOffsets.size() - 1;
  const idx_t  elemCount = elemOffsets.size() - 1;

  IdxVector    inodes    = nodeIndices.toArray ();

  IdxVector    buddies   ( elemCount );

  idx_t        lastElem;
  idx_t        lastNode;

  idx_t        imin, imax;
  idx_t        ibody;
  idx_t        i, j, m, n;


  if ( inodes.size() == 0 )
  {
    imin =  0;
    imax = -1;
  }
  else
  {
    imin = min ( inodes );
    imax = max ( inodes );
  }

  inodes -= imin;

  findSimilarRows (
    buddies,
    Topology ( shape( elemCount, imax - imin + 1 ),
               elemOffsets.toArray(), inodes )
  );

  lastElem = 0;
  lastNode = 0;

  for ( ibody = 0; ibody < bodyCount; ibody++ )
  {
    i = bodyOffsets[ibody];
    m = bodyOffsets[ibody + 1];

    bodyOffsets[ibody] = lastElem;

    for ( ; i < m; i++ )
    {
      if ( buddies[i] >= 0 )
      {
        continue;
      }

      j = elemOffsets[i];
      n = elemOffsets[i + 1];

      elemOffsets[lastElem++] = lastNode;

      for ( ; j < n; j++, lastNode++ )
      {
        inodes[lastNode] = inodes[j] + imin;
      }
    }
  }

  bodyOffsets[bodyCount]  = lastElem;
  elemOffsets[lastElem++] = lastNode;

  elemOffsets.resize ( lastElem );
  nodeIndices.resize ( lastNode );

  elemOffsets.trimToSize ();
  nodeIndices.trimToSize ();
}


//-----------------------------------------------------------------------
//   getSurfaceMesh_
//-----------------------------------------------------------------------


void                  getSurfaceMesh_

  ( Topology&           surfMesh,
    const IdxVector&    bodyOffsets,
    const IdxVector&    ibodies,
    const BodySet&      bodies,
    const ShapeTable&   shapes )

{
  using jem::shape;


  ElementSet   elems      = bodies.getElements ();
  NodeSet      nodes      = elems .getNodes    ();

  const idx_t  nodeCount  = nodes  .size       ();
  const idx_t  ibodyCount = ibodies.size       ();

  TopoVector   elemTopos;
  IdxVector    topoMap;

  IdxBuffer    elemOffsets;
  IdxBuffer    nodeIndices;

  IdxVector    ielems;
  IdxVector    jnodes;

  idx_t        ibody;
  idx_t        ielem;
  idx_t        itopo;
  idx_t        ib;
  idx_t        i, j, m, n;


  ielems.ref     ( bodies.getElemsOf( ibodies ) );
  jnodes.resize  ( elems.maxElemNodeCount() );
  topoMap.resize ( ielems.size() );
  elemTopos.ref  ( shapes.getShapeToposOf( topoMap, ielems ) );

  i = 0;

  for ( ib = 0; ib < ibodyCount; ib++ )
  {
    bodyOffsets[ib] = elemOffsets.size ();

    ibody = ibodies[ib];
    m     = i + bodies.getBodyElemCount ( ibody );

    for ( ; i < m; i++ )
    {
      ielem = ielems[i];
      itopo = topoMap[i];
      n     = elems.getElemNodes ( jnodes, ielem );

      if ( itopo < 0 )
      {
        elemOffsets.pushBack ( nodeIndices.size() );

        for ( j = 0; j < n; j++ )
        {
          nodeIndices.pushBack ( jnodes[j] );
        }
      }
      else
      {
        addBoundaryElems_ ( elemOffsets,      nodeIndices,
                            elemTopos[itopo], jnodes );
      }
    }
  }

  bodyOffsets[ibodyCount] = n = elemOffsets.size ();

  elemOffsets.pushBack ( nodeIndices.size() );

  killInternalBounds_  ( elemOffsets,
                         nodeIndices,
                         bodyOffsets );

  surfMesh = Topology  ( shape ( n, nodeCount ),
                         elemOffsets.toArray (),
                         nodeIndices.toArray () );
}


//-----------------------------------------------------------------------
//   addSurfaceBodies_
//-----------------------------------------------------------------------


void                  addSurfaceBodies_

  ( const XElementSet&  selems,
    const XBodySet&     sbodies,
    const BodySet&      rbodies,
    const IdxVector&    ibodies,
    const IdxVector&    bodyOffsets,
    const Topology&     surfMesh )

{
  IdxVector    elemOffsets = surfMesh.getRowOffsets    ();
  IdxVector    nodeIndices = surfMesh.getColumnIndices ();

  const idx_t  elemCount   = surfMesh.size (0);
  const idx_t  ibodyCount  = ibodies .size ();

  IdxVector    bodyIDs     ( ibodyCount );
  IdxVector    ielems      ( elemCount  );

  idx_t        ib;
  idx_t        i, j, k, m, n;


  rbodies.getData()->getItemMap()->getItemIDs ( bodyIDs,
                                                ibodies );

  selems .reserve ( elemCount  );
  sbodies.reserve ( ibodyCount );

  for ( ib = 0; ib < ibodyCount; ib++ )
  {
    i = bodyOffsets[ib];
    m = bodyOffsets[ib + 1];
    k = 0;

    for ( ; i < m; i++, k++ )
    {
      j = elemOffsets[i];
      n = elemOffsets[i + 1];

      ielems[k] =

        selems.addElement ( nodeIndices[slice(j,n)] );
    }

    sbodies.addBody ( bodyIDs[ib], ielems[slice(BEGIN,k)] );
  }

  selems .trimToSize ();
  sbodies.trimToSize ();
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   updateSurfaces
//-----------------------------------------------------------------------


void                  updateSurfaces

  ( const XBodySet&     sbodies,
    const BodySet&      rbodies,
    const ShapeTable&   shapes )

{
  using jem::iarray;
  using jive::fem::toXElementSet;


  JEM_PRECHECK ( sbodies && rbodies );

  XElementSet  selems = toXElementSet ( sbodies.getElements() );
  ElementSet   relems = rbodies.getElements ();
  NodeSet      nodes  = relems .getNodes    ();

  const idx_t  rbodyCount = rbodies.size    ();
  idx_t        sbodyCount = sbodies.size    ();

  Topology     surfMesh;

  IdxVector    bodyOffsets;
  IdxVector    ibodies;

  idx_t        i, j;


  if ( ! selems )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "surface element set is not an XElementSet"
    );
  }

  if ( sbodyCount > rbodyCount )
  {
    sbodies.clear ();
    sbodyCount = 0;
  }

  if ( sbodyCount == rbodyCount )
  {
    return;
  }

  i = sbodyCount;
  j = rbodyCount;

  bodyOffsets.resize ( j - i + 1 );
  ibodies    .resize ( j - i );

  ibodies  = iarray  ( slice(i,j) );

  getSurfaceMesh_    ( surfMesh,
                       bodyOffsets,
                       ibodies,
                       rbodies,
                       shapes );

  addSurfaceBodies_  ( selems,
                       sbodies,
                       rbodies,
                       ibodies,
                       bodyOffsets,
                       surfMesh );
}


JIVE_END_PACKAGE( mbody )
