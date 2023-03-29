
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
#include <jem/base/Class.h>
#include <jem/base/array/utilities.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/utilities.h>
#include <jive/fem/check.h>
#include <jive/fem/error.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/BoundarySet.h>



JIVE_BEGIN_PACKAGE( fem )


using jem::iarray;

typedef jem::util::ArrayBuffer<idx_t>  IdxBuffer;


//=======================================================================
//   class BoundarySet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundarySet::CLASS_NAME_ = "jive::fem::BoundarySet";


//-----------------------------------------------------------------------
//   checkBoundaries
//-----------------------------------------------------------------------


void BoundarySet::checkBoundaries

  ( const String&    context,
    const Topology&  elemTopo ) const

{
  JEM_PRECHECK ( *this );

  IdxVector  ibounds ( iarray( size() ) );

  checkSomeBoundaries ( context, ibounds, elemTopo );
}


void BoundarySet::checkBoundaries

  ( const String&      context,
    const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  JEM_PRECHECK ( *this );

  IdxVector  ibounds ( iarray( size() ) );

  checkSomeBoundaries ( context, ibounds, topoMap, elemTopos );
}


//-----------------------------------------------------------------------
//   checkSomeBoundaries (given one element topology)
//-----------------------------------------------------------------------


void BoundarySet::checkSomeBoundaries

  ( const String&     context,
    const IdxVector&  ibounds,
    const Topology&   elemTopo ) const

{
  JEM_PRECHECK ( *this );

  ElementSet   elems     = getElements  ();

  const idx_t  ibndCount = ibounds.size ();

  idx_t        ibound;
  idx_t        ielem;
  idx_t        jelem;
  idx_t        ilocal;
  idx_t        ib;
  idx_t        n;


  jelem = n = -1;

  for ( ib = 0; ib < ibndCount; ib++ )
  {
    ibound = ibounds[ib];

    getBoundary ( ielem, ilocal, ibound );

    if ( ilocal >= elemTopo.size(0) )
    {
      localBoundaryIndexError (
        context,
        getBoundaryID ( ibound ),
        ilocal,
        elemTopo.size(0)
      );
    }

    if ( ielem != jelem )
    {
      n     = elems.getElemNodeCount ( ielem );
      jelem = ielem;
    }

    if ( n != elemTopo.size(1) )
    {
      elemNodeCountError (
        context,
        elems.getElemID ( ielem ),
        n,
        elemTopo.size(1)
      );
    }
  }
}


//-----------------------------------------------------------------------
//   checkSomeBoundaries (given multiple element topologies)
//-----------------------------------------------------------------------


void BoundarySet::checkSomeBoundaries

  ( const String&      context,
    const IdxVector&   ibounds,
    const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  JEM_PRECHECK ( *this );

  ElementSet   elems     = getElements    ();

  const idx_t  ibndCount = ibounds  .size ();
  const idx_t  topoCount = elemTopos.size ();

  idx_t        ibound;
  idx_t        ielem;
  idx_t        jelem;
  idx_t        ilocal;
  idx_t        itopo;
  idx_t        ib;
  idx_t        n;


  JIVE_CHECK_TOPO_MAP ( topoMap, elems );

  jelem = n = -1;

  for ( ib = 0; ib < ibndCount; ib++ )
  {
    ibound = ibounds[ib];

    getBoundary ( ielem, ilocal, ibound );

    itopo = topoMap[ielem];

    JIVE_CHECK_TOPO_INDEX ( itopo, topoCount );

    if ( ilocal >= elemTopos[itopo].size(0) )
    {
      localBoundaryIndexError (
        context,
        getBoundaryID ( ibound ),
        ilocal,
        elemTopos[itopo].size(0)
      );
    }

    if ( ielem != jelem )
    {
      n     = elems.getElemNodeCount ( ielem );
      jelem = ielem;
    }

    if ( n != elemTopos[itopo].size(1) )
    {
      elemNodeCountError (
        context,
        elems.getElemID ( ielem ),
        n,
        elemTopos[itopo].size(1)
      );
    }
  }
}


//-----------------------------------------------------------------------
//   getNodesOf (given one element topology)
//-----------------------------------------------------------------------


IdxVector BoundarySet::getNodesOf

  ( const IdxVector&  ibounds,
    const Topology&   elemTopo ) const

{
  JEM_PRECHECK ( *this );

  ElementSet   elems     = getElements  ();

  const idx_t  ibndCount = ibounds.size ();

  IdxVector    ibuf      ( elems.maxElemNodeCount() );

  IdxBuffer    inodes;
  IdxVector    jnodes;

  idx_t        ibound;
  idx_t        ielem;
  idx_t        ilocal;
  idx_t        jlocal;
  idx_t        ib;
  idx_t        i, n;


  inodes.reserve ( 2 * ibndCount );

  jlocal = n = -1;

  for ( ib = 0; ib < ibndCount; ib++ )
  {
    ibound = ibounds[ib];

    getBoundary ( ielem, ilocal, ibound );

    if ( ilocal != jlocal )
    {
      jnodes.ref ( elemTopo.getColumnIndices( ilocal ) );

      n      = jnodes.size ();
      jlocal = ilocal;
    }

    elems.getSomeElemNodes ( ibuf, jnodes, ielem );

    for ( i = 0; i < n; i++ )
    {
      inodes.pushBack ( ibuf[i] );
    }
  }

  inodes.trimToSize ();

  return inodes.toArray ();
}


//-----------------------------------------------------------------------
//   getNodesOf (given multiple element topologies)
//-----------------------------------------------------------------------


IdxVector BoundarySet::getNodesOf

  ( const IdxVector&   ibounds,
    const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  JEM_PRECHECK ( *this );

  ElementSet   elems     = getElements    ();

  const idx_t  ibndCount = ibounds  .size ();
  const idx_t  topoCount = elemTopos.size ();

  IdxVector    ibuf      ( elems.maxElemNodeCount() );

  IdxBuffer    inodes;
  IdxVector    jnodes;

  idx_t        ibound;
  idx_t        ielem;
  idx_t        ilocal;
  idx_t        itopo;
  idx_t        ib;
  idx_t        i, n;


  JIVE_CHECK_TOPO_MAP ( topoMap, elems );

  inodes.reserve ( 2 * ibndCount );

  for ( ib = 0; ib < ibndCount; ib++ )
  {
    ibound = ibounds[ib];

    getBoundary ( ielem, ilocal, ibound );

    itopo  = topoMap[ielem];

    JIVE_CHECK_TOPO_INDEX ( itopo, topoCount );

    jnodes.ref ( elemTopos[itopo].getColumnIndices( ilocal ) );

    elems.getSomeElemNodes ( ibuf, jnodes, ielem );

    n = jnodes.size ();

    for ( i = 0; i < n; i++ )
    {
      inodes.pushBack ( ibuf[i] );
    }
  }

  inodes.trimToSize ();

  return inodes.toArray ();
}


//-----------------------------------------------------------------------
//   getUniqueNodesOf
//-----------------------------------------------------------------------


IdxVector BoundarySet::getUniqueNodesOf

  ( const IdxVector&  ibounds,
    const Topology&   elemTopo ) const

{
  using jive::util::makeUnique;

  IdxVector  inodes    = getNodesOf ( ibounds, elemTopo );

  idx_t      nodeCount = getNodes().size ();

  return makeUnique ( inodes, 0, nodeCount - 1 ).clone ();
}


IdxVector BoundarySet::getUniqueNodesOf

  ( const IdxVector&   ibounds,
    const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  using jive::util::makeUnique;

  IdxVector  inodes    = getNodesOf ( ibounds, topoMap, elemTopos );

  idx_t      nodeCount = getNodes().size ();

  return makeUnique ( inodes, 0, nodeCount - 1 ).clone ();
}


//-----------------------------------------------------------------------
//   getNodesPerBoundary (given one element topology)
//-----------------------------------------------------------------------


BoundarySet::Topology BoundarySet::getNodesPerBoundary

  ( const Topology&  elemTopo ) const

{
  JEM_PRECHECK ( *this );

  ElementSet   elems     = getElements    ();
  NodeSet      nodes     = elems.getNodes ();

  const idx_t  bndCount  =       size     ();
  const idx_t  nodeCount = nodes.size     ();

  IdxVector    offsets   ( bndCount + 1 );
  IdxBuffer    indices;

  IdxVector    inodes    ( elems.maxElemNodeCount() );
  IdxVector    jnodes;

  idx_t        ibound;
  idx_t        ielem;
  idx_t        ilocal;
  idx_t        jlocal;
  idx_t        i, n;


  indices.reserve ( 2 * bndCount );

  jlocal = n = -1;

  for ( ibound = 0; ibound < bndCount; ibound++ )
  {
    getBoundary ( ielem, ilocal, ibound );

    offsets[ibound] = indices.size ();

    if ( ilocal != jlocal )
    {
      jnodes.ref ( elemTopo.getColumnIndices( ilocal ) );

      n      = jnodes.size ();
      jlocal = ilocal;
    }

    elems.getSomeElemNodes ( inodes, jnodes, ielem );

    for ( i = 0; i < n; i++ )
    {
      indices.pushBack ( inodes[i] );
    }
  }

  offsets[bndCount] = indices.size ();

  indices.trimToSize ();

  return Topology ( jem::shape( bndCount, nodeCount ),
                    offsets, indices.toArray() );
}


//-----------------------------------------------------------------------
//   getNodesPerBoundary (given multiple element topologies)
//-----------------------------------------------------------------------


BoundarySet::Topology BoundarySet::getNodesPerBoundary

  ( const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  JEM_PRECHECK ( *this );

  ElementSet   elems     = getElements    ();
  NodeSet      nodes     = elems.getNodes ();

  const idx_t  bndCount  =           size ();
  const idx_t  topoCount = elemTopos.size ();
  const idx_t  nodeCount = nodes    .size ();

  IdxVector    offsets   ( bndCount + 1 );
  IdxBuffer    indices;

  IdxVector    inodes    ( elems.maxElemNodeCount() );
  IdxVector    jnodes;

  idx_t        ibound;
  idx_t        ielem;
  idx_t        ilocal;
  idx_t        itopo;
  idx_t        i, n;


  JIVE_CHECK_TOPO_MAP ( topoMap, elems );

  indices.reserve ( 2 * bndCount );

  for ( ibound = 0; ibound < bndCount; ibound++ )
  {
    getBoundary ( ielem, ilocal, ibound );

    offsets[ibound] = indices.size ();
    itopo           = topoMap[ielem];

    JIVE_CHECK_TOPO_INDEX ( itopo, topoCount );

    jnodes.ref ( elemTopos[itopo].getColumnIndices( ilocal ) );

    elems.getSomeElemNodes ( inodes, jnodes, ielem );

    n = jnodes.size ();

    for ( i = 0; i < n; i++ )
    {
      indices.pushBack ( inodes[i] );
    }
  }

  offsets[bndCount] = indices.size ();

  indices.trimToSize ();

  return Topology ( jem::shape( bndCount, nodeCount ),
                    offsets, indices.toArray() );
}


//-----------------------------------------------------------------------
//   findBuddies
//-----------------------------------------------------------------------


void BoundarySet::findBuddies

 ( const IdxVector&  buddies,
   const Topology&   elemTopo ) const

{
  Topology  bnTopo = getNodesPerBoundary ( elemTopo );

  jem::numeric::findSimilarRows ( buddies, bnTopo );
}


void BoundarySet::findBuddies

 ( const IdxVector&   buddies,
   const IdxVector&   topoMap,
   const TopoVector&  elemTopos ) const

{
  Topology  bnTopo = getNodesPerBoundary ( topoMap, elemTopos );

  jem::numeric::findSimilarRows ( buddies, bnTopo );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void BoundarySet::store

  ( const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  data_->store ( Globdat::BOUNDARIES, globdat );
}


void BoundarySet::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  data_->store ( name, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


BoundarySet BoundarySet::find

  ( const Properties&  globdat )

{
  return find ( Globdat::BOUNDARIES, globdat );
}


BoundarySet BoundarySet::find

  ( const String&      name,
    const Properties&  globdat )

{
  using jem::dynamicCast;
  using jive::util::ItemSet;

  Ref<Data>  data =

    dynamicCast<Data> ( ItemSet::find( name, globdat ) );

  if ( data && isValidData( data ) )
  {
    return BoundarySet ( data );
  }
  else
  {
    return BoundarySet ();
  }
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


BoundarySet BoundarySet::get

  ( const Properties&  globdat,
    const String&      context )

{
  BoundarySet  bounds = find ( Globdat::BOUNDARIES, globdat );

  if ( ! bounds )
  {
    throw jem::IllegalInputException (
      context,
      "no boundaries have been defined"
    );
  }

  return bounds;
}


BoundarySet BoundarySet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  BoundarySet  bounds = find ( name, globdat );

  if ( ! bounds )
  {
    throw jem::IllegalInputException (
      context,
      "undefined boundary set: " + name
    );
  }

  return bounds;
}


//-----------------------------------------------------------------------
//   isValidData
//-----------------------------------------------------------------------


bool BoundarySet::isValidData ( const Ref<Data>& data )
{
  using jem::Class;

  typedef ElementSet::Data  ElemData;
  typedef NodeSet::Data     NodeData;

  ItemSet*  items;
  Class*    type;


  if ( ! data )
  {
    return true;
  }

  items = data->getCompoundItems ();
  type  = items->getClass ();

  if ( ! ElemData::getType()->isBaseOf( type ) )
  {
    return false;
  }

  items = static_cast<ElemData*>( items )->getGroupedItems ();
  type  = items->getClass ();

  return NodeData::getType()->isBaseOf ( type );
}


JIVE_END_PACKAGE( fem )
