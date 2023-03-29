
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
#include <jem/base/array/utilities.h>
#include <jem/util/SparseArray.h>
#include <jive/util/XItemMap.h>
#include <jive/util/StdMemberSet.h>
#include <jive/fem/check.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/XBoundarySet.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::shape;
using jem::iarray;
using jem::newInstance;

typedef jem::util::SparseArray<int,2>  MaskMatrix;


//=======================================================================
//   class XBoundarySet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  XBoundarySet::CLASS_NAME_ = "jive::fem::XBoundarySet";


//-----------------------------------------------------------------------
//   addBoundaries
//-----------------------------------------------------------------------


idx_t XBoundarySet::addBoundaries ( const Topology& elemTopo ) const
{
  const idx_t  elemCount = getElements().size ();

  IdxVector    ielems    ( iarray( elemCount ) );

  return addBoundariesOf ( ielems, elemTopo );
}


idx_t XBoundarySet::addBoundaries

  ( const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  const idx_t  elemCount = getElements().size ();

  IdxVector    ielems    ( iarray( elemCount ) );

  return addBoundariesOf ( ielems, topoMap, elemTopos );
}


//-----------------------------------------------------------------------
//   addBoundariesOf (given one element topology)
//-----------------------------------------------------------------------


idx_t XBoundarySet::addBoundariesOf

  ( const IdxVector&  ielems,
    const Topology&   elemTopo ) const

{
  JEM_PRECHECK ( *this );

  ElementSet   elems      = getElements   ();

  const idx_t  bndCount   =          size ();
  const idx_t  elemCount  = elems   .size ();
  const idx_t  ielemCount = ielems  .size ();
  const idx_t  localCount = elemTopo.size (0);

  MaskMatrix   mask;

  idx_t        ielem;
  idx_t        ibound;
  idx_t        ilocal;
  idx_t        ie;


  reserve      ( bndCount + ielemCount );
  mask.reshape ( shape( elemCount, 8 ) );
  mask.reserve ( bndCount );

  for ( ibound = 0; ibound < bndCount; ibound++ )
  {
    getBoundary ( ielem, ilocal, ibound );
    mask(ielem,ilocal) = 1;
  }

  for ( ie = 0; ie < ielemCount; ie++ )
  {
    ielem = ielems[ie];

    for ( ilocal = 0; ilocal < localCount; ilocal++ )
    {
      if ( ! mask.find( ielem, ilocal ) )
      {
        addBoundary ( ielem, ilocal );
      }
    }
  }

  trimToSize ();

  return bndCount;
}


//-----------------------------------------------------------------------
//   addBoundariesOf (given multiple element topologies)
//-----------------------------------------------------------------------


idx_t XBoundarySet::addBoundariesOf

  ( const IdxVector&   ielems,
    const IdxVector&   topoMap,
    const TopoVector&  elemTopos ) const

{
  JEM_PRECHECK ( *this );

  ElementSet   elems      = getElements  ();

  const idx_t  bndCount   =           size ();
  const idx_t  topoCount  = elemTopos.size ();
  const idx_t  elemCount  = elems    .size ();
  const idx_t  ielemCount = ielems   .size ();

  MaskMatrix   mask;

  idx_t        localCount;
  idx_t        ielem;
  idx_t        ibound;
  idx_t        ilocal;
  idx_t        itopo;
  idx_t        ie;


  JIVE_CHECK_TOPO_MAP ( topoMap, ielems );

  reserve      ( bndCount + ielemCount );
  mask.reshape ( shape( elemCount, 8 ) );
  mask.reserve ( bndCount );

  for ( ibound = 0; ibound < bndCount; ibound++ )
  {
    getBoundary ( ielem, ilocal, ibound );

    mask(ielem,ilocal) = 1;
  }

  for ( ie = 0; ie < ielemCount; ie++ )
  {
    ielem = ielems[ie];
    itopo = topoMap[ie];

    JIVE_CHECK_TOPO_INDEX ( itopo, topoCount );

    localCount = elemTopos[itopo].size(0);

    for ( ilocal = 0; ilocal < localCount; ilocal++ )
    {
      if ( ! mask.find( ielem, ilocal ) )
      {
        addBoundary ( ielem, ilocal );
      }
    }
  }

  trimToSize ();

  return bndCount;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


XBoundarySet XBoundarySet::get

  ( const Properties&  globdat,
    const String&      context )

{
  XBoundarySet  bounds =

    toXBoundarySet ( Super::get( globdat, context ) );

  if ( ! bounds )
  {
    throw jem::IllegalInputException (
      context,
      "the standard boundary set is not editable "
      "(not an XBoundarySet)"
    );
  }

  return bounds;
}


XBoundarySet XBoundarySet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  XBoundarySet  bounds =

    toXBoundarySet ( Super::get( name, globdat, context ) );

  if ( ! bounds )
  {
    throw jem::IllegalInputException (
      context,
      String::format (
        "the boundary set `%s\' is not editable "
        "(not an XBoundarySet)",
        name
      )
    );
  }

  return bounds;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newXBoundarySet
//-----------------------------------------------------------------------


XBoundarySet          newXBoundarySet

  ( const ElementSet&   elems,
    util::StorageMode   mode )

{
  JEM_PRECHECK ( elems );

  return XBoundarySet (
    newInstance<util::StdMemberSet> (
      "boundaries",
      "boundary",
      elems.getData     (),
      util::newXItemMap ( mode )
    )
  );
}


XBoundarySet          newXBoundarySet

  ( const String&       name,
    const ElementSet&   elems,
    util::StorageMode   mode )

{
  JEM_PRECHECK ( elems );

  return XBoundarySet (
    newInstance<util::StdMemberSet> (
      name,
      "boundary",
      elems.getData     (),
      util::newXItemMap ( mode )
    )
  );
}


JIVE_END_PACKAGE( fem )
