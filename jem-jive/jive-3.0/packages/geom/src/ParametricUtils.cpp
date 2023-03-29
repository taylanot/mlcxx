
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


#include <jem/base/array/utilities.h>
#include <jem/base/IllegalArgumentException.h>
#include <jive/util/error.h>
#include <jive/geom/StdBoundary.h>
#include <jive/geom/ShapeBoundary.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/geom/ParametricUtils.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::IllegalArgumentException;


//=======================================================================
//   class ParametricUtils
//=======================================================================

//-----------------------------------------------------------------------
//   checkShapes
//-----------------------------------------------------------------------


void ParametricUtils::checkShapes

  ( const String&    context,
    const StdShape&  xshape,
    const StdShape&  sshape )

{
  if ( xshape.shapeFuncCount() == 0 )
  {
    throw IllegalArgumentException (
      context,
      "xshape has no shape functions"
    );
  }

  if ( xshape.getGeometry() != sshape.getGeometry() )
  {
    throw IllegalArgumentException (
      context,
      String::format (
        "xshape geometry (%s) and "
        "sshape geometry (%s) do not match",
        xshape.getGeometry (),
        sshape.getGeometry ()
      )
    );
  }

  if ( xshape.rank() != sshape.rank() )
  {
    throw IllegalArgumentException (
      context,
      String::format (
        "xshape rank (%d) and sshape rank (%d) do not match",
        xshape.rank (),
        sshape.rank ()
      )
    );
  }
}


//-----------------------------------------------------------------------
//   checkIScheme
//-----------------------------------------------------------------------


void ParametricUtils::checkIScheme

  ( const String&    context,
    const StdShape&  xshape,
    const Matrix&    ischeme )

{
  if ( ischeme.size(0) - 1 != xshape.rank() )
  {
    throw IllegalArgumentException (
      context,
      String::format (
        "rank of integration points (%d) and "
        "rank of xshape (%d) do not match",
        ischeme.size(0) - 1,
        xshape .rank()
      )
    );
  }
}


//-----------------------------------------------------------------------
//   checkBounds
//-----------------------------------------------------------------------


void ParametricUtils::checkBounds

  ( const String&         context,
    const StdShape&       xshape,
    const StdShape&       sshape,
    const ShapeBoundary&  bnd )

{
  IdxVector    offsets    = bnd.topology.getRowOffsets    ();
  IdxVector    indices    = bnd.topology.getColumnIndices ();

  const idx_t  rank       = xshape.rank           ();
  const idx_t  nodeCount  = xshape.shapeFuncCount ();
  const idx_t  funcCount  = sshape.shapeFuncCount ();
  const idx_t  bndCount   = bnd.shapes.size       ();


  if ( bnd.bounds.size() != bndCount )
  {
    throw IllegalArgumentException (
      context,
      "illegal standard boundary shape array (wrong size)"
    );
  }

  if ( bnd.topology.size(0) != bndCount )
  {
    throw IllegalArgumentException (
      context,
      "illegal boundary topology specified (wrong shape)"
    );
  }

  if ( bnd.topology.size(1) != nodeCount )
  {
    throw IllegalArgumentException (
      context,
      "illegal boundary topology specified (wrong shape)"
    );
  }

  if ( min( indices ) < 0 || max( indices ) >= nodeCount )
  {
    throw IllegalArgumentException (
      context,
      "illegal boundary topology specified "
      "(column index out of range)"
    );
  }

  for ( idx_t ibnd = 0; ibnd < bndCount; ibnd++ )
  {
    if ( ! bnd.shapes[ibnd] )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "null boundary shape specified "
          "(bnd.shapes[%d] == nullptr)",
          ibnd
        )
      );
    }

    if ( ! bnd.bounds[ibnd] )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "null standard boundary shape specified "
          "(bnd.shapes[%d] == nullptr)",
          ibnd
        )
      );
    }

    const BShape&  bshape = * bnd.shapes[ibnd];

    if ( bshape.globalRank() != rank )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "boundary shape %d has global rank %d, "
          "but should have rank %d",
          ibnd,
          bshape.globalRank (),
          rank
        )
      );
    }

    if ( bshape.shapeFuncCount() != funcCount )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "boundary %d has %d shape funcs, "
          "but should have %d shape funcs",
          ibnd,
          bshape.shapeFuncCount (),
          funcCount
        )
      );
    }

    if ( bnd.bounds[ibnd]->rank() != rank - 1 )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "standard boundary %d has rank %d, "
          "but should have rank %d",
          ibnd,
          bnd.bounds[ibnd]->rank (),
          rank - 1
        )
      );
    }

    idx_t  n = offsets[ibnd + 1] - offsets[ibnd];

    if ( bshape.nodeCount() != n )
    {
      throw IllegalArgumentException (
        context,
        String::format (
          "boundary %d has %d nodes, but should have %d nodes",
          ibnd,
          bshape.nodeCount (),
          n
        )
      );
    }
  }
}


JIVE_END_PACKAGE( geom )
