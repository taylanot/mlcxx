
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


#include <jem/base/IllegalArgumentException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/geom/StdBoundary.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/geom/ShapeBoundary.h>


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class ShapeBoundary
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


ShapeBoundary::ShapeBoundary ()
{}


ShapeBoundary::ShapeBoundary

  ( const BShapeVector&   sh,
    const SBShapeVector&  bd,
    const Topology&       tp ) :

    shapes   ( sh ),
    bounds   ( bd ),
    topology ( tp )

{
  if ( shapes.size() != bounds.size() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "size mismatch between the boundary shape array and "
      "the standard boundary shape array"
    );
  }

  if ( topology.size(0) != shapes.size() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "size mismatch between the boundary topology and "
      "the boundary shape array"
    );
  }
}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


ShapeBoundary& ShapeBoundary::operator = ( const ShapeBoundary& rhs )
{
  shapes.ref ( rhs.shapes );
  bounds.ref ( rhs.bounds );

  topology = rhs.topology;

  return *this;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


void                    decode

  ( ObjectInput&          in,
    ShapeBoundary&        sb )

{
  decode ( in, sb.topology, sb.bounds, sb.shapes );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


void                    encode

  ( ObjectOutput&         out,
    const ShapeBoundary&  sb )
{
  encode ( out, sb.topology, sb.bounds, sb.shapes );
}


JIVE_END_PACKAGE( geom )
