
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
#include <jive/util/Reordering.h>
#include <jive/util/XPointSet.h>


JEM_DEFINE_CLASS( jive::util::XPointSet );


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class XPointSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


XPointSet::XPointSet ( const String& name ) :

  Super ( name )

{}


XPointSet::~XPointSet ()
{}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void XPointSet::reserve ( idx_t )
{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void XPointSet::trimToSize ()
{}


//-----------------------------------------------------------------------
//   addPoints
//-----------------------------------------------------------------------


idx_t XPointSet::addPoints ( const Matrix& coords )
{
  const idx_t  k = this ->size ();
  const idx_t  n = coords.size (1);

  reserve ( k + n );

  for ( idx_t i = 0; i < n; i++ )
  {
    addPoint ( coords[i] );
  }

  return k;
}


//-----------------------------------------------------------------------
//   erasePoints
//-----------------------------------------------------------------------


void XPointSet::erasePoints ( const IdxVector& ipoints )
{
  reorderPoints ( makeEraseReordering( ipoints, size() ) );
}


//-----------------------------------------------------------------------
//   setCoords
//-----------------------------------------------------------------------


void XPointSet::setCoords ( const Matrix& coords )
{
  JEM_PRECHECK2 ( coords.size(0) == rank() &&
                  coords.size(1) >= size(),
                  "Array shape mismatch" );

  const idx_t  n = size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    setPointCoords ( i, coords[i] );
  }
}


//-----------------------------------------------------------------------
//   setXCoords
//-----------------------------------------------------------------------


void XPointSet::setXCoords

  ( const Vector&  coords,
    idx_t          idim )

{
  JEM_PRECHECK2 ( coords.size() >= size(),
                  "coordinate array too small" );

  const idx_t  n = size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    setPointXCoord ( i, coords[i], idim );
  }
}


//-----------------------------------------------------------------------
//   setSomeCoords
//-----------------------------------------------------------------------


void XPointSet::setSomeCoords

  ( const IdxVector&  ipoints,
    const Matrix&     coords )

{
  JEM_PRECHECK2 ( coords.size(0) == rank() &&
                  coords.size(1) >= ipoints.size(),
                  "Array shape mismatch" );

  const idx_t  n = ipoints.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    setPointCoords ( ipoints[i], coords[i] );
  }
}


//-----------------------------------------------------------------------
//   setSomeXCoords
//-----------------------------------------------------------------------


void XPointSet::setSomeXCoords

  ( const IdxVector&  ipoints,
    const Vector&     coords,
    idx_t               idim )

{
  JEM_PRECHECK2 ( coords.size() >= ipoints.size(),
                  "coordinate array is too small" );

  const idx_t  n = ipoints.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    setPointXCoord ( ipoints[i], coords[i], idim );
  }
}


JIVE_END_PACKAGE( util )
