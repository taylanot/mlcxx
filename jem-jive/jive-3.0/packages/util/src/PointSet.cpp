
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
#include <jem/io/PrintWriter.h>
#include <jive/util/ItemMap.h>
#include <jive/util/PointSet.h>


JEM_DEFINE_CLASS( jive::util::PointSet );


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class PointSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PointSet::PointSet ( const String& name ) :

  Super ( name )

{}


PointSet::~PointSet ()
{}


//-----------------------------------------------------------------------
//   getItemName
//-----------------------------------------------------------------------


String PointSet::getItemName () const
{
  return "point";
}


//-----------------------------------------------------------------------
//   getCoords
//-----------------------------------------------------------------------


void PointSet::getCoords ( const Matrix& coords ) const
{
  JEM_PRECHECK2 ( coords.size(1) >= size(),
                  "output array is too small" );

  const idx_t  n = size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    getPointCoords ( coords[i], i );
  }
}


//-----------------------------------------------------------------------
//   getSomeCoords
//-----------------------------------------------------------------------


void PointSet::getSomeCoords

  ( const Matrix&     coords,
    const IdxVector&  ipoints ) const

{
  JEM_PRECHECK2 ( coords.size(1) >= ipoints.size(),
                  "output array is too small" );

  const idx_t  n = ipoints.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    getPointCoords ( coords[i], ipoints[i] );
  }
}


//-----------------------------------------------------------------------
//   getXCoords
//-----------------------------------------------------------------------


void PointSet::getXCoords

  ( const Vector&  coords,
    idx_t          idim ) const

{
  JEM_PRECHECK2 ( coords.size() >= size(),
                  "output array is too small" );

  const idx_t  n = size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    coords[i] = getPointXCoord ( i, idim );
  }
}


//-----------------------------------------------------------------------
//   getSomeXCoords
//-----------------------------------------------------------------------


void PointSet::getSomeXCoords

  ( const Vector&     coords,
    const IdxVector&  ipoints,
    idx_t             idim ) const

{
  JEM_PRECHECK2 ( coords.size() >= ipoints.size(),
                  "output array is too small" );

  const idx_t  n = ipoints.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    coords[i] = getPointXCoord ( ipoints[i], idim );
  }
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


Matrix PointSet::toMatrix () const
{
  Matrix c ( rank(), size() );

  getCoords ( c );

  return c;
}


//-----------------------------------------------------------------------
//   printSubset
//-----------------------------------------------------------------------


void PointSet::printSubset

  ( PrintWriter&      out,
    const IdxVector&  ipoints ) const

{
  using jem::io::endl;
  using jem::io::space;

  const idx_t  m = this  ->rank ();
  const idx_t  n = ipoints.size ();

  if ( m > 0 )
  {
    const ItemMap&  imap = * getItemMap ();

    Vector          buf  ( m );

    for ( idx_t j = 0; j < n; j++ )
    {
      if ( j > 0 )
      {
        print ( out, endl );
      }

      idx_t  ip = ipoints[j];

      print ( out, imap.getItemID( ip ) );

      this->getPointCoords ( buf, ip );

      for ( idx_t i = 0; i < m; i++ )
      {
        print ( out, space, buf[i] );
      }

      print ( out, ';' );
    }
  }
}


JIVE_END_PACKAGE( util )
