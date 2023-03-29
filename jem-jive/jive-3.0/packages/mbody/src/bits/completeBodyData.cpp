
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
#include <jem/base/limits.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/intrinsics.h>
#include <jive/geom/Shape.h>
#include <jive/geom/ShapeTable.h>
#include <jive/fem/Element.h>
#include <jive/mbody/utilities.h>
#include <jive/mbody/BodyData.h>


using jive::fem::Element;
using jive::geom::ShapeMatrix;


JIVE_BEGIN_PACKAGE( mbody )


//=======================================================================
//   class BodyDataComputer
//=======================================================================


class BodyDataComputer
{
 public:

  static const int        LO_PREC = 1;
  static const int        HI_PREC = 2;


                          BodyDataComputer

    ( const BodyData&       bdata,
      const ShapeTable&     shapes );

  void                    computeData ();


 private:

  void                    getCenter_

    ( double&               volume,
      const Vector&         center,
      const IdxVector&      ielems );

  void                    getInertia2D_

    ( double&               rmass,
      const Vector&         center,
      const IdxVector&      ielems );

  void                    getInertia3D_

    ( const Vector&         rmass,
      const Vector&         center,
      const IdxVector&      ielems );

  bool                    initElement_

    ( Element&              elem,
      idx_t                 ielem,
      idx_t                 prec  );


 private:

  BodyData                bdata_;
  Ref<Element>            elem_;

  IdxVector               shapeMap_;
  ShapeMatrix             shapes_;

  idx_t                   maxPrec_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


BodyDataComputer::BodyDataComputer

  ( const BodyData&    bdata,
    const ShapeTable&  shapes ) :

    bdata_ ( bdata )

{
  BodySet  bodies = bdata.getBodies ();


  elem_    = jem::newInstance<Element> ( bodies.getElements() );
  maxPrec_ = jem::min ( (idx_t) HI_PREC, shapes.maxPrecision() );

  shapeMap_.resize ( elem_->elems.size() );
  shapes_  .ref    ( shapes.getShapes( shapeMap_, 0, maxPrec_ ) );
}


//-----------------------------------------------------------------------
//   computeData
//-----------------------------------------------------------------------


void BodyDataComputer::computeData ()
{
  BodyData
    ::StdColumns  cols      = bdata_.getStdColumns ();

  BodySet         bodies    = bdata_.getBodies     ();

  const idx_t     rank      = bodies.rank ();
  const idx_t     bodyCount = bodies.size ();

  IdxVector       ielembuf  ( bodies.maxBodyElemCount() );
  IdxVector       ielems;

  Vector          buf       (    6 );
  Vector          center    ( rank );
  Vector          rmass;

  double          volume;
  double          mass;
  double          density;

  idx_t           ibody;
  idx_t           n;


  if ( rank == 2 )
  {
    rmass.resize ( 1 );
  }
  else
  {
    rmass.resize ( 6 );
  }

  for ( ibody = 0; ibody < bodyCount; ibody++ )
  {
    n = bodies.getBodyElems ( ielembuf, ibody );

    ielems.ref ( ielembuf[slice(BEGIN,n)] );

    getCenter_ ( volume, center, ielems );

    if ( cols.mass->findValues( buf, ibody ) == 1 )
    {
      mass    = buf[0];
      density = mass / volume;

      cols.density->setValue ( ibody, density );
    }
    else
    {
      density = cols.density->getValue ( ibody );
      mass    = density * volume;

      cols.mass->setValue ( ibody, mass );
    }

    if ( cols.center->findValues( buf, ibody ) == rank )
    {
      center = buf[slice(BEGIN,rank)];
    }
    else
    {
      cols.center->setValues ( ibody, center );
    }

    n = cols.rmass->findValues ( buf, ibody );

    if ( rank == 2 )
    {
      if ( n != 1 )
      {
        getInertia2D_ ( rmass[0], center, ielems );

        rmass[0] *= density;

        cols.rmass->setValue ( ibody, rmass[0] );
      }
    }
    else if ( rank == 3 )
    {
      if ( n != 3 && n != 6 )
      {
        getInertia3D_ ( rmass, center, ielems );

        rmass *= density;

        cols.rmass->setValues ( ibody, rmass );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getCenter_
//-----------------------------------------------------------------------


void BodyDataComputer::getCenter_

  ( double&           volume,
    const Vector&     center,
    const IdxVector&  ielems )

{
  const idx_t  n = ielems.size ();

  Element&     e = * elem_;

  double       w;

  idx_t        prec;
  idx_t        ielem;
  idx_t        ie;
  idx_t        ip;
  idx_t        i;


  prec   = jem::min ( (idx_t) LO_PREC, maxPrec_ );
  volume = 0.0;
  center = 0.0;

  for ( ie = 0; ie < n; ie++ )
  {
    ielem = ielems[ie];

    if ( ! initElement_( e, ielem, prec ) )
    {
      continue;
    }

    for ( ip = 0; ip < e.pntCount; ip++ )
    {
      w       = e.pntWeights[ip];
      volume += w;

      for ( i = 0; i < e.rank; i++ )
      {
        center[i] += (w * e.xpoints(i,ip));
      }
    }
  }

  if ( jem::isTiny( volume ) )
  {
    volume = 1.0;
  }

  center = center / volume;
}


//-----------------------------------------------------------------------
//   getInertia2D_
//-----------------------------------------------------------------------


void BodyDataComputer::getInertia2D_

  ( double&           rmass,
    const Vector&     center,
    const IdxVector&  ielems )

{
  const idx_t  n = ielems.size ();

  Element&     e = * elem_;

  double       dx, dy;
  double       w;

  idx_t        prec;
  idx_t        ielem;
  idx_t        ie;
  idx_t        ip;


  prec  = jem::min ( (idx_t) HI_PREC, maxPrec_ );
  rmass = 0.0;

  for ( ie = 0; ie < n; ie++ )
  {
    ielem = ielems[ie];

    if ( ! initElement_( e, ielem, prec ) )
    {
      continue;
    }

    for ( ip = 0; ip < e.pntCount; ip++ )
    {
      w  = e.pntWeights[ip];
      dx = e.xpoints(0,ip) - center[0];
      dy = e.xpoints(1,ip) - center[1];

      rmass += w * (dx * dx + dy * dy);
    }
  }

  if ( rmass <= 0.0 )
  {
    rmass = 1.0;
  }
}


//-----------------------------------------------------------------------
//   getInertia3D_
//-----------------------------------------------------------------------


void BodyDataComputer::getInertia3D_

  ( const Vector&     rmass,
    const Vector&     center,
    const IdxVector&  ielems )

{
  const idx_t  n = ielems.size ();

  Element&     e = * elem_;

  double       dx, dy, dz;
  double       w;

  idx_t        prec;
  idx_t        ielem;
  idx_t        ie;
  idx_t        ip;
  idx_t        i;


  prec  = jem::min ( (idx_t) HI_PREC, maxPrec_ );
  rmass = 0.0;

  for ( ie = 0; ie < n; ie++ )
  {
    ielem = ielems[ie];

    if ( ! initElement_( e, ielem, prec ) )
    {
      continue;
    }

    for ( ip = 0; ip < e.pntCount; ip++ )
    {
      w  = e.pntWeights[ip];
      dx = e.xpoints(0,ip);
      dy = e.xpoints(1,ip);
      dz = e.xpoints(2,ip);

      rmass[0] += w * (dy * dy + dz * dz);
      rmass[1] += w * (dx * dx + dz * dz);
      rmass[2] += w * (dx * dx + dy * dy);
      rmass[3] += w * dx * dy;
      rmass[4] += w * dx * dz;
      rmass[5] += w * dy * dz;
    }
  }

  for ( i = 0; i < 3; i++ )
  {
    if ( rmass[i] <= 0.0 )
    {
      rmass[i] = 1.0;
    }
  }
}


//-----------------------------------------------------------------------
//   initElement_
//-----------------------------------------------------------------------


bool BodyDataComputer::initElement_

  ( Element&  e,
    idx_t     ielem,
    idx_t     prec )

{
  using jive::geom::Shape;

  Shape*  shape;

  idx_t   i, j;


  j = shapeMap_[ielem];

  if ( j < 0 )
  {
    return false;
  }

  shape = shapes_(prec,j).get ();

  if ( ! shape )
  {
    for ( i = prec - 1; i >= 0; i-- )
    {
      shape = shapes_(i,j).get ();

      if ( shape )
      {
        break;
      }
    }
  }

  if ( ! shape )
  {
    return false;
  }

  if ( shape != e.shape.get() )
  {
    e.setShape ( shape );
  }

  e.loadDataFast ( ielem );

  e.getPointWeights ();
  e.getGlobalPoints ();

  return true;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   completeBodyData
//-----------------------------------------------------------------------


void                  completeBodyData

  ( const BodyData&     bdata,
    const ShapeTable&   shapes )

{
  JEM_PRECHECK ( bdata );

  BodyDataComputer  bcomp ( bdata, shapes );

  bcomp.computeData ();
}


JIVE_END_PACKAGE( mbody )
