
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jive/fem/InternalBoundary.h>


JEM_DEFINE_CLASS( jive::fem::InternalBoundary );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class InternalBoundary
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  InternalBoundary::PRECISION = 1.0e-5;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


InternalBoundary::InternalBoundary ( const BoundarySet& b )
{
  first_     = newInstance<Boundary> ( b );
  second_    = newInstance<Boundary> ( b );
  precision_ = PRECISION;
}


InternalBoundary::InternalBoundary

  ( const Ref<Boundary>&  first,
    const Ref<Boundary>&  second ) :

    first_  ( first  ),
    second_ ( second )

{
  JEM_PRECHECK ( first && second );

  precision_ = PRECISION;
}


InternalBoundary::~InternalBoundary ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void InternalBoundary::reset ()
{
  first_ ->reset ();
  second_->reset ();
}


//-----------------------------------------------------------------------
//   setShapes
//-----------------------------------------------------------------------


void InternalBoundary::setShapes

  ( geom::IShape*  shape0,
    idx_t            ilocal,
    geom::IShape*  shape1,
    idx_t            jlocal )

{
  Boundary&  b0 = * first_;
  Boundary&  b1 = * second_;
  IElement&  e0 = * b0.getElement ();
  IElement&  e1 = * b1.getElement ();


  if ( shape0 != e0.shape.get() || ilocal != b0.ilocal )
  {
    b0.setShape ( shape0, ilocal );
  }

  if ( shape1 != e1.shape.get() || jlocal != b1.ilocal )
  {
    b1.setShape ( shape1, jlocal );
  }
}


//-----------------------------------------------------------------------
//   loadData
//-----------------------------------------------------------------------


void InternalBoundary::loadData

  ( idx_t  ibound,
    idx_t  ielem,
    idx_t  jbound,
    idx_t  jelem )

{
  Boundary&    b0 = * first_;
  Boundary&    b1 = * second_;

  const idx_t  r  = b0.getElement()->rank;
  const idx_t  n  = b0.nodeCount;

  double*      xmin;
  double*      xmax;
  double       x, dx;
  double       eps;

  idx_t        i, j;


  b0.loadData ( ibound, ielem );
  b1.loadData ( jbound, jelem );

  if ( 2 * r > xbuf_.size() )
  {
    xbuf_.resize ( 2 * r );
  }

  xmin = xbuf_.addr ();
  xmax = xmin + r;
  eps  = 0.0;

  if ( n > 0 )
  {
    for ( i = 0; i < r; i++ )
    {
      xmin[i] = xmax[i] = b0.xnodes(i,0);
    }

    for ( j = 1; j < n; j++ )
    {
      for ( i = 0; i < r; i++ )
      {
        x = b0.xnodes(i,j);

        if      ( x < xmin[i] )
        {
          xmin[i] = x;
        }
        else if ( x > xmax[i] )
        {
          xmax[i] = x;
        }
      }
    }

    for ( i = 0; i < r; i++ )
    {
      dx   = xmax[i] - xmin[i];
      eps += dx * dx;
    }

    eps = std::sqrt ( eps );
  }

  eps *= precision_;

  b1.setGlobalPoints ( b0.getGlobalPoints(), eps );
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void InternalBoundary::setPrecision ( double prec )
{
  JEM_PRECHECK ( prec > 0.0 );

  precision_ = prec;
}


JIVE_END_PACKAGE( fem )
