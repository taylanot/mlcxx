
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
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/util/utilities.h>
#include <jive/geom/ShapeTable.h>
#include <jive/geom/InternalShape.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/fem/VirtualIBoundary.h>
#include <jive/fem/IBoundaryIterator.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::IBoundaryIterator );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class IBoundaryIterator
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


IBoundaryIterator::IBoundaryIterator ()
{
  reset_ ();
}


IBoundaryIterator::IBoundaryIterator

  ( const BoundaryGroup&    group,
    const Ref<ShapeTable>&  shapes,
    const String&           context ) :

    Super    ( shapes, context ),
    bndGroup ( group )

{
  init_ ();
}


IBoundaryIterator::IBoundaryIterator

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const String&      context ) :

    bndGroup  (
      BoundaryGroup::get ( conf, props, globdat, context )
    )

{
  ElementSet  elems = getElements ();

  context_    = context;
  shapeTable_ = ShapeTable::get ( elems.getData (),
                                  globdat, context );

  init_ ();
}


IBoundaryIterator::~IBoundaryIterator ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void IBoundaryIterator::readFrom ( ObjectInput& in )
{
  Super::readFrom_ ( in );

  decode ( in, bndGroup );
  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void IBoundaryIterator::writeTo ( ObjectOutput& out ) const
{
  Super::writeTo_ ( out );
  encode          ( out, bndGroup );
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void IBoundaryIterator::resetEvents ()
{
  bndGroup    .resetEvents ();
  shapeTable_->resetEvents ();
}


//-----------------------------------------------------------------------
//   rewind
//-----------------------------------------------------------------------


void IBoundaryIterator::rewind ()
{
  if ( ! updated_ )
  {
    update_ ();
  }

  reset_ ();
}


//-----------------------------------------------------------------------
//   toNext
//-----------------------------------------------------------------------


bool IBoundaryIterator::toNext ( VirtualIBoundary&  bnd )
{
  if ( inext_ >= iperm_.size() )
  {
    return false;
  }

  ipos = iperm_[inext_];
  jpos = iperm_[inext_ + 1];

  if ( ipos < 0 )
  {
    ipos = -ipos - 1;

    if ( inext_ == 0 )
    {
      bnd.reset ();
    }

    bnd.setShapes ( shapes_ [shapeMap_[ipos]].get (),
                    ilocals_[ipos],
                    shapes_ [shapeMap_[jpos]].get (),
                    ilocals_[jpos] );
  }

  bnd.loadData ( ibounds_[ipos], ielems_[ipos],
                 ibounds_[jpos], ielems_[jpos] );

  inext_ += 2;

  return true;
}


//-----------------------------------------------------------------------
//   checkShapes
//-----------------------------------------------------------------------


void IBoundaryIterator::checkShapes ()
{
  if ( ! updated_ )
  {
    update_ ();
  }
}


//-----------------------------------------------------------------------
//   setShapeFlags
//-----------------------------------------------------------------------


void IBoundaryIterator::setShapeFlags ( int flags )
{
  Super::setShapeFlags ( flags | ShapeTable::INTERNAL_TYPE );
}


//-----------------------------------------------------------------------
//   getShapes
//-----------------------------------------------------------------------


IShapeVector IBoundaryIterator::getShapes () const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return shapes_;
}


//-----------------------------------------------------------------------
//   getShapeMap
//-----------------------------------------------------------------------


IdxVector IBoundaryIterator::getShapeMap () const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return shapeMap_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void IBoundaryIterator::init_ ()
{
  using jem::util::connect;

  shapeFlags_ |= ShapeTable::INTERNAL_TYPE;

  reset_  ();

  connect ( bndGroup.newSizeEvent  (), this, & Self::invalidate_ );
  connect ( bndGroup.newOrderEvent (), this, & Self::invalidate_ );
  connect ( shapeTable_->changeEvent,  this, & Self::invalidate_ );

  updated_ = false;
}


//-----------------------------------------------------------------------
//   reset_
//-----------------------------------------------------------------------


void IBoundaryIterator::reset_ ()
{
  ipos   = -1;
  jpos   = -1;
  inext_ =  0;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void IBoundaryIterator::update_ ()
{
  using jive::util::sortColors;

  BoundarySet  bounds = bndGroup.getBoundaries ();

  IdxVector    jperm;
  IdxVector    colors;
  IdxVector    offsets;

  idx_t        ipCount[2];
  idx_t        jshape[2];
  idx_t        ilocal[2];
  idx_t        st[4];

  idx_t        first;
  idx_t        second;

  idx_t        i, j, k, n;


  checkPrecision_ ();

  ibounds_.ref ( bndGroup.getIndices() );

  n = ibounds_.size ();

  if ( n % 2 > 0 )
  {
    throw jem::IllegalInputException (
      context_,
      String::format (
        "internal boundary group has wrong size: %d "
        "(should be even)", n
      )
    );
  }

  shapeMap_.resize ( n );
  ielems_  .resize ( n );
  ilocals_ .resize ( n );

  for ( i = 0; i < n; i++ )
  {
    bounds.getBoundary ( ielems_[i], ilocals_[i], ibounds_[i] );
  }

  shapeTable_->checkShapesOf ( context_,    ielems_,
                               shapeFlags_, precision_ );

  shapes_.ref (
    geom::toIShapes (
      shapeTable_->getShapesOf ( shapeMap_, ielems_, precision_ )
    )
  );

  if ( n == 0 )
  {
    st[0] = st[1] = st[2] = st[3] = 0;
  }
  else
  {
    st[0] = 1;
    st[1] = 1 + max ( ilocals_ );
    st[2] = st[1] * st[1];
    st[3] = st[2] * shapes_.size ();
  }

  iperm_.resize ( n );
  jperm .resize ( n );
  colors.resize ( n / 2 );

  for ( i = j = 0; i < n; i += 2, j++ )
  {
    for ( k = 0; k < 2; k++ )
    {
      jshape [k] = shapeMap_[i + k];
      ilocal [k] = ilocals_ [i + k];
      ipCount[k] =

        shapes_[jshape[k]] -> getBoundaryShape ( ilocal[k] )
                           -> ipointCount      ();
    }

    if ( ipCount[0] >= ipCount[1] )
    {
      first  = 0;
      second = 1;
    }
    else
    {
      first  = 1;
      second = 0;
    }

    jperm[i + 1] = first;

    colors[j] =

      st[0] * ilocal[first]  +
      st[1] * ilocal[second] +
      st[2] * jshape[first]  +
      st[3] * jshape[second];
  }

  offsets.ref ( sortColors( jperm[slice(ALL,2)], colors ) );

  for ( i = 0; i < n; i += 2 )
  {
    k     = 2 * jperm[i];
    first = jperm[k + 1];

    if ( first )
    {
      second = 0;
    }
    else
    {
      second = 1;
    }

    JEM_ASSERT ( first + second == 1 );

    iperm_[i + 0] = k + first;
    iperm_[i + 1] = k + second;
  }

  n = offsets.size() - 1;

  for ( i = 0; i < n; i++ )
  {
    j         = 2 * offsets[i];
    iperm_[j] = -iperm_[j] - 1;
  }

  resetEvents ();

  updated_ = true;
}


JIVE_END_PACKAGE( fem )
