
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
#include <jem/base/RuntimeException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/util/utilities.h>
#include <jive/geom/ShapeTable.h>
#include <jive/fem/VirtualElement.h>
#include <jive/fem/ElementIterator.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::ElementIterator );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class ElementIterator
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ElementIterator::ElementIterator ()
{
  reset_ ();
}


ElementIterator::ElementIterator

  ( const ElementSet&       elems,
    const Ref<ShapeTable>&  shapes,
    const String&           context ) :

    Super     ( shapes, context ),
    elemGroup ( newElementGroup( ALL, elems ) )

{
  init_ ();
}


ElementIterator::ElementIterator

  ( const ElementGroup&     group,
    const Ref<ShapeTable>&  shapes,
    const String&           context ) :

    Super     ( shapes, context ),
    elemGroup ( group )

{
  JEM_PRECHECK ( group );

  init_ ();
}


ElementIterator::ElementIterator

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const String&      context ) :

    elemGroup ( ElementGroup::get ( conf, props,
                                    globdat, context ) )

{
  ElementSet  elems = elemGroup.getElements ();

  context_    = context;
  shapeTable_ = ShapeTable::get ( elems.getData(),
                                  globdat, context );

  init_ ();
}


ElementIterator::~ElementIterator ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ElementIterator::readFrom ( ObjectInput& in )
{
  Super::readFrom_ ( in );

  decode ( in, elemGroup );
  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ElementIterator::writeTo ( ObjectOutput& out ) const
{
  Super::writeTo_ ( out );

  encode ( out, elemGroup );
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void ElementIterator::resetEvents ()
{
  elemGroup   .resetEvents ();
  shapeTable_->resetEvents ();
}


//-----------------------------------------------------------------------
//   rewind
//-----------------------------------------------------------------------


void ElementIterator::rewind ()
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


bool ElementIterator::toNext ( VirtualElement& elem )
{
  if ( inext_ >= iperm_.size() )
  {
    return false;
  }

  ipos = iperm_[inext_];

  if ( ipos < 0 )
  {
    ipos = -ipos - 1;

    if ( inext_ == 0 )
    {
      elem.reset ();
    }

    elem.setShape ( shapes_[shapeMap_[ipos]].get() );
  }

  elem.loadData ( ielems_[ipos] );

  inext_++;

  return true;
}


//-----------------------------------------------------------------------
//   checkShapes
//-----------------------------------------------------------------------


void ElementIterator::checkShapes ()
{
  shapeTable_->checkShapesOf ( context_,
                               elemGroup.getIndices (),
                               shapeFlags_, precision_ );
}


//-----------------------------------------------------------------------
//   getShapes
//-----------------------------------------------------------------------


ShapeVector ElementIterator::getShapes () const
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


IdxVector ElementIterator::getShapeMap () const
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


void ElementIterator::init_ ()
{
  using jem::util::connect;

  reset_  ();

  connect ( elemGroup.newSizeEvent  (), this, & Self::invalidate_ );
  connect ( elemGroup.newOrderEvent (), this, & Self::invalidate_ );
  connect ( shapeTable_->changeEvent,   this, & Self::invalidate_ );

  updated_ = false;
}


//-----------------------------------------------------------------------
//   reset_
//-----------------------------------------------------------------------


inline void ElementIterator::reset_ ()
{
  ipos   = -1_idx;
  inext_ =  0_idx;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void ElementIterator::update_ ()
{
  using jive::util::sortColors;

  IdxVector  offsets;
  idx_t      i, j, n;


  checkPrecision_ ();

  ielems_.ref ( elemGroup.getIndices() );

  shapeTable_->checkShapesOf ( context_,    ielems_,
                               shapeFlags_, precision_ );

  n = ielems_.size ();

  iperm_   .resize ( n );
  shapeMap_.resize ( n );

  shapes_.ref (
    shapeTable_->getShapesOf ( shapeMap_, ielems_, precision_ )
  );

  offsets.ref ( sortColors( iperm_, shapeMap_ ) );

  n = offsets.size() - 1;

  for ( i = 0; i < n; i++ )
  {
    j         = offsets[i];
    iperm_[j] = -iperm_[j] - 1;
  }

  resetEvents ();

  updated_ = true;
}


JIVE_END_PACKAGE( fem )
