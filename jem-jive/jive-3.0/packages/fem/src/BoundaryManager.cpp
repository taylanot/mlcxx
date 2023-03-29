
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
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/log.h>
#include <jive/geom/ShapeTable.h>
#include <jive/fem/AutoBndGroup.h>
#include <jive/fem/BoundaryManager.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::BoundaryManager );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jive::util::COMPACT_STORAGE;
using jive::util::TopoVector;


//=======================================================================
//   class BoundaryManager
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  BoundaryManager::NEW_BUDDIES_ = 1 << 0;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BoundaryManager::BoundaryManager ()
{
  updated_ = false;
}


BoundaryManager::BoundaryManager

  ( const ElementSet&       elems,
    const Ref<ShapeTable>&  shapes ) :

    egroup_ ( newElementGroup( ALL, elems ) ),
    bounds_ ( newXBoundarySet( elems, COMPACT_STORAGE ) ),
    shapes_ ( shapes )

{
  JEM_PRECHECK ( shapes );

  init_ ();
}


BoundaryManager::BoundaryManager

  ( const ElementGroup&     group,
    const Ref<ShapeTable>&  shapes ) :

    egroup_ ( group ),
    bounds_ ( newXBoundarySet( group.getElements(),
                               COMPACT_STORAGE ) ),
    shapes_ ( shapes )

{
  JEM_PRECHECK ( shapes );

  init_ ();
}


BoundaryManager::BoundaryManager

  ( const XBoundarySet&     bounds,
    const Ref<ShapeTable>&  shapes ) :

    egroup_ ( newElementGroup( ALL, bounds.getElements() ) ),
    bounds_ ( bounds ),
    shapes_ ( shapes )

{
  JEM_PRECHECK ( shapes );

  init_ ();
}


BoundaryManager::BoundaryManager

  ( const ElementGroup&     group,
    const XBoundarySet&     bounds,
    const Ref<ShapeTable>&  shapes ) :

    egroup_ ( group  ),
    bounds_ ( bounds ),
    shapes_ ( shapes )

{
  JEM_PRECHECK ( group && bounds && shapes );
  JEM_PRECHECK ( group.getElements() == bounds.getElements() );

  init_ ();
}


BoundaryManager::~BoundaryManager ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void BoundaryManager::resetEvents ()
{
  events_.reset ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void BoundaryManager::readFrom ( ObjectInput& in )
{
  decode   ( in, egroup_, bounds_, shapes_ );
  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void BoundaryManager::writeTo ( ObjectOutput& out ) const
{
  encode ( out, egroup_, bounds_, shapes_ );
}


//-----------------------------------------------------------------------
//   getBoundaries
//-----------------------------------------------------------------------


BoundarySet BoundaryManager::getBoundaries () const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return bounds_;
}


//-----------------------------------------------------------------------
//   getXBoundaries
//-----------------------------------------------------------------------


BoundaryGroup BoundaryManager::getXBoundaries () const
{
  Self*  self = const_cast<Self*> ( this );

  if ( ! updated_ )
  {
    self->update_ ();
  }

  return BoundaryGroup (
    newInstance<AutoBndGroup> ( AutoBndGroup::EXTERNAL, self )
  );
}


//-----------------------------------------------------------------------
//   getIBoundaries
//-----------------------------------------------------------------------


BoundaryGroup BoundaryManager::getIBoundaries () const
{
  Self*  self = const_cast<Self*> ( this );

  if ( ! updated_ )
  {
    self->update_ ();
  }

  return BoundaryGroup (
    newInstance<AutoBndGroup> ( AutoBndGroup::INTERNAL, self )
  );
}


//-----------------------------------------------------------------------
//   getBuddies
//-----------------------------------------------------------------------


IdxVector BoundaryManager::getBuddies () const
{
  if ( ! updated_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return buddies_;
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void BoundaryManager::emitEvents_ ( int events )
{
  if ( events & NEW_BUDDIES_ )
  {
    newBuddiesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void BoundaryManager::init_ ()
{
  ItemSet*  elems = egroup_.getElements().getData ();

  if ( elems != shapes_->getItems() )
  {
    throw jem::IllegalArgumentException (
      CLASS_NAME,
      "invalid shape table: incorrect item set"
    );
  }

  connect_ ();
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void BoundaryManager::connect_ ()
{
  using jem::util::connect;

  connect ( bounds_.newSizeEvent  (), this, & Self::invalidate_ );
  connect ( bounds_.newOrderEvent (), this, & Self::invalidate_ );
  connect ( egroup_.newSizeEvent  (), this, & Self::invalidate_ );
  connect ( shapes_->changeEvent,     this, & Self::invalidate_ );

  events_.reset ();

  updated_ = false;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void BoundaryManager::update_ ()
{
  using jem::iarray;

  ElementSet   elems      = egroup_.getElements ();

  const idx_t  elemCount  = elems  .size ();
  const idx_t  ielemCount = egroup_.size ();

  TopoVector   shapeTopos;
  IdxVector    shapeMap;
  IdxVector    ielems;

  idx_t        i, j;


  print ( JIVE_LOG_DEBUG, CLASS_NAME,
          " : adding new boundaries ...\n" );

  ielems.resize   ( ielemCount );

  ielems = egroup_.getIndices ();

  shapeMap.resize ( ielemCount );
  shapeTopos.ref  ( shapes_->getShapeToposOf( shapeMap, ielems ) );

  // Remove all non-internal shapes.

  j = 0;

  for ( i = 0; i < ielemCount; i++ )
  {
    if ( shapeMap[i] >= 0 )
    {
      shapeMap[j] = shapeMap[i];
      ielems  [j] = ielems  [i];
      j++;
    }
  }

  if ( j < ielemCount )
  {
    shapeMap.reshape ( j );
    ielems  .reshape ( j );
  }

  bounds_.addBoundariesOf ( ielems, shapeMap, shapeTopos );

  print ( JIVE_LOG_DEBUG, CLASS_NAME,
          " : searching for boundary buddies ...\n" );

  buddies_.resize ( bounds_.size() );
  shapeMap.resize ( elemCount );
  ielems  .resize ( elemCount );

  ielems = iarray ( elemCount );

  shapeTopos.ref  ( shapes_->getShapeToposOf( shapeMap, ielems ) );

  bounds_ .findBuddies ( buddies_, shapeMap, shapeTopos );

  egroup_.resetEvents  ();
  bounds_.resetEvents  ();
  events_.reset        ();

  updated_ = true;
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void BoundaryManager::invalidate_ ()
{
  updated_ = false;

  events_.emit ( NEW_BUDDIES_, this );
}


JIVE_END_PACKAGE( fem )
