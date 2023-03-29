
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
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/utilities.h>
#include <jive/geom/ShapeTable.h>
#include <jive/geom/InternalShape.h>
#include <jive/fem/VirtualBoundary.h>
#include <jive/fem/BoundaryIterator.h>


JEM_DEFINE_SERIABLE_CLASS( jive::fem::BoundaryIterator );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class BoundaryIterator
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BoundaryIterator::BoundaryIterator ()
{
  reset_ ();
}


BoundaryIterator::BoundaryIterator

  ( const BoundaryGroup&    group,
    const Ref<ShapeTable>&  shapes,
    const String&           context ) :

    Super    ( shapes, context ),
    bndGroup ( group )

{
  JEM_PRECHECK ( group );

  init_ ();
}


BoundaryIterator::BoundaryIterator

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


BoundaryIterator::~BoundaryIterator ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void BoundaryIterator::readFrom ( ObjectInput& in )
{
  Super::readFrom_ ( in );

  decode ( in, bndGroup );
  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void BoundaryIterator::writeTo ( ObjectOutput& out ) const
{
  Super::writeTo_ ( out );

  encode ( out, bndGroup );
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void BoundaryIterator::resetEvents ()
{
  bndGroup    .resetEvents ();
  shapeTable_->resetEvents ();
}


//-----------------------------------------------------------------------
//   rewind
//-----------------------------------------------------------------------


void BoundaryIterator::rewind ()
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


bool BoundaryIterator::toNext ( VirtualBoundary& bnd )
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
      bnd.reset ();
    }

    bnd.setShape ( shapes_ [shapeMap_[ipos]].get(),
                   ilocals_[ipos] );
  }

  bnd.loadData ( ibounds_[ipos], ielems_[ipos] );

  inext_++;

  return true;
}


//-----------------------------------------------------------------------
//   checkShapes
//-----------------------------------------------------------------------


void BoundaryIterator::checkShapes ()
{
  if ( ! updated_ )
  {
    update_ ();
  }
}


//-----------------------------------------------------------------------
//   setShapeFlags
//-----------------------------------------------------------------------


void BoundaryIterator::setShapeFlags (  int flags )
{
  Super::setShapeFlags ( flags | ShapeTable::INTERNAL_TYPE );
}


//-----------------------------------------------------------------------
//   getNodeIndices
//-----------------------------------------------------------------------


IdxVector BoundaryIterator::getNodeIndices () const
{
  typedef jem::util::ArrayBuffer<idx_t>  IdxBuffer;
  using   jive::util::makeUnique;

  ElementSet    elems      = getElements    ();
  NodeSet       nodes      = elems.getNodes ();
  IShapeVector  shapes     = getShapes      ();
  IdxVector     shapeMap   = getShapeMap    ();

  const idx_t   bndCount   = ibounds_.size  ();

  IdxVector     enodes     ( elems.maxElemNodeCount() );
  IdxVector     bnodes;
  IdxBuffer     inodes;

  idx_t         bnodeCount =  0_idx;
  idx_t         lastShape  = -1_idx;
  idx_t         lastLocal  = -1_idx;


  inodes.reserve ( 2 * bndCount );

  for ( idx_t i = 0; i < bndCount; i++ )
  {
    idx_t  ishape = shapeMap[i];
    idx_t  ilocal = ilocals_[i];

    if ( lastShape != ishape || lastLocal != ilocal )
    {
      bnodes.ref ( shapes[ishape]->getBoundaryNodes( ilocal ) );

      lastShape  = ishape;
      lastLocal  = ilocal;
      bnodeCount = bnodes.size ();
    }

    elems.getSomeElemNodes ( enodes, bnodes, ielems_[i] );

    for ( idx_t j = 0; j < bnodeCount; j++ )
    {
      inodes.pushBack ( enodes[j] );
    }
  }

  return makeUnique ( inodes.toArray(),
                      0, nodes.size() - 1 ).clone ();
}


//-----------------------------------------------------------------------
//   getShapes
//-----------------------------------------------------------------------


IShapeVector BoundaryIterator::getShapes () const
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


IdxVector BoundaryIterator::getShapeMap () const
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


void BoundaryIterator::init_ ()
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


inline void BoundaryIterator::reset_ ()
{
  ipos   = -1;
  inext_ =  0;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void BoundaryIterator::update_ ()
{
  using jive::util::sortColors;

  BoundarySet  bounds = bndGroup.getBoundaries ();

  IdxVector    colors;
  IdxVector    offsets;

  idx_t        st[2];
  idx_t        i, j, n;


  checkPrecision_ ();

  ibounds_.ref ( bndGroup.getIndices() );

  n = ibounds_.size ();

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
    st[0] = st[1] = 0;
  }
  else
  {
    st[0] = 1;
    st[1] = 1 + max ( ilocals_ );
  }

  iperm_.resize ( n );
  colors.resize ( n );

  for ( i = 0; i < n; i++ )
  {
    colors[i] = st[0] * ilocals_[i] + st[1] * shapeMap_[i];
  }

  offsets.ref ( sortColors( iperm_, colors ) );

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
