
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/Reordering.h>
#include <jive/geom/Shape.h>
#include <jive/geom/StdShapeTable.h>


JEM_DEFINE_CLASS( jive::geom::StdShapeTable );


JIVE_BEGIN_PACKAGE( geom )


using jive::util::sizeError;
using jive::util::indexError;
using jive::util::TopoVector;


//=======================================================================
//   internal macros
//=======================================================================


#undef  JIVE_CHECK_MAP_SIZE
#define JIVE_CHECK_MAP_SIZE( map, n )                   \
                                                        \
  if ( map.size() < n )                                 \
                                                        \
    sizeError ( JEM_FUNC, "shape map array",            \
                map.size(), n )


#undef  JIVE_CHECK_ITEM_INDEX
#define JIVE_CHECK_ITEM_INDEX( iitem )                  \
                                                        \
  if ( iitem < 0 || iitem >= map_.size() )              \
                                                        \
    checkItemIndex_ ( iitem )


#undef  JIVE_CHECK_PRECISION
#define JIVE_CHECK_PRECISION( prec )                    \
                                                        \
  if ( prec < 0 || prec >= shapes_.size(0) )            \
                                                        \
    indexError ( JEM_FUNC, "precision",                 \
                 prec, shapes_.size(0) )


#undef  JIVE_CHECK_PRECISIONS
#define JIVE_CHECK_PRECISIONS( minPrec, maxPrec )       \
                                                        \
  if ( maxPrec >= shapes_.size(0) )                     \
                                                        \
    indexError ( JEM_FUNC, "maximum precision",         \
                 minPrec,  shapes_.size(0) );           \
                                                        \
                                                        \
  if ( minPrec < 0 || minPrec > maxPrec )               \
                                                        \
    indexError ( JEM_FUNC, "minimum precision",         \
                 minPrec,  maxPrec + 1 )



//=======================================================================
//   class StdShapeTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdShapeTable::StdShapeTable ()
{}


StdShapeTable::StdShapeTable

  ( const String&       name,
    const Ref<ItemSet>& items,
    idx_t               maxPrec ) :

    Super  ( name  ),
    items_ ( items )

{
  JEM_PRECHECK2 ( items,        "NULL ItemSet" );
  JEM_PRECHECK2 ( maxPrec >= 0, "invalid maximum precision" );

  shapes_.resize ( maxPrec + 1, 0 );

  connect_ ();
}


StdShapeTable::~StdShapeTable ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void StdShapeTable::resetEvents ()
{
  items_->resetEvents ();
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* StdShapeTable::getItems () const
{
  return items_.get ();
}


//-----------------------------------------------------------------------
//   maxPrecision
//-----------------------------------------------------------------------


idx_t StdShapeTable::maxPrecision () const
{
  return shapes_.size(0) - 1;
}


//-----------------------------------------------------------------------
//   getShapes (given a single precision)
//-----------------------------------------------------------------------


ShapeVector StdShapeTable::getShapes

  ( const IdxVector&  map,
    idx_t             prec ) const

{
  if ( map_.size() != items_->size() )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  const idx_t  n = map_.size ();


  JIVE_CHECK_MAP_SIZE  ( map, n );
  JIVE_CHECK_PRECISION ( prec );

  if ( map.size() == n )
  {
    map = map_;
  }
  else
  {
    map[slice(BEGIN,n)] = map_;
  }

  return shapes_(prec,ALL);
}


//-----------------------------------------------------------------------
//   getShapes (given a precision range)
//-----------------------------------------------------------------------


ShapeMatrix StdShapeTable::getShapes

  ( const IdxVector&  map,
    idx_t             minPrec,
    idx_t             maxPrec ) const

{
  if ( map_.size() != items_->size() )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  const idx_t  n = map_.size ();

  JIVE_CHECK_MAP_SIZE   ( map, n );
  JIVE_CHECK_PRECISIONS ( minPrec, maxPrec );

  if ( map.size() == n )
  {
    map = map_;
  }
  else
  {
    map[slice(BEGIN,n)] = map_;
  }

  return shapes_(slice(minPrec,maxPrec + 1),ALL);
}


//-----------------------------------------------------------------------
//   getShapeOf
//-----------------------------------------------------------------------


Shape* StdShapeTable::getShapeOf

  ( idx_t  iitem,
    idx_t  prec ) const

{
  JIVE_CHECK_ITEM_INDEX ( iitem );
  JIVE_CHECK_PRECISION  ( prec  );

  const idx_t  j = map_[iitem];

  if ( j >= 0 )
  {
    return shapes_(prec,j).get ();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   getShapesOf (given a single precision)
//-----------------------------------------------------------------------


ShapeVector StdShapeTable::getShapesOf

  ( const IdxVector&  map,
    const IdxVector&  iitems,
    idx_t             prec ) const

{
  const idx_t  iitemCount = iitems.size ();


  JIVE_CHECK_MAP_SIZE  ( map, iitemCount );
  JIVE_CHECK_PRECISION ( prec );

  for ( idx_t i = 0; i < iitemCount; i++ )
  {
    idx_t  iitem = iitems[i];

    JIVE_CHECK_ITEM_INDEX ( iitem );

    idx_t  j = map_[iitem];

    if ( j >= 0 && shapes_(prec,j) == nullptr )
    {
      j = -1;
    }

    map[i] = j;
  }

  return shapes_(prec,ALL);
}


//-----------------------------------------------------------------------
//   getShapesOf (given a precision range)
//-----------------------------------------------------------------------


ShapeMatrix StdShapeTable::getShapesOf

  ( const IdxVector&  map,
    const IdxVector&  iitems,
    idx_t             minPrec,
    idx_t             maxPrec ) const

{
  const idx_t  iitemCount = iitems.size ();


  JIVE_CHECK_MAP_SIZE   ( map, iitemCount );
  JIVE_CHECK_PRECISIONS ( minPrec, maxPrec );

  for ( idx_t i = 0; i < iitemCount; i++ )
  {
    idx_t  iitem = iitems[i];

    JIVE_CHECK_ITEM_INDEX ( iitem );

    map[i] = map_[iitem];
  }

  return shapes_(slice(minPrec,maxPrec + 1),ALL);
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void StdShapeTable::update_ ()
{
  const idx_t  mapSize   = map_   .size ();
  const idx_t  itemCount = items_->size ();

  if ( itemCount != mapSize )
  {
    map_.reshape ( itemCount );

    if ( itemCount > mapSize )
    {
      map_[slice(mapSize,END)] = -1;
    }
  }

  items_->resetEvents ();
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void StdShapeTable::connect_ ()
{
  using jem::util::connect;

  connect ( items_->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( items_->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void StdShapeTable::itemsResized_ ( idx_t itemCount )
{
  const idx_t  mapSize = map_.size ();

  if      ( itemCount < mapSize )
  {
    map_.reshape ( itemCount );
  }
  else if ( itemCount > mapSize )
  {
    changeEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void StdShapeTable::itemsReordered_ ( const Reordering& reord )
{
  if      ( map_.size() == reord.oldSize() )
  {
    util::reorder ( map_, reord );
  }
  else if ( map_.size() >  reord.start() )
  {
    map_.reshape ( reord.start() );

    changeEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   checkItemIndex_
//-----------------------------------------------------------------------


void StdShapeTable::checkItemIndex_ ( idx_t iitem ) const
{
  const idx_t  itemCount = items_->size ();

  if ( iitem < 0 || iitem >= itemCount )
  {
    indexError ( JEM_FUNC, items_->getItemName(),
                 iitem, itemCount );
  }

  if ( iitem >= map_.size() )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }
}


JIVE_END_PACKAGE( geom )
