
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
#include <jem/base/array/select.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/ObjFlex.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemGroup.h>
#include <jive/geom/Names.h>
#include <jive/geom/Shape.h>
#include <jive/geom/ShapeFactory.h>
#include <jive/geom/CustomShapeTable.h>


JEM_DEFINE_CLASS( jive::geom::CustomShapeTable );


JIVE_BEGIN_PACKAGE( geom )


using jem::checkedCast;
using jem::newInstance;
using jem::util::connect;
using jem::util::disconnect;
using jem::util::ObjFlex;


//=======================================================================
//   class CustomShapeTable::ShapeSet_
//=======================================================================


class CustomShapeTable::ShapeSet_ : public jem::Object
{
 public:

  typedef ShapeSet_         Self;


  inline                    ShapeSet_

    ( const ShapeVector&      shapes,
      const Ref<ItemGroup>&   igroup );

  static Ref<Self>          readFrom

    ( ObjectInput&            in );

  void                      writeTo

    ( ObjectOutput&           out )    const;


 public:

  ShapeVector               shapes;
  Ref<ItemGroup>            igroup;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline CustomShapeTable::ShapeSet_::ShapeSet_

  ( const ShapeVector&     sh,
    const Ref<ItemGroup>&  gr ) :

    shapes ( sh ),
    igroup ( gr )

{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


Ref<CustomShapeTable::ShapeSet_>

  CustomShapeTable::ShapeSet_::readFrom ( ObjectInput& in )

{
  ShapeVector  shapes;

  decode ( in, shapes );

  return newInstance<Self> (
    shapes,
    checkedCast<ItemGroup> ( in.decodeObject() )
  );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void CustomShapeTable::ShapeSet_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, shapes, igroup );
}


//=======================================================================
//   internal macros
//=======================================================================


#undef  JIVE_UPDATE_SELF
#define JIVE_UPDATE_SELF                                \
                                                        \
  if ( ! updated_ )                                     \
                                                        \
    const_cast<Self*>( this )->update_ ()


#undef  JIVE_CHECK_MAP_SIZE
#define JIVE_CHECK_MAP_SIZE( map, n )                   \
                                                        \
  if ( map.size() < n )                                 \
                                                        \
    util::sizeError ( JEM_FUNC, "shape map array",      \
                      map.size(), n )


#undef  JIVE_CHECK_ITEM_INDEX
#define JIVE_CHECK_ITEM_INDEX( iitem, ubound )          \
                                                        \
  if ( iitem < 0 || iitem >= ubound )                   \
                                                        \
    util::indexError ( JEM_FUNC, items_->getItemName(), \
                       iitem, ubound )


#undef  JIVE_CHECK_PRECISION
#define JIVE_CHECK_PRECISION( prec )                    \
                                                        \
  if ( prec < 0 || prec > maxPrec_ )                    \
                                                        \
    util::indexError ( JEM_FUNC, "precision",           \
                       prec, maxPrec_ + 1 )


#undef  JIVE_CHECK_PRECISIONS
#define JIVE_CHECK_PRECISIONS( minPrec, maxPrec )       \
                                                        \
  if ( maxPrec > maxPrec_ )                             \
                                                        \
    util::indexError ( JEM_FUNC, "maximum precision",   \
                       minPrec,  maxPrec_ + 1 );        \
                                                        \
                                                        \
  if ( minPrec < 0 || minPrec > maxPrec )               \
                                                        \
    util::indexError ( JEM_FUNC, "minimum precision",   \
                       minPrec,  maxPrec + 1 )




//=======================================================================
//   class CustomShapeTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


CustomShapeTable::CustomShapeTable ()
{
  maxPrec_ = 0;
  updated_ = false;
}


CustomShapeTable::CustomShapeTable

  ( const String&        name,
    const Ref<ItemSet>&  items ) :

    Super  ( name  ),
    items_ ( items )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  maxPrec_ = 0;
  updated_ = false;

  connect_ ();
}


CustomShapeTable::~CustomShapeTable ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void CustomShapeTable::readFrom ( ObjectInput& in )
{
  idx_t  n;


  maxPrec_ = 0;
  updated_ = false;

  if ( items_ )
  {
    disconnect ( items_->newSizeEvent,  this, & Self::invalidate_ );
    disconnect ( items_->newOrderEvent, this, & Self::invalidate_ );

    items_ = nullptr;
  }

  decode ( in, myName_, items_, myConf_, n );

  if ( shapeSets_.size() )
  {
    clear_ ();
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    addShapeSet_ ( ShapeSet_::readFrom( in ) );
  }

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void CustomShapeTable::writeTo ( ObjectOutput& out ) const
{
  const idx_t  n = shapeSets_.size ();

  encode ( out, myName_, items_, myConf_, n );

  for ( idx_t i = 0; i < n; i++ )
  {
    shapeSets_.getAs<ShapeSet_>(i)->writeTo ( out );
  }
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void CustomShapeTable::resetEvents ()
{
  for ( idx_t i = 0; i < shapeSets_.size(); i++ )
  {
    shapeSets_.getAs<ShapeSet_>(i)->igroup->resetEvents ();
  }

  items_->resetEvents ();
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void CustomShapeTable::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::staticCast;
  using jive::util::joinNames;

  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  const String    myContext = getContext ();
  const String    itemsName = items_->getItemsName ();

  Properties      myProps   = props.getProps ( myName_ );
  Properties      myConf;
  Properties      conf;

  Ref<ItemGroup>  igroup;

  ObjFlex         shapeSets;

  StringVector    setNames;
  String          name;

  idx_t           i, j, n;


  conf   .set ( myName_, myConf );

  myProps.get ( setNames, PropNames::SHAPE_SETS );
  myConf .set ( PropNames::SHAPE_SETS, setNames );

  shapeSets.reserve ( setNames.size() );

  for ( j = 0; j < setNames.size(); j++ )
  {
    Properties   subProps = myProps.getProps  ( setNames[j] );
    Properties   subConf  = myConf .makeProps ( setNames[j] );

    ShapeVector  shapes;


    name = setNames[j];

    subProps.find ( name, itemsName );
    subConf .set  ( itemsName, name );

    igroup = ItemGroup::get ( name, items_, globdat, myContext );

    // Count the number of shapes

    for ( n = 1; n < 100; n++ )
    {
      name = PropNames::SHAPE + String( n );

      if ( ! subProps.contains( name ) )
      {
        break;
      }
    }

    // Create the shapes

    shapes.resize ( n );

    name      = joinNames ( myName_, setNames[j] );
    name      = joinNames ( name,    PropNames::SHAPE );

    shapes[0] = ShapeFactory::newInstance ( name + String( 0 ),
                                            conf, props );

    for ( i = 1; i < n; i++ )
    {
      shapes[i] = ShapeFactory::newInstance ( name + String( i ),
                                              conf, props );
    }

    shapeSets.pushBack ( newInstance<ShapeSet_>( shapes, igroup ) );
  }

  // The new shape sets are OK, so update this object

  clear_ ();

  for ( i = 0; i < shapeSets.size(); i++ )
  {
    addShapeSet_ ( shapeSets.getAs<ShapeSet_>( i ) );
  }

  myConf_.swap ( myConf );

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void CustomShapeTable::getConfig ( const Properties& conf ) const
{
  using jem::util::mergeAndReplace;

  mergeAndReplace ( conf.makeProps( myName_ ), myConf_ );
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* CustomShapeTable::getItems () const
{
  return items_.get ();
}


//-----------------------------------------------------------------------
//   maxPrecision
//-----------------------------------------------------------------------


idx_t CustomShapeTable::maxPrecision () const
{
  return maxPrec_;
}


//-----------------------------------------------------------------------
//   getShapes (given a single precision)
//-----------------------------------------------------------------------


ShapeVector CustomShapeTable::getShapes

  ( const IdxVector&  map,
    idx_t             prec ) const

{
  JIVE_UPDATE_SELF;

  const idx_t  n = shapeMap_.size ();

  JIVE_CHECK_MAP_SIZE  ( map, n );
  JIVE_CHECK_PRECISION ( prec );

  if ( map.size() == n )
  {
    map = shapeMap_;
  }
  else
  {
    map[slice(BEGIN,n)] = shapeMap_;
  }

  return shapes_(prec,ALL);
}


//-----------------------------------------------------------------------
//   getShapes (given a precision range)
//-----------------------------------------------------------------------


ShapeMatrix CustomShapeTable::getShapes

  ( const IdxVector&  map,
    idx_t             minPrec,
    idx_t             maxPrec ) const

{
  JIVE_UPDATE_SELF;

  const idx_t  n = shapeMap_.size ();

  JIVE_CHECK_MAP_SIZE   ( map,     n );
  JIVE_CHECK_PRECISIONS ( minPrec, maxPrec );

  if ( map.size() == n )
  {
    map = shapeMap_;
  }
  else
  {
    map[slice(BEGIN,n)] = shapeMap_;
  }

  return shapes_(slice(minPrec,maxPrec + 1),ALL);
}


//-----------------------------------------------------------------------
//   getShapeOf
//-----------------------------------------------------------------------


Shape* CustomShapeTable::getShapeOf

  ( idx_t  iitem,
    idx_t  prec ) const

{
  JIVE_UPDATE_SELF;
  JIVE_CHECK_ITEM_INDEX ( iitem, shapeMap_.size() );
  JIVE_CHECK_PRECISION  ( prec );

  const idx_t  j = shapeMap_[iitem];

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


ShapeVector CustomShapeTable::getShapesOf

  ( const IdxVector&  map,
    const IdxVector&  iitems,
    idx_t             prec ) const

{
  JIVE_UPDATE_SELF;

  const idx_t  itemCount  = shapeMap_.size ();
  const idx_t  iitemCount = iitems   .size ();


  JIVE_CHECK_MAP_SIZE  ( map, iitemCount );
  JIVE_CHECK_PRECISION ( prec );

  for ( idx_t ii = 0; ii < iitemCount; ii++ )
  {
    idx_t  iitem = iitems[ii];

    JIVE_CHECK_ITEM_INDEX ( iitem, itemCount );

    idx_t  j = shapeMap_[iitem];

    if ( j >= 0 && shapes_(prec,j) == nullptr )
    {
      j = -1;
    }

    map[ii] = j;
  }

  return shapes_(prec,ALL);
}


//-----------------------------------------------------------------------
//   getShapesOf (given a precision range)
//-----------------------------------------------------------------------


ShapeMatrix CustomShapeTable::getShapesOf

  ( const IdxVector&  map,
    const IdxVector&  iitems,
    idx_t             minPrec,
    idx_t             maxPrec ) const

{
  JIVE_UPDATE_SELF;

  const idx_t  itemCount  = shapeMap_.size ();
  const idx_t  iitemCount = iitems   .size ();


  JIVE_CHECK_MAP_SIZE   ( map, iitemCount );
  JIVE_CHECK_PRECISIONS ( minPrec, maxPrec );

  for ( idx_t ii = 0; ii < iitemCount; ii++ )
  {
    idx_t  iitem = iitems[ii];

    JIVE_CHECK_ITEM_INDEX ( iitem, itemCount );

    map[ii] = shapeMap_[iitem];
  }

  return shapes_(slice(minPrec,maxPrec + 1),ALL);
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void CustomShapeTable::invalidate_ ()
{
  updated_ = false;

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


void CustomShapeTable::clear_ ()
{
  for ( idx_t i = 0; i < shapeSets_.size(); i++ )
  {
    ShapeSet_&  s = * shapeSets_.getAs<ShapeSet_> ( i );

    disconnect ( s.igroup->newSizeEvent, this, & Self::invalidate_ );
  }

  shapeSets_.clear ();
  myConf_   .clear ();

  maxPrec_ = 0;
  updated_ = false;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void CustomShapeTable::update_ ()
{
  shapeMap_.resize ( items_->size() );

  shapeMap_ = -1;

  shapes_.resize ( maxPrec_ + 1, shapeSets_.size() );

  for ( idx_t j = 0; j < shapeSets_.size(); j++ )
  {
    const ShapeSet_&  s = * shapeSets_.getAs<ShapeSet_> ( j );

    for ( idx_t i = 0; i < s.shapes.size(); i++ )
    {
      checkShapeFor_ ( *s.shapes[i], *s.igroup );

      shapes_(i,j) = s.shapes[i];
    }

    shapeMap_[s.igroup->getIndices()] = j;

    s.igroup->resetEvents ();
  }

  items_->resetEvents ();

  updated_ = true;
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void CustomShapeTable::connect_ ()
{
  connect ( items_->newSizeEvent,  this, & Self::invalidate_ );
  connect ( items_->newOrderEvent, this, & Self::invalidate_ );
}


//-----------------------------------------------------------------------
//   addShapeSet_
//-----------------------------------------------------------------------


void CustomShapeTable::addShapeSet_ ( const Ref<ShapeSet_>& set )
{
  shapeSets_.pushBack ( set );

  connect ( set->igroup->newSizeEvent, this, & Self::invalidate_ );

  maxPrec_ = jem::max ( maxPrec_, set->shapes.size() - 1 );
  updated_ = false;
}


JIVE_END_PACKAGE( geom )
