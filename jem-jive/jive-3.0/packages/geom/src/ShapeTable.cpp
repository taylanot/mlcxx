
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
#include <jive/util/ItemSet.h>
#include <jive/geom/Globdat.h>
#include <jive/geom/InternalShape.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/geom/InterfaceShape.h>
#include <jive/geom/Grad2Extension.h>
#include <jive/geom/BezierExtension.h>
#include <jive/geom/ShapeTable.h>


JEM_DEFINE_CLASS( jive::geom::ShapeTable );


JIVE_BEGIN_PACKAGE( geom )


using jem::iarray;
using jem::Class;
using jem::IllegalInputException;
using jive::util::TopoVector;


//=======================================================================
//   class ShapeTable
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  ShapeTable::ANY_TYPE       =      0;
const int  ShapeTable::INTERNAL_TYPE  = 1 << 0;
const int  ShapeTable::BOUNDARY_TYPE  = 1 << 1;
const int  ShapeTable::INTERFACE_TYPE = 1 << 2;
const int  ShapeTable::REGULAR_TYPE   = 1 << 3;
const int  ShapeTable::GRAD2_TYPE     = 1 << 4;
const int  ShapeTable::BEZIER_TYPE    = 1 << 5;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ShapeTable::ShapeTable ( const String& name ) :

  Super ( name )

{}


ShapeTable::~ShapeTable ()
{}


//-----------------------------------------------------------------------
//   itemCount
//-----------------------------------------------------------------------


idx_t ShapeTable::itemCount () const
{
  return getItems()->size ();
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String ShapeTable::getContext () const
{
  return NamedObject::makeContext ( "shape table", myName_ );
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void ShapeTable::resetEvents ()
{}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void ShapeTable::configure

  ( const Properties&  props,
    const Properties&  globdat )

{}


void ShapeTable::getConfig

  ( const Properties&  conf ) const

{}


//-----------------------------------------------------------------------
//   getShapes
//-----------------------------------------------------------------------


ShapeVector ShapeTable::getShapes

  ( const IdxVector&  map,
    idx_t             prec ) const

{
  IdxVector  iitems ( iarray( itemCount() ) );

  return getShapesOf ( map, iitems, prec );
}


ShapeVector ShapeTable::getShapes

  ( const IdxVector&  map,
    const IdxVector&  precs ) const

{
  IdxVector  iitems ( iarray( itemCount() ) );

  return getShapesOf ( map, iitems, precs );
}


ShapeMatrix ShapeTable::getShapes

  ( const IdxVector&  map,
    idx_t             minPrec,
    idx_t             maxPrec ) const

{
  IdxVector  iitems ( iarray( itemCount() ) );

  return getShapesOf ( map, iitems, minPrec, maxPrec );
}


//-----------------------------------------------------------------------
//   getShapesOf (given a single precision)
//-----------------------------------------------------------------------


ShapeVector ShapeTable::getShapesOf

  ( const IdxVector&  map,
    const IdxVector&  iitems,
    idx_t             prec ) const

{
  return getShapesOf ( map, iitems, prec, prec ) (0,ALL);
}


//-----------------------------------------------------------------------
//   getShapesOf (given a precision vector)
//-----------------------------------------------------------------------


ShapeVector ShapeTable::getShapesOf

  ( const IdxVector&  map,
    const IdxVector&  iitems,
    const IdxVector&  precs ) const

{
  JEM_PRECHECK2 ( iitems.size() == precs.size(),
                  "Array size mismatch" );

  if ( iitems.size() == 0 )
  {
    return ShapeVector ();
  }

  const idx_t  iitemCount = iitems.size ();

  const idx_t  minPrec    = min ( precs );
  const idx_t  maxPrec    = max ( precs );

  ShapeMatrix  sm = getShapesOf ( map, iitems, minPrec, maxPrec );

  IdxMatrix    remap ( sm.shape() );

  ShapeVector  sv;

  idx_t        ii;
  idx_t        i, j, m, n;


  remap = -1;
  n     =  0;

  for ( ii = 0; ii < iitemCount; ii++ )
  {
    i = precs[ii] - minPrec;
    j = map[ii];

    if ( j < 0 )
    {
      continue;
    }

    if ( ! sm(i,j) )
    {
      map[ii] = -1;
      continue;
    }

    if ( remap(i,j) < 0 )
    {
      remap(i,j) = n++;
    }

    map[ii] = remap(i,j);
  }

  sv.resize ( n );

  m = sm.size(0);
  n = sm.size(1);

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      if ( remap(i,j) >= 0 )
      {
        sv[remap(i,j)] = sm(i,j);
      }
    }
  }

  return sv;
}


//-----------------------------------------------------------------------
//   getShapeTopos
//-----------------------------------------------------------------------


TopoVector ShapeTable::getShapeTopos ( const IdxVector& map ) const
{
  IdxVector  iitems ( iarray( itemCount() ) );

  return getShapeToposOf ( map, iitems );
}


//-----------------------------------------------------------------------
//   getShapeToposOf
//-----------------------------------------------------------------------


TopoVector ShapeTable::getShapeToposOf

  ( const IdxVector&  map,
    const IdxVector&  iitems ) const

{
  Class*       klass = IShape::getType ();

  ShapeVector  sv    = getShapesOf ( map, iitems, 0 );

  const idx_t  n     = sv.size ();

  TopoVector   topos ( n );


  for ( idx_t i = 0; i < n; i++ )
  {
    Shape*  s = sv[i].get ();

    if ( s && klass->isBaseOf( s->getClass() ) )
    {
      topos[i] = static_cast<IShape*>(s)->getBoundaryTopology ();
    }
  }

  return topos;
}


//-----------------------------------------------------------------------
//   checkShapes
//-----------------------------------------------------------------------


void ShapeTable::checkShapes

  ( const String&  context,
    int            flags,
    idx_t          prec ) const

{
  IdxVector  iitems ( iarray( itemCount() ) );

  checkShapesOf ( context, iitems, flags,   prec, prec );
}


void ShapeTable::checkShapes

  ( const String&     context,
    int               flags,
    const IdxVector&  precs ) const

{
  IdxVector  iitems ( iarray( itemCount() ) );

  checkShapesOf ( context, iitems, flags, precs );
}


void ShapeTable::checkShapes

  ( const String&  context,
    int            flags,
    idx_t          minPrec,
    idx_t          maxPrec ) const

{
  IdxVector  iitems ( iarray( itemCount() ) );

  checkShapesOf ( context, iitems, flags, minPrec, maxPrec );
}


//-----------------------------------------------------------------------
//   checkShapesOf (given a single precision)
//-----------------------------------------------------------------------


void ShapeTable::checkShapesOf

  ( const String&     context,
    const IdxVector&  iitems,
    int               flags,
    idx_t             prec ) const

{
  checkShapesOf ( context, iitems, flags, prec, prec );
}


//-----------------------------------------------------------------------
//   checkShapesOf (given a precision vector)
//-----------------------------------------------------------------------


void ShapeTable::checkShapesOf

  ( const String&     context,
    const IdxVector&  iitems,
    int               flags,
    const IdxVector&  precs ) const

{
  const idx_t  iitemCount = iitems.size ();

  IdxVector    map ( iitemCount );

  ShapeVector  sv = getShapesOf ( map, iitems, precs );

  const idx_t  n  = sv.size ();

  IntVector    shapeTypes ( n );

  int          err;
  idx_t        iitem;
  idx_t        ii;
  idx_t        i;


  shapeTypes = 0;

  for ( i = 0; i < n; i++ )
  {
    if ( sv[i] )
    {
      shapeTypes[i] = getTypeFlags ( *sv[i] );
    }
  }

  for ( ii = 0; ii < iitemCount; ii++ )
  {
    iitem = iitems[ii];
    i     = map[ii];

    if ( i < 0 || sv[i] == nullptr )
    {
      noShapeError_ ( context, iitem );
    }

    err = flags & (~shapeTypes[i]);

    if ( err )
    {
      shapeTypeError_ ( context, *sv[i], iitem, err );
    }
  }
}


//-----------------------------------------------------------------------
//   checkShapesOf (given a precision range)
//-----------------------------------------------------------------------


void ShapeTable::checkShapesOf

  ( const String&     context,
    const IdxVector&  iitems,
    int               flags,
    idx_t             minPrec,
    idx_t             maxPrec ) const

{
  const idx_t  iitemCount = iitems.size ();

  IdxVector    map ( iitemCount );

  ShapeMatrix  sm = getShapesOf ( map, iitems, minPrec,  maxPrec );

  const idx_t  m  = sm.size(0);
  const idx_t  n  = sm.size(1);

  IntMatrix    shapeTypes ( m, n );

  int          err;
  idx_t        iitem;
  idx_t        ii;
  idx_t        i, j;


  shapeTypes = 0;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      if ( sm(i,j) )
      {
        shapeTypes(i,j) = getTypeFlags ( *sm(i,j) );
      }
    }
  }

  for ( ii = 0; ii < iitemCount; ii++ )
  {
    iitem = iitems[ii];
    j     = map[ii];

    if ( j < 0 )
    {
      noShapeError_ ( context, iitem );
    }

    for ( i = 0; i < m; i++ )
    {
      if ( ! sm(i,j) )
      {
        noShapeError_ ( context, iitem, i );
      }

      err = flags & (~shapeTypes(i,j));

      if ( err )
      {
        shapeTypeError_ ( context, *sm(i,j), iitem, err );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getTypeFlags
//-----------------------------------------------------------------------


int ShapeTable::getTypeFlags ( const Shape& shape )
{
  Class*  klass = shape.getClass ();
  int     flags = 0;


  if      ( InternalShape::getType()->isBaseOf( klass ) )
  {
    flags |= INTERNAL_TYPE;
  }
  else if ( BoundaryShape::getType()->isBaseOf( klass ) )
  {
    flags |= BOUNDARY_TYPE;
  }
  else if ( InterfaceShape::getType()->isBaseOf( klass ) )
  {
    flags |= INTERFACE_TYPE;
  }

  if ( shape.isRegular() )
  {
    flags |= REGULAR_TYPE;
  }

  if ( shape.hasExtension<Grad2Extension>() )
  {
    flags |= GRAD2_TYPE;
  }

  if ( shape.hasExtension<BezierExtension>() )
  {
    flags |= BEZIER_TYPE;
  }

  return flags;
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void ShapeTable::store ( const Properties& globdat ) const
{
  Ref<Object>  items = getItems ();
  Ref<Self>    self  = const_cast<Self*> ( this );

  Globdat::storeFor ( Globdat::SHAPE_TABLE,
                      self, items, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<ShapeTable> ShapeTable::find

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<Self>  st;

  Globdat::findFor ( st, Globdat::SHAPE_TABLE, items, globdat );

  return st;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<ShapeTable> ShapeTable::get

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat,
    const String&        context )

{
  Ref<Self>  st;
  bool       found;


  found = Globdat::findFor ( st, Globdat::SHAPE_TABLE,
                             items, globdat );

  if ( ! found )
  {
    String  setName  = items->getName     ();
    String  itemName = items->getItemName ();

    throw IllegalInputException (
      context,
      String::format (
        "no shape table has been defined for %s set `%s\'",
        itemName,
        setName
      )
    );
  }

  return st;
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void ShapeTable::emitEvents_ ( int events )
{}


//-----------------------------------------------------------------------
//   noShapeError_
//-----------------------------------------------------------------------


void ShapeTable::noShapeError_

  ( const String&  context,
    idx_t          iitem ) const

{
  const String  itemName = getItems()->getItemName ();
  const idx_t   itemID   = getItems()->getItemID   ( iitem );


  throw IllegalInputException (
    context,
    String::format (
      "no shape has been defined for %s %d", itemName, itemID
    )
  );
}


void ShapeTable::noShapeError_

  ( const String&  context,
    idx_t          iitem,
    idx_t          prec ) const

{
  const String  itemName = getItems()->getItemName ();
  const idx_t   itemID   = getItems()->getItemID   ( iitem );


  throw IllegalInputException (
    context,
    String::format (
      "%s %d does not have a shape with precision %d",
      itemName,
      itemID,
      prec
    )
  );
}


//-----------------------------------------------------------------------
//   shapeTypeError_
//-----------------------------------------------------------------------


void ShapeTable::shapeTypeError_

  ( const String&  context,
    const Shape&   shape,
    idx_t          iitem,
    int            err ) const

{
  const String  shapeName = shape      .getName     ();
  const String  itemName  = getItems()->getItemName ();
  const idx_t   itemID    = getItems()->getItemID   ( iitem );

  String        details;


  if      ( err & INTERNAL_TYPE )
  {
    details = "is not an internal shape";
  }
  else if ( err & BOUNDARY_TYPE )
  {
    details = "is not a boundary shape";
  }
  else if ( err & INTERFACE_TYPE )
  {
    details = "an interface shape";
  }
  else if ( err & REGULAR_TYPE )
  {
    details = "a regular shape";
  }
  else if ( err & GRAD2_TYPE )
  {
    details = "does not implement the Grad2 extension";
  }
  else if ( err & BEZIER_TYPE )
  {
    details = "does not implement the Bezier extension";
  }

  throw IllegalInputException (
    context,
    String::format (
      "shape `%s\' associated with %s %d %s",
      shapeName,
      itemName,
      itemID,
      details
    )
  );
}


//=======================================================================
//   getTopologies
//=======================================================================

//-----------------------------------------------------------------------
//   getTopologies
//-----------------------------------------------------------------------


TopoVector                  getTopologies

  ( const IShapeVector&       shapes )

{
  const idx_t  n = shapes.size ();

  TopoVector   topos ( n );


  for ( idx_t i = 0; i < n; i++ )
  {
    if ( shapes[i] )
    {
      topos[i] = shapes[i]->getBoundaryTopology ();
    }
  }

  return topos;
}


JIVE_END_PACKAGE( mbody )
