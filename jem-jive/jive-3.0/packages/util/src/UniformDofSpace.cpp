
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
#include <jive/util/UniformDofSpace.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::UniformDofSpace );


JIVE_BEGIN_PACKAGE( util )


using jem::min;
using jem::newInstance;


//=======================================================================
//   class UniformDofSpace
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


UniformDofSpace::UniformDofSpace ()
{}


UniformDofSpace::UniformDofSpace

  ( const Ref<ItemSet>&  items,
    const String&        dofType ) :

    items_ ( items )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  typeNames_.addName ( dofType );

  connect_ ();
}


UniformDofSpace::UniformDofSpace

  ( const Ref<ItemSet>&  items,
    const String&        dofType1,
    const String&        dofType2 ) :

    items_ ( items )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  typeNames_.addName ( dofType1 );
  typeNames_.addName ( dofType2 );

  connect_ ();
}


UniformDofSpace::UniformDofSpace

  ( const Ref<ItemSet>&  items,
    const String&        dofType1,
    const String&        dofType2,
    const String&        dofType3 ) :

    items_ ( items )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  typeNames_.addName ( dofType1 );
  typeNames_.addName ( dofType2 );
  typeNames_.addName ( dofType3 );

  connect_ ();
}


UniformDofSpace::UniformDofSpace

  ( const Ref<ItemSet>&  items,
    const StringVector&  dofTypes ) :

    items_     (    items ),
    typeNames_ ( dofTypes )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  connect_ ();
}


UniformDofSpace::~UniformDofSpace ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void UniformDofSpace::resetEvents ()
{
  items_->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void UniformDofSpace::readFrom ( ObjectInput& in )
{
  decode   ( in, items_, typeNames_ );
  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void UniformDofSpace::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items_, typeNames_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> UniformDofSpace::clone () const
{
  return Ref<Object> ( const_cast<Self*>( this ) );
}


//-----------------------------------------------------------------------
//   typeCount
//-----------------------------------------------------------------------


idx_t UniformDofSpace::typeCount () const
{
  return typeNames_.size ();
}


//-----------------------------------------------------------------------
//   dofCount
//-----------------------------------------------------------------------


idx_t UniformDofSpace::dofCount () const
{
  return ( items_->size() * typeNames_.size() );
}


//-----------------------------------------------------------------------
//   findType
//-----------------------------------------------------------------------


idx_t UniformDofSpace::findType ( const String& name ) const
{
  return typeNames_.findName ( name );
}


//-----------------------------------------------------------------------
//   getTypeName
//-----------------------------------------------------------------------


String UniformDofSpace::getTypeName ( idx_t itype ) const
{
  if ( itype < 0 || itype >= typeNames_.size() )
  {
    indexError ( JEM_FUNC,
                 "dof type", itype, typeNames_.size() );
  }

  return typeNames_.getName ( itype );
}


//-----------------------------------------------------------------------
//   getTypeNames
//-----------------------------------------------------------------------


StringVector UniformDofSpace::getTypeNames () const
{
  return typeNames_.toArray ();
}


//-----------------------------------------------------------------------
//   findDofIndex
//-----------------------------------------------------------------------


idx_t UniformDofSpace::findDofIndex

  ( idx_t  iitem,
    idx_t  itype ) const

{
  return ( iitem * typeNames_.size() + itype );
}


//-----------------------------------------------------------------------
//   findDofIndices
//-----------------------------------------------------------------------


idx_t UniformDofSpace::findDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( idofs.size() >= iitems.size() * itypes.size(),
                  "output array is too small" );

  const idx_t  m = iitems.size     ();
  const idx_t  n = itypes.size     ();
  const idx_t  p = typeNames_.size ();

  idx_t        k = 0;


  for ( idx_t i = 0; i < m; i++ )
  {
    idx_t  iitem = iitems[i];

    for ( idx_t j = 0; j < n; j++ )
    {
      idofs[k++] = iitem * p + itypes[j];
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   collectDofIndices
//-----------------------------------------------------------------------


idx_t UniformDofSpace::collectDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( idofs.size() >= iitems.size() * itypes.size(),
                  "output array is too small" );

  const idx_t  m = iitems.size     ();
  const idx_t  n = itypes.size     ();
  const idx_t  p = typeNames_.size ();

  idx_t        k = 0;


  for ( idx_t i = 0; i < m; i++ )
  {
    idx_t  iitem = iitems[i];

    for ( idx_t j = 0; j < n; j++ )
    {
      idofs[k++] = iitem * p + itypes[j];
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofsForItem
//-----------------------------------------------------------------------


idx_t UniformDofSpace::getDofsForItem

  ( const IdxVector&  idofs,
    const IdxVector&  itypes,
    idx_t             iitem ) const

{
  const idx_t  p = typeNames_.size ();
  const idx_t  n = min ( idofs.size(), min( itypes.size(), p ) );

  for ( idx_t itype = 0; itype < n; itype++ )
  {
    idofs [itype] = iitem * p + itype;
    itypes[itype] = itype;
  }

  return n;
}


//-----------------------------------------------------------------------
//   getDofsForType
//-----------------------------------------------------------------------


idx_t UniformDofSpace::getDofsForType

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    idx_t             itype ) const

{
  const idx_t  m = items_->size ();
  const idx_t  n = min ( idofs.size(), min( iitems.size(), m ) );
  const idx_t  p = typeNames_.size ();

  for ( idx_t iitem = 0; iitem < n; iitem++ )
  {
    idofs [iitem] = iitem * p + itype;
    iitems[iitem] = iitem;
  }

  return n;
}


//-----------------------------------------------------------------------
//   decodeDofIndex
//-----------------------------------------------------------------------


void UniformDofSpace::decodeDofIndex

  ( idx_t&  iitem,
    idx_t&  itype,
    idx_t   idof ) const

{
  const idx_t  p = typeNames_.size ();

  iitem = idof / p;
  itype = idof - iitem * p;
}


//-----------------------------------------------------------------------
//   decodeDofIndices
//-----------------------------------------------------------------------


void UniformDofSpace::decodeDofIndices

  ( const IdxVector&  iitems,
    const IdxVector&  itypes,
    const IdxVector&  idofs ) const

{
  JEM_PRECHECK2 ( iitems.size() == idofs.size() &&
                  itypes.size() == idofs.size(),
                  "Array size mismatch" );

  const idx_t  n = idofs     .size ();
  const idx_t  p = typeNames_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  idof = idofs[i];

    iitems[i] = idof / p;
    itypes[i] = idof - iitems[i] * p;
  }
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseIdxMatrix UniformDofSpace::toMatrix () const
{
  const idx_t  itemCount = items_   ->size ();
  const idx_t  typeCount = typeNames_.size ();
  const idx_t  dofCount  = itemCount * typeCount;

  IdxVector    ibuf ( itemCount + 1 + 2 * dofCount );

  IdxVector    itemOffsets;
  IdxVector    typeIndices;
  IdxVector    idofs;

  idx_t        n    = itemCount + 1;
  idx_t        idof = 0;


  itemOffsets.ref ( ibuf[slice(BEGIN,n)]          );
  typeIndices.ref ( ibuf[slice(n,n + dofCount)]   );
  idofs      .ref ( ibuf[slice(n + dofCount,END)] );

  itemOffsets[0] = 0;

  for ( idx_t iitem = 0; iitem < itemCount; iitem++ )
  {
    itemOffsets[iitem + 1] = itemOffsets[iitem] + typeCount;

    for ( idx_t itype = 0; itype < typeCount; itype++ )
    {
      typeIndices[idof] = itype;
      idofs[idof]       = idof;
      idof++;
    }
  }

  return SparseIdxMatrix ( jem::shape ( itemCount, typeCount ),
                           itemOffsets, typeIndices, idofs );
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* UniformDofSpace::getItems () const
{
  return items_.get ();
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void UniformDofSpace::emitEvents_ ( int events )
{}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void UniformDofSpace::connect_ ()
{
  using jem::util::connect;

  connect ( items_->newSizeEvent,  this, & Self::newSizeHandler_  );
  connect ( items_->newOrderEvent, this, & Self::newOrderHandler_ );
}


//-----------------------------------------------------------------------
//   newSizeHandler_
//-----------------------------------------------------------------------


void UniformDofSpace::newSizeHandler_ ( idx_t itemCount )
{
  newSizeEvent.emit ( itemCount * typeNames_.size(), * this );
}


//-----------------------------------------------------------------------
//   newOrderHandler_
//-----------------------------------------------------------------------


void UniformDofSpace::newOrderHandler_ ( const Reordering& reord )
{
  const idx_t*  JEM_RESTRICT  itemMap = reord.rmapPtr ();

  const idx_t   p       = typeNames_.size ();
  const idx_t   start   = reord.start     ();
  const idx_t   oldSize = reord.oldSize   ();
  const idx_t   newSize = reord.newSize   ();

  IdxVector     dofMap  ( oldSize * p );

  idx_t         idof    = p * start;


  for ( idx_t i = 0; i < p * start; i++ )
  {
    dofMap[i] = i;
  }

  for ( idx_t iitem = start; iitem < oldSize; iitem++ )
  {
    idx_t  jitem = itemMap[iitem];
    idx_t  jdof  = jitem * p;

    for ( idx_t i = 0; i < p; i++, idof++, jdof++ )
    {
      dofMap[idof] = jdof;
    }
  }

  newOrderEvent.emit ( Reordering( dofMap, p * newSize ), *this );

  if ( newSize != oldSize )
  {
    newSizeEvent.emit ( p * newSize, *this );
  }
}


JIVE_END_PACKAGE( util )
