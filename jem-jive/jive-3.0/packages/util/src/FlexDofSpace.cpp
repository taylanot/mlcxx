
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


#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/log.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DenseDofSpace.h>
#include <jive/util/SparseDofSpace.h>
#include <jive/util/FlexDofSpace.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::FlexDofSpace );


JIVE_BEGIN_PACKAGE( util )


using jem::checkedCast;
using jem::dynamicCast;
using jem::newInstance;


//=======================================================================
//   class FlexDofSpace
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FlexDofSpace::FlexDofSpace ()
{}


FlexDofSpace::FlexDofSpace ( const Ref<ItemSet>& items )
{
  sparse_ = newInstance<SparseDofSpace> ( items );
  dofs_   = sparse_;

  connect_ ();
}


FlexDofSpace::FlexDofSpace ( const Self& rhs )
{
  dofs_   = checkedCast<XDofSpace>      ( rhs.dofs_ );
  dense_  = dynamicCast<DenseDofSpace>  ( dofs_ );
  sparse_ = dynamicCast<SparseDofSpace> ( dofs_ );

  connect_ ();
}


FlexDofSpace::~FlexDofSpace ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void FlexDofSpace::resetEvents ()
{
  dofs_->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void FlexDofSpace::readFrom ( ObjectInput& in )
{
  decode ( in, dofs_ );

  dense_  = dynamicCast<DenseDofSpace>  ( dofs_ );
  sparse_ = dynamicCast<SparseDofSpace> ( dofs_ );

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void FlexDofSpace::writeTo ( ObjectOutput& out ) const
{
  encode ( out, dofs_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> FlexDofSpace::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   typeCount
//-----------------------------------------------------------------------


idx_t FlexDofSpace::typeCount () const
{
  return dofs_->typeCount ();
}


//-----------------------------------------------------------------------
//   dofCount
//-----------------------------------------------------------------------


idx_t FlexDofSpace::dofCount () const
{
  return dofs_->dofCount ();
}


//-----------------------------------------------------------------------
//   clearDofs
//-----------------------------------------------------------------------


void FlexDofSpace::clearDofs ()
{
  dofs_->clearDofs ();
}


//-----------------------------------------------------------------------
//   clearAll
//-----------------------------------------------------------------------


void FlexDofSpace::clearAll ()
{
  dofs_->clearAll ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void FlexDofSpace::reserve ( idx_t n )
{
  dofs_->reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void FlexDofSpace::trimToSize ()
{
  if ( ! switchMode_() )
  {
    dofs_->trimToSize ();
  }
}


//-----------------------------------------------------------------------
//   addType
//-----------------------------------------------------------------------


idx_t FlexDofSpace::addType ( const String& name )
{
  const idx_t  typeCount = dofs_->typeCount ();
  const idx_t  itype     = dofs_->addType   ( name );

  if ( itype >= typeCount )
  {
    switchMode_ ();
  }

  return itype;
}


//-----------------------------------------------------------------------
//   findType
//-----------------------------------------------------------------------


idx_t FlexDofSpace::findType ( const String& name ) const
{
  return dofs_->findType ( name );
}


//-----------------------------------------------------------------------
//   getTypeName
//-----------------------------------------------------------------------


String FlexDofSpace::getTypeName ( idx_t itype ) const
{
  return dofs_->getTypeName ( itype );
}


//-----------------------------------------------------------------------
//   getTypeNames
//-----------------------------------------------------------------------


StringVector FlexDofSpace::getTypeNames () const
{
  return dofs_->getTypeNames ();
}


//-----------------------------------------------------------------------
//   addDof
//-----------------------------------------------------------------------


idx_t FlexDofSpace::addDof

  ( idx_t  iitem,
    idx_t  itype )

{
  return dofs_->addDof ( iitem, itype );
}


//-----------------------------------------------------------------------
//   addDofs
//-----------------------------------------------------------------------


void FlexDofSpace::addDofs

  ( const IdxVector&  iitems,
    const IdxVector&  itypes )

{
  return dofs_->addDofs ( iitems, itypes );
}


//-----------------------------------------------------------------------
//   reorderDofs
//-----------------------------------------------------------------------


void FlexDofSpace::reorderDofs ( const Reordering& reord )
{
  dofs_->reorderDofs ( reord );
}


//-----------------------------------------------------------------------
//   findDofIndex
//-----------------------------------------------------------------------


idx_t FlexDofSpace::findDofIndex ( idx_t iitem, idx_t itype ) const
{
  return dofs_->findDofIndex ( iitem, itype );
}


//-----------------------------------------------------------------------
//   findDofIndices
//-----------------------------------------------------------------------


idx_t FlexDofSpace::findDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  return dofs_->findDofIndices ( idofs, iitems, itypes );
}


//-----------------------------------------------------------------------
//   collectDofIndices
//-----------------------------------------------------------------------


idx_t FlexDofSpace::collectDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  return dofs_->collectDofIndices ( idofs, iitems, itypes );
}


//-----------------------------------------------------------------------
//   getDofsForItem
//-----------------------------------------------------------------------


idx_t FlexDofSpace::getDofsForItem

  ( const IdxVector&  idofs,
    const IdxVector&  itypes,
    idx_t             iitem ) const

{
  return dofs_->getDofsForItem ( idofs, itypes, iitem );
}


//-----------------------------------------------------------------------
//   getDofsForType
//-----------------------------------------------------------------------


idx_t FlexDofSpace::getDofsForType

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    idx_t             itype ) const

{
  return dofs_->getDofsForType ( idofs, iitems, itype );
}


//-----------------------------------------------------------------------
//   decodeDofIndex
//-----------------------------------------------------------------------


void FlexDofSpace::decodeDofIndex

  ( idx_t&  iitem,
    idx_t&  itype,
    idx_t   idof ) const

{
  dofs_->decodeDofIndex ( iitem, itype, idof );
}


//-----------------------------------------------------------------------
//   decodeDofIndices
//-----------------------------------------------------------------------


void FlexDofSpace::decodeDofIndices

  ( const IdxVector&  iitems,
    const IdxVector&  itypes,
    const IdxVector&  idofs ) const

{
  dofs_->decodeDofIndices ( iitems, itypes, idofs );
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseIdxMatrix FlexDofSpace::toMatrix () const
{
  return dofs_->toMatrix ();
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* FlexDofSpace::getItems () const
{
  return dofs_->getItems ();
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void FlexDofSpace::emitEvents_ ( int events )
{}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void FlexDofSpace::connect_ ()
{
  using jem::util::connect;

  connect ( dofs_->newSizeEvent,  this, & Self::dofsResized_   );
  connect ( dofs_->newOrderEvent, this, & Self::dofsReordered_ );
}


//-----------------------------------------------------------------------
//   switchMode_
//-----------------------------------------------------------------------


bool FlexDofSpace::switchMode_ ()
{
  const idx_t  dofCount  = dofs_->dofCount  ();
  const idx_t  itemCount = dofs_->itemCount ();
  const idx_t  typeCount = dofs_->typeCount ();

  bool         isDense   = (2 * dofCount > itemCount * typeCount);


  if ( (dense_  &&   isDense) ||
       (sparse_ && ! isDense) )
  {
    return false;
  }

  StringVector  typeNames = dofs_->getTypeNames ();
  Ref<ItemSet>  items     = dofs_->getItems     ();

  IdxVector     iitems    ( dofCount );
  IdxVector     itypes    ( dofCount );


  if      ( dense_ )
  {
    print ( JIVE_LOG_DEBUG, toString (),
            " : switching to sparse storage mode ...\n\n" );

    dense_->getAllDofs ( iitems, itypes );

    dofs_   = nullptr;
    dense_  = nullptr;
    sparse_ = newInstance<SparseDofSpace> ( items );
    dofs_   = sparse_;

    sparse_->addTypes   ( typeNames );
    sparse_->setAllDofs ( iitems, itypes );
  }
  else if ( sparse_ )
  {
    print ( JIVE_LOG_DEBUG, toString (),
            " : switching to dense storage mode ...\n\n" );

    sparse_->getAllDofs ( iitems, itypes );

    dofs_   = nullptr;
    sparse_ = nullptr;
    dense_  = newInstance<DenseDofSpace>  ( items );
    dofs_   = dense_;

    dense_->addTypes   ( typeNames );
    dense_->setAllDofs ( iitems, itypes );
  }

  connect_ ();

  return true;
}


//-----------------------------------------------------------------------
//   dofsResized_
//-----------------------------------------------------------------------


void FlexDofSpace::dofsResized_ ( idx_t dofCount )
{
  newSizeEvent.emit ( dofCount, *this );
}


//-----------------------------------------------------------------------
//   dofsReordered_
//-----------------------------------------------------------------------


void FlexDofSpace::dofsReordered_ ( const Reordering& reord )
{
  newOrderEvent.emit ( reord, *this );
}


//-----------------------------------------------------------------------
//   newDofSpace_
//-----------------------------------------------------------------------


Ref<XDofSpace> XDofSpace::newDofSpace_ ( const Ref<ItemSet>& items )
{
  return newInstance<FlexDofSpace> ( items );
}


JIVE_END_PACKAGE( util )
