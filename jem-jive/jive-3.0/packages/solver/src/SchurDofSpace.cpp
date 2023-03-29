
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
#include <jem/base/IllegalOperationException.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/mp/utilities.h>
#include <jive/solver/SchurDofSpace.h>


JEM_DEFINE_CLASS( jive::solver::SchurDofSpace );


JIVE_BEGIN_PACKAGE( solver )


using jive::util::indexError;


//=======================================================================
//   class SchurDofSpace
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SchurDofSpace::SchurDofSpace

  ( const Ref<DofSpace>&  dofs,
    const BorderSet&      borders ) :

    dofs_        ( dofs ),
    recvBorders_ ( borders ),
    sendBorders_ ( borders )

{
  JEM_PRECHECK ( dofs && borders );

  init_ ();
}


SchurDofSpace::SchurDofSpace

  ( const Ref<DofSpace>&  dofs,
    const BorderSet&      recvBorders,
    const BorderSet&      sendBorders ) :

    dofs_        ( dofs ),
    recvBorders_ ( recvBorders ),
    sendBorders_ ( sendBorders )

{
  JEM_PRECHECK ( dofs        &&
                 recvBorders &&
                 sendBorders );

  init_ ();
}


SchurDofSpace::~SchurDofSpace ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SchurDofSpace::start ()
{
  update ();

  started_++;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SchurDofSpace::finish ()
{
  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SchurDofSpace::update ()
{
  if ( ! started_ )
  {
    resetEvents ();
  }

  if ( ! updated_ )
  {
    update_ ();
  }
}


//-----------------------------------------------------------------------
//   typeCount
//-----------------------------------------------------------------------


idx_t SchurDofSpace::typeCount () const
{
  return dofs_->typeCount ();
}


//-----------------------------------------------------------------------
//   dofCount
//-----------------------------------------------------------------------


idx_t SchurDofSpace::dofCount () const
{
  return borderDofs_.size ();
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SchurDofSpace::resetEvents ()
{
  dofs_      ->resetEvents ();
  recvBorders_.resetEvents ();
  sendBorders_.resetEvents ();
}


//-----------------------------------------------------------------------
//   findType
//-----------------------------------------------------------------------


idx_t SchurDofSpace::findType ( const String& name ) const
{
  return dofs_->findType ( name );
}


//-----------------------------------------------------------------------
//   getTypeName
//-----------------------------------------------------------------------


String SchurDofSpace::getTypeName ( idx_t itype ) const
{
  return dofs_->getTypeName ( itype );
}


//-----------------------------------------------------------------------
//   getTypeNames
//-----------------------------------------------------------------------


StringVector SchurDofSpace::getTypeNames () const
{
  return dofs_->getTypeNames ();
}


//-----------------------------------------------------------------------
//   findDofIndex
//-----------------------------------------------------------------------


idx_t SchurDofSpace::findDofIndex

  ( idx_t  iitem,
    idx_t  itype ) const

{
  idx_t  idof = -1;

  if ( updated_ )
  {
    idof = dofs_->findDofIndex ( iitem, itype );
    idof = dofMap_[idof];
  }

  return idof;
}


//-----------------------------------------------------------------------
//   findDofIndices
//-----------------------------------------------------------------------


idx_t SchurDofSpace::findDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  idx_t  k = 0;

  if ( updated_ )
  {
    idx_t      ibuf [8];
    IdxVector  itmp ( ibuf, 8 );


    if ( idofs.size() > 8 )
    {
      itmp.ref ( IdxVector( idofs.size() ) );
    }

    itmp = -1;

    idx_t  n = dofs_->findDofIndices ( itmp, iitems, itypes );

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = itmp[i];

      if ( idof < 0 )
      {
        continue;
      }

      idof = dofMap_[idof];

      if ( idof >= 0 )
      {
        idofs[i] = itmp[i];
        k++;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   collectDofIndices
//-----------------------------------------------------------------------


idx_t SchurDofSpace::collectDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  idx_t  k = 0;

  if ( updated_ )
  {
    idx_t  n = dofs_->collectDofIndices ( idofs, iitems, itypes );

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = dofMap_[idofs[i]];

      if ( idof >= 0 )
      {
        idofs[k++] = idofs[i];
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofsForItem
//-----------------------------------------------------------------------


idx_t SchurDofSpace::getDofsForItem

  ( const IdxVector&  idofs,
    const IdxVector&  itypes,
    idx_t               iitem ) const

{
  idx_t  k = 0;

  if ( updated_ )
  {
    idx_t  n = dofs_->getDofsForItem ( idofs, itypes, iitem );

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = dofMap_[idofs[i]];

      if ( idof >= 0 )
      {
        idofs [k] = idof;
        itypes[k] = itypes[i];

        k++;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofsForType
//-----------------------------------------------------------------------


idx_t SchurDofSpace::getDofsForType

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    idx_t               itype ) const

{
  idx_t  k = 0;

  if ( updated_ )
  {
    idx_t  n = dofs_->getDofsForType ( idofs, iitems, itype );

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = dofMap_[idofs[i]];

      if ( idof >= 0 )
      {
        idofs [k] = idof;
        iitems[k] = iitems[i];

        k++;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   decodeDofIndex
//-----------------------------------------------------------------------


void SchurDofSpace::decodeDofIndex

  ( idx_t&  iitem,
    idx_t&  itype,
    idx_t   idof ) const

{
  if ( idof < 0 || idof >= borderDofs_.size() )
  {
    indexError ( JEM_FUNC, "DOF", idof, borderDofs_.size() );
  }

  dofs_->decodeDofIndex ( iitem, itype, borderDofs_[idof] );
}


//-----------------------------------------------------------------------
//   decodeDofIndices
//-----------------------------------------------------------------------


void SchurDofSpace::decodeDofIndices

  ( const IdxVector&  iitems,
    const IdxVector&  itypes,
    const IdxVector&  idofs ) const

{
  const idx_t  schurSize = borderDofs_.size ();
  const idx_t  n         = idofs      .size ();

  idx_t        ibuf [8];
  IdxVector    itmp ( ibuf, 8 );


  if ( n > 8 )
  {
    itmp.ref ( IdxVector( n ) );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  idof = idofs[i];

    if ( idof < 0 || idof >= schurSize )
    {
      indexError ( JEM_FUNC, "border DOF", idof, schurSize );
    }

    itmp[i] = borderDofs_[idof];
  }

  dofs_->decodeDofIndices ( iitems, itypes, itmp );
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseIdxMatrix SchurDofSpace::toMatrix () const
{
  using jem::shape;
  using jem::numeric::SparseUtils;

  const idx_t  schurSize = borderDofs_.size ();
  const idx_t  rowCount  = dofs_->itemCount ();
  const idx_t  colCount  = dofs_->typeCount ();

  IdxVector    iitems   ( schurSize );
  IdxVector    itypes   ( schurSize );

  IdxVector    offsets  ( rowCount + 1 );
  IdxVector    indices  ( schurSize );
  IdxVector    idofs    ( schurSize );


  offsets = 0_idx;

  if ( updated_ )
  {
    dofs_->decodeDofIndices ( iitems, itypes, borderDofs_ );

    for ( idx_t i = 0; i < schurSize; i++ )
    {
      offsets[iitems[i]]++;
    }

    SparseUtils::sumOffsets ( offsets );

    for ( idx_t i = 0; i < schurSize; i++ )
    {
      idx_t  iitem  = iitems [i];
      idx_t  offset = offsets[iitem]++;

      indices[offset] = itypes[i];
      idofs  [offset] = i;
    }

    SparseUtils::shiftOffsets ( offsets );
  }

  return SparseIdxMatrix ( shape( rowCount, colCount ),
                           offsets, indices, idofs );
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* SchurDofSpace::getItems () const
{
  return dofs_->getItems ();
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void SchurDofSpace::emitEvents_ ( int events )
{}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void SchurDofSpace::init_ ()
{
  using jem::util::connect;

  ItemSet*  items = dofs_->getItems ();

  if ( recvBorders_.getItems() != items ||
       sendBorders_.getItems() != items )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "incompatible exchange borders and dof space "
      "(different item sets)"
    );
  }

  updated_ = false;
  started_ = 0;

  connect ( dofs_->newSizeEvent,        this, & Self::invalidate_ );
  connect ( dofs_->newOrderEvent,       this, & Self::invalidate_ );
  connect ( recvBorders_.changeEvent(), this, & Self::invalidate_ );

  if ( sendBorders_ != recvBorders_ )
  {
    connect ( sendBorders_.changeEvent(),
              this, & Self::invalidate_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void SchurDofSpace::update_ ()
{
  using jem::count;
  using jive::mp::markBorderDofs;

  const idx_t  totalSize = dofs_->dofCount ();

  IdxVector    ibuf ( totalSize * 2 );
  BoolVector   mask ( totalSize );

  idx_t        i, j, k;


  mask = false;

  markBorderDofs ( mask, recvBorders_, *dofs_ );

  if ( sendBorders_ != recvBorders_ )
  {
    markBorderDofs ( mask, sendBorders_, *dofs_ );
  }

  k = count ( mask );
  i = totalSize;
  j = totalSize + i - k;

  dofMap_    .ref ( ibuf[slice(BEGIN,i)] );
  innerDofs_ .ref ( ibuf[slice(i,j)]     );
  borderDofs_.ref ( ibuf[slice(j,END)]   );

  for ( i = j = k = 0; i < totalSize; i++ )
  {
    if ( mask[i] )
    {
      dofMap_    [i]   = k;
      borderDofs_[k++] = i;
    }
    else
    {
      dofMap_    [i]   = -1;
      innerDofs_ [j++] =  i;
    }
  }

  Self::resetEvents ();

  updated_ = true;

  newSizeEvent.emit ( borderDofs_.size(), *this );
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void SchurDofSpace::invalidate_ ()
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      JEM_FUNC,
      "number of Schur degrees of freedom changed "
      "while solving a linear system of equations"
    );
  }

  if ( updated_ )
  {
    IdxVector  empty;

    updated_ = false;

    dofMap_    .ref ( empty );
    innerDofs_ .ref ( empty );
    borderDofs_.ref ( empty );

    newSizeEvent.emit ( 0, *this );
  }
}


JIVE_END_PACKAGE( solver )
