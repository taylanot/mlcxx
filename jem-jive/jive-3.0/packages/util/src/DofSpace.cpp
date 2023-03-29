
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
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/utilities.h>
#include <jem/io/TextOutput.h>
#include <jem/util/Properties.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/Globdat.h>
#include <jive/util/DofSpace.h>


JEM_DEFINE_CLASS( jive::util::DofSpace );


JIVE_BEGIN_PACKAGE( util )


using jem::shape;
using jem::IllegalInputException;
using jem::IllegalArgumentException;


//=======================================================================
//   class DofSpace
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


DofSpace::~DofSpace ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String DofSpace::toString () const
{
  const String  className = getClass()->getName ();
  const String  isetName  = getItems()->getName ();

  if ( isetName.size() > 0 )
  {
    return String::format ( "%s(%s)", className, isetName );
  }
  else
  {
    return className;
  }
}


//-----------------------------------------------------------------------
//   itemCount
//-----------------------------------------------------------------------


idx_t DofSpace::itemCount () const
{
  return getItems()->size ();
}


//-----------------------------------------------------------------------
//   findType
//-----------------------------------------------------------------------


idx_t DofSpace::findType ( const String& name ) const
{
  const idx_t  n = typeCount ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( name == getTypeName(i) )
    {
      return i;
    }
  }

  return -1;
}


//-----------------------------------------------------------------------
//   getTypeNames
//-----------------------------------------------------------------------


StringVector DofSpace::getTypeNames () const
{
  const idx_t   n = typeCount ();

  StringVector  names ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    names[i] = getTypeName ( i );
  }

  return names;
}


//-----------------------------------------------------------------------
//   findDofIndices
//-----------------------------------------------------------------------


idx_t DofSpace::findDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( idofs.size() >= iitems.size() * itypes.size(),
                  "DOF index array too small" );

  const idx_t  m = itypes.size ();
  const idx_t  n = iitems.size ();

  idx_t        k = 0;

  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      idx_t  idof = findDofIndex ( iitems[j], itypes[i] );

      if ( idof >= 0 )
      {
        idofs[j * m + i] = idof;
        k++;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   collectDofIndices
//-----------------------------------------------------------------------


idx_t DofSpace::collectDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( idofs.size() >= iitems.size() * itypes.size(),
                  "DOF index array too small" );

  const idx_t  m = itypes.size ();
  const idx_t  n = iitems.size ();

  idx_t        k = 0;

  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      idx_t  idof = findDofIndex ( iitems[j], itypes[i] );

      if ( idof >= 0 )
      {
        idofs[k++] = idof;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofIndices
//-----------------------------------------------------------------------


void DofSpace::getDofIndices

  ( const IdxVector&  idofs,
    idx_t             iitem,
    const IdxVector&  itypes ) const

{
  idx_t        itmp[1] = { iitem };
  IdxVector    iitems  ( itmp, 1 );

  const idx_t  k = findDofIndices ( idofs, iitems, itypes );

  if ( k != itypes.size() )
  {
    missingDofError_ ( iitems, itypes );
  }
}


void DofSpace::getDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    idx_t             itype ) const

{
  idx_t        itmp[1] = { itype };
  IdxVector    itypes  ( itmp, 1 );

  const idx_t  k = findDofIndices ( idofs, iitems, itypes );

  if ( k != iitems.size() )
  {
    missingDofError_ ( iitems, itypes );
  }
}


//-----------------------------------------------------------------------
//   getDofsForItem
//-----------------------------------------------------------------------


idx_t DofSpace::getDofsForItem

  ( const IdxVector&  idofs,
    const IdxVector&  itypes,
    idx_t             iitem ) const

{
  const idx_t  m = typeCount ();
  const idx_t  n = jem::min  ( idofs.size(), itypes.size() );

  idx_t        k = 0;

  for ( idx_t itype = 0; itype < m && k < n; itype++ )
  {
    idx_t  idof = findDofIndex ( iitem, itype );

    if ( idof >= 0 )
    {
      idofs [k] = idof;
      itypes[k] = itype;

      k++;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofsForType
//-----------------------------------------------------------------------


idx_t DofSpace::getDofsForType

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    idx_t             itype ) const

{
  const idx_t  m = itemCount ();
  const idx_t  n = jem::min  ( idofs.size(), iitems.size() );

  idx_t        k = 0;

  for ( idx_t iitem = 0; iitem < m && k < n; iitem++ )
  {
    idx_t  idof = findDofIndex ( iitem, itype );

    if ( idof >= 0 )
    {
      idofs [k] = idof;
      iitems[k] = iitem;

      k++;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   decodeDofIndices
//-----------------------------------------------------------------------


void DofSpace::decodeDofIndices

  ( const IdxVector&  iitems,
    const IdxVector&  itypes,
    const IdxVector&  idofs ) const

{
  JEM_PRECHECK2 ( iitems.size() == idofs.size() &&
                  itypes.size() == idofs.size(),
                  "Array size mismatch" );

  const idx_t  n = idofs.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    decodeDofIndex ( iitems[i], itypes[i], idofs[i] );
  }
}


//-----------------------------------------------------------------------
//   getDofName
//-----------------------------------------------------------------------


String DofSpace::getDofName ( idx_t idof ) const
{
  idx_t  iitem;
  idx_t  itype;

  decodeDofIndex ( iitem, itype, idof );

  return getDofName ( iitem, itype );
}


String DofSpace::getDofName ( idx_t iitem, idx_t itype ) const
{
  String  typeName = getTypeName           ( itype );
  idx_t   itemID   = getItems()->getItemID ( iitem );

  return String::format ( "%s[%d]", typeName, itemID );
}


//-----------------------------------------------------------------------
//   getDofPath
//-----------------------------------------------------------------------


String DofSpace::getDofPath ( idx_t idof ) const
{
  idx_t  iitem;
  idx_t  itype;

  decodeDofIndex ( iitem, itype, idof );

  return getDofPath ( iitem, itype );
}


String DofSpace::getDofPath ( idx_t iitem, idx_t itype ) const
{
  idx_t itemID = getItems()->getItemID ( iitem );

  return joinNames ( getItems()->getItemPath( itemID ),
                     getTypeName ( itype ) );
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String DofSpace::getContext () const
{
  const ItemSet*  items       = getItems           ();
  const String    itemName    = items->getItemName ();
  const String    itemSetName = items->getName     ();

  if ( itemSetName.size() > 0 )
  {
    return String::format ( "%s dof space `%s\'",
                            itemName, itemSetName );
  }
  else
  {
    return String::format ( "%s dof space", itemName );
  }
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void DofSpace::printTo ( jem::io::TextOutput& out ) const
{
  const ItemSet&   items     = * getItems              ();
  const ItemMap&   itemMap   = * items.getItemMap      ();
  const String     itemName  = items.getItemName       ();

  StringVector     typeNames = getTypeNames            ();
  SparseIdxMatrix  dofMat    = toMatrix                ();

  IdxVector        itemOffsets = dofMat.getRowOffsets    ();
  IdxVector        typeIndices = dofMat.getColumnIndices ();
  IdxVector        dofIndices  = dofMat.getValues        ();

  const idx_t      itemCount   = items    .size ();
  const idx_t      typeCount   = typeNames.size ();


  if ( dofMat.size(0) != itemCount ||
       dofMat.size(1) != typeCount )
  {
    shapeError ( JEM_FUNC,
                 "dof matrix",
                 dofMat.shape(), shape( itemCount, typeCount ) );
  }

  for ( idx_t iitem = 0; iitem < itemCount; iitem++ )
  {
    idx_t  n = itemOffsets[iitem + 1];

    print ( out, "  ", itemName, ' ',
            itemMap.getItemID( iitem ), " : " );

    for ( idx_t  i = itemOffsets[iitem]; i < n; i++ )
    {
      idx_t  itype = typeIndices[i];
      idx_t  idof  = dofIndices[i];

      print ( out, "( \'", typeNames[itype], "\', ", idof, " )" );

      if ( i < n - 1 )
      {
        print ( out, ", " );
      }
    }

    print ( out, '\n' );
  }
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void DofSpace::store ( const Properties& globdat ) const
{
  Ref<Self>  self ( const_cast<Self*>( this ) );

  Globdat::storeFor ( Globdat::DOF_SPACE, self,
                      getItems(), globdat );

  globdat.set       ( Globdat::DOF_SPACE, self );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<DofSpace> DofSpace::find ( const Properties& globdat )
{
  Ref<Self> dofs;

  globdat.find ( dofs, Globdat::DOF_SPACE );

  return dofs;
}


Ref<DofSpace> DofSpace::find

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Ref<Self>  dofs;

  Globdat::findFor ( dofs, Globdat::DOF_SPACE, items, globdat );

  return dofs;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<DofSpace> DofSpace::get

  ( const Properties&  globdat,
    const String&      context )

{
  Ref<Self> dofs;

  if ( ! globdat.find( dofs, Globdat::DOF_SPACE ) )
  {
    throw IllegalInputException (
      context,
      "no primary dof space has been set"
    );
  }

  return dofs;
}


Ref<DofSpace> DofSpace::get

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat,
    const String&        context )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  Ref<Self>  dofs;
  bool       found;

  found =

    Globdat::findFor ( dofs, Globdat::DOF_SPACE, items, globdat );

  if ( ! found )
  {
    String  itemName = items->getItemName ();
    String  isetName = items->getName     ();

    throw IllegalInputException (
      context,
      String::format (
        "no dof space associated with %s set `%s\'",
        itemName,
        isetName
      )
    );
  }

  return dofs;
}


//-----------------------------------------------------------------------
//   noSuchTypeError_
//-----------------------------------------------------------------------


void DofSpace::noSuchTypeError_ ( const String& name ) const
{
  throw IllegalInputException (
    getContext (),
    String::format (
      "undefined degree of freedom type: %s", name
    )
  );
}


//-----------------------------------------------------------------------
//   noSuchDofError_
//-----------------------------------------------------------------------


void DofSpace::noSuchDofError_ ( idx_t iitem, idx_t itype ) const
{
  String  dofPath = getDofPath ( iitem, itype );

  throw IllegalInputException (
    getContext (),
    String::format (
      "undefined degree of freedom: %s", dofPath
    )
  );
}


//-----------------------------------------------------------------------
//   missingDofError_
//-----------------------------------------------------------------------


void DofSpace::missingDofError_

  ( const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  const idx_t  m = iitems.size ();
  const idx_t  n = itypes.size ();

  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      if ( findDofIndex( iitems[i], itypes[j] ) < 0 )
      {
        noSuchDofError_ ( iitems[i], itypes[j] );
      }
    }
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   makeDofOrdering
//-----------------------------------------------------------------------


Reordering          makeDofOrdering

  ( const DofSpace&   dofs,
    const IdxVector&  itemPerm )

{
  JEM_PRECHECK2 ( itemPerm.size() == dofs.itemCount(),
                  "permutation array has wrong size" );

  const idx_t  dofCount  = dofs.dofCount  ();
  const idx_t  typeCount = dofs.typeCount ();
  const idx_t  itemCount = itemPerm.size  ();

  IdxVector    invPerm ( itemCount );
  IdxVector    dofPerm ( dofCount  );
  IdxVector    idofs   ( typeCount );
  IdxVector    itypes  ( typeCount );

  idx_t        inext   = 0;


  dofPerm = -1_idx;

  for ( idx_t iitem = 0; iitem < itemCount; iitem++ )
  {
    invPerm[itemPerm[iitem]] = iitem;
  }

  for ( idx_t iitem = 0; iitem < itemCount; iitem++ )
  {
    idx_t  n = dofs.getDofsForItem ( idofs, itypes, invPerm[iitem] );

    if ( n > 2 )
    {
      jem::sort ( idofs[slice(BEGIN,n)] );
    }
    else if ( n == 2 && idofs[0] > idofs[1] )
    {
      jem::swap ( idofs[0], idofs[1] );
    }

    for ( idx_t i = 0; i < n; i++ )
    {
      dofPerm[idofs[i]] = inext++;
    }
  }

  if ( inext != dofCount )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "invalid dof space"
    );
  }

  if ( dofCount > 0 && min( dofPerm ) < 0 )
  {
    String itemName = dofs.getItems()->getItemName ();

    throw IllegalArgumentException (
      JEM_FUNC,
      String::format (
        "invalid %s permutation specified", itemName
      )
    );
  }

  return Reordering ( dofPerm );
}


JIVE_END_PACKAGE( util )
