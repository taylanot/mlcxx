
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


#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Flex.h>
#include <jem/util/Event.h>
#include <jive/util/private/NameList.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/EventMask.h>
#include <jive/util/Reordering.h>
#include <jive/util/DenseDofSpace.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::DenseDofSpace );


JIVE_BEGIN_PACKAGE( util )


using jem::min;
using jem::shape;
using jem::newInstance;
using jem::util::Flex;


//=======================================================================
//   class DenseDofSpace::Data_
//=======================================================================


class DenseDofSpace::Data_ : public jem::Collectable
{
 public:

  typedef Data_           Self;

  static const int        NEW_SIZE = 1 << 0;

  enum                    DofRecord
  {
                            IITEM,
                            ITYPE,
                            RECSIZE
  };


  inline                  Data_       ();

  explicit                Data_

    ( const Ref<ItemSet>&   items );

  inline void             readFrom

    ( ObjectInput&          in  );

  inline void             writeTo

    ( ObjectOutput&         out )        const;

  inline void             reserve

    ( idx_t                 count );

  void                    resize      ();
  void                    rebuild     ();
  inline idx_t            typeCount   () const;
  inline idx_t            dofCount    () const;

  inline void             checkItemIndex

    ( idx_t                 iitem );

  inline bool             checkItemIndex

    ( idx_t                 iitem )      const;

  inline void             checkTypeIndex

    ( idx_t                 itype );

  inline bool             checkTypeIndex

    ( idx_t                 itype )      const;

  inline void             checkDofIndex

    ( idx_t                 idof )       const;

  inline idx_t            addDof

    ( idx_t                 iitem,
      idx_t                 itype  );

  void                    itemIndexError

    ( idx_t                 iitem )      const;

  void                    typeIndexError

    ( idx_t                 itype )      const;

  void                    dofIndexError

    ( idx_t                 idof )       const;


 public:

  Ref<ItemSet>            items;
  NameList                typeNames;
  IdxMatrix               dmap;
  Flex<idx_t>             dofs;
  EventMask               events;


 private:

  void                    checkItemIndex_

    ( idx_t                 iitem );

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline DenseDofSpace::Data_::Data_ ()
{}


DenseDofSpace::Data_::Data_ ( const Ref<ItemSet>& iset ) :

  items ( iset )

{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


inline void DenseDofSpace::Data_::readFrom ( ObjectInput& in )
{
  decode ( in, items, typeNames, dofs );

  if ( dofs.size() % RECSIZE != 0 )
  {
    jem::io::decodeError ( JEM_FUNC );
  }

  events.reset ();
  rebuild      ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


inline void DenseDofSpace::Data_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items, typeNames, dofs );
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


inline void DenseDofSpace::Data_::reserve ( idx_t n )
{
  dofs.reserve ( n * RECSIZE );
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void DenseDofSpace::Data_::resize ()
{
  const idx_t  m = dmap.size (0);
  const idx_t  n = dmap.size (1);

  idx_t        p = items   ->size ();
  idx_t        q = typeNames.size ();


  if ( p != m || q != n )
  {
    IdxMatrix  newMap ( p, q );

    p = min ( p, m );
    q = min ( q, n );

    newMap = -1_idx;

    newMap(slice(BEGIN,p),slice(BEGIN,q)) =
      dmap(slice(BEGIN,p),slice(BEGIN,q));

    dmap.swap ( newMap );
  }
}


//-----------------------------------------------------------------------
//   rebuild
//-----------------------------------------------------------------------


void DenseDofSpace::Data_::rebuild ()
{
  const idx_t*  JEM_RESTRICT  dp = dofs.addr ();

  const idx_t   dofCount  = dofs.size() / RECSIZE;
  const idx_t   typeCount = typeNames.size ();
  const idx_t   itemCount = items   ->size ();


  if ( dmap.size(0) != itemCount ||
       dmap.size(1) != typeCount )
  {
    dmap.resize ( itemCount, typeCount );
  }

  dmap = -1_idx;

  for ( idx_t idof = 0; idof < dofCount; idof++, dp += RECSIZE )
  {
    dmap(dp[IITEM],dp[ITYPE]) = idof;
  }
}


//-----------------------------------------------------------------------
//   typeCount
//-----------------------------------------------------------------------


inline idx_t DenseDofSpace::Data_::typeCount () const
{
  return typeNames.size ();
}


//-----------------------------------------------------------------------
//   dofCount
//-----------------------------------------------------------------------


inline idx_t DenseDofSpace::Data_::dofCount () const
{
  return (dofs.size() / RECSIZE);
}


//-----------------------------------------------------------------------
//   checkItemIndex
//-----------------------------------------------------------------------


inline void DenseDofSpace::Data_::checkItemIndex ( idx_t iitem )
{
  if ( iitem < 0 || iitem >= dmap.size(0) )
  {
    checkItemIndex_ ( iitem );
  }
}


inline bool DenseDofSpace::Data_::checkItemIndex ( idx_t iitem ) const
{
  if ( iitem < 0 )
  {
    itemIndexError ( iitem );
  }

  return (iitem < dmap.size(0));
}


//-----------------------------------------------------------------------
//   checkTypeIndex
//-----------------------------------------------------------------------


inline void DenseDofSpace::Data_::checkTypeIndex ( idx_t itype )
{
  if ( itype < 0 || itype >= typeCount() )
  {
    typeIndexError ( itype );
  }

  if ( itype >= dmap.size(1) )
  {
    resize ();
  }
}


inline bool DenseDofSpace::Data_::checkTypeIndex ( idx_t itype ) const
{
  if ( itype < 0 || itype >= typeCount() )
  {
    typeIndexError ( itype );
  }

  return (itype < dmap.size(1));
}


//-----------------------------------------------------------------------
//   checkDofIndex
//-----------------------------------------------------------------------


inline void DenseDofSpace::Data_::checkDofIndex ( idx_t idof  ) const
{
  if ( idof < 0 || (RECSIZE * idof) >= dofs.size() )
  {
    dofIndexError ( idof );
  }
}


//-----------------------------------------------------------------------
//   addDof
//-----------------------------------------------------------------------


inline idx_t DenseDofSpace::Data_::addDof

  ( idx_t  iitem,
    idx_t  itype )

{
  idx_t  idof = dmap(iitem,itype);

  if ( idof < 0 )
  {
    idof = dofs.size() / RECSIZE;

    dofs.pushBack ( iitem );
    dofs.pushBack ( itype );

    dmap(iitem,itype) = idof;

    events.set ( NEW_SIZE );
  }

  return idof;
}


//-----------------------------------------------------------------------
//   itemIndexError
//-----------------------------------------------------------------------


void DenseDofSpace::Data_::itemIndexError ( idx_t iitem ) const
{
  indexError ( CLASS_NAME,
               items->getItemName(), iitem, items->size() );
}


//-----------------------------------------------------------------------
//   typeIndexError
//-----------------------------------------------------------------------


void DenseDofSpace::Data_::typeIndexError ( idx_t itype ) const
{
  indexError ( CLASS_NAME, "DOF type", itype, typeCount() );
}


//-----------------------------------------------------------------------
//   dofIndexError
//-----------------------------------------------------------------------


void DenseDofSpace::Data_::dofIndexError ( idx_t idof ) const
{
  indexError ( CLASS_NAME, "DOF", idof, dofCount() );
}


//-----------------------------------------------------------------------
//   checkItemIndex_
//-----------------------------------------------------------------------


void DenseDofSpace::Data_::checkItemIndex_ ( idx_t iitem )
{
  const idx_t  itemCount = items->size ();

  if ( iitem < 0 || iitem >= itemCount )
  {
    itemIndexError ( iitem );
  }

  if ( iitem >= dmap.size(0) )
  {
    resize ();
  }
}


//=======================================================================
//   class DenseDofSpace
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DenseDofSpace::DenseDofSpace ()
{}


DenseDofSpace::DenseDofSpace ( const Ref<ItemSet>& items )
{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  data_ = newInstance<Data_> ( items );

  connect_ ();
}


DenseDofSpace::DenseDofSpace ( const Self& rhs )
{
  data_ = newInstance<Data_> ( * rhs.data_ );

  connect_ ();
}


DenseDofSpace::~DenseDofSpace ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void DenseDofSpace::resetEvents ()
{
  data_->events.reset       ();
  data_->items->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void DenseDofSpace::readFrom ( ObjectInput& in )
{
  data_ = newInstance<Data_> ();

  data_->readFrom ( in );

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void DenseDofSpace::writeTo ( ObjectOutput& out ) const
{
  data_->writeTo ( out );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> DenseDofSpace::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   typeCount
//-----------------------------------------------------------------------


idx_t DenseDofSpace::typeCount () const
{
  return data_->typeCount ();
}


//-----------------------------------------------------------------------
//   dofCount
//-----------------------------------------------------------------------


idx_t DenseDofSpace::dofCount () const
{
  return data_->dofCount ();
}


//-----------------------------------------------------------------------
//   clearDofs
//-----------------------------------------------------------------------


void DenseDofSpace::clearDofs ()
{
  Data_&  d = * data_;

  if ( d.dofs.size() > 0 )
  {
    d.dmap = -1;

    d.dofs  .clear ();
    d.events.reset ();

    newSizeEvent.emit ( 0, *this );
  }
}


//-----------------------------------------------------------------------
//   clearAll
//-----------------------------------------------------------------------


void DenseDofSpace::clearAll ()
{
  data_->typeNames.clear ();

  clearDofs ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void DenseDofSpace::reserve ( idx_t n )
{
  data_->reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void DenseDofSpace::trimToSize ()
{
  data_->resize          ();
  data_->dofs.trimToSize ();
}


//-----------------------------------------------------------------------
//   addType
//-----------------------------------------------------------------------


idx_t DenseDofSpace::addType ( const String& name )
{
  return data_->typeNames.addName ( name );
}


//-----------------------------------------------------------------------
//   findType
//-----------------------------------------------------------------------


idx_t DenseDofSpace::findType ( const String& name ) const
{
  return data_->typeNames.findName ( name );
}


//-----------------------------------------------------------------------
//   getTypeName
//-----------------------------------------------------------------------


String DenseDofSpace::getTypeName ( idx_t itype ) const
{
  const Data_&  d = * data_;

  d.checkTypeIndex ( itype );

  return d.typeNames.getName ( itype );
}


//-----------------------------------------------------------------------
//   getTypeNames
//-----------------------------------------------------------------------


StringVector DenseDofSpace::getTypeNames () const
{
  return data_->typeNames.toArray ();
}


//-----------------------------------------------------------------------
//   addDof
//-----------------------------------------------------------------------


idx_t DenseDofSpace::addDof

  ( idx_t  iitem,
    idx_t  itype )

{
  Data_&  d = * data_;

  d.checkItemIndex ( iitem );
  d.checkTypeIndex ( itype );

  const idx_t  idof  = d.addDof ( iitem, itype );

  d.events.emit ( this );

  return idof;
}


//-----------------------------------------------------------------------
//   addDofs
//-----------------------------------------------------------------------


void DenseDofSpace::addDofs

  ( const IdxVector&  iitems,
    const IdxVector&  itypes )

{
  Data_&       d = * data_;

  const idx_t  m = iitems.size ();
  const idx_t  n = itypes.size ();


  if ( m * n > 1000_idx )
  {
    d.reserve ( d.dofCount() + m * n );
  }

  for ( idx_t j = 0; j < n; j++ )
  {
    d.checkTypeIndex ( itypes[j] );
  }

  for ( idx_t i = 0; i < m; i++ )
  {
    idx_t  iitem = iitems[i];

    d.checkItemIndex ( iitem );

    for ( idx_t j = 0; j < n; j++ )
    {
      d.addDof ( iitem, itypes[j] );
    }
  }

  d.events.emit ( this );
}


//-----------------------------------------------------------------------
//   reorderDofs
//-----------------------------------------------------------------------


void DenseDofSpace::reorderDofs ( const Reordering& reord )
{
  JEM_PRECHECK2 ( reord.oldSize() == dofCount(),
                  "invalid Reordering" );

  Data_&        d = * data_;

  const idx_t*  JEM_RESTRICT  rmap = reord.rmapPtr ();
  idx_t*        JEM_RESTRICT  dofs = d.dofs.addr   ();

  const idx_t   start   = reord.start   ();
  const idx_t   newSize = reord.newSize ();


  if ( reord.isSimple() )
  {
    for ( idx_t idof = start; idof < newSize; idof++ )
    {
      idx_t  i = idof       * Data_::RECSIZE;
      idx_t  j = rmap[idof] * Data_::RECSIZE;

      dofs[i + Data_::IITEM] = dofs[j + Data_::IITEM];
      dofs[i + Data_::ITYPE] = dofs[j + Data_::ITYPE];
    }

    d.dofs.resize ( newSize * Data_::RECSIZE );
  }
  else
  {
    Flex<idx_t>  tmpDofs ( newSize * Data_::RECSIZE );

    idx_t*       JEM_RESTRICT  newDofs = tmpDofs.addr ();


    if ( start > 0 )
    {
      std::memcpy ( newDofs, dofs, (size_t)
                    (start * Data_::RECSIZE) * sizeof(idx_t) );

      newDofs += (start * Data_::RECSIZE);
    }

    for ( idx_t  idof = start; idof < newSize;
          idof++, newDofs += Data_::RECSIZE )
    {
      idx_t  j = rmap[idof] * Data_::RECSIZE;

      newDofs[Data_::IITEM] = dofs[j + Data_::IITEM];
      newDofs[Data_::ITYPE] = dofs[j + Data_::ITYPE];
    }

    d.dofs.swap ( tmpDofs );
  }

  d.rebuild      ();
  d.events.reset ();

  newOrderEvent.emit ( reord, *this );

  if ( reord.newSize() != reord.oldSize() )
  {
    newSizeEvent.emit ( reord.newSize(), *this );
  }
}


//-----------------------------------------------------------------------
//   findDofIndex
//-----------------------------------------------------------------------


idx_t DenseDofSpace::findDofIndex ( idx_t iitem, idx_t itype ) const
{
  const Data_&  d = * data_;

  if ( d.checkItemIndex( iitem ) && d.checkTypeIndex( itype ) )
  {
    return d.dmap(iitem,itype);
  }
  else
  {
    return -1;
  }
}


//-----------------------------------------------------------------------
//   findDofIndices
//-----------------------------------------------------------------------


idx_t DenseDofSpace::findDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( idofs.size() >= iitems.size() * itypes.size(),
                  "DOF index array is too small" );

  const Data_&  d = * data_;

  const idx_t   m = iitems.size ();
  const idx_t   n = itypes.size ();

  const idx_t   lastItem = d.dmap.size(0) - 1;

  const idx_t*  dmap;


  for ( idx_t i = 0; i < m; i++ )
  {
    d.checkItemIndex ( iitems[i] );
  }

  idx_t  k = 0;

  for ( idx_t j = 0; j < n; j++ )
  {
    idx_t  itype = itypes[j];

    if ( ! d.checkTypeIndex( itype ) )
    {
      continue;
    }

    dmap = d.dmap.addr ( itype );

    for ( idx_t i = 0; i < m; i++ )
    {
      idx_t  iitem = iitems[i];

      if ( iitem > lastItem )
      {
        continue;
      }

      idx_t  idof = dmap[iitem];

      if ( idof >= 0 )
      {
        idofs[i * n + j] = idof;
        k++;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   collectDofIndices
//-----------------------------------------------------------------------


idx_t DenseDofSpace::collectDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( idofs.size() >= iitems.size() * itypes.size(),
                  "DOF index array is too small" );

  const Data_&  d = * data_;

  const idx_t   m = iitems.size ();
  const idx_t   n = itypes.size ();

  const idx_t   lastType = d.dmap.size(1) - 1;


  for ( idx_t j = 0; j < n; j++ )
  {
    d.checkTypeIndex ( itypes[j] );
  }

  idx_t  k = 0;

  for ( idx_t i = 0; i < m; i++ )
  {
    idx_t  iitem = iitems[i];

    if ( ! d.checkItemIndex( iitem ) )
    {
      continue;
    }

    for ( idx_t j = 0; j < n; j++ )
    {
      idx_t  itype = itypes[j];

      if ( itype > lastType )
      {
        continue;
      }

      idx_t  idof = d.dmap(iitem,itype);

      if ( idof >= 0 )
      {
        idofs[k++] = idof;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofsForItem
//-----------------------------------------------------------------------


idx_t DenseDofSpace::getDofsForItem

  ( const IdxVector&  idofs,
    const IdxVector&  itypes,
    idx_t             iitem ) const

{
  const Data_&  d    = * data_;
  const idx_t*  dofs = d.dofs.addr ();


  if ( ! d.checkItemIndex( iitem ) )
  {
    return 0;
  }

  idx_t  n = min ( idofs.size(), itypes.size(), d.dmap.size(1) );
  idx_t  k = 0;

  for ( idx_t j = 0; j < n; j++ )
  {
    idx_t  idof = d.dmap(iitem,j);

    if ( idof >= 0 )
    {
      idofs [k] = idof;
      itypes[k] = dofs[idof * Data_::RECSIZE + Data_::ITYPE];
      k++;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofsForType
//-----------------------------------------------------------------------


idx_t DenseDofSpace::getDofsForType

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    idx_t             itype ) const

{
  const Data_&  d    = * data_;
  const idx_t*  dofs = d.dofs.addr ();


  if ( ! d.checkTypeIndex( itype ) )
  {
    return 0;
  }

  idx_t  n = min ( idofs.size(), iitems.size(), d.dmap.size(0) );
  idx_t  k = 0;

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  idof = d.dmap(i,itype);

    if ( idof >= 0 )
    {
      idofs [k] = idof;
      iitems[k] = dofs[idof * Data_::RECSIZE + Data_::IITEM];
      k++;
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   decodeDofIndex
//-----------------------------------------------------------------------


void DenseDofSpace::decodeDofIndex

  ( idx_t&  iitem,
    idx_t&  itype,
    idx_t   idof ) const

{
  const Data_&  d = * data_;

  d.checkDofIndex ( idof );

  iitem = d.dofs[idof * Data_::RECSIZE + Data_::IITEM];
  itype = d.dofs[idof * Data_::RECSIZE + Data_::ITYPE];
}


//-----------------------------------------------------------------------
//   decodeDofIndices
//-----------------------------------------------------------------------


void DenseDofSpace::decodeDofIndices

  ( const IdxVector&  iitems,
    const IdxVector&  itypes,
    const IdxVector&  idofs ) const

{
  JEM_PRECHECK2 ( iitems.size() == idofs.size() &&
                  itypes.size() == idofs.size(),
                  "Array size mismatch" );

  const Data_&  d    = * data_;
  const idx_t   n    = idofs .size ();
  const idx_t*  dofs = d.dofs.addr ();


  for ( idx_t i = 0; i < n; i++ )
  {
    d.checkDofIndex ( idofs[i] );

    idx_t  k = idofs[i] * Data_::RECSIZE;

    iitems[i] = dofs[k + Data_::IITEM];
    itypes[i] = dofs[k + Data_::ITYPE];
  }
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseIdxMatrix DenseDofSpace::toMatrix () const
{
  const Data_&  d = * data_;

  const idx_t   dofCount  = d.dofCount    ();
  const idx_t   itemCount = d.items->size ();
  const idx_t   typeCount = d.typeCount   ();

  IdxVector     ibuf      ( itemCount + 1 + 2 * dofCount );

  IdxVector     itemOffsets;
  IdxVector     typeIndices;
  IdxVector     dofIndices;

  idx_t         j, m, n;


  n = itemCount + 1;

  itemOffsets.ref ( ibuf[slice(BEGIN,n)]          );
  typeIndices.ref ( ibuf[slice(n,n + dofCount)]   );
  dofIndices .ref ( ibuf[slice(n + dofCount,END)] );

  m = min ( itemCount, d.dmap.size(0) );
  n = min ( typeCount, d.dmap.size(1) );
  j = 0;

  for ( idx_t iitem = 0; iitem < m; iitem++ )
  {
    itemOffsets[iitem] = j;

    for ( idx_t itype = 0; itype < n; itype++ )
    {
      idx_t  idof = d.dmap(iitem,itype);

      if ( idof >= 0 )
      {
        typeIndices[j] = itype;
        dofIndices [j] = idof;
        j++;
      }
    }
  }

  itemOffsets[slice(m,END)] = j;

  return SparseIdxMatrix ( shape ( itemCount, typeCount ),
                           itemOffsets,
                           typeIndices,
                           dofIndices );
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* DenseDofSpace::getItems () const
{
  return data_->items.get ();
}


//-----------------------------------------------------------------------
//   setAllDofs
//-----------------------------------------------------------------------


void DenseDofSpace::setAllDofs

  ( const IdxVector&  iitems,
    const IdxVector&  itypes )

{
  JEM_PRECHECK2 ( iitems.size() == itypes.size(),
                  "Array size mismatch" );

  Data_&     d = * data_;

  idx_t*     JEM_RESTRICT  dofs;

  const idx_t  dofCount  = iitems  .size ();
  const idx_t  itemCount = d.items->size ();
  const idx_t  typeCount = d.typeCount   ();


  if ( d.dofs.size() > 0 )
  {
    clearDofs ();
  }

  d.dofs.resize ( dofCount * Data_::RECSIZE );

  dofs = d.dofs.addr ();

  for ( idx_t idof = 0; idof < dofCount;
        idof++, dofs += Data_::RECSIZE )
  {
    idx_t  iitem = iitems[idof];
    idx_t  itype = itypes[idof];

    if ( iitem < 0 || iitem >= itemCount )
    {
      d.itemIndexError ( iitem );
    }

    if ( itype < 0 || itype >= typeCount )
    {
      d.typeIndexError ( itype );
    }

    dofs[Data_::IITEM] = iitem;
    dofs[Data_::ITYPE] = itype;
  }

  d.rebuild      ();
  d.events.emit  ( this );
  d.events.reset ();
}


//-----------------------------------------------------------------------
//   getAllDofs
//-----------------------------------------------------------------------


void DenseDofSpace::getAllDofs

  ( const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( iitems.size() >= dofCount() &&
                  itypes.size() >= dofCount(),
                  "output arrays are too small" );

  const Data_&  d = * data_;

  const idx_t*  JEM_RESTRICT  dofs = d.dofs.addr ();

  const idx_t   dofCount = d.dofCount ();


  if ( iitems.stride() + itypes.stride() == 2L )
  {
    idx_t*  JEM_RESTRICT  ip = iitems.addr ();
    idx_t*  JEM_RESTRICT  tp = itypes.addr ();

    for ( idx_t idof = 0; idof < dofCount;
          idof++, dofs += Data_::RECSIZE )
    {
      ip[idof] = dofs[Data_::IITEM];
      tp[idof] = dofs[Data_::ITYPE];
    }
  }
  else
  {
    for ( idx_t idof = 0; idof < dofCount;
          idof++, dofs += Data_::RECSIZE )
    {
      iitems[idof] = dofs[Data_::IITEM];
      itypes[idof] = dofs[Data_::ITYPE];
    }
  }
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void DenseDofSpace::emitEvents_ ( int events )
{
  if ( events & Data_::NEW_SIZE )
  {
    newSizeEvent.emit ( data_->dofCount(), *this );
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void DenseDofSpace::connect_ ()
{
  using jem::util::connect;

  Data_&  d = * data_;

  connect ( d.items->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( d.items->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void DenseDofSpace::itemsResized_ ( idx_t itemCount )
{
  if ( itemCount == 0 )
  {
    clearDofs ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void DenseDofSpace::itemsReordered_ ( const Reordering& reord )
{
  Data_&        d = * data_;

  const idx_t*  JEM_RESTRICT  itemMap = reord.fmapPtr ();
  idx_t*        JEM_RESTRICT  dofs    = d.dofs.addr   ();

  const idx_t   dofCount = d.dofCount ();

  IdxVector     dofMap   ( dofCount );

  bool          changed  = false;
  idx_t         jdof     = 0;


  for ( idx_t idof = 0; idof < dofCount; idof++ )
  {
    idx_t  i     = idof * Data_::RECSIZE;
    idx_t  iitem = dofs[i + Data_::IITEM];
    idx_t  jitem = itemMap[iitem];

    if ( jitem >= 0 )
    {
      idx_t  j = jdof * Data_::RECSIZE;

      dofs[j + Data_::IITEM] = jitem;
      dofs[j + Data_::ITYPE] = dofs[i + Data_::ITYPE];

      dofMap[idof] = jdof++;
    }
    else
    {
      dofMap[idof] = -1;
    }

    if ( iitem != jitem )
    {
      changed = true;
    }
  }

  if ( jdof < dofCount )
  {
    d.dofs.resize ( jdof * Data_::RECSIZE );
  }

  if ( changed )
  {
    d.rebuild ();
  }
  else
  {
    d.resize  ();
  }

  if ( jdof < dofCount )
  {
    d.events.reset ();

    newOrderEvent.emit ( Reordering( dofMap ), *this );
    newSizeEvent .emit ( jdof,                 *this );
  }
}


JIVE_END_PACKAGE( util )
