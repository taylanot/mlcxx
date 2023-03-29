
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
#include <jem/util/SparseArray.h>
#include <jive/util/private/NameList.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/EventMask.h>
#include <jive/util/Reordering.h>
#include <jive/util/SparseDofSpace.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::SparseDofSpace );


JIVE_BEGIN_PACKAGE( util )


using jem::min;
using jem::shape;
using jem::newInstance;
using jem::util::Flex;
using jem::util::SparseArray;


//=======================================================================
//   class SparseDofSpace::Data_
//=======================================================================


class SparseDofSpace::Data_ : public jem::Collectable
{
 public:

  typedef Data_           Self;
  typedef
    SparseArray<idx_t,2>    DofMap;


  enum                    DofRecord
  {
                            IITEM,
                            ITYPE,
                            NEXT,
                            DOWN,
                            RECSIZE
  };

  static const int        NEW_SIZE = 1 << 0;


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
  DofMap                  dmap;

  Flex<idx_t>             dofs;
  Flex<idx_t>             ilists;
  Flex<idx_t>             tlists;

  EventMask               events;


 private:

  void                    checkItemIndex_

    ( idx_t                 iitem );

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline SparseDofSpace::Data_::Data_ ()
{}


SparseDofSpace::Data_::Data_ ( const Ref<ItemSet>& iset ) :

  items ( iset )

{
  resize         ();
  tlists.reserve ( 4 );
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


inline void SparseDofSpace::Data_::readFrom ( ObjectInput& in )
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


inline void SparseDofSpace::Data_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, items, typeNames, dofs );
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


inline void SparseDofSpace::Data_::reserve ( idx_t n )
{
  dmap.reserve ( n );
  dofs.reserve ( n * RECSIZE );
  resize       ();
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void SparseDofSpace::Data_::resize ()
{
  const idx_t  m = items->size ();
  const idx_t  n = typeCount   ();

  if      ( m != dmap.size(0) )
  {
    dmap.reshape ( shape( m, n + 2 ) );
  }

  if      ( m > ilists.size() )
  {
    ilists.pushBack ( -1, m - ilists.size() );
  }
  else if ( m < ilists.size() )
  {
    ilists.popBack  ( ilists.size() - m );
  }

  if      ( n > tlists.size() )
  {
    tlists.pushBack ( -1, n - tlists.size() );
  }
  else if ( n < tlists.size() )
  {
    tlists.popBack  ( tlists.size() - n );
  }
}


//-----------------------------------------------------------------------
//   rebuild
//-----------------------------------------------------------------------


void SparseDofSpace::Data_::rebuild ()
{
  idx_t*       JEM_RESTRICT  dp = dofs.addr ();
  idx_t*       JEM_RESTRICT  ip;
  idx_t*       JEM_RESTRICT  tp;

  const idx_t  dofCount  = dofs.size() / RECSIZE;
  const idx_t  itemCount = items   ->size ();
  const idx_t  typeCount = typeNames.size ();


  dmap  .clear   ();
  dmap  .reserve ( dofCount );
  dmap  .reshape ( shape( itemCount, typeCount ) );

  ilists.resize  ( itemCount );
  tlists.resize  ( typeCount );

  ilists = -1;
  tlists = -1;

  ip     = ilists.addr ();
  tp     = tlists.addr ();

  for ( idx_t idof = 0; idof < dofCount; idof++, dp += RECSIZE )
  {
    idx_t  iitem = dp[IITEM];
    idx_t  itype = dp[ITYPE];

    dp[NEXT]   = ip[iitem];
    dp[DOWN]   = tp[itype];
    ip[iitem]  = idof;
    tp[itype]  = idof;

    dmap(iitem,itype) = idof;
  }
}


//-----------------------------------------------------------------------
//   typeCount
//-----------------------------------------------------------------------


inline idx_t SparseDofSpace::Data_::typeCount () const
{
  return typeNames.size ();
}


//-----------------------------------------------------------------------
//   dofCount
//-----------------------------------------------------------------------


inline idx_t SparseDofSpace::Data_::dofCount () const
{
  return dmap.nonZeroCount ();
}


//-----------------------------------------------------------------------
//   checkItemIndex
//-----------------------------------------------------------------------


inline void SparseDofSpace::Data_::checkItemIndex ( idx_t iitem )
{
  if ( iitem < 0 || iitem >= ilists.size() )
  {
    checkItemIndex_ ( iitem );
  }
}


inline bool SparseDofSpace::Data_::checkItemIndex ( idx_t iitem ) const
{
  if ( iitem < 0 )
  {
    itemIndexError ( iitem );
  }

  return (iitem < ilists.size());
}


//-----------------------------------------------------------------------
//   checkTypeIndex
//-----------------------------------------------------------------------


inline void SparseDofSpace::Data_::checkTypeIndex ( idx_t itype )
{
  if ( itype < 0 || itype >= typeNames.size() )
  {
    typeIndexError ( itype );
  }

  if ( itype >= tlists.size() )
  {
    resize ();
  }
}


inline bool SparseDofSpace::Data_::checkTypeIndex ( idx_t itype ) const
{
  if ( itype < 0 || itype >= typeNames.size() )
  {
    typeIndexError ( itype );
  }

  return (itype < tlists.size());
}


//-----------------------------------------------------------------------
//   checkDofIndex
//-----------------------------------------------------------------------


inline void SparseDofSpace::Data_::checkDofIndex ( idx_t idof  ) const
{
  if ( idof < 0 || idof >= dofCount() )
  {
    dofIndexError ( idof );
  }
}


//-----------------------------------------------------------------------
//   addDof
//-----------------------------------------------------------------------


inline idx_t SparseDofSpace::Data_::addDof

  ( idx_t  iitem,
    idx_t  itype )

{
  idx_t*  iptr = dmap.find ( iitem, itype );

  if ( iptr )
  {
    return *iptr;
  }

  idx_t  idof = dmap.nonZeroCount ();

  dmap(iitem,itype) = idof;

  dofs.pushBack ( iitem );
  dofs.pushBack ( itype );
  dofs.pushBack ( ilists[iitem] );
  dofs.pushBack ( tlists[itype] );

  ilists[iitem] = idof;
  tlists[itype] = idof;

  events.set ( NEW_SIZE );

  return idof;
}


//-----------------------------------------------------------------------
//   itemIndexError
//-----------------------------------------------------------------------


void SparseDofSpace::Data_::itemIndexError ( idx_t iitem ) const
{
  indexError ( CLASS_NAME,
               items->getItemName(), iitem, items->size() );
}


//-----------------------------------------------------------------------
//   typeIndexError
//-----------------------------------------------------------------------


void SparseDofSpace::Data_::typeIndexError ( idx_t itype ) const
{
  indexError ( CLASS_NAME, "DOF type", itype, typeCount() );
}


//-----------------------------------------------------------------------
//   dofIndexError
//-----------------------------------------------------------------------


void SparseDofSpace::Data_::dofIndexError ( idx_t idof ) const
{
  indexError ( CLASS_NAME, "DOF", idof, dofCount() );
}


//-----------------------------------------------------------------------
//   checkItemIndex_
//-----------------------------------------------------------------------


void SparseDofSpace::Data_::checkItemIndex_ ( idx_t iitem )
{
  const idx_t  itemCount = items->size ();

  if ( iitem < 0 || iitem >= itemCount )
  {
    itemIndexError ( iitem );
  }

  if ( iitem >= ilists.size() )
  {
    resize ();
  }
}


//=======================================================================
//   class SparseDofSpace
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SparseDofSpace::SparseDofSpace ()
{}


SparseDofSpace::SparseDofSpace ( const Ref<ItemSet>& items )
{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  data_ = newInstance<Data_> ( items );

  connect_ ();
}


SparseDofSpace::SparseDofSpace ( const Self& rhs )
{
  data_ = newInstance<Data_> ( * rhs.data_ );

  connect_ ();
}


SparseDofSpace::~SparseDofSpace ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SparseDofSpace::resetEvents ()
{
  data_->events.reset       ();
  data_->items->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void SparseDofSpace::readFrom ( ObjectInput& in )
{
  data_ = newInstance<Data_> ();

  data_->readFrom  ( in );

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void SparseDofSpace::writeTo ( ObjectOutput& out ) const
{
  data_->writeTo ( out );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> SparseDofSpace::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   typeCount
//-----------------------------------------------------------------------


idx_t SparseDofSpace::typeCount () const
{
  return data_->typeCount ();
}


//-----------------------------------------------------------------------
//   dofCount
//-----------------------------------------------------------------------


idx_t SparseDofSpace::dofCount () const
{
  return data_->dofCount ();
}


//-----------------------------------------------------------------------
//   clearDofs
//-----------------------------------------------------------------------


void SparseDofSpace::clearDofs ()
{
  Data_&  d = * data_;

  if ( d.dofs.size() > 0 )
  {
    d.dmap  .clear ();
    d.dofs  .clear ();
    d.ilists.clear ();
    d.events.reset ();

    d.tlists = -1;

    newSizeEvent.emit ( 0, *this );
  }
}


//-----------------------------------------------------------------------
//   clearAll
//-----------------------------------------------------------------------


void SparseDofSpace::clearAll ()
{
  data_->typeNames.clear ();
  data_->tlists   .clear ();

  clearDofs ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void SparseDofSpace::reserve ( idx_t n )
{
  data_->reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void SparseDofSpace::trimToSize ()
{
  Data_&  d = * data_;

  d.dmap  .trimToSize ();
  d.dofs  .trimToSize ();
  d.ilists.trimToSize ();
  d.tlists.trimToSize ();
}


//-----------------------------------------------------------------------
//   addType
//-----------------------------------------------------------------------


idx_t SparseDofSpace::addType ( const String& name )
{
  return data_->typeNames.addName ( name );
}


//-----------------------------------------------------------------------
//   findType
//-----------------------------------------------------------------------


idx_t SparseDofSpace::findType ( const String& name ) const
{
  return data_->typeNames.findName ( name );
}


//-----------------------------------------------------------------------
//   getTypeName
//-----------------------------------------------------------------------


String SparseDofSpace::getTypeName ( idx_t itype ) const
{
  const Data_&  d= * data_;

  d.checkTypeIndex ( itype );

  return d.typeNames.getName ( itype );
}


//-----------------------------------------------------------------------
//   getTypeNames
//-----------------------------------------------------------------------


StringVector SparseDofSpace::getTypeNames () const
{
  return data_->typeNames.toArray ();
}


//-----------------------------------------------------------------------
//   addDof
//-----------------------------------------------------------------------


idx_t SparseDofSpace::addDof

  ( idx_t  iitem,
    idx_t  itype )

{
  Data_&  d = * data_;

  d.checkItemIndex ( iitem );
  d.checkTypeIndex ( itype );

  const idx_t  idof = d.addDof ( iitem, itype );

  d.events.emit ( this );

  return idof;
}


//-----------------------------------------------------------------------
//   addDofs
//-----------------------------------------------------------------------


void SparseDofSpace::addDofs

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


void SparseDofSpace::reorderDofs ( const Reordering& reord )
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

    idx_t* JEM_RESTRICT    newDofs = tmpDofs.addr ();


    if ( start > 0 )
    {
      std::memcpy ( newDofs, dofs, (size_t)
                    (start * Data_::RECSIZE) * sizeof(idx_t) );

      newDofs += (start * Data_::RECSIZE);
    }

    for ( idx_t idof = start; idof < newSize;
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

  if ( reord.oldSize() != reord.newSize() )
  {
    newSizeEvent.emit ( reord.newSize(), *this );
  }
}


//-----------------------------------------------------------------------
//   findDofIndex
//-----------------------------------------------------------------------


idx_t SparseDofSpace::findDofIndex

  ( idx_t  iitem,
    idx_t  itype ) const

{
  const Data_&  d = * data_;

  d.checkItemIndex ( iitem );
  d.checkTypeIndex ( itype );

  const idx_t*  iptr = d.dmap.find ( iitem, itype );

  if ( iptr )
  {
    return *iptr;
  }
  else
  {
    return -1_idx;
  }
}


//-----------------------------------------------------------------------
//   findDofIndices
//-----------------------------------------------------------------------


idx_t SparseDofSpace::findDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( idofs.size() >= iitems.size() * itypes.size(),
                  "DOF index array is too small" );

  const Data_&  d = * data_;

  const idx_t   m = iitems.size ();
  const idx_t   n = itypes.size ();

  const idx_t*  iptr;


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
      iptr = d.dmap.find ( iitem, itypes[j] );

      if ( iptr )
      {
        idofs[i * n + j] = *iptr;
        k++;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   collectDofIndices
//-----------------------------------------------------------------------


idx_t SparseDofSpace::collectDofIndices

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    const IdxVector&  itypes ) const

{
  JEM_PRECHECK2 ( idofs.size() >= iitems.size() * itypes.size(),
                  "DOF index array is too small" );

  const Data_&  d = * data_;

  const idx_t   m = iitems.size ();
  const idx_t   n = itypes.size ();

  const idx_t*  iptr;


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
      iptr = d.dmap.find ( iitem, itypes[j] );

      if ( iptr )
      {
        idofs[k++] = *iptr;
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofsForItem
//-----------------------------------------------------------------------


idx_t SparseDofSpace::getDofsForItem

  ( const IdxVector&  idofs,
    const IdxVector&  itypes,
    idx_t             iitem ) const

{
  const Data_&  d    = * data_;

  const idx_t   n    = min ( idofs.size(), itypes.size() );
  const idx_t*  dofs = d.dofs.addr ();


  if ( ! d.checkItemIndex( iitem ) )
  {
    return 0;
  }

  idx_t  idof = d.ilists[iitem];
  idx_t  k    = 0;

  while ( idof >= 0 && k < n )
  {
    idofs [k] = idof;
    itypes[k] = dofs[idof * Data_::RECSIZE + Data_::ITYPE];
    idof      = dofs[idof * Data_::RECSIZE + Data_::NEXT];
    k++;
  }

  return k;
}


//-----------------------------------------------------------------------
//   getDofsForType
//-----------------------------------------------------------------------


idx_t SparseDofSpace::getDofsForType

  ( const IdxVector&  idofs,
    const IdxVector&  iitems,
    idx_t             itype ) const

{
  const Data_&  d    = * data_;

  const idx_t   n    = min ( idofs.size(), iitems.size() );
  const idx_t*  dofs = d.dofs.addr ();


  if ( ! d.checkTypeIndex( itype ) )
  {
    return 0;
  }

  idx_t  idof = d.tlists[itype];
  idx_t  k    = 0;

  while ( idof >= 0 && k < n )
  {
    idofs [k] = idof;
    iitems[k] = dofs[idof * Data_::RECSIZE + Data_::IITEM];
    idof      = dofs[idof * Data_::RECSIZE + Data_::DOWN];
    k++;
  }

  return k;
}


//-----------------------------------------------------------------------
//   decodeDofIndex
//-----------------------------------------------------------------------


void SparseDofSpace::decodeDofIndex

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


void SparseDofSpace::decodeDofIndices

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


SparseIdxMatrix SparseDofSpace::toMatrix () const
{
  const Data_&  d = * data_;

  const idx_t*  dofs      = d.dofs.addr   ();
  const idx_t   dofCount  = d.dofCount    ();
  const idx_t   itemCount = d.items->size ();
  const idx_t   typeCount = d.typeCount   ();

  IdxVector     ibuf      ( itemCount + 1 + 2 * dofCount );

  IdxVector     itemOffsets;
  IdxVector     typeIndices;
  IdxVector     dofIndices;

  idx_t         j, n;


  n = itemCount + 1;

  itemOffsets.ref ( ibuf[slice(BEGIN,n)]          );
  typeIndices.ref ( ibuf[slice(n,n + dofCount)]   );
  dofIndices .ref ( ibuf[slice(n + dofCount,END)] );

  n = min ( itemCount, d.ilists.size() );
  j = 0;

  for ( idx_t iitem = 0; iitem < n; iitem++ )
  {
    itemOffsets[iitem] = j;

    idx_t  idof = d.ilists[iitem];

    while ( idof >= 0 )
    {
      idx_t  i       = idof * Data_::RECSIZE;

      typeIndices[j] = dofs[i + Data_::ITYPE];
      dofIndices [j] = idof;
      idof           = dofs[i + Data_::NEXT];

      j++;
    }
  }

  itemOffsets[slice(n,END)] = j;

  return SparseIdxMatrix ( shape ( itemCount, typeCount ),
                           itemOffsets,
                           typeIndices,
                           dofIndices );
}


//-----------------------------------------------------------------------
//   setAllDofs
//-----------------------------------------------------------------------


void SparseDofSpace::setAllDofs

  ( const IdxVector&  iitems,
    const IdxVector&  itypes )

{
  JEM_PRECHECK2 ( iitems.size() == itypes.size(),
                  "Array size mismatch" );

  Data_&       d = * data_;

  idx_t*       JEM_RESTRICT  dofs;

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


void SparseDofSpace::getAllDofs

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
//   getItems
//-----------------------------------------------------------------------


ItemSet* SparseDofSpace::getItems () const
{
  return data_->items.get ();
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void SparseDofSpace::emitEvents_ ( int events )
{
  if ( events & Data_::NEW_SIZE )
  {
    newSizeEvent.emit ( data_->dofCount(), *this );
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void SparseDofSpace::connect_ ()
{
  using jem::util::connect;

  Data_&  d = * data_;

  connect ( d.items->newSizeEvent,  this, & Self::itemsResized_   );
  connect ( d.items->newOrderEvent, this, & Self::itemsReordered_ );
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void SparseDofSpace::itemsResized_ ( idx_t itemCount )
{
  if ( itemCount == 0 )
  {
    clearDofs ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void SparseDofSpace::itemsReordered_ ( const Reordering& reord )
{
  Data_&        d = * data_;

  const idx_t*  JEM_RESTRICT  itemMap = reord.fmapPtr ();
  idx_t*        JEM_RESTRICT  dofs    = d.dofs.addr   ();

  const idx_t   dofCount = d.dofCount ();

  IdxVector     dofMap   ( dofCount );

  bool          changed = false;
  idx_t         jdof    = 0;


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
